"""
训练管线 — 完整训练循环、学习率调度、早停、模型保存
用法: python scripts/train.py
配置: configs/base.yaml 中的 model: 和 training: 节
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from src.data.dataset import create_dataloaders
from src.models.fall_risk_predictor import FallRiskPredictor
from src.utils.config import get_config
from src.utils.logger import get_logger, setup_logging

log = get_logger(__name__)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scheduler,
    device: str,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> dict:
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_ce = 0.0
    n = 0

    for batch in loader:
        keypoints = batch["keypoints"].to(device)
        gait_feat = batch.get("gait_feat")
        env_feat = batch.get("env_feat")
        risk_score = batch["risk_score"].to(device)
        risk_level = batch["risk_level"].to(device)

        # 如果没有gait/env特征,用零向量填充
        if gait_feat is None:
            gait_feat = torch.zeros(keypoints.size(0), 4, device=device)
        else:
            gait_feat = gait_feat.to(device)
        if env_feat is None:
            env_feat = torch.zeros(keypoints.size(0), 5, device=device)
        else:
            env_feat = env_feat.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                preds = model(keypoints, gait_feat, env_feat)
                loss_dict = model.compute_loss(preds, risk_score, risk_level)
            scaler.scale(loss_dict["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(keypoints, gait_feat, env_feat)
            loss_dict = model.compute_loss(preds, risk_score, risk_level)
            loss_dict["loss"].backward()
            optimizer.step()

        bs = keypoints.size(0)
        total_loss += loss_dict["loss"].item() * bs
        total_mse += loss_dict["mse"] * bs
        total_ce += loss_dict["ce"] * bs
        n += bs

    if scheduler is not None:
        scheduler.step()

    return {
        "loss": total_loss / max(n, 1),
        "mse": total_mse / max(n, 1),
        "ce": total_ce / max(n, 1),
    }


@torch.no_grad()
def validate(model: nn.Module, loader, device: str) -> dict:
    """验证"""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    correct = 0
    n = 0

    for batch in loader:
        keypoints = batch["keypoints"].to(device)
        gait_feat = batch.get("gait_feat")
        env_feat = batch.get("env_feat")
        risk_score = batch["risk_score"].to(device)
        risk_level = batch["risk_level"].to(device)

        if gait_feat is None:
            gait_feat = torch.zeros(keypoints.size(0), 4, device=device)
        else:
            gait_feat = gait_feat.to(device)
        if env_feat is None:
            env_feat = torch.zeros(keypoints.size(0), 5, device=device)
        else:
            env_feat = env_feat.to(device)

        preds = model(keypoints, gait_feat, env_feat)
        loss_dict = model.compute_loss(preds, risk_score, risk_level)

        bs = keypoints.size(0)
        total_loss += loss_dict["loss"].item() * bs
        total_mae += torch.abs(preds["risk_score"].squeeze(-1) - risk_score).sum().item()
        correct += (preds["risk_probs"].argmax(dim=-1) == risk_level).sum().item()
        n += bs

    return {
        "loss": total_loss / max(n, 1),
        "mae": total_mae / max(n, 1),
        "acc": correct / max(n, 1),
    }


def train(args):
    """完整训练流程"""
    setup_logging()
    config = get_config()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"训练设备: {device}")

    # 数据
    train_loader, val_loader, test_loader = create_dataloaders(
        keypoint_dir=args.data_dir,
        labels=None,  # TODO: 加载真实标签
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seq_len=args.seq_len,
    )

    # 模型
    model = FallRiskPredictor(
        input_dim=99,
        d_model=config.model.d_model,
        nhead=config.model.nhead,
        num_layers=config.model.num_layers,
        encoder_dropout=config.model.dropout,
    ).to(device)

    # 优化器
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.training.weight_decay,
    )

    # 学习率调度: CosineAnnealingWarmRestarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.training.warmup_epochs * 2,
        T_mult=2,
    )

    # 混合精度
    use_amp = device == "cuda" and config.training.mixed_precision
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Checkpoint目录
    ckpt_dir = Path(config.paths.checkpoints)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_mae = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, device, scaler)
        val_metrics = validate(model, val_loader, device)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        log.info(
            f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s) lr={lr:.2e} | "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} | "
            f"val_MAE={val_metrics['mae']:.2f} val_acc={val_metrics['acc']:.4f}"
        )

        # 早停
        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_mae": best_mae,
            }, ckpt_dir / "best.pt")
            log.info(f"  → 保存最佳模型 (MAE={best_mae:.2f})")
        else:
            patience_counter += 1
            if patience_counter >= config.training.patience:
                log.info(f"早停: {config.training.patience}个epoch未改善")
                break

    log.info(f"训练完成,最佳验证MAE={best_mae:.2f}")

    # 测试集评估
    if test_loader is not None:
        test_metrics = validate(model, test_loader, device)
        log.info(f"测试集: loss={test_metrics['loss']:.4f} MAE={test_metrics['mae']:.2f} acc={test_metrics['acc']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="训练跌倒风险预测模型")
    parser.add_argument("--data-dir", default="data/keypoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=90)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
