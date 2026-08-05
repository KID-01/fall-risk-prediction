"""
完整模型组装: FallRiskPredictor
将 TemporalEncoder + MultiModalFusion + RiskPredictionHead 组装为端到端模型

前向传播:
  keypoints (B,T,33,3) → TemporalEncoder → (seq_feat, global_feat)
  global_feat + gait_feat + env_feat → MultiModalFusion → fused_feat
  fused_feat → RiskPredictionHead → {"risk_score", "risk_probs"}
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.multimodal_fusion import MultiModalFusion
from src.models.risk_head import MultiTaskLoss, RiskPredictionHead
from src.models.temporal_encoder import TemporalEncoder


class FallRiskPredictor(nn.Module):
    """
    端到端跌倒风险预测模型

    输入:
      keypoints: (B, T, 33, 3) — 骨骼关键点时序序列
      gait_feat: (B, 4) — 四大步态手工特征
      env_feat: (B, 5) — 五维环境风险特征

    输出:
      {"risk_score": (B,1) [0,100], "risk_probs": (B,4), "risk_logits": (B,4)}
    """

    def __init__(
        self,
        # TemporalEncoder 参数
        input_dim: int = 99,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        encoder_dropout: float = 0.1,
        # MultiModalFusion 参数
        gait_dim: int = 4,
        env_dim: int = 5,
        fusion_dropout: float = 0.1,
        # RiskPredictionHead 参数
        head_dropout: float = 0.1,
        num_classes: int = 4,
    ):
        super().__init__()

        self.temporal_encoder = TemporalEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=encoder_dropout,
        )

        self.fusion = MultiModalFusion(
            temporal_dim=d_model,
            gait_dim=gait_dim,
            env_dim=env_dim,
            fusion_dim=d_model,
            nhead=nhead,
            dropout=fusion_dropout,
        )

        self.risk_head = RiskPredictionHead(
            input_dim=d_model,
            num_classes=num_classes,
            dropout=head_dropout,
        )

    def forward(
        self,
        keypoints: torch.Tensor,
        gait_feat: torch.Tensor,
        env_feat: torch.Tensor,
    ) -> dict:
        """
        Args:
            keypoints: (B, T, 33, 3)
            gait_feat: (B, 4)
            env_feat: (B, 5)
        Returns:
            {"risk_score": (B,1), "risk_probs": (B,4), "risk_logits": (B,4)}
        """
        # 时序编码
        _, global_feat = self.temporal_encoder(keypoints)  # (B, 256)

        # 多模态融合
        fused = self.fusion(global_feat, gait_feat, env_feat)  # (B, 256)

        # 风险预测
        output = self.risk_head(fused)

        return output

    def compute_loss(
        self,
        predictions: dict,
        risk_score_target: torch.Tensor,
        risk_level_target: torch.Tensor,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.3,
    ) -> dict:
        """计算多任务损失"""
        loss_fn = MultiTaskLoss(alpha=alpha, beta=beta, gamma=gamma)
        return loss_fn(predictions, risk_score_target, risk_level_target)

    @torch.no_grad()
    def predict(
        self,
        keypoints: torch.Tensor,
        gait_feat: torch.Tensor,
        env_feat: torch.Tensor,
    ) -> dict:
        """推理模式预测"""
        self.eval()
        output = self.forward(keypoints, gait_feat, env_feat)
        return {
            "risk_score": output["risk_score"].squeeze(-1).cpu().numpy(),
            "risk_probs": output["risk_probs"].cpu().numpy(),
            "risk_level": output["risk_probs"].argmax(dim=-1).cpu().numpy(),
        }
