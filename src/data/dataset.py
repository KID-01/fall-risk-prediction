"""
PyTorch Dataset 与 DataLoader — 支持多模态数据加载、数据增强、批处理
FallRiskDataset: 加载关键点时序序列,返回 dict
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.keypoint_store import KeypointStore
from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)

# MediaPipe Pose 左右对称关键点对 (用于水平翻转)
LEFT_RIGHT_SWAP = [
    (11, 12),  # shoulder
    (13, 14),  # elbow
    (15, 16),  # wrist
    (23, 24),  # hip
    (25, 26),  # knee
    (27, 28),  # ankle
    (29, 30),  # heel
    (31, 32),  # foot_index
]


class FallRiskDataset(Dataset):
    """
    跌倒风险数据集

    __getitem__ 返回:
        {
            "keypoints": (T, 33, 3),       # 时序关键点 [x, y, confidence]
            "risk_score": float,            # 风险评分 [0, 100]
            "risk_level": int,              # 风险等级 0-3 (绿/黄/橙/红)
        }
    """

    def __init__(
        self,
        keypoint_dir: str = "data/keypoints",
        labels: dict[str, dict] | None = None,
        seq_len: int = 90,
        augment: bool = False,
        augment_config: dict | None = None,
    ):
        """
        Args:
            keypoint_dir: 关键点 .npy 文件目录
            labels: {文件名(无扩展名): {"risk_score": float, "risk_level": int}} 标签字典
            seq_len: 时序窗口长度(帧数), 默认90帧≈6秒@15fps
            augment: 是否启用数据增强
            augment_config: 增强参数 {jitter_sigma, flip_prob, crop_ratio, time_scale_prob}
        """
        self.store = KeypointStore(keypoint_dir)
        self.labels = labels or {}
        self.seq_len = seq_len
        self.augment = augment
        self.augment_config = augment_config or {
            "jitter_sigma": 0.005,
            "flip_prob": 0.5,
            "crop_ratio": (0.8, 1.0),
            "time_scale_prob": 0.3,
        }

        self.files = self.store.list_all()
        log.info(f"FallRiskDataset: {len(self.files)} 个样本, seq_len={seq_len}, augment={augment}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        name = self.files[idx]
        data = self.store.load(name)  # (T_orig, 33, 4)

        # 取 [x, y, confidence] 三列
        kps = data[:, :, :3]  # (T_orig, 33, 3)

        # 时序采样: 固定长度窗口
        kps = self._sample_sequence(kps, self.seq_len)

        # 数据增强
        if self.augment:
            kps = self._augment(kps)

        # 标签
        label_info = self.labels.get(name, {"risk_score": 0.0, "risk_level": 0})

        return {
            "keypoints": torch.from_numpy(kps).float(),
            "risk_score": torch.tensor(label_info.get("risk_score", 0.0), dtype=torch.float32),
            "risk_level": torch.tensor(label_info.get("risk_level", 0), dtype=torch.long),
            "name": name,
        }

    def _sample_sequence(self, kps: np.ndarray, seq_len: int) -> np.ndarray:
        """时序采样: 随机起点截取固定长度窗口,不足则重复padding"""
        T = kps.shape[0]
        if T >= seq_len:
            start = np.random.randint(0, T - seq_len + 1)
            return kps[start : start + seq_len]
        else:
            # 不足则循环重复
            repeats = (seq_len // T) + 1
            kps_repeated = np.tile(kps, (repeats, 1, 1))[:seq_len]
            return kps_repeated

    def _augment(self, kps: np.ndarray) -> np.ndarray:
        """数据增强: 关键点抖动 + 水平翻转 + 时序裁剪 + 时间缩放"""
        cfg = self.augment_config

        # 1. 关键点抖动 (高斯噪声)
        sigma = cfg.get("jitter_sigma", 0.005)
        noise = np.random.normal(0, sigma, kps.shape[:2] + (2,))  # 仅对x,y加噪
        kps = kps.copy()
        kps[:, :, :2] += noise

        # 2. 水平翻转 (50%概率)
        if np.random.random() < cfg.get("flip_prob", 0.5):
            kps = self._horizontal_flip(kps)

        # 3. 时序裁剪 (随机截取80%-100%长度)
        crop_min, crop_max = cfg.get("crop_ratio", (0.8, 1.0))
        crop_ratio = np.random.uniform(crop_min, crop_max)
        crop_len = int(kps.shape[0] * crop_ratio)
        if crop_len < kps.shape[0]:
            start = np.random.randint(0, kps.shape[0] - crop_len + 1)
            cropped = kps[start : start + crop_len]
            # resize回原始长度
            kps = self._interpolate_time(cropped, kps.shape[0])

        # 4. 时间缩放 (随机加速/减速)
        if np.random.random() < cfg.get("time_scale_prob", 0.3):
            scale = np.random.uniform(0.8, 1.2)
            new_len = max(1, int(kps.shape[0] * scale))
            kps = self._interpolate_time(kps, self.seq_len)

        return kps

    def _horizontal_flip(self, kps: np.ndarray) -> np.ndarray:
        """水平翻转关键点 (x坐标翻转 + 左右关节交换)"""
        kps = kps.copy()
        kps[:, :, 0] = 1.0 - kps[:, :, 0]  # x翻转
        # 交换左右关节
        for left_idx, right_idx in LEFT_RIGHT_SWAP:
            kps[:, [left_idx, right_idx]] = kps[:, [right_idx, left_idx]]
        return kps

    def _interpolate_time(self, kps: np.ndarray, target_len: int) -> np.ndarray:
        """时间维度线性插值"""
        T = kps.shape[0]
        if T == target_len:
            return kps
        # 为每个关键点的每个坐标做插值
        result = np.zeros((target_len, kps.shape[1], kps.shape[2]), dtype=kps.dtype)
        for j in range(kps.shape[1]):
            for c in range(kps.shape[2]):
                result[:, j, c] = np.interp(
                    np.linspace(0, T - 1, target_len),
                    np.arange(T),
                    kps[:, j, c],
                )
        return result


def create_dataloaders(
    keypoint_dir: str = "data/keypoints",
    labels: dict | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seq_len: int = 90,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练/验证/测试 DataLoader
    按文件名列表分层划分(非被试者ID分层,简化版)
    """
    from torch.utils.data import random_split

    full_dataset = FallRiskDataset(
        keypoint_dir=keypoint_dir,
        labels=labels,
        seq_len=seq_len,
        augment=False,
    )

    n = len(full_dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    # 训练集启用增强
    train_ds.dataset.augment = True

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    log.info(f"DataLoader: train={n_train}, val={n_val}, test={n_test}")
    return train_loader, val_loader, test_loader
