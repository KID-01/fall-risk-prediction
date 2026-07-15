"""
时序特征编码器 — Transformer Encoder
从骨骼关键点时序序列中提取时空模式

架构:
  输入 (B, T, 33, 3) → 展平 (B, T, 99) → 线性投影 (B, T, 256)
  → 可学习位置编码 → 4层Transformer Encoder (8头注意力)
  → 输出 (B, T, 256) 时序特征 + (B, 256) 全局CLS token
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """可学习的正弦位置编码"""

    def __init__(self, d_model: int = 256, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        return x + self.pe[:, : x.size(1)]


class TemporalEncoder(nn.Module):
    """
    Transformer时序编码器

    输入: 关键点序列 (B, T, 33, 3)
    输出: (seq_features (B,T,256), global_feat (B,256))
    """

    def __init__(
        self,
        input_dim: int = 99,       # 33 * 3 = 99
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 500,
    ):
        super().__init__()

        self.d_model = d_model

        # 输入投影: (B, T, 99) → (B, T, 256)
        self.input_proj = nn.Linear(input_dim, d_model)

        # CLS token (用于全局特征聚合)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len + 1)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,   # FFN: 256→1024→256
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,               # Pre-LN (更稳定)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # LayerNorm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, keypoints: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            keypoints: (B, T, 33, 3) 骨骼关键点时序序列
        Returns:
            seq_features: (B, T, 256) 时序特征
            global_feat: (B, 256) 全局特征 (CLS token)
        """
        B, T, K, C = keypoints.shape

        # 展平关键点: (B, T, 33, 3) → (B, T, 99)
        x = keypoints.reshape(B, T, K * C)

        # 线性投影
        x = self.input_proj(x)  # (B, T, d_model)

        # 添加CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)           # (B, T+1, d_model)

        # 位置编码
        x = self.pos_encoding(x)

        # Transformer编码
        x = self.transformer(x)  # (B, T+1, d_model)
        x = self.norm(x)

        # 分离CLS token和序列特征
        global_feat = x[:, 0]        # (B, d_model) — CLS
        seq_features = x[:, 1:]      # (B, T, d_model)

        return seq_features, global_feat
