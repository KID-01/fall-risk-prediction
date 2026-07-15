"""
多模态融合模块 — Cross-Attention融合
三路输入:
  ① 时序编码器全局特征 (B, 256) — 来自TemporalEncoder
  ② 步态手工特征 (B, D_gait) — 来自FeatureCalculator, 经MLP投影到 (B, 128)
  ③ 环境风险特征 (B, 5) — 来自环境感知模块, 经MLP投影到 (B, 64)

融合方式: 以时序特征为Query, 步态+环境特征为Key/Value, 做交叉注意力
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPProjector(nn.Module):
    """MLP投影器: 将任意维度特征投影到目标维度"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or output_dim * 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiModalFusion(nn.Module):
    """
    多模态融合: Cross-Attention

    输入:
      temporal_feat: (B, 256) — 时序编码器全局特征
      gait_feat: (B, D_gait) — 步态手工特征
      env_feat: (B, 5) — 环境风险特征

    输出:
      fused_feature: (B, 256) — 融合后特征
    """

    def __init__(
        self,
        temporal_dim: int = 256,
        gait_dim: int = 4,        # 四大相对特征
        env_dim: int = 5,         # 五维环境风险
        fusion_dim: int = 256,
        nhead: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 投影到统一维度
        self.temporal_proj = nn.Linear(temporal_dim, fusion_dim)
        self.gait_proj = MLPProjector(gait_dim, 128)
        self.env_proj = MLPProjector(env_dim, 64)

        # 融合维度 = 128 + 64 = 192 (Key/Value)
        kv_dim = 128 + 64

        # Cross-Attention: temporal为Query, gait+env为Key/Value
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=nhead,
            kdim=kv_dim,
            vdim=kv_dim,
            dropout=dropout,
            batch_first=True,
        )

        # 残差 + LayerNorm
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 4, fusion_dim),
        )

    def forward(
        self,
        temporal_feat: torch.Tensor,
        gait_feat: torch.Tensor,
        env_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            temporal_feat: (B, 256)
            gait_feat: (B, D_gait)
            env_feat: (B, 5)
        Returns:
            fused: (B, 256)
        """
        B = temporal_feat.size(0)

        # 投影
        q = self.temporal_proj(temporal_feat).unsqueeze(1)  # (B, 1, 256)

        gait = self.gait_proj(gait_feat)    # (B, 128)
        env = self.env_proj(env_feat)        # (B, 64)
        kv = torch.cat([gait, env], dim=-1).unsqueeze(1)  # (B, 1, 192)

        # Cross-Attention
        attn_out, _ = self.cross_attn(q, kv, kv)  # (B, 1, 256)

        # 残差 + LayerNorm
        x = self.norm1(q + attn_out)  # (B, 1, 256)

        # FFN + 残差
        x = self.norm2(x + self.ffn(x))  # (B, 1, 256)

        return x.squeeze(1)  # (B, 256)
