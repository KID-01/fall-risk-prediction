"""
风险评分回归头 + 风险等级分类头 — 双任务输出

回归头: MLP (256→128→64→1) + Sigmoid → [0, 100] 风险评分
分类头: MLP (256→128→4) + Softmax → 4类概率 (绿/黄/橙/红)

多任务损失: α*MSE + β*CrossEntropy + γ*OrdinalLoss
"""
from __future__ import annotations

import torch
import torch.nn as nn


class OrdinalLoss(nn.Module):
    """
    序数损失: 确保预测的风险等级顺序正确
    对于4级分类,转化为3个二分类(>0, >1, >2)
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 4) 分类logits
            targets: (B,) 等级标签 0-3
        """
        # 将等级标签转为序数编码
        B = targets.size(0)
        ordinal_targets = torch.zeros(B, self.num_classes - 1, device=targets.device)
        for k in range(self.num_classes - 1):
            ordinal_targets[:, k] = (targets > k).float()

        # 取logits的累积概率(简化: 用前num_classes-1个logits)
        ordinal_logits = logits[:, : self.num_classes - 1]
        return self.bce(ordinal_logits, ordinal_targets)


class RiskPredictionHead(nn.Module):
    """
    风险预测双任务头

    输入: fused_feat (B, 256)
    输出: {"risk_score": (B,1) [0,100], "risk_probs": (B,4) 绿/黄/橙/红}
    """

    def __init__(
        self,
        input_dim: int = 256,
        num_classes: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 回归头: 256→128→64→1
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # 输出 [0, 1], 后续乘100
        )

        # 分类头: 256→128→4
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, fused_feat: torch.Tensor) -> dict:
        """
        Args:
            fused_feat: (B, 256)
        Returns:
            {"risk_score": (B,1) [0,100], "risk_probs": (B,4), "risk_logits": (B,4)}
        """
        risk_score = self.regressor(fused_feat) * 100.0  # (B, 1) [0, 100]
        risk_logits = self.classifier(fused_feat)         # (B, 4)
        risk_probs = torch.softmax(risk_logits, dim=-1)  # (B, 4)

        return {
            "risk_score": risk_score,
            "risk_probs": risk_probs,
            "risk_logits": risk_logits,
        }


class MultiTaskLoss(nn.Module):
    """多任务损失: α*MSE + β*CrossEntropy + γ*OrdinalLoss"""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.ordinal = OrdinalLoss()

    def forward(
        self,
        predictions: dict,
        risk_score_target: torch.Tensor,
        risk_level_target: torch.Tensor,
    ) -> dict:
        """
        Args:
            predictions: {"risk_score": (B,1), "risk_logits": (B,4)}
            risk_score_target: (B,) 真实评分 [0, 100]
            risk_level_target: (B,) 真实等级 0-3
        Returns:
            {"loss": total, "mse": ..., "ce": ..., "ordinal": ...}
        """
        mse_loss = self.mse(predictions["risk_score"].squeeze(-1), risk_score_target)
        ce_loss = self.ce(predictions["risk_logits"], risk_level_target)
        ord_loss = self.ordinal(predictions["risk_logits"], risk_level_target)

        total = self.alpha * mse_loss + self.beta * ce_loss + self.gamma * ord_loss

        return {
            "loss": total,
            "mse": mse_loss.item(),
            "ce": ce_loss.item(),
            "ordinal": ord_loss.item(),
        }
