"""
双层偏离检测模块
第一层: 短期异常检测 (分钟级, 5分钟窗口, 马氏距离)
第二层: 长期趋势分析 (天级, 14天窗口, 线性回归斜率)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from src.inference.baseline import IndividualBaseline
from src.inference.features import FeatureVector
from src.utils.config import get_config


class DeviationLevel(Enum):
    """偏离程度"""

    NONE = "none"
    SHORT_TERM = "short_term"       # 短期异常
    LONG_TERM = "long_term"         # 长期趋势下降
    BOTH = "both"                   # 短期+长期同时触发


@dataclass
class DeviationResult:
    """偏离检测结果"""

    level: DeviationLevel = DeviationLevel.NONE
    mahalanobis_distance: float = 0.0       # 当前马氏距离
    short_term_triggered: bool = False      # 短期偏离是否触发
    long_term_triggered: bool = False       # 长期趋势是否触发
    z_scores: np.ndarray = field(default_factory=lambda: np.zeros(4))
    trend_slopes: np.ndarray = field(default_factory=lambda: np.zeros(4))
    detail: str = ""


# ============================================================
# 第一层: 短期异常检测
# ============================================================
class ShortTermDetector:
    """短期偏离检测: 5分钟滑动窗口,马氏距离,连续3窗口触发"""

    def __init__(self):
        config = get_config()
        st = config.deviation.short_term
        self.window_seconds = st.window_minutes * 60
        self.stride_seconds = st.stride_seconds
        self.threshold = st.threshold
        self.consecutive_limit = st.consecutive_windows
        self._window_buffer: deque[FeatureVector] = deque()
        self._consecutive_count = 0

    def add_and_check(
        self, feature: FeatureVector, baseline: IndividualBaseline
    ) -> tuple[float, bool]:
        """
        添加特征样本并检查短期偏离
        Returns: (马氏距离, 是否触发)
        """
        self._window_buffer.append(feature)

        # 清理过期样本
        cutoff = feature.timestamp - self.window_seconds
        while self._window_buffer and self._window_buffer[0].timestamp < cutoff:
            self._window_buffer.popleft()

        if not baseline.is_ready or len(self._window_buffer) < 2:
            return 0.0, False

        # 计算窗口内特征均值的马氏距离
        window_mean = np.mean(
            [f.to_array() for f in self._window_buffer], axis=0
        )
        dist = baseline.mahalanobis_distance(window_mean)

        if dist > self.threshold:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 0

        triggered = self._consecutive_count >= self.consecutive_limit
        return dist, triggered

    def get_short_term_frequency(self, window_hours: float = 1.0) -> int:
        """获取最近N小时内的短期偏离次数(简化版)"""
        return self._consecutive_count


# ============================================================
# 第二层: 长期趋势分析
# ============================================================
class LongTermDetector:
    """长期趋势检测: 14天滑动窗口,每日特征均值线性回归"""

    def __init__(self):
        config = get_config()
        lt = config.deviation.long_term
        self.window_days = lt.window_days
        self.slope_threshold = lt.slope_threshold
        self.min_negative_days = lt.min_negative_days
        self._daily_means: list[tuple[float, np.ndarray]] = []  # (day_index, feature_mean)

    def add_daily_mean(self, day_index: float, feature_mean: np.ndarray):
        """添加每日特征均值"""
        self._daily_means.append((day_index, feature_mean))
        # 保留窗口内的数据
        if len(self._daily_means) > self.window_days:
            self._daily_means = self._daily_means[-self.window_days:]

    def check_trend(self) -> tuple[bool, np.ndarray]:
        """
        检查长期趋势
        Returns: (是否触发, 各特征斜率)
        """
        if len(self._daily_means) < self.min_negative_days:
            return False, np.zeros(4)

        days = np.array([d[0] for d in self._daily_means])
        features = np.array([d[1] for d in self._daily_means])  # (N, 4)

        # 对每个特征做线性回归
        slopes = np.zeros(4)
        for i in range(4):
            y = features[:, i]
            if len(y) >= 2:
                slope = np.polyfit(days, y, 1)[0]
                slopes[i] = slope

        # 判断: 斜率负向变化超过阈值
        negative_count = np.sum(slopes < self.slope_threshold)
        triggered = negative_count > 0

        return triggered, slopes


# ============================================================
# 双层检测总管
# ============================================================
class DeviationDetector:
    """双层偏离检测总管"""

    def __init__(self):
        self.short_term = ShortTermDetector()
        self.long_term = LongTermDetector()

    def check(
        self,
        feature: FeatureVector,
        baseline: IndividualBaseline,
    ) -> DeviationResult:
        """综合检测短期+长期偏离"""
        # 短期检测
        dist, short_triggered = self.short_term.add_and_check(feature, baseline)

        # 长期检测
        long_triggered, slopes = self.long_term.check_trend()

        # Z-Score
        z_scores = baseline.z_scores(feature) if baseline.is_ready else np.zeros(4)

        # 综合等级
        if short_triggered and long_triggered:
            level = DeviationLevel.BOTH
        elif short_triggered:
            level = DeviationLevel.SHORT_TERM
        elif long_triggered:
            level = DeviationLevel.LONG_TERM
        else:
            level = DeviationLevel.NONE

        detail_parts = []
        if short_triggered:
            detail_parts.append(f"短期偏离(马氏距离={dist:.2f}>{self.short_term.threshold})")
        if long_triggered:
            detail_parts.append(f"长期趋势下降(斜率={slopes})")

        return DeviationResult(
            level=level,
            mahalanobis_distance=dist,
            short_term_triggered=short_triggered,
            long_term_triggered=long_triggered,
            z_scores=z_scores,
            trend_slopes=slopes,
            detail="; ".join(detail_parts) if detail_parts else "正常",
        )
