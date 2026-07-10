"""
分级预警引擎 — 四级风险分类与响应

风险等级:
  低风险   — 所有特征在基线±1个标准差范围内 → 持续监测
  关注级   — 短期偏离频繁(≥3次/小时) → APP推送
  预警级   — 长期趋势连续7天负向变化 → 短信通知
  高危级   — 近似跌倒/4小时无活动 → 电话通知
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable

from src.inference.deviation import DeviationLevel, DeviationResult
from src.utils.config import get_config


class RiskLevel(Enum):
    """风险等级"""

    LOW = "low"
    ATTENTION = "attention"
    WARNING = "warning"
    CRITICAL = "critical"

    @property
    def label(self) -> str:
        labels = {
            RiskLevel.LOW: "低风险",
            RiskLevel.ATTENTION: "关注级",
            RiskLevel.WARNING: "预警级",
            RiskLevel.CRITICAL: "高危级",
        }
        return labels[self]

    @property
    def priority(self) -> int:
        return {
            RiskLevel.LOW: 0,
            RiskLevel.ATTENTION: 1,
            RiskLevel.WARNING: 2,
            RiskLevel.CRITICAL: 3,
        }[self]


@dataclass
class AlertEvent:
    """预警事件"""

    level: RiskLevel
    timestamp: float
    message: str
    deviation: DeviationResult | None = None
    video_clip_path: str | None = None
    notified: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


# 响应动作类型
AlertAction = Callable[[AlertEvent], None]


class AlertEngine:
    """分级预警引擎"""

    def __init__(self):
        config = get_config()
        alert_cfg = config.alert
        self.short_term_freq_threshold = alert_cfg.short_term_freq_threshold
        self.inactivity_threshold_minutes = alert_cfg.inactivity_threshold_minutes
        self.video_clip_enabled = alert_cfg.video_clip.enabled
        self.video_clip_before = alert_cfg.video_clip.before_seconds
        self.video_clip_after = alert_cfg.video_clip.after_seconds

        self._short_term_count_hourly = 0      # 每小时短期偏离计数
        self._last_reset_time = 0.0            # 上次计数重置时间
        self._last_activity_time = 0.0         # 最后活动时间
        self._actions: dict[RiskLevel, list[AlertAction]] = {
            RiskLevel.LOW: [],
            RiskLevel.ATTENTION: [],
            RiskLevel.WARNING: [],
            RiskLevel.CRITICAL: [],
        }
        self._event_log: list[AlertEvent] = []

    def register_action(self, level: RiskLevel, action: AlertAction):
        """注册某等级的响应动作"""
        self._actions[level].append(action)

    def evaluate(
        self,
        deviation: DeviationResult,
        timestamp: float,
        has_activity: bool = True,
    ) -> AlertEvent:
        """
        评估风险等级并生成预警事件

        Args:
            deviation: 偏离检测结果
            timestamp: 当前时间戳
            has_activity: 当前是否有活动(用于无活动检测)
        Returns:
            AlertEvent
        """
        # 更新活动时间
        if has_activity:
            self._last_activity_time = timestamp

        # 每小时重置短期偏离计数
        if timestamp - self._last_reset_time >= 3600:
            self._short_term_count_hourly = 0
            self._last_reset_time = timestamp

        # 短期偏离计数
        if deviation.short_term_triggered:
            self._short_term_count_hourly += 1

        # 判断无活动时间
        inactivity_minutes = (timestamp - self._last_activity_time) / 60
        is_inactive = inactivity_minutes >= self.inactivity_threshold_minutes

        # 四级风险判定 (从高到低)
        if is_inactive:
            level = RiskLevel.CRITICAL
            message = f"超过{self.inactivity_threshold_minutes}分钟无活动，可能发生意外"
        elif deviation.level == DeviationLevel.BOTH:
            level = RiskLevel.CRITICAL
            message = f"短期异常与长期下降同时触发: {deviation.detail}"
        elif deviation.long_term_triggered:
            level = RiskLevel.WARNING
            message = f"长期趋势下降: {deviation.detail}"
        elif self._short_term_count_hourly >= self.short_term_freq_threshold:
            level = RiskLevel.ATTENTION
            message = f"短期偏离频繁({self._short_term_count_hourly}次/小时)"
        else:
            level = RiskLevel.LOW
            message = "所有特征正常"

        event = AlertEvent(
            level=level,
            timestamp=timestamp,
            message=message,
            deviation=deviation,
        )

        # 执行响应动作(关注级及以上)
        if level.priority > 0:
            for action in self._actions[level]:
                try:
                    action(event)
                    event.notified = True
                except Exception as e:
                    event.message += f" [通知失败: {e}]"

        self._event_log.append(event)
        return event

    def get_events(
        self,
        level: RiskLevel | None = None,
        limit: int = 100,
    ) -> list[AlertEvent]:
        """获取预警事件历史"""
        events = self._event_log
        if level is not None:
            events = [e for e in events if e.level == level]
        return events[-limit:]

    def get_current_level(self) -> RiskLevel:
        """获取当前风险等级(最近一次评估)"""
        if not self._event_log:
            return RiskLevel.LOW
        return self._event_log[-1].level
