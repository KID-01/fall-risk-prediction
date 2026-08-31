"""
分级预警引擎 — 四级风险分类与响应

风险等级:
  低风险   — 所有特征在基线±1个标准差范围内 → 持续监测
  关注级   — 短期偏离频繁(≥3次/小时) → APP推送
  预警级   — 长期趋势连续7天负向变化 → 短信通知
  高危级   — 近似跌倒/4小时无活动 → 电话通知
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
from omegaconf import OmegaConf

from src.inference.audio_analyzer import AudioEvent, SoundCategory
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

    def __init__(self, config: OmegaConf | None = None):
        cfg = config if config is not None else get_config()
        alert_cfg = cfg.alert
        self.short_term_freq_threshold = alert_cfg.short_term_freq_threshold
        # 短期检测确认后，极端偏离应直接进入高危级，而不是等待频次计数。
        self.severe_deviation_distance = cfg.deviation.short_term.threshold * 2
        self.severe_deviation_z = 6.0
        self.inactivity_threshold_minutes = alert_cfg.inactivity_threshold_minutes
        self.video_clip_enabled = alert_cfg.video_clip.enabled
        self.video_clip_before = alert_cfg.video_clip.before_seconds
        self.video_clip_after = alert_cfg.video_clip.after_seconds
        # 音频触发预警阈值
        self.impact_critical_threshold = float(alert_cfg.audio.impact_critical_threshold)
        self.vocal_attention_threshold = float(alert_cfg.audio.vocal_attention_threshold)

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
        self._event_log_lock = threading.Lock()

    def register_action(self, level: RiskLevel, action: AlertAction):
        """注册某等级的响应动作"""
        self._actions[level].append(action)

    def reset(self):
        """清空跨视频计数和事件，保留已注册的通知动作。"""
        self._short_term_count_hourly = 0
        self._last_reset_time = 0.0
        self._last_activity_time = 0.0
        with self._event_log_lock:
            self._event_log.clear()

    def evaluate(
        self,
        deviation: DeviationResult,
        timestamp: float,
        has_activity: bool = True,
        audio_events: list[AudioEvent] | None = None,
        emit: bool = True,
    ) -> AlertEvent:
        """
        评估风险等级并生成预警事件

        Args:
            deviation: 偏离检测结果
            timestamp: 当前时间戳
            has_activity: 当前是否有活动(用于无活动检测)
            audio_events: 音频分析检测到的事件列表 (可选)
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
        elif deviation.level == DeviationLevel.SHORT_TERM:
            max_abs_z = float(np.max(np.abs(deviation.z_scores)))
            if (
                deviation.mahalanobis_distance >= self.severe_deviation_distance
                or max_abs_z >= self.severe_deviation_z
            ):
                level = RiskLevel.CRITICAL
                message = f"严重短期异常，需立即确认: {deviation.detail}"
            else:
                level = RiskLevel.ATTENTION
                message = f"短期异常，建议关注: {deviation.detail}"
        elif deviation.long_term_triggered:
            level = RiskLevel.WARNING
            message = f"长期趋势下降: {deviation.detail}"
        elif self._short_term_count_hourly >= self.short_term_freq_threshold:
            level = RiskLevel.ATTENTION
            message = f"短期偏离频繁({self._short_term_count_hourly}次/小时)"
        else:
            level = RiskLevel.LOW
            message = "所有特征正常"

        # 音频事件升级逻辑
        if audio_events:
            for event in audio_events:
                if event.category == SoundCategory.IMPACT and event.score >= self.impact_critical_threshold:
                    # 撞击声达到阈值 → 直接升级为 CRITICAL
                    if level.priority < RiskLevel.CRITICAL.priority:
                        level = RiskLevel.CRITICAL
                        message += f" | 撞击声触发高危: {event.label} ({event.score:.2f})"
                elif event.category == SoundCategory.VOCAL_DISTRESS and event.score >= self.vocal_attention_threshold:
                    # 人声呼救达到阈值 → 至少升级为 ATTENTION
                    if level.priority < RiskLevel.ATTENTION.priority:
                        level = RiskLevel.ATTENTION
                    message += f" | 人声呼救: {event.label} ({event.score:.2f})"

        event = AlertEvent(
            level=level,
            timestamp=timestamp,
            message=message,
            deviation=deviation,
        )

        if emit:
            self.emit_event(event)
        return event

    def emit_event(self, event: AlertEvent) -> AlertEvent:
        """执行通知并写入内存事件日志。"""
        if event.level.priority > 0:
            for action in self._actions[event.level]:
                try:
                    action(event)
                    event.notified = True
                except Exception as e:
                    event.message += f" [通知失败: {e}]"
        with self._event_log_lock:
            self._event_log.append(event)
        return event

    def get_events(
        self,
        level: RiskLevel | None = None,
        limit: int = 100,
    ) -> list[AlertEvent]:
        """获取预警事件历史"""
        with self._event_log_lock:
            events = list(self._event_log)
        if level is not None:
            events = [e for e in events if e.level == level]
        return events[-limit:]

    def get_current_level(self) -> RiskLevel:
        """获取当前风险等级(最近一次评估)"""
        with self._event_log_lock:
            if not self._event_log:
                return RiskLevel.LOW
            return self._event_log[-1].level
