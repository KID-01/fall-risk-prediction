"""预警引擎音频事件升级测试 — 验证 audio_events 参数能正确触发风险等级升级"""
from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

from src.alerts.engine import AlertEngine, RiskLevel
from src.inference.audio_analyzer import AudioEvent, SoundCategory
from src.inference.deviation import DeviationLevel, DeviationResult


# ============================================================
# 测试辅助
# ============================================================
def _make_cfg(**overrides) -> OmegaConf:
    """构造最小 alert 配置, 支持字段覆盖"""
    cfg = OmegaConf.create(
        {
            "alert": {
                "short_term_freq_threshold": 3,
                "inactivity_threshold_minutes": 240,
                "video_clip": {"enabled": True, "before_seconds": 15, "after_seconds": 15, "format": "mp4"},
                "audio": {
                    "impact_critical_threshold": 0.70,
                    "vocal_attention_threshold": 0.50,
                },
            },
            "deviation": {
                "short_term": {"threshold": 3.0},
            },
        }
    )
    return OmegaConf.merge(cfg, OmegaConf.create(overrides))


def _make_fake_deviation(
    level: DeviationLevel = DeviationLevel.NONE,
    mahalanobis_distance: float = 1.0,
    z_scores: np.ndarray | None = None,
) -> DeviationResult:
    """构造可控的偏离结果"""
    if z_scores is None:
        z_scores = np.array([0.5, -0.3, 0.2, 0.1])
    return DeviationResult(
        level=level,
        mahalanobis_distance=mahalanobis_distance,
        z_scores=z_scores,
        detail="test deviation",
        short_term_triggered=(level in (DeviationLevel.SHORT_TERM, DeviationLevel.BOTH)),
        long_term_triggered=(level in (DeviationLevel.LONG_TERM, DeviationLevel.BOTH)),
    )


def _make_impact_event(score: float, timestamp: float = 10.0) -> AudioEvent:
    return AudioEvent(
        category=SoundCategory.IMPACT,
        label="Thump, thud",
        class_index=460,
        score=score,
        timestamp=timestamp,
    )


def _make_vocal_event(score: float, timestamp: float = 10.0) -> AudioEvent:
    return AudioEvent(
        category=SoundCategory.VOCAL_DISTRESS,
        label="Screaming",
        class_index=14,
        score=score,
        timestamp=timestamp,
    )


# ============================================================
# TestAudioEscalation
# ============================================================
class TestAudioEscalation:
    def test_impact_score_above_critical_threshold_escalates_to_critical(self):
        """impact score >= 0.70 → CRITICAL"""
        engine = AlertEngine(config=_make_cfg())
        dev = _make_fake_deviation(DeviationLevel.NONE)  # 正常基线
        events = [_make_impact_event(0.85)]
        result = engine.evaluate(dev, 1000.0, has_activity=True, audio_events=events)
        assert result.level == RiskLevel.CRITICAL
        assert "撞击声" in result.message
        assert "Thump, thud" in result.message

    def test_impact_score_below_critical_threshold_no_escalation(self):
        """impact score < 0.70 → 不升级 (保持原等级 LOW)"""
        engine = AlertEngine(config=_make_cfg())
        dev = _make_fake_deviation(DeviationLevel.NONE)
        events = [_make_impact_event(0.65)]
        result = engine.evaluate(dev, 1000.0, has_activity=True, audio_events=events)
        assert result.level == RiskLevel.LOW
        assert result.message == "所有特征正常"

    def test_vocal_score_above_attention_threshold_escalates_to_at_least_attention(self):
        """vocal score >= 0.50 → 至少 ATTENTION"""
        engine = AlertEngine(config=_make_cfg())
        dev = _make_fake_deviation(DeviationLevel.NONE)
        events = [_make_vocal_event(0.55)]
        result = engine.evaluate(dev, 1000.0, has_activity=True, audio_events=events)
        assert result.level.priority >= RiskLevel.ATTENTION.priority
        assert "人声呼救" in result.message
        assert "Screaming" in result.message

    def test_vocal_does_not_downgrade_higher_levels(self):
        """vocal 事件不降级更高的等级 (WARNING/CRITICAL 保持不变)"""
        engine = AlertEngine(config=_make_cfg())
        # 先构造一个 WARNING 级别的偏离
        dev = _make_fake_deviation(DeviationLevel.LONG_TERM)
        events = [_make_vocal_event(0.55)]
        result = engine.evaluate(dev, 1000.0, has_activity=True, audio_events=events)
        assert result.level == RiskLevel.WARNING
        assert result.level != RiskLevel.ATTENTION
        assert result.level != RiskLevel.LOW

    def test_impact_does_not_downgrade_critical(self):
        """impact 事件不降级 CRITICAL"""
        engine = AlertEngine(config=_make_cfg())
        dev = _make_fake_deviation(DeviationLevel.BOTH)  # 已经是 CRITICAL
        events = [_make_impact_event(0.65)]  # 即使分数低于阈值
        result = engine.evaluate(dev, 1000.0, has_activity=True, audio_events=events)
        assert result.level == RiskLevel.CRITICAL

    def test_audio_events_none_behavior_unchanged(self):
        """audio_events=None → 行为与旧版完全一致 (回归守卫)"""
        engine = AlertEngine(config=_make_cfg())
        dev = _make_fake_deviation(DeviationLevel.SHORT_TERM)
        # 旧版签名兼容：不传 audio_events 或传 None
        result1 = engine.evaluate(dev, 1000.0, has_activity=True)
        result2 = engine.evaluate(dev, 1000.0, has_activity=True, audio_events=None)
        assert result1.level == result2.level
        assert result1.message == result2.message

    def test_multiple_events_merge_messages(self):
        """多个音频事件消息合并"""
        engine = AlertEngine(config=_make_cfg())
        dev = _make_fake_deviation(DeviationLevel.NONE)
        events = [
            _make_impact_event(0.85),
            _make_vocal_event(0.55),
        ]
        result = engine.evaluate(dev, 1000.0, has_activity=True, audio_events=events)
        assert result.level == RiskLevel.CRITICAL
        assert "撞击声" in result.message
        assert "人声呼救" in result.message

    def test_empty_audio_events_list_no_escalation(self):
        """空列表 → 无升级"""
        engine = AlertEngine(config=_make_cfg())
        dev = _make_fake_deviation(DeviationLevel.NONE)
        result = engine.evaluate(dev, 1000.0, has_activity=True, audio_events=[])
        assert result.level == RiskLevel.LOW
        assert result.message == "所有特征正常"

    def test_custom_thresholds_from_config(self):
        """阈值从配置读取"""
        # 修改阈值：impact 0.60 触发 CRITICAL, vocal 0.40 触发 ATTENTION
        cfg = _make_cfg(alert={"audio": {"impact_critical_threshold": 0.60, "vocal_attention_threshold": 0.40}})
        engine = AlertEngine(config=cfg)
        dev = _make_fake_deviation(DeviationLevel.NONE)
        events = [_make_impact_event(0.65)]
        result = engine.evaluate(dev, 1000.0, has_activity=True, audio_events=events)
        assert result.level == RiskLevel.CRITICAL

    def test_inactivity_still_triggers_critical_even_with_audio(self):
        """无活动超时仍触发 CRITICAL (优先级最高)"""
        engine = AlertEngine(config=_make_cfg())
        dev = _make_fake_deviation(DeviationLevel.NONE)
        events = [_make_impact_event(0.85)]
        # timestamp 差值超过 inactivity_threshold_minutes (240min)
        result = engine.evaluate(dev, 1000.0 + 250 * 60, has_activity=False, audio_events=events)
        assert result.level == RiskLevel.CRITICAL
        assert "无活动" in result.message or "撞击声" in result.message  # 两者其一或同时


# ============================================================
# TestSignatureCompatibility — 旧版签名兼容
# ============================================================
class TestSignatureCompatibility:
    def test_old_signature_still_works(self):
        """不传 audio_events 或显式 None → 旧版行为"""
        engine = AlertEngine(config=_make_cfg())
        dev = _make_fake_deviation(DeviationLevel.SHORT_TERM)
        # 这两种调用方式都要工作
        r1 = engine.evaluate(dev, 1000.0, has_activity=True)
        r2 = engine.evaluate(dev, 1000.0, has_activity=True, audio_events=None)
        assert r1.level == r2.level
        assert r1.message == r2.message

    def test_positional_args_still_work(self):
        """位置参数调用不受影响"""
        engine = AlertEngine(config=_make_cfg())
        dev = _make_fake_deviation(DeviationLevel.LONG_TERM)
        # evaluate(deviation, timestamp, has_activity) 旧版三参数
        result = engine.evaluate(dev, 1000.0, True)
        assert result.level == RiskLevel.WARNING


# ============================================================
# TestAudioEventEscalationIntegration — 与 AlertEngine 内部状态交互
# ============================================================
class TestAudioEventEscalationIntegration:
    def test_audio_events_drained_only_once(self):
        """同一事件只在一次 evaluate 中升级"""
        engine = AlertEngine(config=_make_cfg())
        dev = _make_fake_deviation(DeviationLevel.NONE)
        events = [_make_impact_event(0.85)]

        # 第一次 evaluate 消费事件
        result1 = engine.evaluate(dev, 1000.0, has_activity=True, audio_events=events)
        assert result1.level == RiskLevel.CRITICAL

        # 第二次 evaluate 无事件传入 → 不应再升级
        result2 = engine.evaluate(dev, 2000.0, has_activity=True, audio_events=None)
        # 此时偏离为 NONE 且无音频事件 → 应回到 LOW
        assert result2.level == RiskLevel.LOW

    def test_audio_events_do_not_persist_across_resets(self):
        """reset() 后音频事件不残留"""
        engine = AlertEngine(config=_make_cfg())
        dev = _make_fake_deviation(DeviationLevel.NONE)
        events = [_make_impact_event(0.85)]
        engine.evaluate(dev, 1000.0, has_activity=True, audio_events=events)
        engine.reset()
        # reset 后不应有残留事件影响
        result = engine.evaluate(dev, 1000.0, has_activity=True)
        assert result.level == RiskLevel.LOW


# ============================================================
# TestEdgeCases
# ============================================================
class TestEdgeCases:
    def test_extreme_impact_score_1_0(self):
        """分数 1.0 正常处理"""
        engine = AlertEngine(config=_make_cfg())
        dev = _make_fake_deviation(DeviationLevel.NONE)
        events = [_make_impact_event(1.0)]
        result = engine.evaluate(dev, 1000.0, has_activity=True, audio_events=events)
        assert result.level == RiskLevel.CRITICAL

    def test_zero_score_no_escalation(self):
        """分数 0.0 不触发"""
        engine = AlertEngine(config=_make_cfg())
        dev = _make_fake_deviation(DeviationLevel.NONE)
        events = [_make_impact_event(0.0)]
        result = engine.evaluate(dev, 1000.0, has_activity=True, audio_events=events)
        assert result.level == RiskLevel.LOW

    def test_negative_score_handled(self):
        """负分数视为无效 → 不触发"""
        engine = AlertEngine(config=_make_cfg())
        dev = _make_fake_deviation(DeviationLevel.NONE)
        events = [_make_impact_event(-0.1)]
        result = engine.evaluate(dev, 1000.0, has_activity=True, audio_events=events)
        assert result.level == RiskLevel.LOW

    def test_same_audio_category_is_cooled_down(self):
        """同类音频告警在冷却期内不重复升级，原始事件仍由监控层保存。"""
        engine = AlertEngine(config=_make_cfg())
        dev = _make_fake_deviation(DeviationLevel.NONE)
        first = engine.evaluate(dev, 1000.0, audio_events=[_make_impact_event(0.9)])
        second = engine.evaluate(dev, 1010.0, audio_events=[_make_impact_event(0.95)])
        third = engine.evaluate(dev, 1040.0, audio_events=[_make_impact_event(0.95)])
        assert first.level == RiskLevel.CRITICAL
        assert second.level == RiskLevel.LOW
        assert third.level == RiskLevel.CRITICAL
