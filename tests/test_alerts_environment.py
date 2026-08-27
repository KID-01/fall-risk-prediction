"""Environment-aware alert escalation tests."""
from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

from src.alerts.engine import AlertEngine, RiskLevel
from src.inference.deviation import DeviationLevel, DeviationResult
from src.inference.environment_analyzer import EnvironmentAnalysisResult


def _config():
    return OmegaConf.create(
        {
            "alert": {
                "short_term_freq_threshold": 3,
                "inactivity_threshold_minutes": 240,
                "video_clip": {"enabled": True, "before_seconds": 15, "after_seconds": 15},
                "audio": {
                    "impact_critical_threshold": 0.7,
                    "vocal_attention_threshold": 0.5,
                    "cooldown_seconds": 30,
                },
            },
            "deviation": {"short_term": {"threshold": 3.0}},
            "environment": {"alert_cooldown_seconds": 30},
        }
    )


def _deviation(level=DeviationLevel.NONE):
    return DeviationResult(
        level=level,
        short_term_triggered=level in {DeviationLevel.SHORT_TERM, DeviationLevel.BOTH},
        long_term_triggered=level in {DeviationLevel.LONG_TERM, DeviationLevel.BOTH},
        z_scores=np.zeros(4),
        detail="test",
    )


def _environment(state, hazards=None):
    return EnvironmentAnalysisResult(
        timestamp=1000.0,
        motion_heuristic_score=0.7,
        environment_risk_score=0.8,
        motion_state="HIGH" if state == "HIGH" else "MEDIUM",
        environment_state="HIGH",
        overall_state=state,
        person_present=True,
        top_hazards=hazards or [],
        reason_codes=["motion_high", "environment_high"],
    )


class TestEnvironmentAlertEscalation:
    def test_medium_escalates_low_to_attention(self):
        result = AlertEngine(_config()).evaluate(
            _deviation(), 1000.0, environment_result=_environment("MEDIUM")
        )
        assert result.level == RiskLevel.ATTENTION

    def test_high_escalates_low_to_critical_with_hazard(self):
        result = AlertEngine(_config()).evaluate(
            _deviation(),
            1000.0,
            environment_result=_environment("HIGH", [{"class": "suitcase"}]),
        )
        assert result.level == RiskLevel.CRITICAL
        assert "suitcase" in result.message

    def test_low_and_unknown_do_not_escalate(self):
        engine = AlertEngine(_config())
        assert engine.evaluate(_deviation(), 1000.0, environment_result=_environment("LOW")).level == RiskLevel.LOW
        assert engine.evaluate(_deviation(), 1100.0, environment_result=_environment("UNKNOWN")).level == RiskLevel.LOW

    def test_environment_does_not_downgrade_warning(self):
        result = AlertEngine(_config()).evaluate(
            _deviation(DeviationLevel.LONG_TERM),
            1000.0,
            environment_result=_environment("MEDIUM"),
        )
        assert result.level == RiskLevel.WARNING

    def test_cooldown_suppresses_repeat_notification_not_risk(self):
        engine = AlertEngine(_config())
        first = engine.evaluate(_deviation(), 1000.0, environment_result=_environment("HIGH"))
        repeated = engine.evaluate(_deviation(), 1010.0, environment_result=_environment("HIGH"))
        later = engine.evaluate(_deviation(), 1040.0, environment_result=_environment("HIGH"))
        assert first.notification_suppressed is False
        assert repeated.level == RiskLevel.CRITICAL
        assert repeated.notification_suppressed is True
        assert later.notification_suppressed is False

    def test_reset_clears_environment_cooldown(self):
        engine = AlertEngine(_config())
        engine.evaluate(_deviation(), 1000.0, environment_result=_environment("HIGH"))
        engine.reset()
        result = engine.evaluate(_deviation(), 1010.0, environment_result=_environment("HIGH"))
        assert result.notification_suppressed is False
