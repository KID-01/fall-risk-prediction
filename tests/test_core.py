"""
核心模块单元测试 — features / baseline / deviation / alerts
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.inference.features import (
    FeatureVector,
    WalkingRhythmCalculator,
    StepAmplitudeCalculator,
    TrunkStabilityCalculator,
    ActivityDensityCalculator,
    FeatureCalculator,
)
from src.inference.baseline import IndividualBaseline, BaselineManager
from src.inference.deviation import (
    ShortTermDetector,
    LongTermDetector,
    DeviationDetector,
    DeviationLevel,
)
from src.alerts.engine import AlertEngine, RiskLevel, AlertEvent


# ============================================================
# 测试辅助: 合成 KeypointFrame
# ============================================================
def _make_frame(
    *,
    timestamp: float = 0.0,
    left_hip: tuple[float, float] = (0.5, 0.6),
    right_hip: tuple[float, float] = (0.5, 0.6),
    left_shoulder: tuple[float, float] = (0.48, 0.3),
    right_shoulder: tuple[float, float] = (0.52, 0.3),
    left_ankle: tuple[float, float] = (0.45, 0.9),
    right_ankle: tuple[float, float] = (0.55, 0.9),
    visibility: float = 0.9,
    is_valid: bool = True,
) -> "KeypointFrame":
    """构造测试用关键点帧"""
    from src.utils.keypoints import KeypointFrame, PoseKeypoint

    kps = np.zeros((33, 4))
    # 设默认可见
    kps[:, 3] = visibility
    kps[PoseKeypoint.LEFT_HIP] = [*left_hip, 0.0, visibility]
    kps[PoseKeypoint.RIGHT_HIP] = [*right_hip, 0.0, visibility]
    kps[PoseKeypoint.LEFT_SHOULDER] = [*left_shoulder, 0.0, visibility]
    kps[PoseKeypoint.RIGHT_SHOULDER] = [*right_shoulder, 0.0, visibility]
    kps[PoseKeypoint.LEFT_ANKLE] = [*left_ankle, 0.0, visibility]
    kps[PoseKeypoint.RIGHT_ANKLE] = [*right_ankle, 0.0, visibility]
    kps[PoseKeypoint.LEFT_KNEE] = [0.47, 0.75, 0.0, visibility]
    kps[PoseKeypoint.RIGHT_KNEE] = [0.53, 0.75, 0.0, visibility]
    return KeypointFrame(timestamp=timestamp, keypoints=kps, is_valid=is_valid)


# ============================================================
# FeatureVector
# ============================================================
class TestFeatureVector:
    def test_to_array(self):
        fv = FeatureVector(1.5, 0.64, 2.0, 0.5)
        arr = fv.to_array()
        assert arr.shape == (4,)
        assert np.allclose(arr, [1.5, 0.64, 2.0, 0.5])

    def test_from_array(self):
        fv = FeatureVector.from_array(np.array([1.2, 0.5, 3.0, 0.8]), timestamp=42.0)
        assert fv.walking_rhythm == 1.2
        assert fv.step_amplitude == 0.5
        assert fv.trunk_stability == 3.0
        assert fv.activity_density == 0.8
        assert fv.timestamp == 42.0

    def test_to_array_from_array_roundtrip(self):
        original = FeatureVector(1.0, 0.3, 5.0, 0.7, timestamp=10.0)
        arr = original.to_array()
        restored = FeatureVector.from_array(arr, timestamp=10.0)
        assert restored.walking_rhythm == original.walking_rhythm
        assert restored.step_amplitude == original.step_amplitude
        assert restored.trunk_stability == original.trunk_stability
        assert restored.activity_density == original.activity_density


# ============================================================
# WalkingRhythmCalculator
# ============================================================
class TestWalkingRhythmCalculator:
    def test_returns_zero_on_few_frames(self):
        calc = WalkingRhythmCalculator()
        frames = [_make_frame(timestamp=0.0), _make_frame(timestamp=0.1)]
        assert calc.calculate(frames) == 0.0

    def test_returns_zero_on_no_valid_frames(self):
        calc = WalkingRhythmCalculator()
        frames = [_make_frame(is_valid=False, timestamp=float(i)) for i in range(10)]
        assert calc.calculate(frames) == 0.0

    def test_detects_sinusoidal_rhythm(self):
        calc = WalkingRhythmCalculator(min_freq=0.5, max_freq=5.0)
        frames = []
        freq_hz = 2.0
        t = 0.0
        for i in range(200):
            y_offset = 0.05 * math.sin(2 * math.pi * freq_hz * t)
            hip_y = 0.6 + y_offset
            frames.append(_make_frame(
                timestamp=t,
                left_hip=(0.5, hip_y),
                right_hip=(0.5, hip_y),
            ))
            t += 0.033  # ~30fps

        result = calc.calculate(frames)
        assert result > 0
        # 应该能检测到 2Hz 附近的主频(允许一定误差)
        assert 1.5 <= result <= 2.5, f"Expected ~2.0Hz, got {result}"


# ============================================================
# StepAmplitudeCalculator
# ============================================================
class TestStepAmplitudeCalculator:
    def test_returns_zero_on_few_frames(self):
        calc = StepAmplitudeCalculator()
        assert calc.calculate([_make_frame(timestamp=0.0)]) == 0.0

    def test_returns_zero_when_no_torso_height(self):
        calc = StepAmplitudeCalculator()
        kps = np.zeros((33, 4))
        from src.utils.keypoints import KeypointFrame
        frame = KeypointFrame(timestamp=0.0, keypoints=kps, is_valid=True)
        assert calc.calculate([frame, frame]) == 0.0

    def test_computes_normalized_amplitude(self):
        calc = StepAmplitudeCalculator()
        frames = []
        for i in range(30):
            swing = 0.05 * math.sin(2 * math.pi * i / 15)
            frames.append(_make_frame(
                timestamp=i * 0.033,
                left_ankle=(0.45 + swing, 0.9),
                right_ankle=(0.55 + swing, 0.9),
            ))
        result = calc.calculate(frames)
        # 躯干高度约 0.3, 踝摆动峰峰值约 0.1 => 0.1/0.3 ≈ 0.33
        assert 0.1 < result < 0.6, f"Expected ~0.33, got {result}"


# ============================================================
# TrunkStabilityCalculator
# ============================================================
class TestTrunkStabilityCalculator:
    def test_returns_zero_on_few_frames(self):
        calc = TrunkStabilityCalculator()
        assert calc.calculate([_make_frame(timestamp=0.0)]) == 0.0

    def test_stable_trunk_returns_small_angle(self):
        calc = TrunkStabilityCalculator()
        frames = [_make_frame(timestamp=float(i)) for i in range(10)]
        result = calc.calculate(frames)
        assert result < 5.0, f"Upright trunk should be near 0, got {result}"

    def test_leaning_trunk_returns_larger_angle(self):
        calc = TrunkStabilityCalculator()
        frames = []
        for i in range(10):
            lean = i * 0.01
            frames.append(_make_frame(
                timestamp=float(i),
                left_shoulder=(0.48 + lean, 0.3),
                right_shoulder=(0.52 + lean, 0.3),
            ))
        result = calc.calculate(frames)
        assert result > 1.0, f"Leaning trunk should have >1 deg range, got {result}"

    def test_missing_keypoints_returns_zero(self):
        calc = TrunkStabilityCalculator()
        frames = [_make_frame(timestamp=0.0, visibility=0.0) for _ in range(5)]
        result = calc.calculate(frames)
        assert result == 0.0


# ============================================================
# ActivityDensityCalculator
# ============================================================
class TestActivityDensityCalculator:
    def test_returns_zero_on_few_frames(self):
        calc = ActivityDensityCalculator()
        assert calc.calculate([_make_frame(timestamp=0.0)]) == 0.0

    def test_still_frames_low_density(self):
        calc = ActivityDensityCalculator(motion_threshold=0.01)
        frames = [_make_frame(timestamp=float(i)) for i in range(10)]
        result = calc.calculate(frames)
        assert result == 0.0

    def test_moving_frames_high_density(self):
        calc = ActivityDensityCalculator(motion_threshold=0.01)
        frames = []
        for i in range(10):
            hip_x = 0.5 + i * 0.02
            frames.append(_make_frame(
                timestamp=float(i),
                left_hip=(hip_x - 0.02, 0.6),
                right_hip=(hip_x + 0.02, 0.6),
            ))
        result = calc.calculate(frames)
        assert result > 0.5, f"Moving frames should have high density, got {result}"


# ============================================================
# FeatureCalculator
# ============================================================
class TestFeatureCalculator:
    def test_calculate_returns_feature_vector(self):
        calc = FeatureCalculator()
        frames = [_make_frame(timestamp=float(i)) for i in range(30)]
        result = calc.calculate(frames)
        assert isinstance(result, FeatureVector)
        assert result.step_amplitude >= 0
        assert result.trunk_stability >= 0
        assert 0 <= result.activity_density <= 1

    def test_calculate_empty_frames(self):
        calc = FeatureCalculator()
        result = calc.calculate([])
        assert result.walking_rhythm == 0.0
        assert result.step_amplitude == 0.0
        assert result.trunk_stability == 0.0
        assert result.activity_density == 0.0


# ============================================================
# IndividualBaseline
# ============================================================
class TestIndividualBaseline:
    def setup_method(self):
        self.baseline = IndividualBaseline(
            person_id="test",
            mean=np.array([1.5, 0.6, 3.0, 0.5]),
            std=np.array([0.2, 0.1, 1.0, 0.2]),
            cov_inv=np.eye(4),
            sample_count=200,
            collection_days=7,
            is_ready=True,
        )

    def test_mahalanobis_zero_for_mean(self):
        fv = FeatureVector(1.5, 0.6, 3.0, 0.5)
        dist = self.baseline.mahalanobis_distance(fv)
        assert dist == 0.0

    def test_mahalanobis_positive_for_deviation(self):
        fv = FeatureVector(2.5, 0.8, 5.0, 0.7)
        dist = self.baseline.mahalanobis_distance(fv)
        assert dist > 0

    def test_mahalanobis_returns_zero_when_not_ready(self):
        baseline = IndividualBaseline(
            person_id="test", mean=np.zeros(4), std=np.ones(4),
            cov_inv=np.eye(4), is_ready=False,
        )
        fv = FeatureVector(1.5, 0.6, 3.0, 0.5)
        assert baseline.mahalanobis_distance(fv) == 0.0

    def test_z_scores(self):
        z = self.baseline.z_scores(FeatureVector(1.7, 0.7, 4.0, 0.6))
        np.testing.assert_allclose(z, [1.0, 1.0, 1.0, 0.5], atol=0.01)

    def test_z_scores_not_ready(self):
        baseline = IndividualBaseline(
            person_id="test", mean=np.zeros(4), std=np.ones(4),
            cov_inv=np.eye(4), is_ready=False,
        )
        z = baseline.z_scores(FeatureVector(1.5, 0.6, 3.0, 0.5))
        np.testing.assert_array_equal(z, np.zeros(4))

    def test_to_dict_from_dict_roundtrip(self):
        d = self.baseline.to_dict()
        restored = IndividualBaseline.from_dict(d)
        assert restored.person_id == self.baseline.person_id
        np.testing.assert_array_equal(restored.mean, self.baseline.mean)
        np.testing.assert_array_equal(restored.std, self.baseline.std)
        np.testing.assert_array_equal(restored.cov_inv, self.baseline.cov_inv)
        assert restored.is_ready == self.baseline.is_ready


# ============================================================
# BaselineManager
# ============================================================
class TestBaselineManager:
    def setup_method(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.db_path = str(self.tmpdir / "test_baseline.db")
        self.manager = BaselineManager(db_path=self.db_path)

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_add_and_get_samples(self):
        fv = FeatureVector(1.5, 0.6, 3.0, 0.5, timestamp=100.0)
        self.manager.add_sample("alice", fv)
        samples = self.manager.get_samples("alice")
        assert samples.shape == (1, 4)

    def test_get_samples_empty(self):
        samples = self.manager.get_samples("nonexistent")
        assert samples.shape == (0, 4)

    def test_compute_baseline_not_ready_with_few_samples(self):
        for i in range(3):
            fv = FeatureVector(1.5, 0.6, 3.0, 0.5, timestamp=float(i))
            self.manager.add_sample("bob", fv)
        baseline = self.manager.compute_baseline("bob")
        assert baseline.is_ready is False
        assert baseline.sample_count == 3

    def test_compute_baseline_ready(self):
        for i in range(150):
            fv = FeatureVector(
                1.5 + 0.1 * np.random.randn(),
                0.6 + 0.05 * np.random.randn(),
                3.0 + 0.2 * np.random.randn(),
                0.5 + 0.1 * np.random.randn(),
                timestamp=float(i),
            )
            self.manager.add_sample("carol", fv)
        baseline = self.manager.compute_baseline("carol")
        assert baseline.is_ready is True
        assert baseline.sample_count == 150

    def test_load_baseline(self):
        for i in range(150):
            self.manager.add_sample("dave", FeatureVector(1.5, 0.6, 3.0, 0.5, timestamp=float(i)))
        self.manager.compute_baseline("dave")
        loaded = self.manager.load_baseline("dave")
        assert loaded is not None
        assert loaded.is_ready is True
        assert loaded.person_id == "dave"

    def test_load_nonexistent(self):
        loaded = self.manager.load_baseline("ghost")
        assert loaded is None

    def test_reset_baseline(self):
        self.manager.add_sample("eve", FeatureVector(1.5, 0.6, 3.0, 0.5, timestamp=0.0))
        self.manager.compute_baseline("eve")
        self.manager.reset_baseline("eve")
        loaded = self.manager.load_baseline("eve")
        assert loaded is None
        assert self.manager.get_samples("eve").shape == (0, 4)


# ============================================================
# ShortTermDetector
# ============================================================
class TestShortTermDetector:
    def setup_method(self):
        self.baseline = IndividualBaseline(
            person_id="test",
            mean=np.array([1.5, 0.6, 3.0, 0.5]),
            std=np.array([0.2, 0.1, 1.0, 0.2]),
            cov_inv=np.eye(4),
            sample_count=200,
            collection_days=7,
            is_ready=True,
        )

    def test_not_triggered_with_normal_data(self):
        detector = ShortTermDetector()
        for i in range(10):
            fv = FeatureVector(1.5, 0.6, 3.0, 0.5, timestamp=float(i))
            _, triggered = detector.add_and_check(fv, self.baseline)
        assert triggered is False

    def test_triggered_with_abnormal_data(self):
        detector = ShortTermDetector()
        for i in range(10):
            fv = FeatureVector(10.0, 5.0, 20.0, 0.0, timestamp=float(i))
            _, triggered = detector.add_and_check(fv, self.baseline)
        assert triggered is True

    def test_not_triggered_without_ready_baseline(self):
        baseline = IndividualBaseline(
            person_id="test", mean=np.zeros(4), std=np.ones(4),
            cov_inv=np.eye(4), is_ready=False,
        )
        detector = ShortTermDetector()
        for i in range(10):
            fv = FeatureVector(10.0, 5.0, 20.0, 0.0, timestamp=float(i))
            _, triggered = detector.add_and_check(fv, baseline)
        assert triggered is False


# ============================================================
# LongTermDetector
# ============================================================
class TestLongTermDetector:
    def test_not_triggered_with_few_days(self):
        detector = LongTermDetector()
        for i in range(3):
            detector.add_daily_mean(float(i), np.array([1.5, 0.6, 3.0, 0.5]))
        triggered, slopes = detector.check_trend()
        assert triggered is False

    def test_not_triggered_with_stable_trend(self):
        detector = LongTermDetector()
        for i in range(14):
            detector.add_daily_mean(float(i), np.array([1.5, 0.6, 3.0, 0.5]))
        triggered, slopes = detector.check_trend()
        assert not triggered

    def test_triggered_with_declining_trend(self):
        detector = LongTermDetector()
        for i in range(14):
            v = 1.5 - i * 0.1
            detector.add_daily_mean(float(i), np.array([v, 0.6, 3.0, 0.5]))
        triggered, slopes = detector.check_trend()
        assert triggered
        assert slopes[0] < 0


# ============================================================
# DeviationDetector
# ============================================================
class TestDeviationDetector:
    def setup_method(self):
        self.baseline = IndividualBaseline(
            person_id="test",
            mean=np.array([1.5, 0.6, 3.0, 0.5]),
            std=np.array([0.2, 0.1, 1.0, 0.2]),
            cov_inv=np.eye(4),
            sample_count=200,
            collection_days=7,
            is_ready=True,
        )

    def test_normal_returns_none(self):
        detector = DeviationDetector()
        fv = FeatureVector(1.5, 0.6, 3.0, 0.5, timestamp=0.0)
        result = detector.check(fv, self.baseline)
        assert result.level == DeviationLevel.NONE

    def test_short_term_only(self):
        detector = DeviationDetector()
        for i in range(10):
            fv = FeatureVector(10.0, 5.0, 20.0, 0.0, timestamp=float(i))
            result = detector.check(fv, self.baseline)
        assert result.level == DeviationLevel.SHORT_TERM
        assert result.short_term_triggered is True


# ============================================================
# RiskLevel
# ============================================================
class TestRiskLevel:
    def test_priority_order(self):
        assert RiskLevel.LOW.priority == 0
        assert RiskLevel.ATTENTION.priority == 1
        assert RiskLevel.WARNING.priority == 2
        assert RiskLevel.CRITICAL.priority == 3

    def test_labels(self):
        assert RiskLevel.LOW.label == "低风险"
        assert RiskLevel.ATTENTION.label == "关注级"
        assert RiskLevel.WARNING.label == "预警级"
        assert RiskLevel.CRITICAL.label == "高危级"


# ============================================================
# AlertEngine
# ============================================================
class TestAlertEngine:
    def setup_method(self):
        self.engine = AlertEngine()
        self.baseline = IndividualBaseline(
            person_id="test",
            mean=np.array([1.5, 0.6, 3.0, 0.5]),
            std=np.array([0.2, 0.1, 1.0, 0.2]),
            cov_inv=np.eye(4),
            sample_count=200,
            collection_days=7,
            is_ready=True,
        )

    def _make_deviation(self, level: DeviationLevel) -> "DeviationResult":
        from src.inference.deviation import DeviationResult
        return DeviationResult(
            level=level,
            short_term_triggered=(level in (DeviationLevel.SHORT_TERM, DeviationLevel.BOTH)),
            long_term_triggered=(level in (DeviationLevel.LONG_TERM, DeviationLevel.BOTH)),
            mahalanobis_distance=2.0,
            z_scores=np.zeros(4),
            trend_slopes=np.zeros(4),
            detail="test",
        )

    def test_low_risk(self):
        dev = self._make_deviation(DeviationLevel.NONE)
        event = self.engine.evaluate(dev, timestamp=0.0, has_activity=True)
        assert event.level == RiskLevel.LOW

    def test_attention_risk_with_frequent_deviations(self):
        dev = self._make_deviation(DeviationLevel.SHORT_TERM)
        for i in range(6):
            event = self.engine.evaluate(dev, timestamp=float(i * 60), has_activity=True)
        assert event.level == RiskLevel.ATTENTION

    def test_critical_when_both(self):
        dev = self._make_deviation(DeviationLevel.BOTH)
        event = self.engine.evaluate(dev, timestamp=0.0, has_activity=True)
        assert event.level == RiskLevel.CRITICAL

    def test_critical_when_inactive(self):
        dev = self._make_deviation(DeviationLevel.NONE)
        event = self.engine.evaluate(dev, timestamp=15000.0, has_activity=False)
        assert event.level == RiskLevel.CRITICAL

    def test_event_log(self):
        dev = self._make_deviation(DeviationLevel.NONE)
        self.engine.evaluate(dev, timestamp=0.0, has_activity=True)
        assert len(self.engine.get_events()) == 1

    def test_get_current_level(self):
        assert self.engine.get_current_level() == RiskLevel.LOW
        dev = self._make_deviation(DeviationLevel.BOTH)
        self.engine.evaluate(dev, timestamp=0.0, has_activity=True)
        assert self.engine.get_current_level() == RiskLevel.CRITICAL

    def test_register_action_called(self):
        calls = []

        def dummy_action(event: AlertEvent):
            calls.append(event)

        self.engine.register_action(RiskLevel.ATTENTION, dummy_action)
        dev = self._make_deviation(DeviationLevel.SHORT_TERM)
        for i in range(6):
            self.engine.evaluate(dev, timestamp=float(i * 60), has_activity=True)
        assert len(calls) > 0
        assert calls[0].notified is True

    def test_register_action_not_called_for_low(self):
        calls = []

        def dummy_action(event: AlertEvent):
            calls.append(event)

        self.engine.register_action(RiskLevel.ATTENTION, dummy_action)
        dev = self._make_deviation(DeviationLevel.NONE)
        self.engine.evaluate(dev, timestamp=0.0, has_activity=True)
        assert len(calls) == 0

    def test_get_events_filter_by_level(self):
        self.engine.evaluate(self._make_deviation(DeviationLevel.NONE), 0.0, True)
        dev = self._make_deviation(DeviationLevel.BOTH)
        self.engine.evaluate(dev, 1.0, True)
        low_events = self.engine.get_events(level=RiskLevel.LOW)
        critical_events = self.engine.get_events(level=RiskLevel.CRITICAL)
        assert len(low_events) == 1
        assert len(critical_events) == 1


# ============================================================
# AlertEvent
# ============================================================
class TestAlertEvent:
    def test_created_at_auto_set(self):
        event = AlertEvent(level=RiskLevel.LOW, timestamp=0.0, message="test")
        assert event.created_at is not None
        assert isinstance(event.created_at, str)

    def test_defaults(self):
        event = AlertEvent(level=RiskLevel.LOW, timestamp=0.0, message="test")
        assert event.deviation is None
        assert event.video_clip_path is None
        assert event.notified is False
