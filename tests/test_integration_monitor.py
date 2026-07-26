"""
集成测试 — Monitor 内部循环（绕过视频/检测/提点，直接注入关键点帧）
验证: 特征计算 → 基线采集 → 偏离检测 → 预警评估 链路完整
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np

from src.alerts.engine import AlertEvent, RiskLevel
from src.api.database import Database
from src.data.human_detector import DetectionBox
from src.inference.features import FeatureVector
from src.inference.monitor import FallRiskMonitor
from src.utils.keypoints import KeypointFrame, PoseKeypoint


def _make_normal_pose(timestamp: float) -> KeypointFrame:
    """生成一帧正常行走姿态的关键点"""
    kps = np.zeros((33, 4))
    kps[:, 3] = 0.9
    swing = 0.02 * np.sin(2 * np.pi * 2.0 * timestamp)
    kps[PoseKeypoint.LEFT_HIP] = [0.48 + swing, 0.6, 0.0, 0.9]
    kps[PoseKeypoint.RIGHT_HIP] = [0.52 + swing, 0.6, 0.0, 0.9]
    kps[PoseKeypoint.LEFT_SHOULDER] = [0.47, 0.3, 0.0, 0.9]
    kps[PoseKeypoint.RIGHT_SHOULDER] = [0.53, 0.3, 0.0, 0.9]
    kps[PoseKeypoint.LEFT_ANKLE] = [0.44 + swing * 2, 0.9, 0.0, 0.9]
    kps[PoseKeypoint.RIGHT_ANKLE] = [0.56 + swing * 2, 0.9, 0.0, 0.9]
    kps[PoseKeypoint.LEFT_KNEE] = [0.47, 0.75, 0.0, 0.9]
    kps[PoseKeypoint.RIGHT_KNEE] = [0.53, 0.75, 0.0, 0.9]
    return KeypointFrame(timestamp=timestamp, keypoints=kps, is_valid=True)


def test_monitor_pipeline_with_synthetic_data():
    monitor = FallRiskMonitor()

    # 注入 30 帧合成关键点
    frames = [_make_normal_pose(i * 0.1) for i in range(30)]
    monitor._keypoint_buffer = list(frames)
    monitor.status.recent_keypoints = list(frames)
    monitor.person_id = "intg_test"
    monitor.status.frames_valid = len(frames)

    # 阶段4: 特征计算
    feature = monitor.feature_calculator.calculate(frames)
    monitor.status.last_feature = feature
    assert feature.walking_rhythm >= 0
    assert feature.step_amplitude >= 0
    assert 0 <= feature.activity_density <= 1

    # 阶段5: 基线采集（注入足够样本让基线就绪，min_samples=100）
    for i in range(150):
        monitor.baseline_manager.add_sample(
            monitor.person_id,
            FeatureVector(
                1.5 + 0.05 * np.random.randn(),
                0.6 + 0.02 * np.random.randn(),
                2.0 + 0.3 * np.random.randn(),
                0.5 + 0.05 * np.random.randn(),
                timestamp=100.0 + i,
            ),
        )
    baseline = monitor.baseline_manager.compute_baseline(monitor.person_id)
    monitor.status.baseline_ready = baseline.is_ready
    monitor.status.baseline_samples = baseline.sample_count
    assert baseline.is_ready, "基线应已就绪"
    assert baseline.sample_count > 0

    # 阶段6: 偏离检测
    deviation = monitor.deviation_detector.check(feature, baseline)
    monitor.status.last_deviation = deviation

    # 阶段7: 预警评估
    alert = monitor.alert_engine.evaluate(deviation, feature.timestamp, has_activity=True)
    monitor.status.last_alert = alert
    monitor.status.current_risk_level = alert.level

    # 验证整体状态
    status = monitor.get_status()
    assert status["baseline_ready"] is True
    assert status["baseline_samples"] >= 30
    assert status["frames_valid"] == 30
    assert status["current_risk_level"] in ("low", "attention", "warning", "critical")
    assert status["last_feature"] is not None
    assert len(status["last_feature"]) == 4

    print(f"  risk_level={status['current_risk_level']}")
    print(f"  baseline_ready={status['baseline_ready']}, samples={status['baseline_samples']}")
    print(f"  frames_valid={status['frames_valid']}")


def test_monitor_pipeline_persists_to_db():
    monitor = FallRiskMonitor()
    person_id = "p0a_test"

    for i in range(150):
        monitor.baseline_manager.add_sample(
            person_id,
            FeatureVector(
                1.5 + 0.05 * np.random.randn(),
                0.6 + 0.02 * np.random.randn(),
                2.0 + 0.3 * np.random.randn(),
                0.5 + 0.05 * np.random.randn(),
                timestamp=100.0 + i,
            ),
        )
    baseline = monitor.baseline_manager.compute_baseline(person_id)
    assert baseline.is_ready

    kp_frame = _make_normal_pose(timestamp=200.0)

    with patch("src.inference.monitor.VideoCapture") as mock_vc_cls:
        mock_cap = MagicMock()
        mock_cap.__enter__.return_value = mock_cap
        mock_cap.frames.return_value = [
            MagicMock(
                frame=np.zeros((480, 640, 3), dtype=np.uint8),
                timestamp=200.0 + i * 0.1,
            )
            for i in range(15)
        ]
        mock_vc_cls.return_value = mock_cap

        orig_detect = monitor.human_detector.detect_best
        orig_extract = monitor.keypoint_extractor.extract
        try:
            monitor.human_detector.detect_best = MagicMock(
                return_value=DetectionBox(
                    x1=0.1, y1=0.1, x2=0.9, y2=0.9, confidence=0.95,
                )
            )
            monitor.keypoint_extractor.extract = MagicMock(return_value=kp_frame)

            monitor.start(source="mock_source", person_id=person_id)
            for _ in range(40):
                if monitor.status.frames_processed >= 15:
                    break
                time.sleep(0.1)
            monitor.stop()

            db = Database()
            records = db.query_risk_records(person_id=person_id)
            assert len(records) > 0

            if monitor.status.last_alert and monitor.status.last_alert.level != RiskLevel.LOW:
                alerts = db.query_alert_events(person_id=person_id)
                assert len(alerts) > 0

        finally:
            monitor.human_detector.detect_best = orig_detect
            monitor.keypoint_extractor.extract = orig_extract
            db = Database()
            with db._get_conn() as conn:
                conn.execute("DELETE FROM risk_records WHERE person_id=?", (person_id,))
                conn.execute("DELETE FROM alert_events WHERE person_id=?", (person_id,))


def test_monitor_pipeline_persists_alert_event_to_db():
    monitor = FallRiskMonitor()
    person_id = "p0a_alert_test"

    # 构建基线 (150 样本)
    for i in range(150):
        monitor.baseline_manager.add_sample(
            person_id,
            FeatureVector(
                1.5 + 0.05 * np.random.randn(),
                0.6 + 0.02 * np.random.randn(),
                2.0 + 0.3 * np.random.randn(),
                0.5 + 0.05 * np.random.randn(),
                timestamp=100.0 + i,
            ),
        )
    baseline = monitor.baseline_manager.compute_baseline(person_id)
    assert baseline.is_ready

    kp_frame = _make_normal_pose(timestamp=200.0)

    with patch("src.inference.monitor.VideoCapture") as mock_vc_cls:
        mock_cap = MagicMock()
        mock_cap.__enter__.return_value = mock_cap
        mock_cap.frames.return_value = [
            MagicMock(
                frame=np.zeros((480, 640, 3), dtype=np.uint8),
                timestamp=200.0 + i * 0.1,
            )
            for i in range(15)
        ]
        mock_vc_cls.return_value = mock_cap

        orig_detect = monitor.human_detector.detect_best
        orig_extract = monitor.keypoint_extractor.extract
        orig_evaluate = monitor.alert_engine.evaluate
        try:
            monitor.human_detector.detect_best = MagicMock(
                return_value=DetectionBox(
                    x1=0.1, y1=0.1, x2=0.9, y2=0.9, confidence=0.95,
                )
            )
            monitor.keypoint_extractor.extract = MagicMock(return_value=kp_frame)

            # 强制 alert_engine.evaluate 返回 ATTENTION 级别
            monitor.alert_engine.evaluate = MagicMock(
                return_value=AlertEvent(
                    level=RiskLevel.ATTENTION,
                    timestamp=200.0,
                    message="测试告警持久化",
                )
            )

            monitor.start(source="mock_source", person_id=person_id)
            for _ in range(40):
                if monitor.status.frames_processed >= 15:
                    break
                time.sleep(0.1)
            monitor.stop()

            db = Database()
            alerts = db.query_alert_events(person_id=person_id)
            assert len(alerts) > 0, f"预期至少一条告警, 实际 {len(alerts)}"
            assert alerts[0]["alert_level"] == "attention", (
                f"预期 alert_level=attention, 实际 {alerts[0]['alert_level']}"
            )

        finally:
            monitor.human_detector.detect_best = orig_detect
            monitor.keypoint_extractor.extract = orig_extract
            monitor.alert_engine.evaluate = orig_evaluate
            db = Database()
            with db._get_conn() as conn:
                conn.execute("DELETE FROM risk_records WHERE person_id=?", (person_id,))
                conn.execute("DELETE FROM alert_events WHERE person_id=?", (person_id,))


if __name__ == "__main__":
    test_monitor_pipeline_with_synthetic_data()
    test_monitor_pipeline_persists_to_db()
    test_monitor_pipeline_persists_alert_event_to_db()
    print("✅ Monitor 集成测试全部通过!")
