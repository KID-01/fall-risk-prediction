"""Environment analyzer logic and monitor lifecycle tests."""
from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.alerts.engine import RiskLevel
from src.api.database import Database
from src.inference.environment_analyzer import (
    EnvironmentAnalysisResult,
    EnvironmentRiskAnalyzer,
    causal_persist,
    choose_primary,
    compute_environment_risk,
    fuse_states,
    proximity_factor,
)
from src.inference.monitor import FallRiskMonitor, MonitorStatus


def _environment_config(**overrides):
    base = {
        "enabled": True,
        "model_path": "checkpoints/yolo26n.pt",
        "device": "cpu",
        "confidence_threshold": 0.25,
        "person_confidence_threshold": 0.25,
        "image_size": 640,
        "analysis_interval_ms": 1000,
        "result_max_age_seconds": 3,
        "persistence_frames": 1,
        "motion_thresholds": [0.3, 0.6],
        "environment_thresholds": [0.4, 0.7],
        "object_confidence_threshold": 0.25,
        "near_threshold": 0.4,
        "far_threshold": 1.5,
        "foot_band_gate": True,
        "top_k_hazards": 3,
        "object_risk_table": {"chair": 0.6, "suitcase": 0.7},
        "neutral_classes": ["tv"],
    }
    base.update(overrides)
    return OmegaConf.create({"environment": base})


def _box(label="person", conf=0.9, x1=100, y1=100, x2=200, y2=400):
    return {
        "label": label,
        "conf": conf,
        "x1": float(x1),
        "y1": float(y1),
        "x2": float(x2),
        "y2": float(y2),
    }


def _result(state="LOW", timestamp=100.0, environment_score=0.0):
    return EnvironmentAnalysisResult(
        timestamp=timestamp,
        motion_heuristic_score=0.0,
        environment_risk_score=environment_score,
        motion_state="LOW",
        environment_state="LOW",
        overall_state=state,
        person_present=True,
        top_hazards=[],
        reason_codes=["motion_low", "environment_low"],
    )


class TestEnvironmentGeometry:
    def test_primary_prefers_previous_overlap(self):
        previous = _box(x1=0, y1=0, x2=100, y2=200)
        overlapping = _box(x1=10, y1=10, x2=90, y2=190)
        larger = _box(x1=300, y1=0, x2=500, y2=400)
        assert choose_primary([larger, overlapping], previous, 0.25) is overlapping

    def test_primary_without_history_uses_largest_box(self):
        small = _box(x1=0, y1=0, x2=50, y2=100)
        large = _box(x1=0, y1=0, x2=100, y2=300)
        assert choose_primary([small, large], None, 0.25) is large

    @pytest.mark.parametrize(
        ("distance", "expected"),
        [(0.2, 1.0), (0.4, 1.0), (1.5, 0.0), (2.0, 0.0)],
    )
    def test_proximity_bounds(self, distance, expected):
        assert proximity_factor(distance, 0.4, 1.5) == expected

    def test_near_hazard_contributes(self):
        person = _box()
        suitcase = _box("suitcase", x1=150, y1=360, x2=230, y2=420)
        score, hazards = compute_environment_risk(
            person, [suitcase], _environment_config().environment
        )
        assert score is not None and score > 0
        assert hazards[0]["class"] == "suitcase"

    def test_neutral_and_unknown_objects_are_ignored(self):
        person = _box()
        objects = [
            _box("tv", x1=150, y1=360, x2=230, y2=420),
            _box("book", x1=150, y1=360, x2=230, y2=420),
        ]
        score, hazards = compute_environment_risk(
            person, objects, _environment_config().environment
        )
        assert score == 0.0
        assert hazards == []

    def test_missing_person_is_unknown_not_zero_risk(self):
        score, hazards = compute_environment_risk(
            None, [_box("chair")], _environment_config().environment
        )
        assert score is None
        assert hazards == []


class TestEnvironmentFusion:
    @pytest.mark.parametrize(
        ("motion", "environment", "expected"),
        [
            ("LOW", "LOW", "LOW"),
            ("LOW", "MEDIUM", "LOW"),
            ("LOW", "HIGH", "LOW"),
            ("MEDIUM", "LOW", "MEDIUM"),
            ("MEDIUM", "MEDIUM", "MEDIUM"),
            ("MEDIUM", "HIGH", "HIGH"),
            ("HIGH", "LOW", "HIGH"),
            ("HIGH", "MEDIUM", "HIGH"),
            ("HIGH", "HIGH", "HIGH"),
        ],
    )
    def test_rule_table(self, motion, environment, expected):
        overall, _, _ = fuse_states(motion, environment)
        assert overall == expected

    def test_environment_high_alone_is_context_only(self):
        overall, elevated, reasons = fuse_states("LOW", "HIGH")
        assert overall == "LOW"
        assert elevated is True
        assert "context_elevated" in reasons

    def test_missing_motion_remains_unknown(self):
        overall, _, reasons = fuse_states("UNKNOWN", "HIGH")
        assert overall == "UNKNOWN"
        assert reasons == ["motion_missing", "environment_high"]

    def test_causal_persistence_uses_majority(self):
        assert causal_persist(["LOW", "HIGH", "HIGH"], 3) == "HIGH"
        assert causal_persist(["UNKNOWN", "UNKNOWN"], 3) == "UNKNOWN"


class _FakeModel:
    names = {0: "person", 28: "suitcase"}

    def __init__(self, results):
        self.results = list(results)

    def predict(self, **kwargs):
        return [self.results.pop(0)]


def _model_result(person_coords, include_suitcase=False):
    rows = [person_coords]
    confidences = [0.95]
    classes = [0]
    if include_suitcase:
        rows.append([person_coords[0], person_coords[3] - 20, person_coords[2] + 40, person_coords[3] + 20])
        confidences.append(0.9)
        classes.append(28)
    boxes = SimpleNamespace(
        xyxy=torch.tensor(rows, dtype=torch.float32),
        conf=torch.tensor(confidences, dtype=torch.float32),
        cls=torch.tensor(classes, dtype=torch.float32),
    )
    return SimpleNamespace(boxes=boxes, names=_FakeModel.names)


class TestEnvironmentAnalyzer:
    def test_analyze_parses_model_and_tracks_downward_motion(self):
        model = _FakeModel(
            [
                _model_result([100, 0, 200, 100], include_suitcase=True),
                _model_result([100, 400, 200, 500], include_suitcase=True),
            ]
        )
        analyzer = EnvironmentRiskAnalyzer(_environment_config(), model=model)
        first = analyzer.analyze(np.zeros((640, 640, 3), dtype=np.uint8), 0.0, 100.0)
        second = analyzer.analyze(np.zeros((640, 640, 3), dtype=np.uint8), 1.0, 101.0)
        assert first.person_present is True
        assert first.top_hazards[0]["class"] == "suitcase"
        assert second.motion_heuristic_score == pytest.approx(0.8)
        assert second.motion_state == "HIGH"
        assert second.overall_state == "HIGH"

    def test_missing_weight_fails_without_download(self, tmp_path):
        config = _environment_config(model_path=str(tmp_path / "missing.pt"))
        analyzer = EnvironmentRiskAnalyzer(config)
        with pytest.raises(RuntimeError, match="权重不存在"):
            analyzer.analyze(np.zeros((10, 10, 3), dtype=np.uint8), 0.0, 1.0)


class TestEnvironmentMonitorLifecycle:
    def _monitor(self):
        monitor = object.__new__(FallRiskMonitor)
        monitor.config = _environment_config(analysis_interval_ms=0)
        monitor.status = MonitorStatus(environment_enabled=True, environment_status="STARTING")
        monitor._ensure_environment_runtime()
        return monitor

    def test_latest_frame_replaces_stale_queue_item(self):
        monitor = self._monitor()
        monitor._submit_environment_frame(np.zeros((2, 2, 3)), 1.0)
        monitor._submit_environment_frame(np.ones((2, 2, 3)), 2.0)
        frame, timestamp = monitor._environment_queue.get_nowait()
        assert timestamp == 2.0
        assert frame.mean() == 1.0

    def test_worker_updates_running_status(self):
        monitor = self._monitor()
        monitor._environment_queue.put((np.zeros((2, 2, 3)), 1.0))
        analyzer = MagicMock()

        def analyze(*args):
            monitor._environment_stop_flag.set()
            return _result(timestamp=time.time())

        analyzer.analyze.side_effect = analyze
        monitor.environment_analyzer = analyzer
        monitor._run_environment()
        assert monitor.status.environment_status == "RUNNING"
        assert monitor.status.last_environment_result is not None

    def test_worker_failure_isolated_as_unavailable(self):
        monitor = self._monitor()
        monitor._environment_queue.put((np.zeros((2, 2, 3)), 1.0))
        monitor.environment_analyzer = MagicMock()
        monitor.environment_analyzer.analyze.side_effect = RuntimeError("model failed")
        monitor._run_environment()
        assert monitor.status.environment_enabled is False
        assert monitor.status.environment_status == "UNAVAILABLE"
        assert monitor.status.environment_error == "model failed"

    def test_stale_result_is_not_used_for_alerting(self):
        monitor = self._monitor()
        monitor.status.last_environment_result = _result(timestamp=10.0)
        assert monitor._current_environment_result(20.0) is None

    def test_get_status_serializes_environment_fields(self):
        monitor = self._monitor()
        monitor.status.current_risk_level = RiskLevel.LOW
        monitor.status.last_environment_result = _result(timestamp=100.0, environment_score=0.3)
        payload = monitor.get_status()
        assert payload["environment_enabled"] is True
        assert payload["environment_status"] == "STARTING"
        assert payload["last_environment_result"]["environment_risk_score"] == 0.3


def test_database_persists_environment_payload(tmp_path):
    database = object.__new__(Database)
    database._initialized = True
    database.db_path = str(tmp_path / "environment.db")
    database._init_tables()
    database.insert_risk_record(
        risk_score=2.5,
        risk_level="attention",
        env_features=_result(environment_score=0.6).to_dict(),
    )
    record = database.query_risk_records(limit=1)[0]
    payload = json.loads(record["env_features_json"])
    assert payload["environment_risk_score"] == 0.6
    assert payload["overall_state"] == "LOW"
