from __future__ import annotations

import numpy as np

from src.data.environment_detector import EnvironmentBox
from src.data.human_detector import DetectionBox, PrimaryPersonTracker
from src.inference.visual_risk import (
    CausalTrajectoryProvider,
    EngineeringRiskFusion,
    compute_clutter_risk,
    compute_environment_risk,
    compute_interaction_risk,
    compute_lighting_risk,
    unavailable_wet_floor,
)


def _person(x1=100, y1=40, x2=160, y2=180, confidence=0.9):
    return DetectionBox(x1, y1, x2, y2, confidence)


class TestPrimaryPersonTracker:
    def test_keeps_previous_person_when_larger_visitor_appears(self):
        tracker = PrimaryPersonTracker()
        original = _person(100, 40, 160, 180)
        assert tracker.select([original]) == original
        same_person = _person(104, 42, 164, 182, 0.85)
        visitor = _person(300, 20, 430, 220, 0.99)
        assert tracker.select([visitor, same_person]) == same_person


class TestEnvironmentRisk:
    def setup_method(self):
        self.config = {
            "object_risk_table": {"chair": 0.6, "suitcase": 0.7},
            "neutral_classes": [],
            "object_conf_min": 0.25,
            "near_thr": 0.4,
            "far_thr": 1.5,
            "foot_band_gate": True,
            "risk_clip": 1.0,
            "top_k_hazards": 3,
        }

    def test_near_foot_hazard_scores_higher_than_far_object(self):
        person = _person(300, 200, 360, 400)
        far = EnvironmentBox(20, 20, 80, 120, 0.9, "chair")
        near = EnvironmentBox(330, 380, 375, 405, 0.91, "suitcase")
        far_result = compute_environment_risk(person, [far], self.config)
        near_result = compute_environment_risk(person, [near], self.config)
        assert near_result["score"] > far_result["score"]
        assert near_result["top_hazards"][0]["normalized_distance"] < 0.4

    def test_missing_person_is_unknown(self):
        result = compute_environment_risk(None, [], self.config)
        assert result["available"] is False
        assert result["state"] == "UNKNOWN"


class TestRiskExtensions:
    def test_lighting_dark_is_higher_than_bright(self):
        config = {
            "dark_pixel": 45,
            "mean_dark": 70.0,
            "contrast_low": 25.0,
            "dark_ratio_high": 0.55,
        }
        dark = compute_lighting_risk(np.zeros((32, 32, 3), dtype=np.uint8), config)
        bright = compute_lighting_risk(np.full((32, 32, 3), 220, dtype=np.uint8), config)
        assert dark["risk_index"] > bright["risk_index"]
        assert dark["state"] == "HIGH"

    def test_clutter_and_interaction_use_person_corridor(self):
        person = _person()
        chair = EnvironmentBox(105, 160, 180, 230, 0.9, "chair")
        clutter = compute_clutter_risk(
            person,
            [chair],
            {
                "obstacle_classes": ["chair"],
                "corridor_width_ratio": 0.65,
                "corridor_length_ratio": 1.25,
                "count_high": 3,
                "top_k": 3,
            },
            (240, 320, 3),
        )
        assert clutter["risk_index"] > 0

        trajectory = {
            "available": True,
            "horizon_s": 1.0,
            "predicted_points": [[100, 100], [120, 120], [140, 170]],
        }
        interaction = compute_interaction_risk(
            trajectory,
            [chair],
            unavailable_wet_floor(),
            {"obstacle_classes": ["chair"], "corridor_radius_px": 20},
        )
        assert interaction["risk_index"] > 0
        assert interaction["intersections"][0]["class"] == "chair"

    def test_interaction_scores_path_overlap_from_v032(self):
        trajectory = {
            "available": True,
            "horizon_s": 1.0,
            "predicted_points": [[100, 100], [120, 120], [140, 140], [160, 160]],
        }
        chair = EnvironmentBox(130, 130, 170, 180, 0.9, "chair")
        result = compute_interaction_risk(
            trajectory,
            [chair],
            unavailable_wet_floor(),
            {"obstacle_classes": ["chair"], "corridor_radius_px": 20},
        )
        intersection = result["intersections"][0]
        assert intersection["path_overlap_ratio"] > 0
        assert result["evidence"]["max_path_overlap_ratio"] == intersection["path_overlap_ratio"]

        # A corridor hit is represented explicitly and contributes to the score.
        farther = EnvironmentBox(170, 170, 200, 210, 0.9, "chair")
        overlap_result = compute_interaction_risk(
            trajectory,
            [farther],
            unavailable_wet_floor(),
            {"obstacle_classes": ["chair"], "corridor_radius_px": 20},
        )
        assert overlap_result["intersections"][0]["path_overlap_ratio"] > 0

    def test_trajectory_requires_history(self):
        provider = CausalTrajectoryProvider(
            {
                "history_points": 8,
                "min_points": 4,
                "horizon_s": 1.0,
                "steps": 5,
                "moving_speed_px_s": 8,
                "speed_scale_px_s": 100,
            }
        )
        assert provider.update(0.0, _person())["available"] is False
        for index in range(1, 4):
            result = provider.update(index * 0.1, _person(100 + index * 2, 40, 160 + index * 2, 180))
        assert result["available"] is True
        assert len(result["predicted_points"]) == 5

    def test_wet_floor_is_explicit_unknown(self):
        result = unavailable_wet_floor()
        assert result["available"] is False
        assert result["state"] == "UNKNOWN"
        assert result["risk_index"] is None


class TestEngineeringRiskFusion:
    def test_environment_high_alone_stops_at_warning_after_confirmation(self):
        fusion = EngineeringRiskFusion(3, 5)
        for _ in range(2):
            decision = fusion.evaluate(
                human_index=10,
                environment_index=90,
                interaction_index=0,
                base_level="low",
            )
            assert decision.overall_level == "low"
        decision = fusion.evaluate(
            human_index=10,
            environment_index=90,
            interaction_index=0,
            base_level="low",
        )
        assert decision.overall_level == "warning"
        assert decision.overall_score == 74.9

    def test_human_and_environment_high_reaches_critical_after_confirmation(self):
        fusion = EngineeringRiskFusion(3, 5)
        for _ in range(3):
            decision = fusion.evaluate(
                human_index=50,
                environment_index=90,
                interaction_index=0,
                base_level="low",
            )
        assert decision.overall_level == "critical"
        assert decision.overall_score == 90

    def test_acute_human_critical_is_immediate(self):
        fusion = EngineeringRiskFusion(3, 5)
        decision = fusion.evaluate(
            human_index=80,
            environment_index=0,
            interaction_index=0,
            base_level="critical",
            acute_critical=True,
        )
        assert decision.overall_level == "critical"

    def test_downgrade_requires_five_confirmations(self):
        fusion = EngineeringRiskFusion(3, 5)
        for _ in range(3):
            fusion.evaluate(
                human_index=10,
                environment_index=90,
                interaction_index=0,
                base_level="low",
            )
        for _ in range(4):
            decision = fusion.evaluate(
                human_index=10,
                environment_index=10,
                interaction_index=0,
                base_level="low",
            )
            assert decision.overall_level == "warning"
        decision = fusion.evaluate(
            human_index=10,
            environment_index=10,
            interaction_index=0,
            base_level="low",
        )
        assert decision.overall_level == "low"
