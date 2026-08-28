# -*- coding: utf-8 -*-
"""Unit tests for optional v0.3 engineering risk extensions."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from fall_mvp.risk_extensions import (  # noqa: E402
    CausalTrajectoryProvider,
    compute_clutter_risk,
    compute_interaction_risk,
    compute_lighting_risk,
    unavailable_wet_floor,
)


class TestRiskExtensions(unittest.TestCase):
    def test_lighting_dark_is_higher_risk_than_bright(self):
        cfg = {"dark_pixel": 45, "mean_dark": 70.0, "contrast_low": 25.0, "dark_ratio_high": 0.55}
        dark = np.zeros((32, 32, 3), dtype=np.uint8)
        bright = np.full((32, 32, 3), 220, dtype=np.uint8)
        dark_result = compute_lighting_risk(dark, cfg)
        bright_result = compute_lighting_risk(bright, cfg)
        self.assertGreater(dark_result["risk_index"], bright_result["risk_index"])
        self.assertEqual(dark_result["state"], "HIGH")

    def test_clutter_detects_obstacle_in_foot_corridor(self):
        cfg = {
            "obstacle_classes": ["chair"],
            "corridor_width_ratio": 0.65,
            "corridor_length_ratio": 1.25,
            "count_high": 3,
            "top_k": 3,
        }
        person = {"x1": 100, "y1": 40, "x2": 160, "y2": 180}
        chair = {"label": "chair", "conf": 0.9, "x1": 105, "y1": 160, "x2": 180, "y2": 230}
        result = compute_clutter_risk(person, [chair], cfg, (240, 320, 3))
        self.assertTrue(result["available"])
        self.assertGreater(result["risk_index"], 0)
        self.assertEqual(result["obstacles"][0]["class"], "chair")

    def test_trajectory_requires_history_then_extrapolates(self):
        provider = CausalTrajectoryProvider(
            {"history_points": 8, "min_points": 4, "horizon_s": 1.0, "steps": 5,
             "moving_speed_px_s": 8.0, "speed_scale_px_s": 100.0}
        )
        person = {"x1": 10, "x2": 30, "y2": 50}
        first = provider.update(0.0, person)
        self.assertFalse(first["available"])
        for i in range(1, 4):
            moving = {"x1": 10 + i * 2, "x2": 30 + i * 2, "y2": 50 + i * 3}
            result = provider.update(i * 0.1, moving)
        self.assertTrue(result["available"])
        self.assertEqual(len(result["predicted_points"]), 5)
        self.assertGreater(result["predicted_points"][-1][0], result["observed_points"][-1][0])

    def test_interaction_detects_predicted_path_near_obstacle(self):
        trajectory = {
            "available": True,
            "horizon_s": 1.0,
            "predicted_points": [[100, 100], [120, 120], [140, 140], [160, 160]],
        }
        objects = [{"label": "chair", "conf": 0.9, "x1": 130, "y1": 130, "x2": 170, "y2": 180}]
        wet = unavailable_wet_floor()
        cfg = {"obstacle_classes": ["chair"], "corridor_radius_px": 20.0}
        result = compute_interaction_risk(trajectory, objects, wet, cfg)
        self.assertTrue(result["available"])
        self.assertGreater(result["risk_index"], 0)
        self.assertEqual(result["intersections"][0]["class"], "chair")

    def test_unavailable_wet_floor_is_explicit_unknown(self):
        result = unavailable_wet_floor()
        self.assertFalse(result["available"])
        self.assertEqual(result["state"], "UNKNOWN")
        self.assertIsNone(result["risk_index"])
        self.assertEqual(result["regions"], [])


if __name__ == "__main__":
    unittest.main()
