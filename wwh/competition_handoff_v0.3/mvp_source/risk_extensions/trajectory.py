"""Causal linear trajectory extrapolation from historical foot points."""
from __future__ import annotations

from collections import deque

import numpy as np


class CausalTrajectoryProvider:
    """Keep historical bottom-center points and extrapolate without future frames."""

    def __init__(self, config: dict):
        self.config = config
        self.history = deque(maxlen=int(config.get("history_points", 12)))

    def reset(self) -> None:
        self.history.clear()

    def update(self, timestamp: float, person: dict | None) -> dict:
        if person is None:
            return {
                "source": "causal_linear_trajectory_v0",
                "available": False,
                "state": "UNKNOWN",
                "risk_index": None,
                "quality": "person_missing",
                "observed_points": [],
                "predicted_points": [],
            }
        point = (
            float(timestamp),
            (float(person["x1"]) + float(person["x2"])) / 2.0,
            float(person["y2"]),
        )
        self.history.append(point)
        min_points = int(self.config.get("min_points", 4))
        if len(self.history) < min_points:
            return {
                "source": "causal_linear_trajectory_v0",
                "available": False,
                "state": "UNKNOWN",
                "risk_index": None,
                "quality": "insufficient_history",
                "observed_points": [[round(x, 2), round(y, 2)] for _, x, y in self.history],
                "predicted_points": [],
            }
        arr = np.asarray(self.history, dtype=float)
        times = arr[:, 0] - arr[-1, 0]
        if float(np.ptp(times)) <= 1e-6:
            return {
                "source": "causal_linear_trajectory_v0",
                "available": False,
                "state": "UNKNOWN",
                "risk_index": None,
                "quality": "zero_time_span",
                "observed_points": [[round(x, 2), round(y, 2)] for _, x, y in self.history],
                "predicted_points": [],
            }
        vx, bx = np.polyfit(times, arr[:, 1], 1)
        vy, by = np.polyfit(times, arr[:, 2], 1)
        horizon = float(self.config.get("horizon_s", 1.0))
        steps = max(int(self.config.get("steps", 6)), 2)
        future = np.linspace(0.0, horizon, steps + 1)[1:]
        predicted = [[float(vx * t + bx), float(vy * t + by)] for t in future]
        speed = float((vx**2 + vy**2) ** 0.5)
        return {
            "source": "causal_linear_trajectory_v0",
            "available": True,
            "state": "LOW" if speed < float(self.config.get("moving_speed_px_s", 8.0)) else "MEDIUM",
            "risk_index": round(min(100.0, speed / max(float(self.config.get("speed_scale_px_s", 100.0)), 1.0) * 100), 2),
            "quality": "OK",
            "velocity_px_s": [round(float(vx), 2), round(float(vy), 2)],
            "observed_points": [[round(x, 2), round(y, 2)] for _, x, y in self.history],
            "predicted_points": [[round(x, 2), round(y, 2)] for x, y in predicted],
            "horizon_s": horizon,
        }
