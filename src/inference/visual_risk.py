"""人体、环境、轨迹交互与综合工程风险计算。"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.data.environment_detector import EnvironmentBox
from src.data.human_detector import DetectionBox

LEVEL_PRIORITY = {"low": 0, "attention": 1, "warning": 2, "critical": 3}
LEVEL_BANDS = {
    "low": (0.0, 29.9),
    "attention": (30.0, 49.9),
    "warning": (50.0, 74.9),
    "critical": (75.0, 100.0),
}


def risk_state(index: float | None, thresholds: tuple[float, float] = (40.0, 70.0)) -> str:
    if index is None:
        return "UNKNOWN"
    if index < thresholds[0]:
        return "LOW"
    if index < thresholds[1]:
        return "MEDIUM"
    return "HIGH"


def provider_result(
    source: str,
    index: float | None,
    *,
    evidence: dict | None = None,
    reasons: list[str] | None = None,
    available: bool = True,
    **extra,
) -> dict:
    result = {
        "source": source,
        "available": bool(available),
        "risk_index": None if index is None else round(float(np.clip(index, 0, 100)), 2),
        "state": risk_state(index) if available else "UNKNOWN",
        "evidence": evidence or {},
        "reason_codes": reasons or [],
    }
    result.update(extra)
    return result


def compute_lighting_risk(frame: np.ndarray | None, config: dict) -> dict:
    if frame is None or frame.size == 0:
        return provider_result(
            "lighting_v0", None, available=False, reasons=["frame_unavailable"]
        )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))
    contrast = float(np.std(gray))
    dark_ratio = float(np.mean(gray < int(config.get("dark_pixel", 45))))
    mean_threshold = max(float(config.get("mean_dark", 70.0)), 1.0)
    contrast_threshold = max(float(config.get("contrast_low", 25.0)), 1.0)
    dark_ratio_threshold = max(float(config.get("dark_ratio_high", 0.55)), 1e-6)
    darkness = np.clip((mean_threshold - mean) / mean_threshold, 0, 1)
    low_contrast = np.clip((contrast_threshold - contrast) / contrast_threshold, 0, 1)
    dark_excess = np.clip(dark_ratio / dark_ratio_threshold, 0, 1)
    index = 100 * (0.5 * darkness + 0.3 * dark_excess + 0.2 * low_contrast)
    reasons: list[str] = []
    if mean < mean_threshold:
        reasons.append("low_mean_luminance")
    if dark_ratio >= dark_ratio_threshold:
        reasons.append("high_dark_pixel_ratio")
    if contrast < contrast_threshold:
        reasons.append("low_contrast")
    return provider_result(
        "lighting_v0",
        float(index),
        evidence={
            "mean_luminance": round(mean, 2),
            "contrast_std": round(contrast, 2),
            "dark_pixel_ratio": round(dark_ratio, 4),
        },
        reasons=reasons,
    )


def _point_box_distance(px: float, py: float, box: EnvironmentBox) -> float:
    dx = max(box.x1 - px, 0.0, px - box.x2)
    dy = max(box.y1 - py, 0.0, py - box.y2)
    return math.hypot(dx, dy)


def _proximity_factor(distance: float, near_threshold: float, far_threshold: float) -> float:
    if distance <= near_threshold:
        return 1.0
    if distance >= far_threshold:
        return 0.0
    return 1.0 - (distance - near_threshold) / (far_threshold - near_threshold)


def _foot_band_factor(obj: EnvironmentBox, person: DetectionBox) -> float:
    midpoint = (person.y1 + person.y2) / 2.0
    overlap = max(0.0, min(obj.y2, person.y2) - max(obj.y1, midpoint))
    return min(1.0, overlap / max(obj.y2 - obj.y1, 1e-3) * 2.0)


def compute_environment_risk(
    person: DetectionBox | None,
    objects: list[EnvironmentBox],
    config: dict,
) -> dict:
    if person is None:
        return {
            "source": "env_risk_v0",
            "available": False,
            "score": None,
            "state": "UNKNOWN",
            "top_hazards": [],
            "reason_codes": ["person_missing"],
        }

    weights = dict(config.get("object_risk_table", {}))
    neutral = set(config.get("neutral_classes", []))
    confidence_min = float(config.get("object_conf_min", 0.25))
    near_threshold = float(config.get("near_thr", 0.4))
    far_threshold = float(config.get("far_thr", 1.5))
    foot_gate = bool(config.get("foot_band_gate", True))
    foot_x = (person.x1 + person.x2) / 2.0
    foot_y = person.y2
    person_height = max(person.height, 1e-3)
    total = 0.0
    hazards: list[dict] = []
    for obj in objects:
        if obj.label in neutral or obj.label not in weights or obj.confidence < confidence_min:
            continue
        normalized_distance = _point_box_distance(foot_x, foot_y, obj) / person_height
        foot_overlap = _foot_band_factor(obj, person) if foot_gate else 1.0
        contribution = (
            float(weights[obj.label])
            * _proximity_factor(normalized_distance, near_threshold, far_threshold)
            * min(obj.confidence, 1.0)
            * foot_overlap
        )
        total += contribution
        if contribution > 0:
            hazards.append(
                {
                    "class": obj.label,
                    "label": obj.label,
                    "confidence": round(obj.confidence, 4),
                    "normalized_distance": round(normalized_distance, 3),
                    "foot_band_overlap": round(foot_overlap, 3),
                    "risk_contribution": round(contribution, 4),
                    "bbox": [obj.x1, obj.y1, obj.x2, obj.y2],
                }
            )
    score = min(float(config.get("risk_clip", 1.0)), total)
    hazards.sort(key=lambda item: item["risk_contribution"], reverse=True)
    state = risk_state(score * 100)
    return {
        "source": "env_risk_v0",
        "available": True,
        "score": round(score, 4),
        "state": state,
        "top_hazards": hazards[: int(config.get("top_k_hazards", 3))],
        "reason_codes": [f"environment_{state.lower()}"],
    }


def _intersection(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def compute_clutter_risk(
    person: DetectionBox | None,
    objects: list[EnvironmentBox],
    config: dict,
    frame_shape: tuple[int, ...],
) -> dict:
    if person is None:
        return provider_result("clutter_v0", None, available=False, reasons=["person_missing"])
    image_height, image_width = frame_shape[:2]
    foot_x = (person.x1 + person.x2) / 2.0
    half_width = float(config.get("corridor_width_ratio", 0.65)) * max(person.width, 1.0)
    length = float(config.get("corridor_length_ratio", 1.25)) * max(person.height, 1.0)
    corridor = (
        max(0.0, foot_x - half_width),
        max(0.0, person.y2 - 0.15 * person.height),
        min(float(image_width), foot_x + half_width),
        min(float(image_height), person.y2 + length),
    )
    allowed = set(config.get("obstacle_classes", []))
    overlaps: list[dict] = []
    for obj in objects:
        if allowed and obj.label not in allowed:
            continue
        box = (obj.x1, obj.y1, obj.x2, obj.y2)
        intersection = _intersection(corridor, box)
        if intersection <= 0:
            continue
        overlap = intersection / max((obj.x2 - obj.x1) * (obj.y2 - obj.y1), 1.0)
        overlaps.append(
            {
                "class": obj.label,
                "confidence": round(obj.confidence, 4),
                "corridor_overlap": round(overlap, 4),
                "bbox": [round(value, 2) for value in box],
            }
        )
    overlaps.sort(
        key=lambda item: item["corridor_overlap"] * item["confidence"], reverse=True
    )
    count_term = min(1.0, len(overlaps) / max(float(config.get("count_high", 3)), 1.0))
    overlap_term = max(
        (item["corridor_overlap"] * item["confidence"] for item in overlaps), default=0.0
    )
    index = 100 * (0.45 * count_term + 0.55 * min(1.0, overlap_term))
    return provider_result(
        "clutter_v0",
        index,
        evidence={
            "corridor": [round(value, 2) for value in corridor],
            "obstacle_count": len(overlaps),
        },
        reasons=["obstacle_in_foot_corridor"] if overlaps else [],
        obstacles=overlaps[: int(config.get("top_k", 3))],
    )


class CausalTrajectoryProvider:
    def __init__(self, config: dict):
        self.config = config
        self.history: deque[tuple[float, float, float]] = deque(
            maxlen=int(config.get("history_points", 12))
        )

    def reset(self) -> None:
        self.history.clear()

    def update(self, timestamp: float, person: DetectionBox | None) -> dict:
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
        point = (timestamp, (person.x1 + person.x2) / 2.0, person.y2)
        self.history.append(point)
        if len(self.history) < int(self.config.get("min_points", 4)):
            return {
                "source": "causal_linear_trajectory_v0",
                "available": False,
                "state": "UNKNOWN",
                "risk_index": None,
                "quality": "insufficient_history",
                "observed_points": [[round(x, 2), round(y, 2)] for _, x, y in self.history],
                "predicted_points": [],
            }
        values = np.asarray(self.history, dtype=float)
        times = values[:, 0] - values[-1, 0]
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
        velocity_x, intercept_x = np.polyfit(times, values[:, 1], 1)
        velocity_y, intercept_y = np.polyfit(times, values[:, 2], 1)
        horizon = float(self.config.get("horizon_s", 1.0))
        steps = max(int(self.config.get("steps", 6)), 2)
        future_times = np.linspace(0.0, horizon, steps + 1)[1:]
        predicted = [
            [float(velocity_x * value + intercept_x), float(velocity_y * value + intercept_y)]
            for value in future_times
        ]
        speed = math.hypot(float(velocity_x), float(velocity_y))
        index = min(
            100.0,
            speed / max(float(self.config.get("speed_scale_px_s", 100.0)), 1.0) * 100,
        )
        return {
            "source": "causal_linear_trajectory_v0",
            "available": True,
            "state": "LOW"
            if speed < float(self.config.get("moving_speed_px_s", 8.0))
            else "MEDIUM",
            "risk_index": round(index, 2),
            "quality": "OK",
            "velocity_px_s": [round(float(velocity_x), 2), round(float(velocity_y), 2)],
            "observed_points": [[round(x, 2), round(y, 2)] for _, x, y in self.history],
            "predicted_points": [[round(x, 2), round(y, 2)] for x, y in predicted],
            "horizon_s": horizon,
        }


def unavailable_wet_floor() -> dict:
    return provider_result(
        "wet_floor_unavailable",
        None,
        available=False,
        reasons=["detector_unavailable"],
        regions=[],
    )


def _point_distance_to_box(point: list[float], box: EnvironmentBox) -> float:
    x, y = point
    dx = max(box.x1 - x, 0.0, x - box.x2)
    dy = max(box.y1 - y, 0.0, y - box.y2)
    return math.hypot(dx, dy)


def _path_overlap_ratio(
    points: list[list[float]],
    box: tuple[float, float, float, float],
    radius: float,
) -> float:
    """预测路径点落入障碍物 corridor 扩展框的比例。"""
    x1, y1, x2, y2 = box
    hits = sum(
        x1 - radius <= point[0] <= x2 + radius
        and y1 - radius <= point[1] <= y2 + radius
        for point in points
    )
    return hits / len(points) if points else 0.0


def compute_interaction_risk(
    trajectory: dict,
    objects: list[EnvironmentBox],
    wet_floor: dict,
    config: dict,
) -> dict:
    if not trajectory.get("available") or not trajectory.get("predicted_points"):
        return provider_result(
            "interaction_v0",
            None,
            available=False,
            reasons=["trajectory_unavailable"],
            intersections=[],
        )
    allowed = set(config.get("obstacle_classes", []))
    radius = float(config.get("corridor_radius_px", 24.0))
    points = trajectory["predicted_points"]
    intersections: list[dict] = []
    for obj in objects:
        if allowed and obj.label not in allowed:
            continue
        distances = [_point_distance_to_box(point, obj) for point in points]
        minimum = min(distances)
        overlap = _path_overlap_ratio(
            points, (obj.x1, obj.y1, obj.x2, obj.y2), radius
        )
        if minimum <= radius or overlap > 0:
            step = distances.index(minimum)
            intersections.append(
                {
                    "type": "obstacle",
                    "class": obj.label,
                    "confidence": round(obj.confidence, 4),
                    "min_distance_px": round(minimum, 2),
                    "path_overlap_ratio": round(overlap, 4),
                    "time_to_interaction_s": round(
                        float(trajectory.get("horizon_s", 1.0)) * (step + 1) / len(points),
                        3,
                    ),
                }
            )
    if wet_floor.get("available"):
        for region in wet_floor.get("regions", []):
            values = region.get("bbox", [])
            if len(values) != 4:
                continue
            virtual = EnvironmentBox(*map(float, values), float(region.get("confidence", 0)), "wet_floor")
            distances = [_point_distance_to_box(point, virtual) for point in points]
            minimum = min(distances)
            overlap = _path_overlap_ratio(points, tuple(values), radius)
            if minimum <= radius or overlap > 0:
                step = distances.index(minimum)
                intersections.append(
                    {
                        "type": "wet_floor",
                        "class": "wet_floor",
                        "confidence": virtual.confidence,
                        "min_distance_px": round(minimum, 2),
                        "path_overlap_ratio": round(overlap, 4),
                        "time_to_interaction_s": round(
                            float(trajectory.get("horizon_s", 1.0)) * (step + 1) / len(points),
                            3,
                        ),
                    }
                )
    if not intersections:
        return provider_result(
            "interaction_v0", 0.0, evidence={"path_points": len(points)}, intersections=[]
        )
    nearest_time = min(item["time_to_interaction_s"] for item in intersections)
    confidence = max(item["confidence"] for item in intersections)
    overlap = max(item["path_overlap_ratio"] for item in intersections)
    urgency = max(
        0.0,
        1.0 - nearest_time / max(float(trajectory.get("horizon_s", 1.0)), 1e-6),
    )
    index = 100 * min(1.0, 0.55 * overlap + 0.25 * urgency + 0.2 * confidence)
    return provider_result(
        "interaction_v0",
        index,
        evidence={
            "path_points": len(points),
            "nearest_time_s": round(nearest_time, 3),
            "max_path_overlap_ratio": round(overlap, 4),
        },
        reasons=["predicted_path_intersects_hazard"],
        intersections=intersections,
    )


class MotionRiskTracker:
    """根据主人物框中心向下速度计算 motion_heuristic_v0。"""

    def __init__(self, scale: float = 5.0):
        self.scale = scale
        self.previous: tuple[float, float, float] | None = None

    def reset(self) -> None:
        self.previous = None

    def update(self, timestamp: float, person: DetectionBox | None) -> float | None:
        if person is None:
            self.previous = None
            return None
        center_y = (person.y1 + person.y2) / 2.0
        score = 0.0
        if self.previous is not None:
            previous_time, previous_center, previous_height = self.previous
            elapsed = timestamp - previous_time
            if elapsed > 1e-3:
                velocity = (center_y - previous_center) / elapsed / max(previous_height, 1e-3)
                score = float(np.clip(velocity / self.scale, 0, 1))
        self.previous = (timestamp, center_y, max(person.height, 1e-3))
        return score


@dataclass
class FusionDecision:
    overall_score: float
    overall_level: str
    candidate_level: str
    human_risk_index: float | None
    environment_risk_index: float | None
    interaction_risk_index: float | None
    reason_codes: list[str] = field(default_factory=list)
    pending_direction: str | None = None
    pending_count: int = 0
    pending_required: int = 0
    context_elevated: bool = False


class EngineeringRiskFusion:
    """分层保守融合，并对环境型升级和降级执行迟滞。"""

    def __init__(self, upgrade_confirmations: int = 3, downgrade_confirmations: int = 5):
        self.upgrade_confirmations = upgrade_confirmations
        self.downgrade_confirmations = downgrade_confirmations
        self.current_level = "low"
        self._pending_level: str | None = None
        self._pending_count = 0

    def reset(self) -> None:
        self.current_level = "low"
        self._pending_level = None
        self._pending_count = 0

    @staticmethod
    def _context_level(
        human_index: float | None,
        environment_index: float | None,
        interaction_index: float | None,
    ) -> tuple[str, list[str]]:
        human_state = risk_state(human_index)
        environment_state = risk_state(environment_index)
        interaction_state = risk_state(interaction_index)
        reasons = [
            f"human_{human_state.lower()}",
            f"environment_{environment_state.lower()}",
            f"interaction_{interaction_state.lower()}",
        ]
        if human_state == "HIGH":
            return "critical", reasons
        if human_state == "MEDIUM" and (
            environment_state == "HIGH" or interaction_state == "HIGH"
        ):
            reasons.append("human_context_compound")
            return "critical", reasons
        if environment_state == "HIGH" or interaction_state == "HIGH":
            reasons.append("context_high")
            return "warning", reasons
        if "MEDIUM" in (human_state, environment_state, interaction_state):
            return "attention", reasons
        return "low", reasons

    def evaluate(
        self,
        *,
        human_index: float | None,
        environment_index: float | None,
        interaction_index: float | None,
        base_level: str,
        base_reason_codes: list[str] | None = None,
        acute_critical: bool = False,
    ) -> FusionDecision:
        context_level, context_reasons = self._context_level(
            human_index, environment_index, interaction_index
        )
        base_level = base_level if base_level in LEVEL_PRIORITY else "low"
        target_level = max((base_level, context_level), key=LEVEL_PRIORITY.get)
        context_driven = LEVEL_PRIORITY[context_level] > LEVEL_PRIORITY[base_level]

        pending_direction = None
        pending_required = 0
        if acute_critical and target_level == "critical":
            self.current_level = "critical"
            self._pending_level = None
            self._pending_count = 0
        elif LEVEL_PRIORITY[target_level] > LEVEL_PRIORITY[self.current_level]:
            if not context_driven:
                self.current_level = target_level
                self._pending_level = None
                self._pending_count = 0
            else:
                pending_direction = "upgrade"
                pending_required = self.upgrade_confirmations
                self._advance_pending(target_level)
                if self._pending_count >= self.upgrade_confirmations:
                    self.current_level = target_level
                    self._pending_level = None
                    self._pending_count = 0
                    pending_direction = None
        elif LEVEL_PRIORITY[target_level] < LEVEL_PRIORITY[self.current_level]:
            pending_direction = "downgrade"
            pending_required = self.downgrade_confirmations
            self._advance_pending(target_level)
            if self._pending_count >= self.downgrade_confirmations:
                self.current_level = target_level
                self._pending_level = None
                self._pending_count = 0
                pending_direction = None
        else:
            self._pending_level = None
            self._pending_count = 0

        raw_score = max(
            [
                value
                for value in (human_index, environment_index, interaction_index)
                if value is not None
            ]
            or [0.0]
        )
        minimum, maximum = LEVEL_BANDS[self.current_level]
        score = float(np.clip(raw_score, minimum, maximum))
        reason_codes = list(dict.fromkeys((base_reason_codes or []) + context_reasons))
        if pending_direction:
            reason_codes.append(f"pending_{pending_direction}")
        return FusionDecision(
            overall_score=round(score, 2),
            overall_level=self.current_level,
            candidate_level=target_level,
            human_risk_index=human_index,
            environment_risk_index=environment_index,
            interaction_risk_index=interaction_index,
            reason_codes=reason_codes,
            pending_direction=pending_direction,
            pending_count=self._pending_count,
            pending_required=pending_required,
            context_elevated=LEVEL_PRIORITY[context_level] > LEVEL_PRIORITY[base_level],
        )

    def _advance_pending(self, target_level: str) -> None:
        if self._pending_level == target_level:
            self._pending_count += 1
        else:
            self._pending_level = target_level
            self._pending_count = 1
