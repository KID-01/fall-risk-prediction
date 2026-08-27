"""Environment-aware engineering risk analysis.

The scores produced here are deterministic engineering heuristics. They are
not calibrated fall probabilities and must not be presented as such.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig

from src.utils.config import get_config


@dataclass
class EnvironmentAnalysisResult:
    timestamp: float
    motion_heuristic_score: float | None
    environment_risk_score: float | None
    motion_state: str
    environment_state: str
    overall_state: str
    person_present: bool
    context_elevated: bool = False
    top_hazards: list[dict] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _iou(a: dict, b: dict) -> float:
    x1, y1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    x2, y2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(1e-6, (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]))
    area_b = max(1e-6, (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))
    return intersection / (area_a + area_b - intersection)


def choose_primary(persons: list[dict], previous: dict | None, min_confidence: float) -> dict | None:
    candidates = [p for p in persons if p.get("conf", 0.0) >= min_confidence]
    if not candidates:
        return None
    if previous is not None:
        overlapping = [p for p in candidates if _iou(previous, p) > 0.0]
        if overlapping:
            return max(overlapping, key=_box_area)
    return max(candidates, key=_box_area)


def _box_area(box: dict) -> float:
    return max(0.0, box["x2"] - box["x1"]) * max(0.0, box["y2"] - box["y1"])


def _point_box_distance(px: float, py: float, box: dict) -> float:
    dx = max(box["x1"] - px, 0.0, px - box["x2"])
    dy = max(box["y1"] - py, 0.0, py - box["y2"])
    return float((dx * dx + dy * dy) ** 0.5)


def proximity_factor(distance: float, near_threshold: float, far_threshold: float) -> float:
    if distance <= near_threshold:
        return 1.0
    if distance >= far_threshold:
        return 0.0
    return 1.0 - (distance - near_threshold) / (far_threshold - near_threshold)


def _foot_band_factor(obj: dict, person: dict) -> float:
    middle = (person["y1"] + person["y2"]) / 2.0
    overlap = max(0.0, min(obj["y2"], person["y2"]) - max(obj["y1"], middle))
    return min(1.0, overlap / max(obj["y2"] - obj["y1"], 1e-3) * 2.0)


def compute_environment_risk(person: dict | None, objects: list[dict], config: Any) -> tuple[float | None, list[dict]]:
    if person is None:
        return None, []

    weights = dict(config.get("object_risk_table", {}))
    neutral = set(config.get("neutral_classes", []))
    object_confidence = float(config.get("object_confidence_threshold", 0.25))
    near_threshold = float(config.get("near_threshold", 0.4))
    far_threshold = float(config.get("far_threshold", 1.5))
    use_foot_band = bool(config.get("foot_band_gate", True))
    top_k = int(config.get("top_k_hazards", 3))

    person_height = max(person["y2"] - person["y1"], 1e-3)
    foot_x = (person["x1"] + person["x2"]) / 2.0
    foot_y = person["y2"]
    total = 0.0
    hazards: list[dict] = []

    for obj in objects:
        label = obj.get("label", "")
        confidence = float(obj.get("conf", 0.0))
        if label in neutral or label not in weights or confidence < object_confidence:
            continue
        normalized_distance = _point_box_distance(foot_x, foot_y, obj) / person_height
        band_factor = _foot_band_factor(obj, person) if use_foot_band else 1.0
        contribution = (
            float(weights[label])
            * proximity_factor(normalized_distance, near_threshold, far_threshold)
            * min(confidence, 1.0)
            * band_factor
        )
        total += contribution
        if contribution > 0:
            hazards.append(
                {
                    "class": label,
                    "confidence": round(confidence, 4),
                    "normalized_distance": round(normalized_distance, 3),
                    "risk_contribution": round(contribution, 4),
                }
            )

    hazards.sort(key=lambda item: -item["risk_contribution"])
    return round(min(total, 1.0), 4), hazards[:top_k]


def state_from_score(score: float | None, thresholds: tuple[float, float]) -> str:
    if score is None:
        return "UNKNOWN"
    if score < thresholds[0]:
        return "LOW"
    if score >= thresholds[1]:
        return "HIGH"
    return "MEDIUM"


def fuse_states(motion_state: str, environment_state: str) -> tuple[str, bool, list[str]]:
    if motion_state == "UNKNOWN":
        reasons = ["motion_missing"]
        if environment_state != "UNKNOWN":
            reasons.append(f"environment_{environment_state.lower()}")
        return "UNKNOWN", False, reasons

    reasons = [f"motion_{motion_state.lower()}"]
    if environment_state != "UNKNOWN":
        reasons.append(f"environment_{environment_state.lower()}")

    context_elevated = motion_state == "LOW" and environment_state in {"MEDIUM", "HIGH"}
    if context_elevated:
        reasons.append("context_elevated")
        return "LOW", True, reasons
    if motion_state == "HIGH":
        return "HIGH", False, reasons
    if motion_state == "MEDIUM" and environment_state == "HIGH":
        return "HIGH", False, reasons
    if motion_state == "MEDIUM":
        return "MEDIUM", False, reasons
    return "LOW", False, reasons


def causal_persist(states: list[str], window_size: int) -> str:
    if not states:
        return "UNKNOWN"
    if window_size <= 1:
        return states[-1]
    window = states[-window_size:]
    known = [state for state in window if state != "UNKNOWN"]
    if not known:
        return "UNKNOWN"
    best, count = Counter(known).most_common(1)[0]
    return best if count > len(window) / 2 else (window[-1] if window[-1] != "UNKNOWN" else best)


class EnvironmentRiskAnalyzer:
    """Run COCO object detection and deterministic environment-aware fusion."""

    def __init__(self, config: DictConfig | dict | None = None, model: Any | None = None):
        root_config = config if config is not None else get_config()
        self.config = root_config.get("environment", root_config)
        model_path = Path(str(self.config.get("model_path", "checkpoints/yolo26n.pt"))).expanduser()
        if not model_path.is_absolute():
            model_path = Path(__file__).parents[2] / model_path
        self.model_path = model_path
        self.device = str(self.config.get("device", "cpu"))
        self.confidence_threshold = float(self.config.get("confidence_threshold", 0.25))
        self.person_confidence_threshold = float(
            self.config.get("person_confidence_threshold", self.confidence_threshold)
        )
        self.motion_thresholds = tuple(self.config.get("motion_thresholds", [0.3, 0.6]))
        self.environment_thresholds = tuple(self.config.get("environment_thresholds", [0.4, 0.7]))
        self.persistence_frames = int(self.config.get("persistence_frames", 3))
        self._model = model
        self._previous_person: dict | None = None
        self._previous_motion: tuple[float, float, float] | None = None
        self._state_history: list[str] = []

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.is_file():
            raise RuntimeError(f"环境检测权重不存在: {self.model_path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("未安装 ultralytics，无法启用环境检测") from exc
        self._model = YOLO(str(self.model_path))

    def _parse_detections(self, result: Any) -> tuple[list[dict], list[dict]]:
        objects: list[dict] = []
        persons: list[dict] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return persons, objects

        names = getattr(result, "names", None) or getattr(self._model, "names", {})
        xyxy = np.asarray(boxes.xyxy.cpu().numpy())
        if len(xyxy) == 0:
            return persons, objects
        confidences = np.asarray(boxes.conf.cpu().numpy())
        classes = np.asarray(boxes.cls.int().cpu().numpy())
        for index, coords in enumerate(xyxy):
            class_id = int(classes[index])
            label = names.get(class_id, str(class_id)) if hasattr(names, "get") else names[class_id]
            item = {
                "label": str(label),
                "conf": float(confidences[index]),
                "x1": float(coords[0]),
                "y1": float(coords[1]),
                "x2": float(coords[2]),
                "y2": float(coords[3]),
            }
            (persons if item["label"] == "person" else objects).append(item)
        return persons, objects

    def _motion_score(self, person: dict | None, source_timestamp: float) -> float | None:
        if person is None:
            self._previous_motion = None
            return None
        center_y = (person["y1"] + person["y2"]) / 2.0
        height = max(person["y2"] - person["y1"], 1e-3)
        score = 0.0
        if self._previous_motion is not None:
            previous_time, previous_center, previous_height = self._previous_motion
            elapsed = source_timestamp - previous_time
            if elapsed > 1e-3:
                downward_velocity = (center_y - previous_center) / elapsed / previous_height
                score = max(0.0, min(1.0, downward_velocity / 5.0))
        self._previous_motion = (source_timestamp, center_y, height)
        return round(score, 4)

    def analyze(
        self,
        frame: np.ndarray,
        source_timestamp: float,
        observed_at: float,
    ) -> EnvironmentAnalysisResult:
        self._ensure_model()
        results = self._model.predict(
            source=frame,
            conf=self.confidence_threshold,
            imgsz=int(self.config.get("image_size", 640)),
            device=self.device,
            verbose=False,
        )
        if not results:
            raise RuntimeError("环境检测模型未返回结果")

        persons, objects = self._parse_detections(results[0])
        person = choose_primary(persons, self._previous_person, self.person_confidence_threshold)
        self._previous_person = person
        motion_score = self._motion_score(person, source_timestamp)
        environment_score, hazards = compute_environment_risk(person, objects, self.config)
        motion_state = state_from_score(motion_score, self.motion_thresholds)
        environment_state = state_from_score(environment_score, self.environment_thresholds)
        overall_state, context_elevated, reasons = fuse_states(motion_state, environment_state)
        self._state_history.append(overall_state)
        self._state_history = self._state_history[-max(self.persistence_frames, 1):]
        persisted_state = causal_persist(self._state_history, self.persistence_frames)

        return EnvironmentAnalysisResult(
            timestamp=observed_at,
            motion_heuristic_score=motion_score,
            environment_risk_score=environment_score,
            motion_state=motion_state,
            environment_state=environment_state,
            overall_state=persisted_state,
            person_present=person is not None,
            context_elevated=context_elevated,
            top_hazards=hazards,
            reason_codes=reasons,
        )

    def close(self) -> None:
        self._model = None
        self._previous_person = None
        self._previous_motion = None
        self._state_history.clear()
