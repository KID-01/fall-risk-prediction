"""YOLO 通用环境目标检测，明确拆分人物与环境目标。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data.human_detector import DetectionBox
from src.utils.config import get_config


@dataclass(frozen=True)
class EnvironmentBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str

    def to_dict(self) -> dict[str, float | str]:
        return {
            "label": self.label,
            "confidence": float(self.confidence),
            "x1": float(self.x1),
            "y1": float(self.y1),
            "x2": float(self.x2),
            "y2": float(self.y2),
        }


@dataclass(frozen=True)
class EnvironmentDetectionResult:
    persons: list[DetectionBox]
    objects: list[EnvironmentBox]


class EnvironmentDetector:
    """使用仓库中的 YOLO 通用检测权重识别环境目标。"""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        config = get_config()
        env_config = config.get("environment_detection", {})
        self.model_name = model_name or str(env_config.get("model", "yolo26n"))
        self.confidence_threshold = float(env_config.get("confidence_threshold", 0.25))
        self.image_size = int(env_config.get("image_size", 640))
        self.device = device or config.human_detection.device
        self._model = None
        self.error: str | None = None

    def _ensure_model(self):
        if self._model is not None:
            return
        from ultralytics import YOLO

        model_path = Path(get_config().paths.checkpoints) / f"{self.model_name}.pt"
        self._model = YOLO(str(model_path))

    def detect_result(self, frame: np.ndarray) -> EnvironmentDetectionResult:
        self._ensure_model()
        results = self._model(
            frame,
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        persons: list[DetectionBox] = []
        objects: list[EnvironmentBox] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = str(names.get(cls_id, cls_id) if isinstance(names, dict) else names[cls_id])
                if label == "person":
                    persons.append(DetectionBox(x1, y1, x2, y2, confidence))
                else:
                    objects.append(EnvironmentBox(x1, y1, x2, y2, confidence, label))
        persons.sort(key=lambda item: (item.area, item.confidence), reverse=True)
        objects.sort(key=lambda item: item.confidence, reverse=True)
        return EnvironmentDetectionResult(persons=persons, objects=objects)

    def detect(self, frame: np.ndarray) -> list[EnvironmentBox]:
        """兼容旧调用，只返回非人体环境目标。"""
        return self.detect_result(frame).objects
