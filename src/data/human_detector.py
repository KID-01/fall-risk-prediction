"""
人体检测模块 — YOLOv8n轻量级目标检测
仅当检测到完整人体时才触发后续关键点提取,节省计算资源
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.utils.config import get_config


@dataclass
class DetectionBox:
    """人体检测框"""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def is_full_body(self) -> bool:
        """判断是否为完整人体(框的高宽比 > 1.2)"""
        return self.width > 0 and self.height / self.width > 1.2


class HumanDetector:
    """YOLOv8n 人体检测器"""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        config = get_config()
        self.model_name = model_name or config.human_detection.model
        self.confidence_threshold = config.human_detection.confidence_threshold
        self.person_class_id = config.human_detection.person_class_id
        self.device = device or config.human_detection.device
        self._model = None

    def _ensure_model(self):
        """延迟加载模型(避免未安装ultralytics时启动报错)"""
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as e:
                raise ImportError("未安装 ultralytics,请运行: pip install ultralytics") from e
            config = get_config()
            model_dir = Path(config.paths.checkpoints)
            model_path = model_dir / f"{self.model_name}.pt"
            self._model = YOLO(str(model_path))

    def detect(self, frame: np.ndarray) -> list[DetectionBox]:
        """
        检测帧中的人体
        Returns: 人体检测框列表(按置信度降序)
        """
        self._ensure_model()
        results = self._model(
            frame,
            conf=self.confidence_threshold,
            classes=[self.person_class_id],
            device=self.device,
            verbose=False,
        )
        boxes: list[DetectionBox] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                boxes.append(
                    DetectionBox(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=float(box.conf[0]),
                    )
                )
        boxes.sort(key=lambda b: b.confidence, reverse=True)
        return boxes

    def detect_best(self, frame: np.ndarray) -> DetectionBox | None:
        """返回置信度最高的完整人体框,无则返回None"""
        boxes = self.detect(frame)
        for box in boxes:
            if box.is_full_body:
                return box
        return boxes[0] if boxes else None
