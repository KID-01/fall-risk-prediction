from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from src.data.environment_detector import EnvironmentDetector


def _box(label_id: int, confidence: float, coords: list[float]):
    box = MagicMock()
    box.cls = np.array([label_id])
    box.conf = np.array([confidence])
    box.xyxy = np.array([coords], dtype=float)
    return box


def test_detector_separates_persons_from_environment_objects():
    detector = EnvironmentDetector()
    model = MagicMock()
    result = MagicMock()
    result.names = {0: "person", 56: "chair"}
    result.boxes = [
        _box(0, 0.95, [10, 20, 100, 220]),
        _box(56, 0.88, [120, 100, 240, 260]),
    ]
    model.return_value = [result]
    detector._model = model

    detected = detector.detect_result(np.zeros((300, 400, 3), dtype=np.uint8))

    assert len(detected.persons) == 1
    assert len(detected.objects) == 1
    assert detected.objects[0].label == "chair"
    assert detector.detect(np.zeros((300, 400, 3), dtype=np.uint8))[0].label == "chair"
