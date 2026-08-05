"""
数据预处理管线 — 帧提取 → 人体检测 → ROI裁剪
从视频帧中裁剪人体区域,resize到统一尺寸,保持宽高比(padding黑边)
"""
from __future__ import annotations

import cv2
import numpy as np

from src.data.human_detector import DetectionBox, HumanDetector
from src.utils.logger import get_logger

log = get_logger(__name__)


class ROICropper:
    """人体ROI裁剪器: 检测框外扩20%, resize到目标尺寸, padding黑边"""

    def __init__(
        self,
        target_size: tuple[int, int] = (256, 256),
        expand_ratio: float = 0.2,
    ):
        self.target_size = target_size
        self.expand_ratio = expand_ratio

    def crop(self, frame: np.ndarray, box: DetectionBox) -> np.ndarray:
        """
        裁剪人体ROI
        Args:
            frame: 原始帧 (H, W, 3) BGR
            box: 人体检测框
        Returns:
            cropped: (target_h, target_w, 3) BGR
        """
        h, w = frame.shape[:2]

        # 外扩20%
        bw = box.width
        bh = box.height
        expand_w = bw * self.expand_ratio
        expand_h = bh * self.expand_ratio

        x1 = int(max(0, box.x1 - expand_w / 2))
        y1 = int(max(0, box.y1 - expand_h / 2))
        x2 = int(min(w, box.x2 + expand_w / 2))
        y2 = int(min(h, box.y2 + expand_h / 2))

        cropped = frame[y1:y2, x1:x2]

        # resize保持宽高比, padding黑边
        return self._resize_with_padding(cropped)

    def _resize_with_padding(self, image: np.ndarray) -> np.ndarray:
        """resize到目标尺寸,保持宽高比,不足部分用黑边填充"""
        target_h, target_w = self.target_size
        h, w = image.shape[:2]

        if h == 0 or w == 0:
            return np.zeros((target_h, target_w, 3), dtype=np.uint8)

        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 创建黑色画布,居中放置
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        return canvas


class PreprocessPipeline:
    """完整预处理管线: 人体检测 → ROI裁剪"""

    def __init__(
        self,
        target_size: tuple[int, int] = (256, 256),
        confidence_threshold: float = 0.5,
    ):
        self.detector = HumanDetector()
        self.cropper = ROICropper(target_size=target_size)
        self.confidence_threshold = confidence_threshold

    def process(self, frame: np.ndarray) -> tuple[np.ndarray | None, DetectionBox | None]:
        """
        处理单帧: 检测人体 → 裁剪ROI
        Returns:
            (cropped_roi, detection_box) — 无人时返回 (None, None)
        """
        box = self.detector.detect_best(frame)
        if box is None:
            return None, None

        roi = self.cropper.crop(frame, box)
        log.debug(f"ROI裁剪: box=({box.x1:.0f},{box.y1:.0f},{box.x2:.0f},{box.y2:.0f}) conf={box.confidence:.2f}")
        return roi, box

    def process_batch(self, frames: list[np.ndarray]) -> list[tuple[np.ndarray, DetectionBox]]:
        """批量处理,返回成功裁剪的 (roi, box) 列表"""
        results = []
        for frame in frames:
            roi, box = self.process(frame)
            if roi is not None and box is not None:
                results.append((roi, box))
        return results
