"""
帧质量过滤模块 — 置信度过滤 + 下肢可见性检查
仅保留质量足够的关键点帧用于后续特征分析
"""
from __future__ import annotations

from src.utils.keypoints import KeypointFrame, check_frame_quality


class FrameFilter:
    """关键点帧质量过滤器"""

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        min_visible_lower: int = 4,
    ):
        self.confidence_threshold = confidence_threshold
        self.min_visible_lower = min_visible_lower

    def filter(self, frame: KeypointFrame) -> KeypointFrame:
        """过滤单帧,更新is_valid标记"""
        is_valid, reason = check_frame_quality(
            frame, self.confidence_threshold, self.min_visible_lower
        )
        frame.is_valid = is_valid
        frame.invalid_reason = reason
        return frame

    def filter_batch(self, frames: list[KeypointFrame]) -> list[KeypointFrame]:
        """批量过滤,返回有效帧列表"""
        valid_frames = []
        for frame in frames:
            self.filter(frame)
            if frame.is_valid:
                valid_frames.append(frame)
        return valid_frames
