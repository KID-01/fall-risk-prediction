"""
关键点提取模块 — MediaPipe Pose 2D人体关键点
输出33个关键点的2D坐标及可见度评分
"""
from __future__ import annotations

import numpy as np

from src.data.video_capture import VideoFrame
from src.utils.config import get_config
from src.utils.keypoints import KeypointFrame


class KeypointExtractor:
    """MediaPipe Pose 关键点提取器"""

    def __init__(self, model_complexity: int | None = None):
        config = get_config()
        self.model_complexity = model_complexity if model_complexity is not None else config.pose_estimation.model_complexity
        self.confidence_threshold = config.pose_estimation.confidence_threshold
        self.min_visible_lower = config.pose_estimation.min_visible_lower_keypoints
        self._pose = None

    def _ensure_model(self):
        """延迟加载MediaPipe模型"""
        if self._pose is None:
            try:
                import mediapipe as mp
            except ImportError as e:
                raise ImportError(
                    "未安装 mediapipe,请运行: pip install mediapipe"
                ) from e
            self._mp_pose = mp.solutions.pose
            self._pose = self._mp_pose.Pose(
                static_image_mode=False,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.confidence_threshold,
                min_tracking_confidence=self.confidence_threshold,
                smooth_landmarks=True,
            )

    def extract(self, video_frame: VideoFrame) -> KeypointFrame | None:
        """
        从视频帧中提取关键点
        Returns: KeypointFrame 或 None(未检测到人体)
        """
        self._ensure_model()
        import cv2

        rgb = cv2.cvtColor(video_frame.frame, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)

        if result.pose_landmarks is None:
            return None

        # 构建 (33, 4) 数组 [x, y, z, visibility]
        keypoints = np.zeros((33, 4), dtype=np.float32)
        for i, lm in enumerate(result.pose_landmarks.landmark):
            keypoints[i] = [lm.x, lm.y, lm.z, lm.visibility]

        kp_frame = KeypointFrame(
            timestamp=video_frame.timestamp,
            keypoints=keypoints,
        )

        # 帧质量检查
        from src.utils.keypoints import check_frame_quality
        is_valid, reason = check_frame_quality(
            kp_frame,
            self.confidence_threshold,
            self.min_visible_lower,
        )
        kp_frame.is_valid = is_valid
        kp_frame.invalid_reason = reason

        return kp_frame

    def close(self):
        if self._pose is not None:
            self._pose.close()
            self._pose = None

    def __del__(self):
        self.close()
