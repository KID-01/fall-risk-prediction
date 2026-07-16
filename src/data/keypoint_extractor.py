"""
关键点提取模块 — MediaPipe Pose 2D人体关键点 (Tasks API)
输出33个关键点的2D坐标及可见度评分
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.video_capture import VideoFrame
from src.utils.config import get_config
from src.utils.keypoints import KeypointFrame

# 默认模型路径(置于 checkpoints/ 下)
_DEFAULT_MODEL_PATH = str(Path(__file__).parents[2] / "checkpoints" / "pose_landmarker_lite.task")


class KeypointExtractor:
    """MediaPipe PoseLandmarker 关键点提取器 (Tasks API)"""

    def __init__(self, model_path: str | None = None):
        config = get_config()
        self.confidence_threshold = config.pose_estimation.confidence_threshold
        self.min_visible_lower = config.pose_estimation.min_visible_lower_keypoints
        self._model_path = model_path or _DEFAULT_MODEL_PATH
        self._landmarker = None

    def _ensure_model(self):
        """延迟加载MediaPipe PoseLandmarker"""
        if self._landmarker is not None:
            return
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as e:
            raise ImportError("未安装 mediapipe (0.10+), 请运行: pip install mediapipe") from e

        model_path = self._model_path
        if not Path(model_path).exists():
            # 尝试从 checkpoints/ 加载
            fallback = str(Path(__file__).parents[2] / "checkpoints" / "pose_landmarker_lite.task")
            if Path(fallback).exists():
                model_path = fallback

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=self.confidence_threshold,
            min_tracking_confidence=self.confidence_threshold,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def extract(self, video_frame: VideoFrame) -> KeypointFrame | None:
        """
        从视频帧中提取关键点
        Returns: KeypointFrame 或 None(未检测到人体)
        """
        self._ensure_model()
        import cv2
        import mediapipe as mp

        rgb = cv2.cvtColor(video_frame.frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return None

        # 取第一个人体的关键点
        landmarks = result.pose_landmarks[0]

        # 构建 (33, 4) 数组 [x, y, z, visibility]
        keypoints = np.zeros((33, 4), dtype=np.float32)
        for i, lm in enumerate(landmarks):
            keypoints[i] = [lm.x, lm.y, lm.z, lm.visibility if hasattr(lm, "visibility") else 1.0]

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
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __del__(self):
        self.close()
