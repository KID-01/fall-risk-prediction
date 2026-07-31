"""
YOLO-Pose 关键点提取模块 — YOLOv8n-pose 人体姿态估计
将COCO 17关键点输出映射为与 MediaPipe 相同的33关键点 KeypointFrame 格式,
下游 features.py 等模块无需任何改动
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.data.video_capture import VideoFrame
from src.utils.config import get_config
from src.utils.keypoints import KeypointFrame, check_frame_quality, convert_coco_to_mediapipe
from src.utils.logger import get_logger

log = get_logger(__name__)

# 默认模型路径(置于 checkpoints/ 下)
_DEFAULT_MODEL_NAME = "yolov8n-pose.pt"
_DEFAULT_MODEL_PATH = str(Path(__file__).parents[2] / "checkpoints" / _DEFAULT_MODEL_NAME)


class YoloPoseExtractor:
    """YOLOv8n-pose 关键点提取器, 输出与 KeypointExtractor 相同的 (33,4) KeypointFrame"""

    def __init__(self, model_path: str | None = None, device: str | None = None):
        config = get_config()
        self.confidence_threshold = config.pose_estimation.confidence_threshold
        self.min_visible_lower = config.pose_estimation.min_visible_lower_keypoints
        self._model_path = model_path or _DEFAULT_MODEL_PATH
        self.device = device or config.human_detection.device
        self._model = None

    def _ensure_model(self):
        """延迟加载YOLOv8n-pose模型(缺失时自动下载到 checkpoints/)"""
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("未安装 ultralytics, 请运行: pip install ultralytics") from e

        model_path = Path(self._model_path)
        if not model_path.exists():
            self._download_model(model_path)
        self._model = YOLO(str(model_path))

    @staticmethod
    def _download_model(model_path: Path):
        """下载yolov8n-pose模型到指定路径"""
        log.warning(f"模型不存在, 开始下载: {model_path.name}")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from ultralytics.utils.downloads import attempt_download_asset

            attempt_download_asset(model_path)
        except Exception:
            # 兜底: 从 GitHub Releases 直接下载
            import urllib.request

            url = (
                f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{model_path.name}"
            )
            urllib.request.urlretrieve(url, model_path)

    @staticmethod
    def _pick_best_person(result: Any) -> np.ndarray | None:
        """从单张图像的推理结果中挑选置信度最高的人体关键点 (17,3)"""
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None:
            return None
        data = getattr(keypoints, "data", None)
        if data is None:
            return None
        if hasattr(data, "cpu"):
            data = data.cpu().numpy()
        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 3 or data.shape[0] == 0:
            return None
        # 以关键点平均置信度作为人物得分
        scores = data[..., 2].mean(axis=1)
        return data[int(np.argmax(scores))]

    def extract(self, video_frame: VideoFrame) -> KeypointFrame | None:
        """
        从视频帧中提取关键点
        Returns: KeypointFrame(33,4) 或 None(未检测到人体)
        """
        self._ensure_model()
        results = self._model(
            video_frame.frame,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        h, w = video_frame.frame.shape[:2]
        for result in results:
            coco_kpts = self._pick_best_person(result)
            if coco_kpts is None:
                continue

            keypoints = convert_coco_to_mediapipe(coco_kpts)
            # 归一化到 [0,1], 与 MediaPipe 输出语义一致
            if w > 0 and h > 0:
                keypoints[:, 0] /= w
                keypoints[:, 1] /= h

            kp_frame = KeypointFrame(timestamp=video_frame.timestamp, keypoints=keypoints)

            # 帧质量检查
            is_valid, reason = check_frame_quality(
                kp_frame, self.confidence_threshold, self.min_visible_lower
            )
            kp_frame.is_valid = is_valid
            kp_frame.invalid_reason = reason
            return kp_frame
        return None

    def close(self):
        """释放资源(YOLO模型无需显式释放, 占位以保持接口一致)"""
        self._model = None

    def __del__(self):
        self.close()
