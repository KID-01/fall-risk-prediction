"""
YOLO-Pose 关键点提取模块 — YOLO26n-pose 人体姿态估计
将COCO 17关键点输出映射为与 MediaPipe 相同的33关键点 KeypointFrame 格式,
下游 features.py 等模块无需任何改动
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.data.human_detector import DetectionBox, PrimaryPersonTracker
from src.data.video_capture import VideoFrame
from src.utils.config import get_config
from src.utils.keypoints import KeypointFrame, check_frame_quality, convert_coco_to_mediapipe
from src.utils.logger import get_logger

log = get_logger(__name__)

# 默认模型路径(置于 checkpoints/ 下)
_DEFAULT_MODEL_NAME = "yolo26n-pose.pt"
_DEFAULT_MODEL_PATH = str(Path(__file__).parents[2] / "checkpoints" / _DEFAULT_MODEL_NAME)


@dataclass
class PosePerson:
    box: DetectionBox
    keypoint_frame: KeypointFrame
    keypoint_score: float

    @property
    def quality_state(self) -> str:
        return "OK" if self.keypoint_frame.is_valid else "LOW_QUALITY"

    @property
    def quality_reason(self) -> str:
        return self.keypoint_frame.invalid_reason or ""


@dataclass
class PoseExtractionResult:
    people: list[PosePerson]
    primary: PosePerson | None


class YoloPoseExtractor:
    """YOLO-Pose 关键点提取器, 输出与 KeypointExtractor 相同的 (33,4) KeypointFrame"""

    def __init__(self, model_path: str | None = None, device: str | None = None):
        config = get_config()
        self.confidence_threshold = float(
            config.pose_estimation.get("model_confidence_threshold", 0.25)
        )
        self.keypoint_visibility_threshold = float(
            config.pose_estimation.get(
                "keypoint_visibility_threshold",
                config.pose_estimation.confidence_threshold,
            )
        )
        self.min_visible_lower = config.pose_estimation.min_visible_lower_keypoints
        self.image_size = int(config.pose_estimation.get("image_size", 640))
        configured_name = str(config.pose_estimation.get("model_type", _DEFAULT_MODEL_NAME))
        if not configured_name.endswith(".pt"):
            configured_name += ".pt"
        configured_path = Path(__file__).parents[2] / "checkpoints" / configured_name
        self._model_path = model_path or str(configured_path)
        self.device = device or config.human_detection.device
        self._model = None
        self.primary_tracker = PrimaryPersonTracker()

    def _ensure_model(self) -> None:
        """延迟加载 YOLO-Pose 模型(缺失时自动下载到 checkpoints/)"""
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
    def _download_model(model_path: Path) -> None:
        """下载 YOLO-Pose 模型到指定路径"""
        log.warning(f"模型不存在, 开始下载: {model_path.name}")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from ultralytics.utils.downloads import attempt_download_asset

            attempt_download_asset(model_path)
        except Exception:
            # 兜底: 从 GitHub Releases 直接下载
            import urllib.request

            url = (
                f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{model_path.name}"
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

    @staticmethod
    def _keypoint_data(result: Any) -> np.ndarray | None:
        keypoints = getattr(result, "keypoints", None)
        data = getattr(keypoints, "data", None) if keypoints is not None else None
        if data is None:
            return None
        if hasattr(data, "cpu"):
            data = data.cpu().numpy()
        array = np.asarray(data, dtype=np.float32)
        return array if array.ndim == 3 and array.shape[0] > 0 else None

    @staticmethod
    def _box_data(result: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
        boxes = getattr(result, "boxes", None)
        xyxy = getattr(boxes, "xyxy", None) if boxes is not None else None
        conf = getattr(boxes, "conf", None) if boxes is not None else None
        if xyxy is None or conf is None:
            return None, None
        try:
            if hasattr(xyxy, "cpu"):
                xyxy = xyxy.cpu().numpy()
            if hasattr(conf, "cpu"):
                conf = conf.cpu().numpy()
            box_array = np.asarray(xyxy, dtype=np.float32)
            conf_array = np.asarray(conf, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return None, None
        if box_array.ndim != 2 or box_array.shape[1] != 4:
            return None, None
        return box_array, conf_array

    @staticmethod
    def _box_from_keypoints(coco_keypoints: np.ndarray, fallback_confidence: float) -> DetectionBox:
        visible = coco_keypoints[:, 2] > 0
        points = coco_keypoints[visible, :2] if np.any(visible) else coco_keypoints[:, :2]
        if points.size == 0:
            return DetectionBox(0, 0, 0, 0, fallback_confidence)
        return DetectionBox(
            x1=float(np.min(points[:, 0])),
            y1=float(np.min(points[:, 1])),
            x2=float(np.max(points[:, 0])),
            y2=float(np.max(points[:, 1])),
            confidence=fallback_confidence,
        )

    def reset(self) -> None:
        self.primary_tracker.reset()

    def extract_result(self, video_frame: VideoFrame) -> PoseExtractionResult:
        """一次 Pose 推理返回全部人体、主人物框和主人物关键点。"""
        self._ensure_model()
        results = self._model(
            video_frame.frame,
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )

        height, width = video_frame.frame.shape[:2]
        if width <= 0 or height <= 0:
            log.warning(f"帧尺寸异常 ({width}x{height}), 跳过该帧")
            return PoseExtractionResult(people=[], primary=None)

        people: list[PosePerson] = []
        for result in results:
            keypoint_data = self._keypoint_data(result)
            if keypoint_data is None:
                continue
            box_data, box_confidences = self._box_data(result)
            for index, coco_keypoints in enumerate(keypoint_data):
                keypoint_score = float(np.mean(coco_keypoints[:, 2]))
                if box_data is not None and index < len(box_data):
                    confidence = (
                        float(box_confidences[index])
                        if box_confidences is not None and index < len(box_confidences)
                        else keypoint_score
                    )
                    box = DetectionBox(*map(float, box_data[index]), confidence=confidence)
                else:
                    box = self._box_from_keypoints(coco_keypoints, keypoint_score)

                keypoints = convert_coco_to_mediapipe(coco_keypoints)
                keypoints[:, 0] /= width
                keypoints[:, 1] /= height
                keypoint_frame = KeypointFrame(
                    timestamp=video_frame.timestamp,
                    keypoints=keypoints,
                )
                is_valid, reason = check_frame_quality(
                    keypoint_frame,
                    self.keypoint_visibility_threshold,
                    self.min_visible_lower,
                )
                keypoint_frame.is_valid = is_valid
                keypoint_frame.invalid_reason = reason
                people.append(PosePerson(box, keypoint_frame, keypoint_score))

        primary_box = self.primary_tracker.select([person.box for person in people])
        primary = next((person for person in people if person.box is primary_box), None)
        if primary is None and primary_box is not None:
            primary = next((person for person in people if person.box == primary_box), None)
        return PoseExtractionResult(people=people, primary=primary)

    def extract(self, video_frame: VideoFrame) -> KeypointFrame | None:
        """
        从视频帧中提取关键点
        Returns: KeypointFrame(33,4) 或 None(未检测到人体)
        """
        result = self.extract_result(video_frame)
        return result.primary.keypoint_frame if result.primary else None

    def close(self):
        """释放资源(YOLO模型无需显式释放, 占位以保持接口一致)"""
        self._model = None
        self.primary_tracker.reset()

    def __del__(self):
        self.close()
