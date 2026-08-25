"""
YOLO-Pose 关键点提取器单元测试 — 纯逻辑测试, 无真实模型下载
覆盖: COCO->MediaPipe 映射转换 / 工厂后端选择 / extract 模拟推理
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.data.keypoint_extractor import KeypointExtractor, create_keypoint_extractor
from src.data.video_capture import VideoFrame
from src.data.yolo_pose_extractor import YoloPoseExtractor
from src.utils.keypoints import COCO_TO_MEDIAPIPE, PoseKeypoint, convert_coco_to_mediapipe


# ============================================================
# 测试辅助
# ============================================================
def _make_video_frame(width: int = 640, height: int = 480) -> VideoFrame:
    """构造测试用视频帧"""
    return VideoFrame(
        frame=np.zeros((height, width, 3), dtype=np.uint8),
        timestamp=1.5,
        frame_idx=0,
    )


def _make_fake_result(keypoints_data: np.ndarray) -> MagicMock:
    """构造假的YOLO pose推理结果"""
    result = MagicMock()
    keypoints = MagicMock()
    keypoints.data = keypoints_data
    result.keypoints = keypoints
    return result


def _make_synthetic_coco(n_people: int, conf: float) -> np.ndarray:
    """合成COCO 17关键点数组(含躯干+下肢, 保证通过帧质量检查)"""
    coco = np.zeros((n_people, 17, 3), dtype=np.float32)
    for p in range(n_people):
        base = p * 0.5
        coco[p, 5] = [0.1 + base, 0.2, conf]  # 左肩
        coco[p, 6] = [0.3 + base, 0.2, conf]  # 右肩
        coco[p, 11] = [0.15 + base, 0.7, conf]  # 左髋
        coco[p, 12] = [0.25 + base, 0.7, conf]  # 右髋
        coco[p, 13] = [0.14 + base, 0.8, conf]  # 左膝
        coco[p, 14] = [0.26 + base, 0.8, conf]  # 右膝
        coco[p, 15] = [0.13 + base, 0.95, conf]  # 左踝
        coco[p, 16] = [0.27 + base, 0.95, conf]  # 右踝
    return coco


# ============================================================
# COCO 17关键点 -> MediaPipe 33关键点 映射
# ============================================================
class TestCocoToMediapipeMapping:
    def test_output_shape_and_zero_fill(self):
        """33行全填充, 未映射点位(脸部/躯干中心等)为零"""
        coco = np.zeros((17, 3), dtype=np.float32)
        for i in range(17):
            coco[i] = [0.1 + i * 0.01, 0.2 + i * 0.01, 0.5 + i * 0.02]
        mp = convert_coco_to_mediapipe(coco)
        assert mp.shape == (33, 4)
        assert mp.dtype == np.float32
        mapped = set(COCO_TO_MEDIAPIPE.values())
        for idx in range(33):
            if idx not in mapped:
                np.testing.assert_array_equal(mp[idx], np.zeros(4))

    def test_specified_mapping_positions(self):
        """验证规格中的映射: 肩/肘/腕与髋/膝/踝"""
        coco = np.zeros((17, 3), dtype=np.float32)
        for i in range(17):
            coco[i] = [i, i + 0.5, 0.9]
        mp = convert_coco_to_mediapipe(coco)
        # MP 11,12,13,14,15,16 <- COCO 5,6,7,8,9,10 (肩/肘/腕)
        for coco_idx, mp_idx in zip(range(5, 11), range(11, 17), strict=True):
            np.testing.assert_allclose(mp[mp_idx, :2], coco[coco_idx, :2])
            assert mp[mp_idx, 2] == 0.0  # z = 0
            assert mp[mp_idx, 3] == 0.9  # visibility = conf
        # MP 23,24,25,26,27,28 <- COCO 11,12,13,14,15,16 (髋/膝/踝)
        for coco_idx, mp_idx in zip(range(11, 17), range(23, 29), strict=True):
            np.testing.assert_allclose(mp[mp_idx, :2], coco[coco_idx, :2])
            assert mp[mp_idx, 3] == 0.9

    def test_extended_mapping_with_21_keypoints(self):
        """输入21关键点时, 脚部扩展点位(COCO 17-20 -> MP 29-32)也能映射"""
        coco = np.zeros((21, 3), dtype=np.float32)
        for i in range(21):
            coco[i] = [i * 0.01, i * 0.02, 0.8]
        mp = convert_coco_to_mediapipe(coco)
        for coco_idx in range(17, 21):
            mp_idx = COCO_TO_MEDIAPIPE[coco_idx]
            np.testing.assert_allclose(mp[mp_idx, :2], coco[coco_idx, :2])
            assert mp[mp_idx, 3] == 0.8

    def test_accepts_4_column_input(self):
        """支持 (N, 4) [x, y, z, conf] 输入"""
        coco = np.zeros((17, 4), dtype=np.float32)
        for i in range(17):
            coco[i] = [i, i, 0.0, 0.7]
        mp = convert_coco_to_mediapipe(coco)
        assert mp[COCO_TO_MEDIAPIPE[5], 3] == 0.7
        assert mp.shape == (33, 4)

    def test_rejects_invalid_shape(self):
        """非 (N, 3/4) 输入应报错"""
        with pytest.raises(ValueError):
            convert_coco_to_mediapipe(np.zeros((17, 2), dtype=np.float32))


# ============================================================
# create_keypoint_extractor 工厂
# ============================================================
class TestCreateKeypointExtractor:
    @staticmethod
    def _cfg(backend: str | None) -> OmegaConf:
        cfg = {
            "pose_estimation": {
                "confidence_threshold": 0.5,
                "min_visible_lower_keypoints": 4,
            },
            "human_detection": {"device": "cpu"},
        }
        if backend is not None:
            cfg["pose_estimation"]["backend"] = backend
        return OmegaConf.create(cfg)

    def test_selects_mediapipe(self):
        with patch("src.data.keypoint_extractor.get_config", return_value=self._cfg("mediapipe")):
            extractor = create_keypoint_extractor()
        assert isinstance(extractor, KeypointExtractor)

    def test_selects_yolo_pose(self):
        with patch("src.data.keypoint_extractor.get_config", return_value=self._cfg("yolo_pose")):
            extractor = create_keypoint_extractor()
        assert isinstance(extractor, YoloPoseExtractor)

    def test_defaults_to_mediapipe_when_backend_missing(self):
        with patch("src.data.keypoint_extractor.get_config", return_value=self._cfg(None)):
            extractor = create_keypoint_extractor()
        assert isinstance(extractor, KeypointExtractor)


# ============================================================
# YoloPoseExtractor.extract (mock 模型, 无网络)
# ============================================================
class TestYoloPoseExtractor:
    def test_model_loaded_lazily(self):
        """构造时不加载模型"""
        extractor = YoloPoseExtractor()
        assert extractor._model is None

    def test_extract_returns_keypoint_frame(self):
        """推理结果转换为 (33,4) KeypointFrame, 坐标归一化, z=0, visibility=conf"""
        extractor = YoloPoseExtractor()
        fake_model = MagicMock()
        fake_model.return_value = [_make_fake_result(_make_synthetic_coco(1, conf=0.9))]
        extractor._model = fake_model

        kp_frame = extractor.extract(_make_video_frame())

        assert kp_frame is not None
        assert kp_frame.timestamp == 1.5
        assert kp_frame.keypoints.shape == (33, 4)
        # 归一化后坐标 (x/w, y/h)
        assert kp_frame.keypoints[PoseKeypoint.LEFT_SHOULDER, 0] == pytest.approx(0.1 / 640)
        assert kp_frame.keypoints[PoseKeypoint.LEFT_SHOULDER, 1] == pytest.approx(0.2 / 480)
        # z=0, visibility=conf
        assert kp_frame.keypoints[PoseKeypoint.LEFT_SHOULDER, 2] == 0.0
        assert kp_frame.keypoints[PoseKeypoint.LEFT_SHOULDER, 3] == 0.9
        assert kp_frame.keypoints[PoseKeypoint.LEFT_HIP, 3] == 0.9
        assert kp_frame.keypoints[PoseKeypoint.LEFT_ANKLE, 3] == 0.9
        # 未映射点位(鼻子)为零
        np.testing.assert_array_equal(kp_frame.keypoints[PoseKeypoint.NOSE], np.zeros(4))
        # 下肢/躯干可见数足够, 通过帧质量检查
        assert kp_frame.is_valid is True

    def test_extract_returns_none_when_no_person(self):
        """无人体检测结果时返回 None"""
        extractor = YoloPoseExtractor()
        fake_model = MagicMock()
        fake_model.return_value = [_make_fake_result(np.zeros((0, 17, 3), dtype=np.float32))]
        extractor._model = fake_model

        assert extractor.extract(_make_video_frame()) is None

    def test_extract_skips_result_without_keypoints(self):
        """推理结果缺少 keypoints 时返回 None"""
        extractor = YoloPoseExtractor()
        fake_model = MagicMock()
        fake_model.return_value = [MagicMock()]
        extractor._model = fake_model

        assert extractor.extract(_make_video_frame()) is None

    def test_extract_picks_highest_confidence_person(self):
        """多人时挑选平均置信度最高的人体"""
        extractor = YoloPoseExtractor()
        coco = np.zeros((2, 17, 3), dtype=np.float32)
        coco[0] = _make_synthetic_coco(1, conf=0.3)[0]
        coco[1] = _make_synthetic_coco(1, conf=0.9)[0]
        coco[1, 5] = [0.4, 0.2, 0.9]
        fake_model = MagicMock()
        fake_model.return_value = [_make_fake_result(coco)]
        extractor._model = fake_model

        kp_frame = extractor.extract(_make_video_frame())

        assert kp_frame is not None
        # 应选中第2个人(高置信度): 左肩 x=0.4
        assert kp_frame.keypoints[PoseKeypoint.LEFT_SHOULDER, 0] == pytest.approx(0.4 / 640)

    def test_ensure_model_downloads_missing_model(self, tmp_path):
        """模型缺失时自动下载到指定路径 (mock 下载与加载, 无网络)"""
        model_path = tmp_path / "yolo26n-pose.pt"
        extractor = YoloPoseExtractor(model_path=str(model_path))
        with (
            patch("ultralytics.utils.downloads.attempt_download_asset") as mock_download,
            patch("ultralytics.YOLO") as mock_yolo_cls,
        ):
            extractor._ensure_model()
        mock_download.assert_called_once()
        mock_yolo_cls.assert_called_once_with(str(model_path))
        assert extractor._model is not None
