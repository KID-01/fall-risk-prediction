"""
MediaPipe Pose 33个关键点定义与工具函数
索引参考: https://google.github.io/mediapipe/solutions/pose.html
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class PoseKeypoint(IntEnum):
    """MediaPipe Pose 33个关键点索引"""

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# 关键点名称映射
KEYPOINT_NAMES: dict[int, str] = {int(k): k.name.lower() for k in PoseKeypoint}

# 本项目重点关注的关键点组
LOWER_BODY_KEYPOINTS = [
    PoseKeypoint.LEFT_HIP,
    PoseKeypoint.RIGHT_HIP,
    PoseKeypoint.LEFT_KNEE,
    PoseKeypoint.RIGHT_KNEE,
    PoseKeypoint.LEFT_ANKLE,
    PoseKeypoint.RIGHT_ANKLE,
]

TRUNK_KEYPOINTS = [
    PoseKeypoint.LEFT_SHOULDER,
    PoseKeypoint.RIGHT_SHOULDER,
    PoseKeypoint.LEFT_HIP,
    PoseKeypoint.RIGHT_HIP,
]

# 髋关节(行走节拍)
HIP_KEYPOINTS = [PoseKeypoint.LEFT_HIP, PoseKeypoint.RIGHT_HIP]

# 踝关节(步幅)
ANKLE_KEYPOINTS = [PoseKeypoint.LEFT_ANKLE, PoseKeypoint.RIGHT_ANKLE]


# ==== COCO 17关键点 -> MediaPipe 33关键点 映射 ====
# COCO 17点姿态关键点顺序(COCO keypoints):
# 0鼻子 1左眼 2右眼 3左耳 4右耳 5左肩 6右肩 7左肘 8右肘 9左腕 10右腕
# 11左髋 12右髋 13左膝 14右膝 15左踝 16右踝
COCO_TO_MEDIAPIPE: dict[int, int] = {
    # 上肢: 肩/肘/腕
    5: PoseKeypoint.LEFT_SHOULDER,  # MP 11
    6: PoseKeypoint.RIGHT_SHOULDER,  # MP 12
    7: PoseKeypoint.LEFT_ELBOW,  # MP 13
    8: PoseKeypoint.RIGHT_ELBOW,  # MP 14
    9: PoseKeypoint.LEFT_WRIST,  # MP 15
    10: PoseKeypoint.RIGHT_WRIST,  # MP 16
    # 下肢: 髋/膝/踝
    11: PoseKeypoint.LEFT_HIP,  # MP 23
    12: PoseKeypoint.RIGHT_HIP,  # MP 24
    13: PoseKeypoint.LEFT_KNEE,  # MP 25
    14: PoseKeypoint.RIGHT_KNEE,  # MP 26
    15: PoseKeypoint.LEFT_ANKLE,  # MP 27
    16: PoseKeypoint.RIGHT_ANKLE,  # MP 28
    # 脚部(COCO扩展点, 标准17点模型不输出, 兼容21点及以上模型)
    17: PoseKeypoint.LEFT_HEEL,  # MP 29
    18: PoseKeypoint.RIGHT_HEEL,  # MP 30
    19: PoseKeypoint.LEFT_FOOT_INDEX,  # MP 31
    20: PoseKeypoint.RIGHT_FOOT_INDEX,  # MP 32
}


def convert_coco_to_mediapipe(coco_kpts: np.ndarray) -> np.ndarray:
    """
    将COCO姿态关键点数组转换为MediaPipe 33点布局

    Args:
        coco_kpts: shape (N, 3) [x, y, conf] 或 (N, 4) [x, y, z, conf] 的COCO关键点数组
    Returns:
        shape (33, 4) [x, y, z, visibility] 的MediaPipe布局数组,
        未映射点位(z=0, visibility=0)填充, 始终返回完整的33行
    """
    coco_kpts = np.asarray(coco_kpts, dtype=np.float32)
    if coco_kpts.ndim != 2 or coco_kpts.shape[1] not in (3, 4):
        raise ValueError(f"coco_kpts 应为 (N, 3/4) 数组, 实际 shape={coco_kpts.shape}")

    if coco_kpts.shape[1] == 3:
        data = np.zeros((coco_kpts.shape[0], 4), dtype=np.float32)
        data[:, :2] = coco_kpts[:, :2]
        data[:, 3] = coco_kpts[:, 2]
    else:
        data = coco_kpts[:, :4]

    mp_kpts = np.zeros((33, 4), dtype=np.float32)
    for coco_idx, mp_idx in COCO_TO_MEDIAPIPE.items():
        if coco_idx < data.shape[0]:
            mp_kpts[mp_idx] = data[coco_idx]
    return mp_kpts


@dataclass
class KeypointFrame:
    """单帧关键点数据"""

    timestamp: float                      # 时间戳(秒)
    keypoints: np.ndarray                 # shape (33, 4) [x, y, z, visibility]
    is_valid: bool = True                 # 是否通过质量过滤
    invalid_reason: str = ""

    def get(self, idx: int) -> np.ndarray:
        """获取指定关键点的坐标 [x, y, z, visibility]"""
        return self.keypoints[idx]

    def get_xy(self, idx: int) -> np.ndarray:
        """获取指定关键点的2D坐标 [x, y]"""
        return self.keypoints[idx, :2]

    def is_visible(self, idx: int, threshold: float = 0.5) -> bool:
        """关键点是否可见"""
        return self.keypoints[idx, 3] > threshold

    def count_visible(self, indices: list[int], threshold: float = 0.5) -> int:
        """统计指定关键点组中可见的数量"""
        return sum(1 for i in indices if self.is_visible(i, threshold))

    def torso_height(self) -> float:
        """躯干高度(肩中点到髋中点的距离),用于归一化"""
        ls = self.get_xy(PoseKeypoint.LEFT_SHOULDER)
        rs = self.get_xy(PoseKeypoint.RIGHT_SHOULDER)
        lh = self.get_xy(PoseKeypoint.LEFT_HIP)
        rh = self.get_xy(PoseKeypoint.RIGHT_HIP)
        shoulder_mid = (ls + rs) / 2
        hip_mid = (lh + rh) / 2
        return float(np.linalg.norm(shoulder_mid - hip_mid))


def check_frame_quality(
    frame: KeypointFrame,
    confidence_threshold: float = 0.5,
    min_visible_lower: int = 4,
) -> tuple[bool, str]:
    """
    检查帧质量是否满足分析要求
    返回: (是否通过, 不通过原因)
    """
    # 检查下肢关键点可见数量
    visible_count = frame.count_visible(LOWER_BODY_KEYPOINTS, confidence_threshold)
    if visible_count < min_visible_lower:
        return False, f"下肢关键点可见数不足: {visible_count}/{len(LOWER_BODY_KEYPOINTS)}"

    # 检查躯干关键点
    trunk_visible = frame.count_visible(TRUNK_KEYPOINTS, confidence_threshold)
    if trunk_visible < 3:
        return False, f"躯干关键点可见数不足: {trunk_visible}/{len(TRUNK_KEYPOINTS)}"

    return True, ""
