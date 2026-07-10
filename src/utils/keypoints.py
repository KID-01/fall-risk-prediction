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
