"""
四大相对特征计算模块
1. 行走节拍频率 — 髋关节y坐标FFT主频
2. 步幅相对幅度 — 踝关节摆动幅度/躯干高度归一化
3. 躯干稳定指数 — 肩髋连线与垂直方向夹角变化范围
4. 活动密度 — 单位时间站立/行走帧占比

所有特征均为相对值,不依赖物理尺度标定
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.utils.keypoints import KeypointFrame, PoseKeypoint


@dataclass
class FeatureVector:
    """四大特征向量"""

    walking_rhythm: float          # 行走节拍频率 (Hz)
    step_amplitude: float          # 步幅相对幅度 (归一化)
    trunk_stability: float         # 躯干稳定指数 (角度变化范围,度)
    activity_density: float        # 活动密度 (0~1)
    timestamp: float = 0.0         # 时间戳

    def to_array(self) -> np.ndarray:
        """转为numpy数组(用于马氏距离计算)"""
        return np.array([
            self.walking_rhythm,
            self.step_amplitude,
            self.trunk_stability,
            self.activity_density,
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray, timestamp: float = 0.0) -> FeatureVector:
        return cls(
            walking_rhythm=float(arr[0]),
            step_amplitude=float(arr[1]),
            trunk_stability=float(arr[2]),
            activity_density=float(arr[3]),
            timestamp=timestamp,
        )

    FEATURE_NAMES = ["walking_rhythm", "step_amplitude", "trunk_stability", "activity_density"]


# ============================================================
# 特征1: 行走节拍频率
# ============================================================
class WalkingRhythmCalculator:
    """通过髋关节y坐标时序FFT提取行走主频"""

    def __init__(self, min_freq: float = 0.5, max_freq: float = 5.0):
        self.min_freq = min_freq
        self.max_freq = max_freq

    def calculate(self, frames: Sequence[KeypointFrame]) -> float:
        """
        计算行走节拍频率
        Args:
            frames: 有效关键点帧序列
        Returns:
            主频(Hz),无有效数据返回0
        """
        valid = [f for f in frames if f.is_valid]
        if len(valid) < 4:
            return 0.0

        # 提取髋关节y坐标时序(取左右髋平均值)
        timestamps = np.array([f.timestamp for f in valid])
        hip_y = np.array([
            (f.get_xy(PoseKeypoint.LEFT_HIP)[1] + f.get_xy(PoseKeypoint.RIGHT_HIP)[1]) / 2
            for f in valid
        ])

        # 去均值
        hip_y = hip_y - np.mean(hip_y)

        # 估计采样率
        if len(timestamps) > 1:
            dt = np.mean(np.diff(timestamps))
            fs = 1.0 / dt if dt > 0 else 10.0
        else:
            fs = 10.0

        # FFT
        n = len(hip_y)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        spectrum = np.abs(np.fft.rfft(hip_y))

        # 在有效频率范围内找主频
        mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
        if not np.any(mask) or np.max(spectrum[mask]) == 0:
            return 0.0

        dominant_idx = np.argmax(spectrum[mask])
        dominant_freq = freqs[mask][dominant_idx]
        return float(dominant_freq)


# ============================================================
# 特征2: 步幅相对幅度
# ============================================================
class StepAmplitudeCalculator:
    """踝关节摆动幅度,经躯干高度归一化"""

    def calculate(self, frames: Sequence[KeypointFrame]) -> float:
        """
        计算步幅相对幅度
        Returns: 归一化摆动幅度,无有效数据返回0
        """
        valid = [f for f in frames if f.is_valid]
        if len(valid) < 2:
            return 0.0

        # 收集踝关节x坐标和躯干高度
        left_ankle_x = []
        right_ankle_x = []
        torso_heights = []

        for f in valid:
            if f.is_visible(PoseKeypoint.LEFT_ANKLE):
                left_ankle_x.append(f.get_xy(PoseKeypoint.LEFT_ANKLE)[0])
            if f.is_visible(PoseKeypoint.RIGHT_ANKLE):
                right_ankle_x.append(f.get_xy(PoseKeypoint.RIGHT_ANKLE)[0])
            th = f.torso_height()
            if th > 0:
                torso_heights.append(th)

        if not torso_heights:
            return 0.0

        avg_torso = np.mean(torso_heights)

        # 踝关节x坐标摆动范围(左右踝分别计算取平均)
        swing = 0.0
        count = 0
        if len(left_ankle_x) >= 2:
            swing += np.ptp(left_ankle_x)
            count += 1
        if len(right_ankle_x) >= 2:
            swing += np.ptp(right_ankle_x)
            count += 1

        if count == 0:
            return 0.0

        swing /= count
        return float(swing / avg_torso) if avg_torso > 0 else 0.0


# ============================================================
# 特征3: 躯干稳定指数
# ============================================================
class TrunkStabilityCalculator:
    """肩髋连线与垂直方向夹角的变化范围"""

    @staticmethod
    def _trunk_angle(frame: KeypointFrame) -> float | None:
        """计算单帧躯干偏转角(度)"""
        if not all(
            frame.is_visible(kp)
            for kp in [PoseKeypoint.LEFT_SHOULDER, PoseKeypoint.RIGHT_SHOULDER,
                       PoseKeypoint.LEFT_HIP, PoseKeypoint.RIGHT_HIP]
        ):
            return None

        shoulder_mid = (
            frame.get_xy(PoseKeypoint.LEFT_SHOULDER)
            + frame.get_xy(PoseKeypoint.RIGHT_SHOULDER)
        ) / 2
        hip_mid = (
            frame.get_xy(PoseKeypoint.LEFT_HIP)
            + frame.get_xy(PoseKeypoint.RIGHT_HIP)
        ) / 2

        # 躯干向量(从髋到肩)
        trunk_vec = shoulder_mid - hip_mid
        # 垂直向量 [0, -1] (图像坐标系y向下)
        angle = np.degrees(np.arctan2(abs(trunk_vec[0]), abs(trunk_vec[1])))
        return float(angle)

    def calculate(self, frames: Sequence[KeypointFrame]) -> float:
        """
        计算躯干稳定指数(角度变化范围)
        Returns: 角度范围(度),越大越不稳定
        """
        angles = []
        for f in frames:
            if f.is_valid:
                angle = self._trunk_angle(f)
                if angle is not None:
                    angles.append(angle)

        if len(angles) < 2:
            return 0.0

        return float(np.ptp(angles))


# ============================================================
# 特征4: 活动密度
# ============================================================
class ActivityDensityCalculator:
    """单位时间内站立/行走帧占比"""

    def __init__(self, motion_threshold: float = 0.01):
        self.motion_threshold = motion_threshold

    def calculate(self, frames: Sequence[KeypointFrame]) -> float:
        """
        计算活动密度
        Returns: 0~1,表示活动帧占比
        """
        valid = [f for f in frames if f.is_valid]
        if len(valid) < 2:
            return 0.0

        # 用髋关节中点位移判断是否在运动
        motion_count = 0
        for i in range(1, len(valid)):
            prev_hip = (
                valid[i - 1].get_xy(PoseKeypoint.LEFT_HIP)
                + valid[i - 1].get_xy(PoseKeypoint.RIGHT_HIP)
            ) / 2
            curr_hip = (
                valid[i].get_xy(PoseKeypoint.LEFT_HIP)
                + valid[i].get_xy(PoseKeypoint.RIGHT_HIP)
            ) / 2
            displacement = np.linalg.norm(curr_hip - prev_hip)
            if displacement > self.motion_threshold:
                motion_count += 1

        return motion_count / (len(valid) - 1)


# ============================================================
# 特征计算总管
# ============================================================
class FeatureCalculator:
    """四大特征计算总管"""

    def __init__(self):
        config = None
        try:
            from src.utils.config import get_config
            config = get_config()
        except Exception:
            pass

        if config is not None:
            wr = config.features.walking_rhythm
            self.rhythm_calc = WalkingRhythmCalculator(wr.fft_min_freq, wr.fft_max_freq)
            ad = config.features.activity_density
            self.density_calc = ActivityDensityCalculator(ad.motion_threshold)
        else:
            self.rhythm_calc = WalkingRhythmCalculator()
            self.density_calc = ActivityDensityCalculator()

        self.amplitude_calc = StepAmplitudeCalculator()
        self.stability_calc = TrunkStabilityCalculator()

    def calculate(self, frames: Sequence[KeypointFrame]) -> FeatureVector:
        """计算一个窗口内所有四大特征"""
        valid = [f for f in frames if f.is_valid]
        timestamp = valid[-1].timestamp if valid else 0.0

        return FeatureVector(
            walking_rhythm=self.rhythm_calc.calculate(frames),
            step_amplitude=self.amplitude_calc.calculate(frames),
            trunk_stability=self.stability_calc.calculate(frames),
            activity_density=self.density_calc.calculate(frames),
            timestamp=timestamp,
        )
