"""
在视频帧上叠加检测结果：骨架关键点、风险等级、帧计数
"""
from __future__ import annotations

import cv2
import numpy as np

from src.utils.keypoints import KeypointFrame, PoseKeypoint

# 骨架连接线定义 (MediaPipe 33点)
SKELETON_CONNECTIONS = [
    # 躯干
    (PoseKeypoint.LEFT_SHOULDER, PoseKeypoint.RIGHT_SHOULDER),
    (PoseKeypoint.LEFT_SHOULDER, PoseKeypoint.LEFT_HIP),
    (PoseKeypoint.RIGHT_SHOULDER, PoseKeypoint.RIGHT_HIP),
    (PoseKeypoint.LEFT_HIP, PoseKeypoint.RIGHT_HIP),
    # 左臂
    (PoseKeypoint.LEFT_SHOULDER, PoseKeypoint.LEFT_ELBOW),
    (PoseKeypoint.LEFT_ELBOW, PoseKeypoint.LEFT_WRIST),
    # 右臂
    (PoseKeypoint.RIGHT_SHOULDER, PoseKeypoint.RIGHT_ELBOW),
    (PoseKeypoint.RIGHT_ELBOW, PoseKeypoint.RIGHT_WRIST),
    # 左腿
    (PoseKeypoint.LEFT_HIP, PoseKeypoint.LEFT_KNEE),
    (PoseKeypoint.LEFT_KNEE, PoseKeypoint.LEFT_ANKLE),
    # 右腿
    (PoseKeypoint.RIGHT_HIP, PoseKeypoint.RIGHT_KNEE),
    (PoseKeypoint.RIGHT_KNEE, PoseKeypoint.RIGHT_ANKLE),
]

# 关键点颜色 (BGR)
KEYPOINT_COLOR = (0, 255, 0)        # 绿色
SKELETON_COLOR = (0, 200, 0)        # 深绿
# 风险等级颜色
RISK_COLORS = {
    "low": (34, 197, 94),
    "attention": (234, 179, 8),
    "warning": (249, 115, 22),
    "critical": (239, 68, 68),
}
RISK_LABELS = {
    "low": "低风险",
    "attention": "关注级",
    "warning": "预警级",
    "critical": "高危级",
}


def draw_overlay(
    frame: np.ndarray,
    kp_frame: KeypointFrame | None,
    risk_level: str = "low",
    baseline_ready: bool = False,
    frames_processed: int = 0,
) -> np.ndarray:
    """在帧上叠加骨架、风险等级等信息，返回新图像（不修改原图）"""
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # 1. 绘制骨架
    if kp_frame is not None and kp_frame.is_valid:
        _draw_skeleton(canvas, kp_frame, w, h)

    # 2. 绘制顶部信息栏
    _draw_info_bar(canvas, risk_level, baseline_ready, frames_processed, w)

    return canvas


def _draw_skeleton(canvas: np.ndarray, kp_frame: KeypointFrame, w: int, h: int):
    """绘制骨架线和关键点"""
    kpts = kp_frame.keypoints  # (33, 4) [x, y, z, visibility]

    # 骨架连线
    for a, b in SKELETON_CONNECTIONS:
        if kpts[a, 3] > 0.5 and kpts[b, 3] > 0.5:
            pt1 = (int(kpts[a, 0] * w), int(kpts[a, 1] * h))
            pt2 = (int(kpts[b, 0] * w), int(kpts[b, 1] * h))
            cv2.line(canvas, pt1, pt2, SKELETON_COLOR, 2, cv2.LINE_AA)

    # 关键点圆点
    for i in range(33):
        if kpts[i, 3] > 0.5:
            pt = (int(kpts[i, 0] * w), int(kpts[i, 1] * h))
            cv2.circle(canvas, pt, 4, KEYPOINT_COLOR, -1, cv2.LINE_AA)


def _draw_info_bar(
    canvas: np.ndarray,
    risk_level: str,
    baseline_ready: bool,
    frames_processed: int,
    w: int,
):
    """绘制顶部半透明信息栏"""
    bar_h = 40
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)

    color = RISK_COLORS.get(risk_level, RISK_COLORS["low"])
    label = RISK_LABELS.get(risk_level, "未知")

    # 风险等级
    status_text = "基线采集中" if not baseline_ready else label
    cv2.putText(canvas, status_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2, cv2.LINE_AA)

    # 帧计数
    fps_text = f"帧: {frames_processed}"
    (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(canvas, fps_text, (w - tw - 12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 1, cv2.LINE_AA)


def encode_jpeg(frame: np.ndarray, quality: int = 75) -> bytes:
    """将 BGR 帧编码为 JPEG 字节"""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()
