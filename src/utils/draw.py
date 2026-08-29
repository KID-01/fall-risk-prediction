"""
在视频帧上叠加检测结果：骨架关键点、风险等级、帧计数
"""
from __future__ import annotations

import cv2
import numpy as np

from src.data.environment_detector import EnvironmentBox
from src.data.human_detector import DetectionBox
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
    human_box: DetectionBox | None = None,
    human_boxes: list[DetectionBox] | None = None,
    environment_boxes: list[EnvironmentBox] | None = None,
    illumination: float | None = None,
    top_hazards: list[dict] | None = None,
    trajectory: dict | None = None,
    risk_score: float | None = None,
    human_risk_score: float | None = None,
    environment_risk_score: float | None = None,
    interaction_risk_score: float | None = None,
) -> np.ndarray:
    """在帧上叠加骨架、风险等级等信息，返回新图像（不修改原图）"""
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    hazard_labels = {item.get("label") or item.get("class") for item in top_hazards or []}

    # 1. 绘制环境目标与所有人体框
    for box in environment_boxes or []:
        color = (40, 80, 235) if box.label in hazard_labels else (220, 140, 30)
        _draw_box(
            canvas,
            box.x1,
            box.y1,
            box.x2,
            box.y2,
            f"{box.label} {box.confidence:.2f}",
            color,
        )

    for box in human_boxes or []:
        if human_box is not None and box == human_box:
            continue
        _draw_box(
            canvas,
            box.x1,
            box.y1,
            box.x2,
            box.y2,
            f"person {box.confidence:.2f}",
            (160, 160, 160),
        )

    if human_box is not None:
        _draw_box(
            canvas,
            human_box.x1,
            human_box.y1,
            human_box.x2,
            human_box.y2,
            f"primary {human_box.confidence:.2f}",
            (0, 220, 80),
        )
        cv2.circle(
            canvas,
            (int((human_box.x1 + human_box.x2) / 2), int(human_box.y2)),
            5,
            (0, 255, 255),
            -1,
            cv2.LINE_AA,
        )

    predicted_points = (trajectory or {}).get("predicted_points", [])
    for start, end in zip(predicted_points, predicted_points[1:], strict=False):
        cv2.line(
            canvas,
            (int(start[0]), int(start[1])),
            (int(end[0]), int(end[1])),
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # 2. 绘制骨架
    if kp_frame is not None and kp_frame.is_valid:
        _draw_skeleton(canvas, kp_frame, w, h)

    # 3. 绘制顶部信息栏
    _draw_info_bar(canvas, risk_level, baseline_ready, frames_processed, w, risk_score)
    if illumination is not None:
        cv2.putText(canvas, f"light {illumination:.0f}/255", (12, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    _draw_risk_extension_panel(
        canvas,
        risk_level,
        risk_score,
        human_risk_score,
        environment_risk_score,
        interaction_risk_score,
    )

    return canvas


def _draw_risk_extension_panel(
    canvas: np.ndarray,
    risk_level: str,
    overall_score: float | None,
    human_score: float | None,
    environment_score: float | None,
    interaction_score: float | None,
) -> None:
    """在分析帧右上角显示 v0.3.2 分项工程指数（非概率）。"""
    h, w = canvas.shape[:2]
    panel_w = min(286, max(180, w - 16))
    panel_h = 112
    x = max(8, w - panel_w - 8)
    y = 46
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
    color = RISK_COLORS.get(risk_level, RISK_COLORS["low"])
    overall_text = "UNKNOWN" if overall_score is None else f"{overall_score:.1f} / 100"
    cv2.putText(
        canvas,
        f"ENGINEERING RISK: {overall_text}",
        (x + 10, y + 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        2,
        cv2.LINE_AA,
    )
    rows = (
        ("HUMAN", human_score),
        ("ENV", environment_score),
        ("INTERACTION", interaction_score),
    )
    for index, (label, value) in enumerate(rows):
        value_text = "UNKNOWN" if value is None else f"{value:.1f} / 100"
        cv2.putText(
            canvas,
            f"{label}: {value_text}",
            (x + 10, y + 46 + index * 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )


def _draw_box(canvas: np.ndarray, x1: float, y1: float, x2: float, y2: float,
              label: str, color: tuple[int, int, int]):
    p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
    cv2.rectangle(canvas, p1, p2, color, 2, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(0, p1[1] - th - 6)
    cv2.rectangle(canvas, (p1[0], top), (p1[0] + tw + 6, p1[1]), color, -1)
    cv2.putText(canvas, label, (p1[0] + 3, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 1, cv2.LINE_AA)


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
    risk_score: float | None = None,
):
    """绘制顶部半透明信息栏"""
    bar_h = 40
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)

    color = RISK_COLORS.get(risk_level, RISK_COLORS["low"])
    label = RISK_LABELS.get(risk_level, "未知")

    # 风险等级
    status_text = f"{label} {risk_score:.1f}" if risk_score is not None else label
    if not baseline_ready:
        status_text += " | baseline collecting"
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
