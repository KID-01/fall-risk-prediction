"""
跌倒后确认状态机（post-impact fall confirmation）

工程确认层，独立于科研级跌倒前预测器。只使用当前帧的人体框和既往状态：
只有画面中持续出现"又宽又扁"的人体框（疑似横卧）才输出 FALL；
人体框丢失本身永远不构成新的跌倒触发（防止人离场/遮挡误报）。
所有判断都是因果的：只看当前和历史帧，不看未来帧。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class PostImpactResult:
    """一次 post-impact 确认更新的输出契约。"""

    confirmed: bool
    state: str  # FALL / UNKNOWN
    phase: str  # NORMAL / FALL_CANDIDATE / FALL_CONFIRMED / FALL_PERSISTING / RECOVERY
    reason_codes: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """序列化为前端/持久化用的普通字典。"""
        return {
            "confirmed": self.confirmed,
            "state": self.state,
            "phase": self.phase,
            "reason_codes": list(self.reason_codes),
            "evidence": dict(self.evidence),
        }


def _bbox(person) -> dict | None:
    """把人体框转成状态机使用的几何量。

    几何证据对齐参考开源跌倒检测器的设计：
      aspect            = 框宽高比（宽大于高 -> 疑似横卧）
      aspect_derivative = 宽高比变化率（形态变化速度）
      height_ratio      = 高度相对站立参考的比例（塌陷证据）
      cy                = 框中心高度（垂直下移的代理量）
    """
    if person is None:
        return None
    if isinstance(person, dict):
        x1, y1, x2, y2 = (float(person[k]) for k in ("x1", "y1", "x2", "y2"))
    else:
        x1, y1, x2, y2 = (
            float(person.x1),
            float(person.y1),
            float(person.x2),
            float(person.y2),
        )
    width = max(x2 - x1, 1e-6)
    height = max(y2 - y1, 1e-6)
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "w": width,
        "h": height,
        "aspect": width / height,
        "cy": (y1 + y2) / 2.0,
        "area": width * height,
    }


class PostImpactDetector:
    """对"画面中明显横卧的人"做因果有限状态确认。"""

    def __init__(self, config: dict | None = None):
        config = dict(config or {})
        self.horiz_aspect = float(config.get("horiz_aspect_thr", 1.6))
        self.low_height_ratio = float(config.get("low_height_ratio", 0.55))
        self.sustain_frames = int(config.get("sustain_frames", 6))
        self.missing_hold_frames = int(config.get("missing_hold_frames", 30))
        self.candidate_missing_hold = int(config.get("candidate_missing_hold", 12))
        self.history = deque(
            maxlen=max(self.sustain_frames + 2, int(config.get("history_frames", 8)))
        )
        self.stand_h: float | None = None
        self.has_been_present = False
        self.phase = "NORMAL"
        self.missing_count = 0
        self.recovery_count = 0
        self.aspect_history: deque = deque(maxlen=3)
        self.last_box: dict | None = None

    def reset(self) -> None:
        """新视频源开始时清空全部状态（站立参考高度等不能跨会话残留）。"""
        self.history.clear()
        self.stand_h = None
        self.has_been_present = False
        self.phase = "NORMAL"
        self.missing_count = 0
        self.recovery_count = 0
        self.aspect_history.clear()
        self.last_box = None

    def _result(
        self, confirmed: bool, box: dict | None, positive_frames: int, reasons: list[str]
    ) -> PostImpactResult:
        """构造稳定的输出契约。"""
        return PostImpactResult(
            confirmed=confirmed,
            state="FALL" if confirmed else "UNKNOWN",
            phase=self.phase,
            reason_codes=reasons,
            evidence={
                "history_frames": min(len(self.history), self.sustain_frames),
                "positive_frames": positive_frames,
                "has_been_present": self.has_been_present,
                "stand_h": round(self.stand_h, 2) if self.stand_h else None,
                "last_box": box,
                "missing_hold_frames": self.missing_count,
                "recovery_frames": self.recovery_count,
            },
        )

    def update(self, person) -> PostImpactResult:
        """消费一帧当前观测，不使用任何未来信息。"""
        box = _bbox(person)
        if box is not None:
            self.has_been_present = True
            if box["aspect"] < self.horiz_aspect:
                # 竖直站姿时更新站立参考高度（只记最大值，跪坐不会拉低参考）
                self.stand_h = max(self.stand_h or 0.0, box["h"])

        aspect = box["aspect"] if box else None
        self.aspect_history.append(aspect)
        aspect_prev = (
            self.aspect_history[-2]
            if len(self.aspect_history) >= 2 and self.aspect_history[-2] is not None
            else None
        )
        aspect_der = (
            (aspect - aspect_prev)
            if (aspect is not None and aspect_prev is not None)
            else None
        )
        cy = box["cy"] if box else None
        height_ratio = (box["h"] / self.stand_h) if (box and self.stand_h) else None

        # 新的跌倒确认必须有画面内"宽扁框"；丢框只能临时保持已有告警
        lying = bool(box and aspect >= self.horiz_aspect)
        collapsed = bool(
            box
            and self.stand_h
            and box["h"] <= self.low_height_ratio * self.stand_h
            and aspect >= self.horiz_aspect
        )
        terminal = lying or collapsed
        # 丢帧不进历史：检测器对横卧人体间歇丢失时，累计应冻结而非被打断
        if box is not None:
            self.history.append(terminal)
        positive_frames = sum(self.history)

        cues = []
        if box:
            if aspect is not None and aspect >= self.horiz_aspect:
                cues.append(f"aspect={aspect:.2f}")
            if height_ratio is not None and height_ratio <= self.low_height_ratio:
                cues.append(f"height_ratio={height_ratio:.2f}")
            if cy is not None and self.stand_h:
                cues.append(f"cy={cy:.0f}")
        if aspect_der is not None:
            cues.append(f"aspect_der={aspect_der:+.2f}")
        self.last_box = box

        if self.phase == "NORMAL":
            self.missing_count = 0
            if terminal:
                self.phase = "FALL_CANDIDATE"
        elif self.phase == "FALL_CANDIDATE":
            if terminal:
                self.missing_count = 0
                if positive_frames >= self.sustain_frames:
                    self.phase = "FALL_CONFIRMED"
            elif box is None:
                # 候选期短暂丢帧不打断累计（看不见人 ≠ 人站起来了）；
                # 持续丢失超过容忍上限才回退，防离场误挂
                self.missing_count += 1
                if self.missing_count > self.candidate_missing_hold:
                    self.phase = "NORMAL"
                    self.missing_count = 0
            else:
                self.phase = "NORMAL"
                self.missing_count = 0
        elif self.phase in ("FALL_CONFIRMED", "FALL_PERSISTING"):
            if terminal:
                self.phase = "FALL_PERSISTING"
                self.missing_count = 0
                self.recovery_count = 0
            elif box is None:
                # 已确认后短暂丢框（遮挡/检测抖动）保持告警，超时才转恢复
                self.missing_count += 1
                self.phase = "FALL_PERSISTING"
                if self.missing_count > self.missing_hold_frames:
                    self.phase = "RECOVERY"
                    self.recovery_count = 0
            else:
                self.recovery_count += 1
                if self.recovery_count >= self.sustain_frames:
                    self.phase = "RECOVERY"
                    self.missing_count = 0
        else:  # RECOVERY
            if terminal:
                self.phase = "FALL_CANDIDATE"
                self.recovery_count = 0
            elif box is not None:
                self.recovery_count += 1
                if self.recovery_count >= self.sustain_frames:
                    self.phase = "NORMAL"
                    self.recovery_count = 0

        confirmed = self.phase in ("FALL_CONFIRMED", "FALL_PERSISTING")
        reasons = (
            ["post_impact_confirmed", "sustained_supine_in_frame"]
            if confirmed
            else ["post_impact_pending"]
        )
        if confirmed and box is None:
            reasons.append("alarm_hold_after_detection_gap")
        if cues:
            reasons.append("evidence:" + "|".join(cues))
        result = self._result(confirmed, box, positive_frames, reasons)
        result.evidence["aspect"] = round(aspect, 3) if aspect is not None else None
        result.evidence["aspect_derivative"] = (
            round(aspect_der, 3) if aspect_der is not None else None
        )
        result.evidence["height_ratio"] = (
            round(height_ratio, 3) if height_ratio is not None else None
        )
        return result
