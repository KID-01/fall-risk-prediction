# -*- coding: utf-8 -*-
"""Day-3：确定性规则融合（模块 D 概念）。

分离三概念：
- motion_heuristic_score  = 工程身体运动行为指标（非校准概率）
- environment_risk_score = 工程上下文危险指标
- overall_risk_state     = 确定性工程融合输出
任何一个都不是校准跌倒概率 / 临床跌倒概率 / 科学验证风险概率。

原则（v0.1 强制）：
- Motion 是瞬时主赢警信号；Environment 是上下文修正/放大器；
- environment 单独不能触发 HIGH imminent-fall 警告；
- 全部阈值/规则在 config；工程启发；不做科学调参。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# (motion_state, env_state) -> overall（两态均非 UNKNOWN 时查表）
FUSION_TABLE: dict[tuple[str, str], str] = {
    ("LOW", "LOW"): "LOW",
    ("LOW", "MEDIUM"): "LOW",   # env 只做 context_elevated
    ("LOW", "HIGH"): "LOW",     # env 单独不能 HIGH（motion LOW）
    ("MEDIUM", "LOW"): "MEDIUM",
    ("MEDIUM", "MEDIUM"): "MEDIUM",
    ("MEDIUM", "HIGH"): "HIGH",  # motion 中 + 高危环境 -> 更强警告
    ("HIGH", "LOW"): "HIGH",
    ("HIGH", "MEDIUM"): "HIGH",
    ("HIGH", "HIGH"): "HIGH",
}

STATES = ["LOW", "MEDIUM", "HIGH"]


@dataclass
class FusionResult:
    person_present: bool
    person_match: Optional[bool]
    motion_state: str
    environment_state: str
    overall_state: str
    reason_codes: list[str] = field(default_factory=list)
    context_elevated: bool = False
    motion_score: Optional[float] = None
    environment_score: Optional[float] = None
    top_hazards: list[dict] = field(default_factory=list)


def state_from_score(score: Optional[float], thr_low: float, thr_high: float) -> str:
    if score is None:
        return "UNKNOWN"
    if score < thr_low:
        return "LOW"
    if score >= thr_high:
        return "HIGH"
    return "MEDIUM"


def fuse(
    motion_score: Optional[float],
    env_score: Optional[float],
    env_top_hazards: list[dict],
    person_present: bool,
    person_match: Optional[bool],
    motion_thr: tuple[float, float],
    env_thr: tuple[float, float],
) -> FusionResult:
    """确定性融合；reason codes 可解释；缺失数据 != 零风险。"""
    r = FusionResult(
        person_present=person_present,
        person_match=person_match,
        motion_state="UNKNOWN",
        environment_state="UNKNOWN",
        overall_state="UNKNOWN",
        motion_score=motion_score,
        environment_score=env_score,
        top_hazards=env_top_hazards,
    )

    if not person_present:
        r.overall_state = "UNKNOWN"
        r.reason_codes = ["person_missing"]
        return r
    if person_match is False:
        r.overall_state = "UNKNOWN"
        r.reason_codes = ["person_branch_mismatch"]
        return r

    ms = state_from_score(motion_score, motion_thr[0], motion_thr[1])
    es = state_from_score(env_score, env_thr[0], env_thr[1])
    r.motion_state = ms
    r.environment_state = es

    if ms == "UNKNOWN":
        # motion 分支缺失：环境仍可报 contextual hazard；整体 UNKNOWN
        r.overall_state = "UNKNOWN"
        r.reason_codes = ["motion_missing"] + ([f"environment_{es.lower()}"] if es != "UNKNOWN" else [])
        return r

    r.reason_codes = [f"motion_{ms.lower()}"]
    if es != "UNKNOWN":
        r.reason_codes.append(f"environment_{es.lower()}")

    if ms == "LOW" and es in ("MEDIUM", "HIGH"):
        r.context_elevated = True
        r.reason_codes.append("context_elevated")

    r.overall_state = FUSION_TABLE.get((ms, es), "UNKNOWN")
    return r


def causal_persist(states: list[str], n: int) -> str:
    """因果稳定化：最近 n 帧（含当前）多数；n=1 即关闭（直接用当前）。全部不知道才 UNKNOWN。"""
    if n <= 1:
        return states[-1] if states else "UNKNOWN"
    win = states[-n:] if len(states) >= n else states
    from collections import Counter

    known = [s for s in win if s != "UNKNOWN"]
    if not known:
        return "UNKNOWN"
    cnt = Counter(known)
    best, num = cnt.most_common(1)[0]
    # 多数同前者采用；并列偏好最近
    return best if num > len(win) / 2 else (win[-1] if win[-1] != "UNKNOWN" else best)


def reason_str(r: FusionResult) -> str:
    return "+".join(r.reason_codes) if r.reason_codes else "none"