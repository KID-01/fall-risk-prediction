# -*- coding: utf-8 -*-
"""Day-2：Environment Risk Engine（模块 C，透明确定性规则引擎）。

- person selection：确定性主人物（与上一帧主人物重叠优先，否则最大 bbox）；
- foot reference：person bbox 底边中心（图像平面近似；可选 ankle 调试比较，不阻塞）；
- proximity：foot 点到 object bbox 的最近图像平面距离，除以 person bbox 高 => d_norm；
- d_norm 是单目图像平面邻近代理，非物理米制距离；
- contribution = class_weight * proximity_factor * confidence（置信度因子）；
- Environment Risk Score = sum(contributions) clipped [0,1]；top_hazards 取贡献前 k。
- 无训练；权重在 config；不宣称校准跌倒概率。
"""
from __future__ import annotations

from dataclasses import asdict

from .contract import EnvRiskFrame, default_config


def _point_box_dist(px: float, py: float, box) -> float:
    """foot 点到 axis-aligned bbox 的最近图像平面距离（像素）；框内=0。"""
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return float((dx * dx + dy * dy) ** 0.5)


def _iou(a, b) -> float:
    x1 = max(a["x1"], b["x1"]); y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"]); y2 = min(a["y2"], b["y2"])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    ar = max(1e-6, (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]))
    br = max(1e-6, (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))
    return inter / (ar + br - inter)


def choose_primary(persons: list[dict], prev_primary: dict | None, min_conf: float) -> dict | None:
    """确定性主人物：优先与上一帧主人物 IoU>0 的；否则最大 bbox；否则最高 conf。"""
    cands = [p for p in persons if p.get("conf", 0) >= min_conf]
    if not cands:
        return None
    if prev_primary is not None:
        keep = [p for p in cands if _iou(prev_primary, p) > 0]
        if keep:
            return max(keep, key=lambda p: (p["x2"] - p["x1"]) * (p["y2"] - p["y1"]))
    return max(cands, key=lambda p: (p["x2"] - p["x1"]) * (p["y2"] - p["y1"]))


def proximity_factor(d_norm: float, near_thr: float, far_thr: float) -> float:
    """分段线性：d<=near->1；d>=far->0；中间线性。"""
    if d_norm <= near_thr:
        return 1.0
    if d_norm >= far_thr:
        return 0.0
    return 1.0 - (d_norm - near_thr) / (far_thr - near_thr)


def _foot_band_factor(obj, person) -> float:
    """物体与 person 下半部带的重叠比例（0..1）：非"近脚/通道"物体被抑制。
    band = [person_mid_y, person_y2]。纯横向（无垂直重叠）→ 0。"""
    y1 = person["y1"]; y2 = person["y2"]
    mid = (y1 + y2) / 2.0
    o1, o2 = obj["y1"], obj["y2"]
    over = max(0.0, min(o2, y2) - max(o1, mid))
    obj_h = max(o2 - o1, 1e-3)
    return min(1.0, over / obj_h * 2.0)


def compute_env_risk(person: dict | None, objects: list[dict], cfg: dict) -> EnvRiskFrame:
    er = cfg.get("environment_risk", {})
    table = er.get("object_risk_table", {})
    neutral = set(er.get("neutral_classes", []))
    near_thr = er.get("near_thr", 0.4)
    far_thr = er.get("far_thr", 1.5)
    conf_min = er.get("object_conf_min", 0.25)
    top_k = er.get("top_k_hazards", 3)
    clip = er.get("risk_clip", 1.0)
    foot_gate = er.get("foot_band_gate", True)

    frame = EnvRiskFrame(timestamp=0.0, source="env_risk_v0", person=person)
    if person is None:
        return frame
    p_h = person["y2"] - person["y1"]
    foot = {"px": (person["x1"] + person["x2"]) / 2.0, "py": person["y2"]}
    total = 0.0
    hazards = []
    for o in objects:
        label = o.get("label", "?")
        if label in neutral:
            continue
        w = table.get(label)
        if w is None or o.get("conf", 0) < conf_min:
            continue
        d_px = _point_box_dist(foot["px"], foot["py"], o)
        d_norm = d_px / max(p_h, 1e-3)
        band = _foot_band_factor(o, person) if foot_gate else 1.0
        contrib = float(w) * proximity_factor(d_norm, near_thr, far_thr) * min(o.get("conf", 0), 1.0) * band
        total += contrib
        if contrib > 0:
            hazards.append({
                "class": label, "confidence": round(o.get("conf", 0), 4),
                "normalized_distance": round(d_norm, 3),
                "foot_band_overlap": round(band, 3),
                "risk_contribution": round(contrib, 4),
            })
    frame.environment_risk_score = round(min(total, clip), 4)
    frame.top_hazards = sorted(hazards, key=lambda h: -h["risk_contribution"])[:top_k]
    frame.objects = [asdict(o) if not isinstance(o, dict) else o for o in objects]
    return frame


def sanity_cases(cfg: dict) -> list[dict]:
    """工程 sanity（非科研）：三人-物几何 case，验证普通家具不自动主导。"""
    person = {"x1": 300, "y1": 200, "x2": 360, "y2": 400, "conf": 0.9, "track_id": 1}  # 高=200px
    p_h = person["y2"] - person["y1"]

    def run(name, objs):
        f = compute_env_risk(person, objs, cfg)
        return {"case": name, "env_score": f.environment_risk_score, "top": f.top_hazards[:1]}

    cases = [
        run("A_chair_far",      [{"label": "chair", "conf": 0.9, "x1": 20,   "y1": 20,  "x2": 80,  "y2": 120}]),
        run("B_chair_near_side", [{"label": "chair", "conf": 0.9, "x1": 250,  "y1": 300, "x2": 280, "y2": 390}]),
        run("C_suitcase_at_feet", [{"label": "suitcase", "conf": 0.91, "x1": 330, "y1": 380, "x2": 375, "y2": 405}]),
    ]
    for c in cases:
        print(f"[sanity] {c['case']}: env_score={c['env_score']} top={c['top']}")
    # 断言：A 低、C 高于 A；B 中等
    return cases