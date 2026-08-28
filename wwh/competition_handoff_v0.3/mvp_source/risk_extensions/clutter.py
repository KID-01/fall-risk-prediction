"""Obstacle/clutter risk around the person's foot corridor."""
from __future__ import annotations

from .contract import provider_result


def _intersection(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def compute_clutter_risk(person: dict | None, objects: list[dict], config: dict, frame_shape) -> dict:
    """Estimate clutter in a causal foot corridor using existing detection boxes."""
    if person is None:
        return provider_result("clutter_v0", None, available=False, reasons=["person_missing"])
    h_img, w_img = frame_shape[:2]
    pw = max(float(person["x2"] - person["x1"]), 1.0)
    ph = max(float(person["y2"] - person["y1"]), 1.0)
    foot_x = (float(person["x1"]) + float(person["x2"])) / 2.0
    foot_y = float(person["y2"])
    half_width = float(config.get("corridor_width_ratio", 0.65)) * pw
    length = float(config.get("corridor_length_ratio", 1.25)) * ph
    corridor = (
        max(0.0, foot_x - half_width),
        max(0.0, foot_y - 0.15 * ph),
        min(float(w_img), foot_x + half_width),
        min(float(h_img), foot_y + length),
    )
    allowed = set(config.get("obstacle_classes", []))
    overlaps = []
    for obj in objects:
        if allowed and obj.get("label") not in allowed:
            continue
        box = (float(obj["x1"]), float(obj["y1"]), float(obj["x2"]), float(obj["y2"]))
        area = max((box[2] - box[0]) * (box[3] - box[1]), 1.0)
        inter = _intersection(corridor, box)
        if inter <= 0:
            continue
        overlaps.append(
            {
                "class": obj.get("label", "unknown"),
                "confidence": round(float(obj.get("conf", 0.0)), 4),
                "corridor_overlap": round(inter / area, 4),
                "bbox": [round(v, 2) for v in box],
            }
        )
    overlaps.sort(key=lambda x: x["corridor_overlap"] * x["confidence"], reverse=True)
    count_term = min(1.0, len(overlaps) / max(float(config.get("count_high", 3)), 1.0))
    overlap_term = max((x["corridor_overlap"] * x["confidence"] for x in overlaps), default=0.0)
    index = 100.0 * (0.45 * count_term + 0.55 * min(1.0, overlap_term))
    reasons = ["obstacle_in_foot_corridor"] if overlaps else []
    return provider_result(
        "clutter_v0",
        index,
        evidence={"corridor": [round(v, 2) for v in corridor], "obstacle_count": len(overlaps)},
        reasons=reasons,
        obstacles=overlaps[: int(config.get("top_k", 3))],
    )
