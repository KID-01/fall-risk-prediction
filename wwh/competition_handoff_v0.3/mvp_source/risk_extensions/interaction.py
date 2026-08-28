"""Interaction risk between causal predicted path and hazard regions."""
from __future__ import annotations

import math

from .contract import provider_result


def _point_distance_to_box(point: list[float], box: tuple[float, float, float, float]) -> float:
    x, y = point
    dx = max(box[0] - x, 0.0, x - box[2])
    dy = max(box[1] - y, 0.0, y - box[3])
    return math.hypot(dx, dy)


def compute_interaction_risk(trajectory: dict, objects: list[dict], wet_floor: dict, config: dict) -> dict:
    """Score intersections/proximity along a predicted causal path."""
    if not trajectory.get("available") or not trajectory.get("predicted_points"):
        return provider_result(
            "interaction_v0", None, available=False, reasons=["trajectory_unavailable"], intersections=[]
        )
    allowed = set(config.get("obstacle_classes", []))
    corridor_px = float(config.get("corridor_radius_px", 24.0))
    points = trajectory["predicted_points"]
    intersections = []
    for obj in objects:
        if allowed and obj.get("label") not in allowed:
            continue
        box = (float(obj["x1"]), float(obj["y1"]), float(obj["x2"]), float(obj["y2"]))
        distances = [_point_distance_to_box(p, box) for p in points]
        min_dist = min(distances)
        if min_dist <= corridor_px:
            step = distances.index(min_dist)
            intersections.append(
                {
                    "type": "obstacle",
                    "class": obj.get("label", "unknown"),
                    "confidence": round(float(obj.get("conf", 0.0)), 4),
                    "min_distance_px": round(min_dist, 2),
                    "time_to_interaction_s": round(
                        float(trajectory.get("horizon_s", 1.0)) * (step + 1) / len(points), 3
                    ),
                }
            )
    if wet_floor.get("available"):
        for region in wet_floor.get("regions", []):
            box = tuple(float(v) for v in region.get("bbox", []))
            if len(box) != 4:
                continue
            distances = [_point_distance_to_box(p, box) for p in points]
            min_dist = min(distances)
            if min_dist <= corridor_px:
                step = distances.index(min_dist)
                intersections.append(
                    {
                        "type": "wet_floor",
                        "class": "wet_floor",
                        "confidence": round(float(region.get("confidence", 0.0)), 4),
                        "min_distance_px": round(min_dist, 2),
                        "time_to_interaction_s": round(
                            float(trajectory.get("horizon_s", 1.0)) * (step + 1) / len(points), 3
                        ),
                    }
                )
    if not intersections:
        return provider_result(
            "interaction_v0", 0.0, evidence={"path_points": len(points)}, reasons=[], intersections=[]
        )
    nearest = min(x["time_to_interaction_s"] for x in intersections)
    confidence = max(x["confidence"] for x in intersections)
    urgency = max(0.0, 1.0 - nearest / max(float(trajectory.get("horizon_s", 1.0)), 1e-6))
    index = 100.0 * min(1.0, 0.6 * urgency + 0.4 * confidence)
    return provider_result(
        "interaction_v0",
        index,
        evidence={"path_points": len(points), "nearest_time_s": round(nearest, 3)},
        reasons=["predicted_path_intersects_hazard"],
        intersections=intersections,
    )
