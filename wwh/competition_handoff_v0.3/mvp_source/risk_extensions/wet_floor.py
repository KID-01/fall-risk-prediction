"""Wet-floor provider placeholder until a validated detector is available."""
from __future__ import annotations

from .contract import provider_result


def unavailable_wet_floor() -> dict:
    """Return an explicit UNKNOWN contract instead of fabricating wet-floor detections."""
    return provider_result(
        "wet_floor_unavailable",
        None,
        available=False,
        reasons=["detector_unavailable"],
        regions=[],
    )
