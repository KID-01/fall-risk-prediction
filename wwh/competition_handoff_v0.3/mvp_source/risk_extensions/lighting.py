"""Frame-level lighting risk diagnostic (engineering index, not probability)."""
from __future__ import annotations

import cv2
import numpy as np

from .contract import provider_result


def compute_lighting_risk(frame, config: dict) -> dict:
    """Compute underexposure/low-contrast risk from the current frame only."""
    if frame is None or getattr(frame, "size", 0) == 0:
        return provider_result(
            "lighting_v0", None, available=False, reasons=["frame_unavailable"]
        )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))
    contrast = float(np.std(gray))
    dark_ratio = float(np.mean(gray < int(config.get("dark_pixel", 45))))

    mean_thr = max(float(config.get("mean_dark", 70.0)), 1.0)
    contrast_thr = max(float(config.get("contrast_low", 25.0)), 1.0)
    dark_thr = max(float(config.get("dark_ratio_high", 0.55)), 1e-6)
    darkness = max(0.0, min(1.0, (mean_thr - mean) / mean_thr))
    low_contrast = max(0.0, min(1.0, (contrast_thr - contrast) / contrast_thr))
    dark_excess = max(0.0, min(1.0, dark_ratio / dark_thr))
    index = 100.0 * (0.5 * darkness + 0.3 * dark_excess + 0.2 * low_contrast)
    reasons = []
    if mean < mean_thr:
        reasons.append("low_mean_luminance")
    if dark_ratio >= dark_thr:
        reasons.append("high_dark_pixel_ratio")
    if contrast < contrast_thr:
        reasons.append("low_contrast")
    return provider_result(
        "lighting_v0",
        index,
        evidence={
            "mean_luminance": round(mean, 2),
            "contrast_std": round(contrast, 2),
            "dark_pixel_ratio": round(dark_ratio, 4),
        },
        reasons=reasons,
    )
