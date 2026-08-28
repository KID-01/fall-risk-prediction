"""Shared helpers for optional v0.3 engineering risk providers."""
from __future__ import annotations


def risk_state(index: float | None, thresholds: tuple[float, float] = (40.0, 70.0)) -> str:
    """Map an engineering risk index to LOW/MEDIUM/HIGH/UNKNOWN."""
    if index is None:
        return "UNKNOWN"
    if index < thresholds[0]:
        return "LOW"
    if index < thresholds[1]:
        return "MEDIUM"
    return "HIGH"


def provider_result(source: str, index: float | None, *, evidence=None, reasons=None, available=True, **extra) -> dict:
    """Create a stable provider result dictionary."""
    result = {
        "source": source,
        "available": bool(available),
        "risk_index": None if index is None else round(float(max(0.0, min(100.0, index))), 2),
        "state": risk_state(index) if available else "UNKNOWN",
        "evidence": evidence or {},
        "reason_codes": reasons or [],
    }
    result.update(extra)
    return result
