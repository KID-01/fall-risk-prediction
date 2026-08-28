"""Optional v0.3 engineering risk extensions.

These providers are disabled by default and do not alter the frozen v0.2 fusion semantics.
"""

from .clutter import compute_clutter_risk
from .interaction import compute_interaction_risk
from .lighting import compute_lighting_risk
from .trajectory import CausalTrajectoryProvider
from .wet_floor import unavailable_wet_floor

__all__ = [
    "CausalTrajectoryProvider",
    "compute_clutter_risk",
    "compute_interaction_risk",
    "compute_lighting_risk",
    "unavailable_wet_floor",
]
