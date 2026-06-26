"""v2 — NVIDIA Warp GPU port of DiReBM, validated against the v1 reference oracle.

Built incrementally. So far: the physics kernels (recover + equilibrium + BGK collision). The
latticeless propagation (dispersion → control points → resampling) on wp.HashGrid follows.

float32 on the GPU (Blackwell fp64 is slow); validated against the float64 oracle within ~1e-5.
"""

from .physics import collide, default_device, equilibrium
from .propagation import create_control_points, disperse, refine_control_points

__all__ = [
    "collide",
    "equilibrium",
    "default_device",
    "disperse",
    "create_control_points",
    "refine_control_points",
]
