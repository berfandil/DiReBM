"""DiReBM — Dispersion-Resampling Boltzmann Method.

A latticeless Lattice-Boltzmann fluid solver. v1 is a numpy reference solver (correctness
oracle); v2 ports to NVIDIA Warp (GPU). See docs/ARCHITECTURE.md for the map.
"""

from .constants import (
    ALPHA,
    DEFAULT_TAU,
    DT,
    DX,
    KAPPA_HARD,
    KAPPA_SOFT,
    C,
    D,
    Q,
    W,
)
from .physics import collide, equilibrium, recover

__version__ = "0.0.1"

__all__ = [
    "__version__",
    # constants
    "D",
    "Q",
    "C",
    "W",
    "DX",
    "DT",
    "KAPPA_HARD",
    "KAPPA_SOFT",
    "ALPHA",
    "DEFAULT_TAU",
    # physics
    "recover",
    "equilibrium",
    "collide",
]
