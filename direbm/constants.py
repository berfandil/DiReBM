"""D2Q7 model constants and default simulation parameters.

We work in lattice units (dx = dt = 1), the standard LBM convention. All seven D2Q7 directions
are unit length — a requirement of the method (see research/idea.md §3.1). Direction ordering
matches the reference C++ implementation (thesis code 5.1).
"""

import numpy as np

D = 2  # spatial dimensionality
Q = 7  # number of discrete velocities

_S = np.sqrt(3.0) / 2.0

# Discrete velocity set c_i (D2Q7). c_0 is the rest direction; c_1..c_6 are the unit hexagon.
C = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, _S],
        [-0.5, _S],
        [-1.0, 0.0],
        [-0.5, -_S],
        [0.5, -_S],
    ],
    dtype=np.float64,
)

# Lattice weights W_i (D2Q7): W_0 = 1/2, rest 1/12. Sum to 1.
W = np.array([1.0 / 2.0] + [1.0 / 12.0] * 6, dtype=np.float64)

# Lattice speed of sound squared. For D2Q7 with the weights above the second moment
# Σ_i W_i c_iα c_iβ = (1/4) δ_αβ, so cs² = 1/4. (The thesis wrote the equilibrium with the
# D2Q9 cs²=1/3 coefficients 3/4.5/1.5 — inconsistent for D2Q7 and it leaks mass; we use the
# consistent cs²=1/4 form. See direbm/physics.equilibrium.)
CS2 = 1.0 / 4.0

# Lattice spacing / timestep (lattice units).
DX = 1.0
DT = 1.0

# Control-point classification thresholds on the perceived-direction count κ (thesis §4.3.3).
KAPPA_HARD = 4  # κ ≤ KAPPA_HARD            → hard_outer
KAPPA_SOFT = 5  # KAPPA_HARD < κ ≤ KAPPA_SOFT → soft_outer; κ > KAPPA_SOFT → inner

# Control-point density factor: creation threshold radius is DX / ALPHA. Larger → denser → more
# precise, more compute. Thesis: α<3 unstable; tested α=4,5.
ALPHA = 4.0

# Default relaxation time. τ > 1/2 required (viscosity ν = (1/3)(τ − 1/2) ≥ 0). 0.6 is a safe,
# clearly-diffusive default; experiments override for a target viscosity.
DEFAULT_TAU = 0.6

__all__ = ["D", "Q", "C", "W", "CS2", "DX", "DT", "KAPPA_HARD", "KAPPA_SOFT", "ALPHA", "DEFAULT_TAU"]
