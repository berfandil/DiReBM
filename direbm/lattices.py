"""Velocity lattices for DiReBM. The method needs **unit-length** directions (so dispersion moves
every component by the same dx) plus isotropic velocity moments (so the equilibrium recovers
Navier–Stokes).

- 2D: **D2Q7** — the six unit-length hexagon directions + rest (the thesis choice).
- 3D: **D3Q13-ico** — the twelve unit-length icosahedron-vertex directions + rest. Icosahedral
  symmetry is isotropic to high order, so a single weight gives isotropic 2nd and 4th moments —
  unlike the cubic FCC 12-neighbour set, where c_x⁴ and c_x²c_y² isotropy cannot both hold (so FCC
  D3Q13 is anisotropic). See docs/decisions/0002-3d-icosahedral.md.

A `Lattice` bundles the directions C (Q×D), weights W (Q,), and sound speed cs². Physics functions
default to D2Q7, so existing 2D code is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import CS2 as _CS2_2D
from .constants import C as _C2D
from .constants import W as _W2D


@dataclass(frozen=True)
class Lattice:
    C: np.ndarray  # (Q, D) discrete velocities; row 0 is the rest direction
    W: np.ndarray  # (Q,) weights, sum to 1
    cs2: float  # lattice sound speed squared (= Σ W_i c_iα c_iβ along the diagonal)
    name: str

    @property
    def D(self) -> int:
        return self.C.shape[1]

    @property
    def Q(self) -> int:
        return self.C.shape[0]


D2Q7 = Lattice(C=_C2D, W=_W2D, cs2=_CS2_2D, name="D2Q7")

# --- D3Q13 icosahedral ---------------------------------------------------------------------------
_PHI = (1.0 + np.sqrt(5.0)) / 2.0
_ICO = np.array(
    [
        [0, 1, _PHI], [0, 1, -_PHI], [0, -1, _PHI], [0, -1, -_PHI],
        [1, _PHI, 0], [1, -_PHI, 0], [-1, _PHI, 0], [-1, -_PHI, 0],
        [_PHI, 0, 1], [_PHI, 0, -1], [-_PHI, 0, 1], [-_PHI, 0, -1],
    ],
    dtype=np.float64,
)
_ICO /= np.linalg.norm(_ICO[0])  # all twelve have the same norm → unit length

# 2nd moment: Σ_i W_i c_iα c_iβ = cs² δ with cs² = 4w (Σ over the 12 unit dirs of c_x² = 4).
# 4th moment (isotropic for icosahedral): Σ W c_x²c_y² = 0.8w and Σ W c_x⁴ = 3·(0.8w). Recovering
# Navier–Stokes needs this to equal cs⁴, i.e. 0.8w = (4w)² → w = 1/20. Then cs² = 1/5, W_0 = 2/5.
C3 = np.vstack([np.zeros(3), _ICO])  # (13, 3)
W3 = np.array([2.0 / 5.0] + [1.0 / 20.0] * 12, dtype=np.float64)
D3Q13 = Lattice(C=C3, W=W3, cs2=1.0 / 5.0, name="D3Q13-ico")

__all__ = ["Lattice", "D2Q7", "D3Q13"]
