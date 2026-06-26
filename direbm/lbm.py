"""D2Q7 hexagonal Lattice-Boltzmann baseline — the macroscopic reference for DiReBM.

The seven D2Q7 directions are not square-grid aligned, but on a *hexagonal* lattice they are the
six nearest neighbours. We store the hex lattice in skewed axial integer coordinates (i, j),
where physical position = i·(1,0) + j·(1/2, √3/2). In those coordinates the directions become
integer index shifts, so streaming is a plain roll:

    c_0 (0,0)      c_1 (1,0)→(1,0)   c_2 (½,√3/2)→(0,1)   c_3 (−½,√3/2)→(−1,1)
    c_4 (−1,0)→(−1,0)   c_5 (−½,−√3/2)→(0,−1)   c_6 (½,−√3/2)→(1,−1)

Uses the same equilibrium (cs²=1/4) and τ as the DiReBM solver, so a comparison isolates the
effect of DiReBM's latticeless propagation. Periodic boundaries (roll wrap).
"""

from __future__ import annotations

import numpy as np

from .physics import collide, equilibrium, recover

# (Δi, Δj) axial-index shift for each direction c_0..c_6.
_OFFSETS = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
_S = np.sqrt(3.0) / 2.0


class HexLBM:
    def __init__(self, ni: int, nj: int, tau: float, rho0: float = 1.0):
        self.ni = int(ni)
        self.nj = int(nj)
        self.tau = float(tau)
        rho = np.full((self.nj, self.ni), float(rho0))
        self.f = equilibrium(rho, np.zeros((self.nj, self.ni, 2)))  # (nj, ni, Q)
        self.center = (self.ni // 2, self.nj // 2)  # (i, j)

    def positions(self):
        """Physical (x, y) of every node, centred on the pulse node. Both shape (nj, ni)."""
        ii, jj = np.meshgrid(np.arange(self.ni), np.arange(self.nj))
        x = ii + jj * 0.5
        y = jj * _S
        ci, cj = self.center
        return x - (ci + cj * 0.5), y - cj * _S

    def radius(self):
        x, y = self.positions()
        return np.sqrt(x * x + y * y)

    def set_pulse(self, rho_pulse: float):
        ci, cj = self.center
        self.f[cj, ci] = equilibrium(np.float64(rho_pulse), np.zeros(2))

    def stream(self):
        new = np.empty_like(self.f)
        for k, (di, dj) in enumerate(_OFFSETS):
            new[..., k] = np.roll(self.f[..., k], shift=(dj, di), axis=(0, 1))
        self.f = new

    def collision(self):
        self.f = collide(self.f, self.tau)

    def step(self):
        self.collision()
        self.stream()

    def macroscopic(self):
        """Return (rho (nj,ni), u (nj,ni,2))."""
        return recover(self.f)


__all__ = ["HexLBM"]
