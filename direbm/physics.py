"""Boltzmann/BGK physics shared by the DiReBM solver and the LBM baseline.

Distribution-function arrays are shaped (..., Q): the last axis indexes the lattice directions.
Macroscopic fields: density ρ (...,) and velocity u (..., D). Everything in lattice units.

Dimension-generic: each function takes a `lattice` (default `D2Q7`, so existing 2D code is
unchanged); pass `D3Q13` for 3D. These are the model-agnostic pieces (thesis §3.4): equilibrium,
macroscopic recovery, and the BGK collision.
"""

import numpy as np

from .lattices import D2Q7


def recover(f, lattice=D2Q7):
    """Recover macroscopic (ρ, u) from distribution values f of shape (..., Q).

    ρ = Σ_i f_i,   u = (1/ρ) Σ_i f_i c_i.
    """
    f = np.asarray(f, dtype=np.float64)
    rho = f.sum(axis=-1)
    momentum = f @ lattice.C  # Σ_i f_i c_i, contracting the Q axis against (Q, D)
    safe_rho = np.where(rho == 0.0, 1.0, rho)  # vacuum: report u=0 rather than NaN
    u = momentum / safe_rho[..., None]
    return rho, u


def equilibrium(rho, u, lattice=D2Q7):
    """Equilibrium distribution f_eq(ρ, u), shape (..., Q).

    Standard second-order LBM equilibrium with the lattice's sound speed cs²:

        f_eq_i = ρ W_i [ 1 + (c_i·u)/cs² + (c_i·u)²/(2 cs⁴) − (u·u)/(2 cs²) ].

    Conserves mass and momentum exactly given an isotropic 2nd moment, and recovers Navier–Stokes
    given an isotropic 4th moment (D2Q7 hex and D3Q13 icosahedral both qualify).
    """
    rho = np.asarray(rho, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    inv = 1.0 / lattice.cs2
    cu = u @ lattice.C.T  # (..., Q) = c_i · u
    usq = (u * u).sum(axis=-1)  # (...,) = u · u
    bracket = 1.0 + inv * cu + 0.5 * inv * inv * cu * cu - 0.5 * inv * usq[..., None]
    return rho[..., None] * lattice.W * bracket


def collide(f, tau, lattice=D2Q7):
    """One BGK collision step (thesis eq. 3.14): relax f toward equilibrium by 1/τ.

    f*_i = f_i + (f_eq_i − f_i)/τ. Conserves mass and momentum exactly.
    """
    rho, u = recover(f, lattice)
    feq = equilibrium(rho, u, lattice)
    return f + (feq - f) / tau


__all__ = ["recover", "equilibrium", "collide"]
