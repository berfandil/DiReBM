"""Boltzmann/BGK physics shared by the DiReBM solver and the LBM baseline.

Distribution-function arrays are shaped (..., Q): the last axis indexes the D2Q7 directions.
Macroscopic fields: density ρ (...,) and velocity u (..., D). Everything in lattice units.

These are the well-specified, model-agnostic pieces (thesis §3.4): equilibrium, macroscopic
recovery, and the BGK collision. The DiReBM-specific dispersion/resampling lives elsewhere.
"""

import numpy as np

from .constants import CS2, C, W


def recover(f):
    """Recover macroscopic (ρ, u) from distribution values f of shape (..., Q).

    ρ = Σ_i f_i,   u = (1/ρ) Σ_i f_i c_i.
    """
    f = np.asarray(f, dtype=np.float64)
    rho = f.sum(axis=-1)
    # Σ_i f_i c_i  — contract the Q axis against the (Q, D) direction table.
    momentum = f @ C
    # Guard ρ = 0 (vacuum): velocity is undefined there; report zero rather than NaN.
    safe_rho = np.where(rho == 0.0, 1.0, rho)
    u = momentum / safe_rho[..., None]
    return rho, u


def equilibrium(rho, u):
    """Equilibrium distribution f_eq(ρ, u), shape (..., Q).

    Standard second-order LBM equilibrium with this lattice's sound speed cs² = CS2:

        f_eq_i = ρ W_i [ 1 + (c_i·u)/cs² + (c_i·u)²/(2 cs⁴) − (u·u)/(2 cs²) ]

    For D2Q7 (cs² = 1/4) the coefficients are 4, 8, 2. This conserves mass and momentum exactly
    and is isotropic — unlike the thesis eq. 3.19, which used the D2Q9 cs²=1/3 coefficients
    (3, 4.5, 1.5) with D2Q7 weights and so leaks mass. (The reference C++ additionally squared
    the (u·u) term — a separate quartic typo.)
    """
    rho = np.asarray(rho, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    inv = 1.0 / CS2
    cu = u @ C.T  # (..., Q) = c_i · u
    usq = (u * u).sum(axis=-1)  # (...,) = u · u
    bracket = 1.0 + inv * cu + 0.5 * inv * inv * cu * cu - 0.5 * inv * usq[..., None]
    return rho[..., None] * W * bracket


def collide(f, tau):
    """One BGK collision step (thesis eq. 3.14): relax f toward equilibrium by 1/τ.

    f*_i = f_i + (f_eq_i − f_i)/τ = (1 − 1/τ) f_i + (1/τ) f_eq_i.

    Conserves mass and momentum exactly (f_eq shares ρ, u with f).
    """
    rho, u = recover(f)
    feq = equilibrium(rho, u)
    return f + (feq - f) / tau


__all__ = ["recover", "equilibrium", "collide"]
