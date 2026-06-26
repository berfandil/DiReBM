"""Reconstruct macroscopic fields (ρ, u) from scattered moments by spatial binning.

Critical subtlety: a moment's per-point ρ = Σ_i f_i is NOT the macroscopic density — it is the
mass carried by one sample point, which scales as 1 / (local sample-point density). The physical
density field is mass per unit AREA. So to compare against a grid solver (LBM) you must bin the
moments' mass and momentum onto a regular grid and divide by cell area. (See exp_rest_state: the
per-moment ρ falls while the binned field stays ≈ ρ_rest.)
"""

from __future__ import annotations

import numpy as np

from .constants import C


def bin_fields(x, f, xmin, xmax, h):
    """Bin moments onto a regular grid; return (rho, u, extent).

    x: (N, 2) positions. f: (N, Q) distributions. Grid spans [xmin, xmax]^2 with cell size h.
    rho: (ny, nx) density = (cell mass)/(h^2). u: (ny, nx, 2) = (cell momentum)/(cell mass),
    zero where empty. extent = (xmin, xmax, xmin, xmax) for imshow(origin="lower").
    """
    x = np.asarray(x, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    edges = np.arange(xmin, xmax + h, h)
    mass = f.sum(axis=1)  # per-moment mass Σ_i f_i
    momentum = f @ C  # per-moment momentum Σ_i f_i c_i = ρu  (N, 2)

    # histogram2d uses (x, y) -> indexed [ix, iy]; transpose to [iy, ix] for image conventions.
    mass_grid, _, _ = np.histogram2d(x[:, 0], x[:, 1], bins=[edges, edges], weights=mass)
    momx, _, _ = np.histogram2d(x[:, 0], x[:, 1], bins=[edges, edges], weights=momentum[:, 0])
    momy, _, _ = np.histogram2d(x[:, 0], x[:, 1], bins=[edges, edges], weights=momentum[:, 1])
    mass_grid, momx, momy = mass_grid.T, momx.T, momy.T

    area = h * h
    rho = mass_grid / area
    safe = np.where(mass_grid == 0.0, 1.0, mass_grid)
    u = np.stack([momx / safe, momy / safe], axis=-1)
    u[mass_grid == 0.0] = 0.0
    return rho, u, (xmin, xmax, xmin, xmax)


__all__ = ["bin_fields"]
