"""Warp GPU kernels for the model-agnostic physics (mirror of direbm.physics, float32).

Moments are stored column-style: positions are separate; the distribution f is a 2D array
(N, 7). Kernels index f[i, k]. Constants (directions C, weights W) are uploaded as device arrays.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..constants import CS2
from ..constants import C as _C_NP
from ..constants import W as _W_NP

wp.init()

_INV = float(1.0 / CS2)  # 1/cs² = 4 for D2Q7


def default_device() -> str:
    return "cuda:0" if wp.get_cuda_devices() else "cpu"


def _consts(device):
    c = wp.array(_C_NP.astype(np.float32), dtype=wp.vec2, device=device)
    w = wp.array(_W_NP.astype(np.float32), dtype=wp.float32, device=device)
    return c, w


@wp.func
def _equilibrium_row(
    rho: wp.float32,
    u: wp.vec2,
    k: int,
    C: wp.array(dtype=wp.vec2),
    W: wp.array(dtype=wp.float32),
):
    inv = wp.float32(_INV)
    cu = wp.dot(C[k], u)
    usq = wp.dot(u, u)
    return rho * W[k] * (1.0 + inv * cu + 0.5 * inv * inv * cu * cu - 0.5 * inv * usq)


@wp.kernel
def _collide(
    f: wp.array2d(dtype=wp.float32),
    C: wp.array(dtype=wp.vec2),
    W: wp.array(dtype=wp.float32),
    tau: wp.float32,
):
    i = wp.tid()
    rho = wp.float32(0.0)
    mom = wp.vec2(0.0, 0.0)
    for k in range(7):
        fk = f[i, k]
        rho += fk
        mom += fk * C[k]
    u = mom / rho
    for k in range(7):
        feq = _equilibrium_row(rho, u, k, C, W)
        f[i, k] = f[i, k] + (feq - f[i, k]) / tau


@wp.kernel
def _equilibrium(
    rho: wp.array(dtype=wp.float32),
    u: wp.array(dtype=wp.vec2),
    C: wp.array(dtype=wp.vec2),
    W: wp.array(dtype=wp.float32),
    out: wp.array2d(dtype=wp.float32),
):
    i = wp.tid()
    for k in range(7):
        out[i, k] = _equilibrium_row(rho[i], u[i], k, C, W)


def collide(f_np, tau, device=None):
    """BGK collision of distributions f_np (N, 7) on the GPU. Returns a new (N, 7) float32 array."""
    device = device or default_device()
    f = wp.array(np.ascontiguousarray(f_np, dtype=np.float32), dtype=wp.float32, device=device)
    c, w = _consts(device)
    wp.launch(_collide, dim=f.shape[0], inputs=[f, c, w, float(tau)], device=device)
    wp.synchronize()
    return f.numpy()


def equilibrium(rho_np, u_np, device=None):
    """Equilibrium distribution for macroscopic (rho (N,), u (N, 2)). Returns (N, 7) float32."""
    device = device or default_device()
    rho = wp.array(np.ascontiguousarray(rho_np, dtype=np.float32), dtype=wp.float32, device=device)
    u = wp.array(np.ascontiguousarray(u_np, dtype=np.float32), dtype=wp.vec2, device=device)
    out = wp.zeros((rho.shape[0], 7), dtype=wp.float32, device=device)
    c, w = _consts(device)
    wp.launch(_equilibrium, dim=rho.shape[0], inputs=[rho, u, c, w, out], device=device)
    wp.synchronize()
    return out.numpy()


__all__ = ["collide", "equilibrium", "default_device"]
