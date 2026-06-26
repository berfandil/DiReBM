"""Warp GPU port of the D2Q7 hexagonal LBM baseline (mirror of direbm.lbm.HexLBM, float32).

Regular grid → trivially parallel: a collision kernel (one thread per node) and a streaming
kernel (gather each direction from its axial-shifted neighbour, periodic). Same equilibrium and τ
as the CPU baseline; validated against it. Used for a fair GPU-vs-GPU speed comparison with the
DiReBM GPU solver.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..physics import equilibrium as _np_equilibrium
from ..physics import recover as _np_recover
from .physics import _consts, _equilibrium_row, default_device

# Axial-index shifts (Δi, Δj) for c_0..c_6 (same as direbm.lbm).
_OFFX = [0, 1, 0, -1, -1, 0, 1]
_OFFY = [0, 0, 1, 1, 0, -1, -1]
_S = np.sqrt(3.0) / 2.0


@wp.kernel
def _lbm_collide(
    f: wp.array3d(dtype=wp.float32),
    C: wp.array(dtype=wp.vec2),
    W: wp.array(dtype=wp.float32),
    tau: wp.float32,
):
    j, i = wp.tid()
    rho = wp.float32(0.0)
    mom = wp.vec2(0.0, 0.0)
    for k in range(7):
        fk = f[j, i, k]
        rho += fk
        mom += fk * C[k]
    u = mom / rho
    for k in range(7):
        feq = _equilibrium_row(rho, u, k, C, W)
        f[j, i, k] = f[j, i, k] + (feq - f[j, i, k]) / tau


@wp.kernel
def _lbm_stream(
    f: wp.array3d(dtype=wp.float32),
    offx: wp.array(dtype=wp.int32),
    offy: wp.array(dtype=wp.int32),
    ni: wp.int32,
    nj: wp.int32,
    out: wp.array3d(dtype=wp.float32),
):
    j, i = wp.tid()
    for k in range(7):
        si = ((i - offx[k]) % ni + ni) % ni
        sj = ((j - offy[k]) % nj + nj) % nj
        out[j, i, k] = f[sj, si, k]


@wp.kernel
def _lbm_set_center(
    f: wp.array3d(dtype=wp.float32),
    ci: wp.int32,
    cj: wp.int32,
    rho: wp.float32,
    W: wp.array(dtype=wp.float32),
):
    k = wp.tid()
    f[cj, ci, k] = rho * W[k]


class GpuHexLBM:
    def __init__(self, ni, nj, tau, rho0=1.0, device=None):
        self.ni = int(ni)
        self.nj = int(nj)
        self.tau = float(tau)
        self.device = device or default_device()
        f0 = _np_equilibrium(np.full((self.nj, self.ni), float(rho0)), np.zeros((self.nj, self.ni, 2)))
        self.f = wp.array(f0.astype(np.float32), dtype=wp.float32, device=self.device)
        self.out = wp.zeros((self.nj, self.ni, 7), dtype=wp.float32, device=self.device)
        self.C, self.W = _consts(self.device)
        self.offx = wp.array(np.array(_OFFX, dtype=np.int32), dtype=wp.int32, device=self.device)
        self.offy = wp.array(np.array(_OFFY, dtype=np.int32), dtype=wp.int32, device=self.device)
        self.center = (self.ni // 2, self.nj // 2)

    def set_pulse(self, rho_pulse):
        ci, cj = self.center
        wp.launch(
            _lbm_set_center,
            dim=7,
            inputs=[self.f, int(ci), int(cj), float(rho_pulse), self.W],
            device=self.device,
        )

    def step(self):
        dev = self.device
        dim = (self.nj, self.ni)
        wp.launch(_lbm_collide, dim=dim, inputs=[self.f, self.C, self.W, self.tau], device=dev)
        wp.launch(
            _lbm_stream,
            dim=dim,
            inputs=[self.f, self.offx, self.offy, self.ni, self.nj, self.out],
            device=dev,
        )
        self.f, self.out = self.out, self.f

    def macroscopic(self):
        return _np_recover(self.f.numpy())

    def radius(self):
        ii, jj = np.meshgrid(np.arange(self.ni), np.arange(self.nj))
        ci, cj = self.center
        x = (ii + jj * 0.5) - (ci + cj * 0.5)
        y = (jj - cj) * _S
        return np.sqrt(x * x + y * y)


__all__ = ["GpuHexLBM"]
