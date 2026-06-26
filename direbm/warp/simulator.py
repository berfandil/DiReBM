"""GPU DiReBM simulator (v2): the full step on device, validated against the v1 oracle.

State (moments) lives on the device between steps; the per-step intermediates (components, control
points) are created and freed each step. Control-point and moment counts vary step to step, so the
arrays are reallocated each step rather than grown in place (the simple, correct approach; a
capacity+compaction scheme is a later optimization).

Differences from v1 (all macroscopic-equivalent, validated by tests/experiments):
  - control points via cell-thinning (one per dx/α cell) instead of greedy density-threshold;
  - soft_outer treated as inner (no anti-anisotropy spawn — the deferred step-3 issue);
  - float32.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..constants import ALPHA, DX, KAPPA_HARD
from ..constants import W as _W_NP
from .physics import _collide, _consts, default_device
from .propagation import _disperse, create_control_points, refine_control_points, resample


class GpuSimulator:
    def __init__(
        self,
        pos_m,
        f_m,
        tau,
        *,
        dx=DX,
        alpha=ALPHA,
        kappa_hard=KAPPA_HARD,
        rho_rest=1.0,
        device=None,
    ):
        dev = device or default_device()
        self.device = dev
        self.pos_m = wp.array(np.ascontiguousarray(pos_m, np.float32), dtype=wp.vec2, device=dev)
        self.f_m = wp.array(np.ascontiguousarray(f_m, np.float32), dtype=wp.float32, device=dev)
        self.tau = float(tau)
        self.dx = float(dx)
        self.cell = float(dx / alpha)  # cell-thinning cell size = dx/α
        self.kappa_hard = int(kappa_hard)
        self.C, self.W = _consts(self.device)
        self.f_rest = wp.array((rho_rest * _W_NP).astype(np.float32), dtype=wp.float32, device=self.device)
        self.iteration = 0

    def step(self):
        dev = self.device
        m = self.pos_m.shape[0]

        # collision (in place on the moments)
        wp.launch(_collide, dim=m, inputs=[self.f_m, self.C, self.W, self.tau], device=dev)

        # dispersion → components
        nc = 7 * m
        pos_c = wp.zeros(nc, dtype=wp.vec2, device=dev)
        val_c = wp.zeros(nc, dtype=wp.float32, device=dev)
        dir_c = wp.zeros(nc, dtype=wp.int32, device=dev)
        wp.launch(
            _disperse,
            dim=m,
            inputs=[self.pos_m, self.f_m, self.C, self.dx, pos_c, val_c, dir_c],
            device=dev,
        )

        # create → refine → resample
        cp_pos, _ = create_control_points(pos_c, self.cell, device=dev)
        cp_pos = refine_control_points(cp_pos, pos_c, val_c, dir_c, self.dx, self.kappa_hard, device=dev)
        self.pos_m, self.f_m = resample(pos_c, val_c, dir_c, cp_pos, self.dx, self.f_rest, device=dev)
        self.iteration += 1

    def moments(self):
        """Return (positions (N,2), f (N,7)) as numpy."""
        return self.pos_m.numpy(), self.f_m.numpy()


__all__ = ["GpuSimulator"]
