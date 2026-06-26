"""Warp GPU propagation: dispersion and control-point creation (cell-thinning).

Dispersion is embarrassingly parallel (moment → 7 components). Control-point creation replaces
v1's greedy, order-dependent density-threshold with a deterministic, parallel **cell-thinning**:
bucket components into cells of size dx/α and emit one control point per occupied cell, at the
centroid of that cell's components. Implemented with radix sort + run-length encode + a centroid
kernel. (Macroscopically equivalent to the greedy version; validated against the v1 oracle.)
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .physics import _consts, default_device

_CELL_BIAS = 16384  # cell-index offset so packed keys stay non-negative (assumes |cell| < BIAS)
_CELL_STRIDE = 32768


@wp.kernel
def _disperse(
    pos_m: wp.array(dtype=wp.vec2),
    f_m: wp.array2d(dtype=wp.float32),
    C: wp.array(dtype=wp.vec2),
    dx: wp.float32,
    pos_c: wp.array(dtype=wp.vec2),
    val_c: wp.array(dtype=wp.float32),
    dir_c: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    for k in range(7):
        idx = i * 7 + k
        pos_c[idx] = pos_m[i] + C[k] * dx
        val_c[idx] = f_m[i, k]
        dir_c[idx] = k


@wp.kernel
def _cell_key(
    pos_c: wp.array(dtype=wp.vec2),
    cs: wp.float32,
    keys: wp.array(dtype=wp.int32),
    vals: wp.array(dtype=wp.int32),
):
    c = wp.tid()
    p = pos_c[c]
    ix = int(wp.floor(p[0] / cs)) + _CELL_BIAS
    iy = int(wp.floor(p[1] / cs)) + _CELL_BIAS
    keys[c] = ix * _CELL_STRIDE + iy
    vals[c] = c


@wp.kernel
def _centroid(
    vals_sorted: wp.array(dtype=wp.int32),
    pos_c: wp.array(dtype=wp.vec2),
    offsets: wp.array(dtype=wp.int32),
    lengths: wp.array(dtype=wp.int32),
    cp_pos: wp.array(dtype=wp.vec2),
):
    r = wp.tid()
    start = offsets[r]
    n = lengths[r]
    s = wp.vec2(0.0, 0.0)
    for j in range(n):
        s += pos_c[vals_sorted[start + j]]
    cp_pos[r] = s / wp.float32(n)


def disperse(pos_m, f_m, dx, device=None):
    """Moment → component cloud. Returns (pos_c, val_c, dir_c) as device wp arrays (length 7*M)."""
    device = device or default_device()
    m = len(pos_m)
    nc = 7 * m
    posm = wp.array(np.ascontiguousarray(pos_m, dtype=np.float32), dtype=wp.vec2, device=device)
    fm = wp.array(np.ascontiguousarray(f_m, dtype=np.float32), dtype=wp.float32, device=device)
    pos_c = wp.zeros(nc, dtype=wp.vec2, device=device)
    val_c = wp.zeros(nc, dtype=wp.float32, device=device)
    dir_c = wp.zeros(nc, dtype=wp.int32, device=device)
    c, _ = _consts(device)
    wp.launch(_disperse, dim=m, inputs=[posm, fm, c, float(dx), pos_c, val_c, dir_c], device=device)
    wp.synchronize()
    return pos_c, val_c, dir_c


def create_control_points(pos_c, cs, device=None):
    """Cell-thin a component cloud → control-point positions. Returns (cp_pos wp.array, P)."""
    device = device or default_device()
    nc = pos_c.shape[0]
    keys = wp.zeros(2 * nc, dtype=wp.int32, device=device)
    vals = wp.zeros(2 * nc, dtype=wp.int32, device=device)
    wp.launch(_cell_key, dim=nc, inputs=[pos_c, float(cs), keys, vals], device=device)

    wp.utils.radix_sort_pairs(keys, vals, nc)

    run_values = wp.zeros(nc, dtype=wp.int32, device=device)
    run_lengths = wp.zeros(nc, dtype=wp.int32, device=device)
    run_count = wp.zeros(1, dtype=wp.int32, device=device)
    wp.utils.runlength_encode(keys, run_values, run_lengths, run_count, value_count=nc)
    wp.synchronize()
    p = int(run_count.numpy()[0])

    offsets = wp.zeros(nc, dtype=wp.int32, device=device)
    wp.utils.array_scan(run_lengths, offsets, inclusive=False)

    cp_pos = wp.zeros(p, dtype=wp.vec2, device=device)
    wp.launch(_centroid, dim=p, inputs=[vals, pos_c, offsets, run_lengths, cp_pos], device=device)
    wp.synchronize()
    return cp_pos, p


__all__ = ["disperse", "create_control_points"]
