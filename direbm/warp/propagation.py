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


@wp.kernel
def _to_vec3(a: wp.array(dtype=wp.vec2), out: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    out[i] = wp.vec3(a[i][0], a[i][1], 0.0)


@wp.kernel
def _refine(
    cp3: wp.array(dtype=wp.vec3),
    grid_id: wp.uint64,
    pos_c3: wp.array(dtype=wp.vec3),
    val_c: wp.array(dtype=wp.float32),
    dir_c: wp.array(dtype=wp.int32),
    dx: wp.float32,
    kappa_hard: wp.int32,
    new_pos: wp.array(dtype=wp.vec2),
):
    r = wp.tid()
    p = cp3[r]
    mask = wp.int32(0)
    sw = wp.float32(0.0)
    sx = wp.float32(0.0)
    sy = wp.float32(0.0)
    query = wp.hash_grid_query(grid_id, p, dx)
    idx = wp.int32(0)
    while wp.hash_grid_query_next(query, idx):
        q = pos_c3[idx]
        if wp.length(q - p) <= dx:
            mask = mask | (wp.int32(1) << dir_c[idx])
            w = wp.exp(val_c[idx])
            sw += w
            sx += w * q[0]
            sy += w * q[1]
    kappa = wp.int32(0)
    for b in range(7):
        if ((mask >> b) & wp.int32(1)) == wp.int32(1):
            kappa += 1
    # soft_outer is treated as inner here (no anti-anisotropy spawn yet — deferred, see idea.md §3.4)
    if kappa <= kappa_hard:
        new_pos[r] = wp.vec2(p[0], p[1])  # hard_outer: keep (preserve the free surface)
    else:
        new_pos[r] = wp.vec2(sx / sw, sy / sw)  # inner: exp(f)-weighted centroid


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
    # The control-point count P must reach the host to size cp_pos and the launch dims below.
    # This is the one unavoidable per-step sync (numpy() synchronizes); the others were removed.
    p = int(run_count.numpy()[0])

    offsets = wp.zeros(nc, dtype=wp.int32, device=device)
    wp.utils.array_scan(run_lengths, offsets, inclusive=False)

    cp_pos = wp.zeros(p, dtype=wp.vec2, device=device)
    wp.launch(_centroid, dim=p, inputs=[vals, pos_c, offsets, run_lengths, cp_pos], device=device)
    return cp_pos, p


@wp.kernel
def _scatter(
    comp_pos3: wp.array(dtype=wp.vec3),
    val_c: wp.array(dtype=wp.float32),
    dir_c: wp.array(dtype=wp.int32),
    grid_id: wp.uint64,
    cp_pos3: wp.array(dtype=wp.vec3),
    dx: wp.float32,
    cp_f: wp.array2d(dtype=wp.float32),
):
    # Resampling phase 1: distribute each component's f into nearby control points, weighted by
    # dx - distance, normalized over the component's near control points. Atomic (many → one).
    c = wp.tid()
    p = comp_pos3[c]
    sw = wp.float32(0.0)
    q = wp.hash_grid_query(grid_id, p, dx)
    idx = wp.int32(0)
    while wp.hash_grid_query_next(q, idx):
        dd = wp.length(cp_pos3[idx] - p)
        if dd <= dx:
            sw += dx - dd
    if sw <= 0.0:
        return
    k = dir_c[c]
    fval = val_c[c]
    q2 = wp.hash_grid_query(grid_id, p, dx)
    idx2 = wp.int32(0)
    while wp.hash_grid_query_next(q2, idx2):
        dd = wp.length(cp_pos3[idx2] - p)
        if dd <= dx:
            wp.atomic_add(cp_f, idx2, k, ((dx - dd) / sw) * fval)


@wp.kernel
def _gapfill_emit(
    cp_pos3: wp.array(dtype=wp.vec3),
    grid_id: wp.uint64,
    cp_f: wp.array2d(dtype=wp.float32),
    f_rest: wp.array(dtype=wp.float32),
    dx: wp.float32,
    eps: wp.float32,
    out_pos: wp.array(dtype=wp.vec2),
    out_f: wp.array2d(dtype=wp.float32),
):
    # Resampling phase 2: fill directions no component delivered with the rest equilibrium,
    # mass-scaled over nearby also-empty control points; emit a moment per control point.
    # Reads cp_f read-only (writes go to out_f) so neighbour emptiness is the pre-fill snapshot.
    r = wp.tid()
    p = cp_pos3[r]
    for i in range(7):
        fi = cp_f[r, i]
        if fi < eps:
            sw = wp.float32(0.0)
            q = wp.hash_grid_query(grid_id, p, dx)
            idx = wp.int32(0)
            while wp.hash_grid_query_next(q, idx):
                dd = wp.length(cp_pos3[idx] - p)
                if dd <= dx and cp_f[idx, i] < eps:
                    sw += dx - dd
            if sw > 0.0:
                fi = (dx / sw) * f_rest[i]
        out_f[r, i] = fi
    out_pos[r] = wp.vec2(p[0], p[1])


def resample(pos_c, val_c, dir_c, cp_pos, dx, f_rest, device=None):
    """Resampling (thesis §4.3.4): scatter component f into control points, fill gaps from the rest
    equilibrium, emit one moment per control point. Returns (out_pos vec2, out_f (P,7))."""
    device = device or default_device()
    nc = pos_c.shape[0]
    p = cp_pos.shape[0]
    pos_c3 = wp.zeros(nc, dtype=wp.vec3, device=device)
    cp3 = wp.zeros(p, dtype=wp.vec3, device=device)
    wp.launch(_to_vec3, dim=nc, inputs=[pos_c, pos_c3], device=device)
    wp.launch(_to_vec3, dim=p, inputs=[cp_pos, cp3], device=device)

    grid = wp.HashGrid(128, 128, 1, device=device)
    grid.build(points=cp3, radius=float(dx))

    cp_f = wp.zeros((p, 7), dtype=wp.float32, device=device)
    wp.launch(_scatter, dim=nc, inputs=[pos_c3, val_c, dir_c, grid.id, cp3, float(dx), cp_f], device=device)

    out_pos = wp.zeros(p, dtype=wp.vec2, device=device)
    out_f = wp.zeros((p, 7), dtype=wp.float32, device=device)
    wp.launch(
        _gapfill_emit,
        dim=p,
        inputs=[cp3, grid.id, cp_f, f_rest, float(dx), 1e-6, out_pos, out_f],
        device=device,
    )
    return out_pos, out_f


def refine_control_points(cp_pos, pos_c, val_c, dir_c, dx, kappa_hard, device=None):
    """Refine control-point positions (thesis §4.3.3, simplified): query each control point's
    components within dx → perceived-direction count κ → hard_outer (κ≤kappa_hard) keep, else move
    to the exp(f)-weighted component centroid. Returns the new positions (wp.array vec2).

    Simplification vs v1: soft_outer is treated as inner (no anti-anisotropy spawn) — that
    correction is the deferred step-3 issue. Validate macroscopically against v1.
    """
    device = device or default_device()
    nc = pos_c.shape[0]
    p = cp_pos.shape[0]
    pos_c3 = wp.zeros(nc, dtype=wp.vec3, device=device)
    cp3 = wp.zeros(p, dtype=wp.vec3, device=device)
    wp.launch(_to_vec3, dim=nc, inputs=[pos_c, pos_c3], device=device)
    wp.launch(_to_vec3, dim=p, inputs=[cp_pos, cp3], device=device)

    grid = wp.HashGrid(128, 128, 1, device=device)
    grid.build(points=pos_c3, radius=float(dx))

    new_pos = wp.zeros(p, dtype=wp.vec2, device=device)
    wp.launch(
        _refine,
        dim=p,
        inputs=[cp3, grid.id, pos_c3, val_c, dir_c, float(dx), int(kappa_hard), new_pos],
        device=device,
    )
    return new_pos


__all__ = ["disperse", "create_control_points", "refine_control_points", "resample"]
