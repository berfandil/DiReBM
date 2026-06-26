"""Warp dispersion + cell-thinning control-point creation, vs a numpy reference."""

import numpy as np
import warp as wp

from direbm.constants import C
from direbm.warp import default_device
from direbm.warp.propagation import create_control_points, disperse, refine_control_points


def _ring(center, radius, dirs, vals):
    """Components placed on a small ring around `center`, one per direction in `dirs`."""
    ang = np.linspace(0.0, 2 * np.pi, len(dirs), endpoint=False)
    pos = np.array(center) + radius * np.c_[np.cos(ang), np.sin(ang)]
    return pos, np.array(dirs, dtype=np.int32), np.array(vals, dtype=np.float64)


def _wp_refine(cp_np, pos_np, dir_np, val_np, dx, kappa_hard):
    dev = default_device()
    cp = wp.array(cp_np.astype(np.float32), dtype=wp.vec2, device=dev)
    pc = wp.array(pos_np.astype(np.float32), dtype=wp.vec2, device=dev)
    vc = wp.array(val_np.astype(np.float32), dtype=wp.float32, device=dev)
    dc = wp.array(dir_np.astype(np.int32), dtype=wp.int32, device=dev)
    return refine_control_points(cp, pc, vc, dc, dx=dx, kappa_hard=kappa_hard).numpy()


def _cell_thin_ref(pos_c, cs):
    """One control point per occupied dx/α cell, at the centroid of its components."""
    cells = np.floor(pos_c / cs).astype(np.int64)
    keys = cells[:, 0] * 1_000_003 + cells[:, 1]
    uniq, inv = np.unique(keys, return_inverse=True)
    sums = np.zeros((len(uniq), 2))
    cnt = np.zeros(len(uniq))
    np.add.at(sums, inv, pos_c)
    np.add.at(cnt, inv, 1)
    return sums / cnt[:, None]


def _sorted_rows(a):
    return a[np.lexsort((a[:, 1], a[:, 0]))]


def test_dispersion_matches_formula():
    rng = np.random.default_rng(0)
    pos_m = rng.uniform(-5.0, 5.0, size=(20, 2))
    f_m = rng.uniform(0.01, 1.0, size=(20, 7))
    pos_c, val_c, dir_c = disperse(pos_m, f_m, dx=1.0)

    expected = (pos_m[:, None, :] + C[None, :, :] * 1.0).reshape(-1, 2)
    assert np.allclose(pos_c.numpy(), expected, atol=1e-5)
    assert np.allclose(val_c.numpy(), f_m.reshape(-1), atol=1e-6)
    assert np.array_equal(dir_c.numpy(), np.tile(np.arange(7), 20))


def test_cell_thinning_matches_reference():
    rng = np.random.default_rng(1)
    pos_m = rng.uniform(-5.0, 5.0, size=(50, 2))
    f_m = rng.uniform(0.01, 1.0, size=(50, 7))
    pos_c, _, _ = disperse(pos_m, f_m, dx=1.0)
    pc = pos_c.numpy()

    cs = 1.0 / 4.0  # dx / alpha
    cp, p = create_control_points(pos_c, cs)
    ref = _cell_thin_ref(pc, cs)

    assert p == len(ref)
    assert np.allclose(_sorted_rows(cp.numpy()), _sorted_rows(ref), atol=1e-4)


def test_thinning_reduces_count_and_is_bounded():
    rng = np.random.default_rng(2)
    pos_m = rng.uniform(-3.0, 3.0, size=(40, 2))
    f_m = rng.uniform(0.01, 1.0, size=(40, 7))
    pos_c, _, _ = disperse(pos_m, f_m, dx=1.0)
    _, p = create_control_points(pos_c, cs=1.0 / 4.0)
    assert 0 < p <= pos_c.shape[0]


def test_refine_hard_keeps_inner_moves():
    # CP A: 7 directions present (κ=7 > kappa_hard) → inner → moves to exp(f)-weighted centroid.
    # CP B (far): only 3 directions (κ=3 ≤ kappa_hard) → hard_outer → stays put.
    posA, dirA, valA = _ring((0.0, 0.0), 0.2, [0, 1, 2, 3, 4, 5, 6], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    posB, dirB, valB = _ring((10.0, 0.0), 0.2, [0, 1, 2], [0.1, 0.2, 0.3])
    pos = np.vstack([posA, posB])
    dirs = np.concatenate([dirA, dirB])
    vals = np.concatenate([valA, valB])
    cp = np.array([[0.0, 0.0], [10.0, 0.0]])

    new = _wp_refine(cp, pos, dirs, vals, dx=1.0, kappa_hard=4)

    w = np.exp(valA)
    expected_a = (w[:, None] * posA).sum(0) / w.sum()
    assert np.allclose(new[0], expected_a, atol=1e-4)  # inner moved to centroid
    assert np.allclose(new[1], [10.0, 0.0], atol=1e-5)  # hard stayed


def test_refine_kappa_threshold():
    # κ=4 (≤4) → hard, stays; κ=5 (>4) → inner, moves.
    # Unequal weights: a wrong "inner" classification would shift the centroid off centre, so
    # asserting it stays genuinely confirms hard_outer.
    pos4, dir4, val4 = _ring((0.0, 0.0), 0.2, [0, 1, 2, 3], [0.1, 0.3, 0.5, 0.7])
    # κ=5 with UNEQUAL weights so the exp(f)-weighted centroid is genuinely off the ring centre.
    pos5, dir5, val5 = _ring((20.0, 0.0), 0.2, [0, 1, 2, 3, 4], [0.1, 0.3, 0.5, 0.7, 0.9])
    pos = np.vstack([pos4, pos5])
    dirs = np.concatenate([dir4, dir5])
    vals = np.concatenate([val4, val5])
    cp = np.array([[0.0, 0.0], [20.0, 0.0]])

    new = _wp_refine(cp, pos, dirs, vals, dx=1.0, kappa_hard=4)

    assert np.allclose(new[0], [0.0, 0.0], atol=1e-5)  # κ=4 hard → stays
    assert not np.allclose(new[1], [20.0, 0.0], atol=1e-3)  # κ=5 inner → moved off centre
