"""Warp dispersion + cell-thinning control-point creation, vs a numpy reference."""

import numpy as np

from direbm.constants import C
from direbm.warp.propagation import create_control_points, disperse


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
