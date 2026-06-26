"""Field reconstruction: binned density/velocity recover the macroscopic state."""

import numpy as np

from direbm import bin_fields, equilibrium


def _grid_moments(half, u=(0.0, 0.0), rho=1.0):
    f_one = equilibrium(np.float64(rho), np.asarray(u, dtype=np.float64))
    xs, fs = [], []
    for gx in range(-half, half + 1):
        for gy in range(-half, half + 1):
            xs.append([float(gx), float(gy)])
            fs.append(f_one)
    return np.array(xs), np.stack(fs)


def test_binned_density_of_unit_rest_field_is_one():
    # One unit-mass moment per unit cell → density field = 1 everywhere it is populated.
    x, f = _grid_moments(3, rho=1.0)
    rho, _, _ = bin_fields(x, f, -3, 4, h=1.0)
    populated = rho > 0
    assert np.allclose(rho[populated], 1.0)


def test_binned_velocity_recovers_uniform_flow():
    u0 = (0.05, -0.02)
    x, f = _grid_moments(3, u=u0, rho=1.0)
    rho, u, _ = bin_fields(x, f, -3, 4, h=1.0)
    pop = rho > 0
    assert np.allclose(u[pop], np.array(u0), atol=1e-12)
