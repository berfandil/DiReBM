"""Warp GPU physics kernels must match the v1 float64 oracle (within float32 tolerance)."""

import numpy as np

from direbm import collide as ref_collide
from direbm import equilibrium as ref_equilibrium
from direbm import recover
from direbm.warp import collide as wp_collide
from direbm.warp import default_device
from direbm.warp import equilibrium as wp_equilibrium


def test_warp_equilibrium_matches_reference():
    rng = np.random.default_rng(0)
    rho = rng.uniform(0.5, 2.0, size=128)
    u = rng.uniform(-0.1, 0.1, size=(128, 2))
    ref = ref_equilibrium(rho, u)
    got = wp_equilibrium(rho, u)
    assert np.allclose(got, ref, rtol=1e-4, atol=1e-5)


def test_warp_collide_matches_reference():
    rng = np.random.default_rng(1)
    f = rng.uniform(0.01, 1.0, size=(256, 7))
    ref = ref_collide(f, 0.6)
    got = wp_collide(f, 0.6)
    assert np.allclose(got, ref, rtol=1e-4, atol=1e-5)


def test_warp_collide_conserves_mass_and_momentum():
    rng = np.random.default_rng(2)
    f = rng.uniform(0.01, 1.0, size=(128, 7))
    rho0, u0 = recover(f)
    got = wp_collide(f, 0.6)
    rho1, u1 = recover(got)
    assert np.allclose(rho1, rho0, rtol=1e-5, atol=1e-5)
    assert np.allclose(rho1[:, None] * u1, rho0[:, None] * u0, atol=1e-5)


def test_runs_on_reported_device():
    # Sanity: the chosen device is one Warp actually exposes.
    assert default_device() in ("cpu", "cuda:0")
