"""Physics invariants for the D2Q7 model: weights, equilibrium moments, BGK conservation."""

import numpy as np

from direbm import C, Q, W, collide, equilibrium, recover
from direbm.constants import DEFAULT_TAU


def test_directions_unit_length():
    # c_0 is the rest direction; c_1..c_6 must be unit length (method requirement).
    norms = np.linalg.norm(C[1:], axis=1)
    assert np.allclose(norms, 1.0)
    assert np.allclose(C[0], 0.0)


def test_weights_sum_to_one():
    assert np.isclose(W.sum(), 1.0)


def test_equilibrium_recovers_its_own_macros():
    # f_eq(ρ, u) must have macroscopic moments exactly (ρ, u).
    rng = np.random.default_rng(0)
    rho = rng.uniform(0.5, 2.0, size=16)
    u = rng.uniform(-0.1, 0.1, size=(16, 2))
    feq = equilibrium(rho, u)
    rho_r, u_r = recover(feq)
    assert np.allclose(rho_r, rho)
    assert np.allclose(u_r, u, atol=1e-12)


def test_equilibrium_at_rest_is_weighted_density():
    # u = 0 → f_eq_i = ρ W_i.
    rho = np.array([1.0, 3.0])
    u = np.zeros((2, 2))
    feq = equilibrium(rho, u)
    assert np.allclose(feq, rho[:, None] * W[None, :])


def test_collision_conserves_mass_and_momentum():
    rng = np.random.default_rng(1)
    f = rng.uniform(0.01, 1.0, size=(32, Q))
    rho0, u0 = recover(f)
    mom0 = rho0[:, None] * u0
    f1 = collide(f, DEFAULT_TAU)
    rho1, u1 = recover(f1)
    mom1 = rho1[:, None] * u1
    assert np.allclose(rho1, rho0)
    assert np.allclose(mom1, mom0)


def test_collision_relaxes_toward_equilibrium():
    # A non-equilibrium f must move closer to its equilibrium after collision.
    f = np.zeros((1, Q))
    f[0, 0] = 1.0  # all mass on the rest direction: far from equilibrium
    rho, u = recover(f)
    feq = equilibrium(rho, u)
    before = np.abs(f - feq).sum()
    after = np.abs(collide(f, DEFAULT_TAU) - feq).sum()
    assert after < before
