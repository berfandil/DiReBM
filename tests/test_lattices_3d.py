"""D3Q13 icosahedral lattice: unit dirs, isotropic moments, and conservation in 3D."""

import numpy as np

from direbm import D3Q13, collide, equilibrium, recover


def test_shape_and_unit_directions():
    assert D3Q13.D == 3 and D3Q13.Q == 13
    assert np.allclose(D3Q13.C[0], 0.0)  # rest
    assert np.allclose(np.linalg.norm(D3Q13.C[1:], axis=1), 1.0)  # twelve unit dirs
    assert np.isclose(D3Q13.W.sum(), 1.0)


def test_second_moment_isotropic():
    # Σ_i W_i c_iα c_iβ = cs² δ
    m2 = np.einsum("i,ia,ib->ab", D3Q13.W, D3Q13.C, D3Q13.C)
    assert np.allclose(m2, D3Q13.cs2 * np.eye(3), atol=1e-12)


def test_fourth_moment_isotropic_and_matches_cs4():
    cs4 = D3Q13.cs2**2
    cxxxx = float(np.einsum("i,i->", D3Q13.W, D3Q13.C[:, 0] ** 4))
    cxxyy = float(np.einsum("i,i->", D3Q13.W, D3Q13.C[:, 0] ** 2 * D3Q13.C[:, 1] ** 2))
    assert np.isclose(cxxyy, cs4, atol=1e-12)  # isotropic constant equals cs⁴ (needed for NS)
    assert np.isclose(cxxxx, 3.0 * cs4, atol=1e-12)  # 3:1 ratio = isotropy


def test_equilibrium_recovers_macros_3d():
    rng = np.random.default_rng(0)
    rho = rng.uniform(0.5, 2.0, size=16)
    u = rng.uniform(-0.1, 0.1, size=(16, 3))
    feq = equilibrium(rho, u, lattice=D3Q13)
    rho_r, u_r = recover(feq, lattice=D3Q13)
    assert np.allclose(rho_r, rho)
    assert np.allclose(u_r, u, atol=1e-12)


def test_collision_conserves_mass_and_momentum_3d():
    rng = np.random.default_rng(1)
    f = rng.uniform(0.01, 1.0, size=(32, 13))
    rho0, u0 = recover(f, D3Q13)
    f1 = collide(f, 0.6, lattice=D3Q13)
    rho1, u1 = recover(f1, D3Q13)
    assert np.allclose(rho1, rho0)
    assert np.allclose(rho1[:, None] * u1, rho0[:, None] * u0)
