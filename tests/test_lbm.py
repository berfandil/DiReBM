"""D2Q7 hexagonal LBM baseline: rest steady state, mass conservation, symmetric spread."""

import numpy as np

from direbm.lbm import HexLBM


def test_rest_is_steady_state():
    lbm = HexLBM(ni=21, nj=21, tau=0.6, rho0=1.0)
    rho0, _ = lbm.macroscopic()
    for _ in range(20):
        lbm.step()
    rho1, u1 = lbm.macroscopic()
    assert np.allclose(rho1, rho0)
    assert np.allclose(u1, 0.0, atol=1e-12)


def test_streaming_collision_conserve_total_mass():
    lbm = HexLBM(ni=31, nj=31, tau=0.7, rho0=1.0)
    lbm.set_pulse(2.0)
    m0 = lbm.f.sum()
    for _ in range(15):
        lbm.step()
    assert np.isclose(lbm.f.sum(), m0)


def test_pulse_spreads_and_stays_bounded():
    lbm = HexLBM(ni=61, nj=61, tau=0.6, rho0=1.0)
    lbm.set_pulse(1.5)
    for _ in range(10):
        lbm.step()
    rho, _ = lbm.macroscopic()
    # Disturbance spread off the centre but stays finite and positive.
    assert np.isfinite(rho).all()
    assert (rho > 0).all()
    # The centre density relaxed back down from the initial spike.
    ci, cj = lbm.center
    assert rho[cj, ci] < 1.5
