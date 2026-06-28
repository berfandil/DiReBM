"""The reference solver runs in 3D on the D3Q13 icosahedral lattice."""

import numpy as np

from direbm import D3Q13, equilibrium
from direbm.reference import Moment, Simulator


def test_3d_single_pulse_spreads():
    f = equilibrium(np.float64(1.5), np.zeros(3), lattice=D3Q13)
    sim = Simulator([Moment(f=f, x=np.zeros(3))], tau=0.6, lattice=D3Q13, soft_mode="off")
    for _ in range(3):
        sim.step()
    x, rho, _ = sim.macroscopic()
    assert x.shape[1] == 3  # 3D positions
    assert len(x) > 1  # spread into many moments
    assert np.isfinite(rho).all()
    assert (rho > 0).all()
    assert np.linalg.norm(x, axis=1).max() > 1.0  # disturbance radiated outward


def test_3d_rest_field_density_preserved():
    # A small block of rest moments: binned 3D density should stay near rest in the interior.
    f_rest = equilibrium(np.float64(1.0), np.zeros(3), lattice=D3Q13)
    moms = [
        Moment(f=f_rest.copy(), x=np.array([float(i), float(j), float(k)]))
        for i in range(-2, 3)
        for j in range(-2, 3)
        for k in range(-2, 3)
    ]
    sim = Simulator(moms, tau=0.6, lattice=D3Q13, soft_mode="off", rho_rest=1.0)
    sim.step()
    x, rho, _ = sim.macroscopic()
    interior = (np.abs(x) < 1.5).all(axis=1)
    mass = rho[interior].sum()
    vol = 3.0**3  # |x|<1.5 cube side 3
    assert abs(mass / vol - 1.0) < 0.3  # field density ≈ rest (mass / volume)
