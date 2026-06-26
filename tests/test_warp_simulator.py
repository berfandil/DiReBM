"""Full GPU DiReBM step: rest preservation, pulse spread, and macroscopic agreement with v1."""

import numpy as np

from direbm import bin_fields, equilibrium
from direbm.reference import Moment, Simulator
from direbm.warp import GpuSimulator


def _rest_field(half, pulse_rho=None):
    f_rest = equilibrium(np.float64(1.0), np.zeros(2))
    pos, f = [], []
    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            x = np.array([float(i), float(j)])
            fi = f_rest.copy()
            if pulse_rho is not None and i == 0 and j == 0:
                fi = equilibrium(np.float64(pulse_rho), np.zeros(2))
            pos.append(x)
            f.append(fi)
    return np.array(pos), np.stack(f)


def test_gpu_rest_field_preserved():
    pos, f = _rest_field(6)
    sim = GpuSimulator(pos, f, tau=0.6, rho_rest=1.0)
    for _ in range(4):
        sim.step()
    xp, fp = sim.moments()
    assert np.isfinite(fp).all()
    assert (fp.sum(axis=1) > 0).all()
    rho, _, _ = bin_fields(xp, fp, -3, 3, h=1.0)
    pop = rho > 0
    assert abs(rho[pop].mean() - 1.0) < 0.2  # macroscopic density field stays ≈ rest


def test_gpu_pulse_spreads():
    pos = np.zeros((1, 2))
    f = equilibrium(np.float64(1.5), np.zeros(2))[None, :]
    sim = GpuSimulator(pos, f, tau=0.6)
    for _ in range(5):
        sim.step()
    xp, fp = sim.moments()
    assert len(xp) > 1
    assert np.isfinite(fp).all()
    assert np.linalg.norm(xp, axis=1).max() > 3.0  # disturbance spread outward


def test_gpu_matches_v1_macroscopically():
    pos, f = _rest_field(6, pulse_rho=1.5)
    v1 = Simulator([Moment(f=f[k].copy(), x=pos[k].copy()) for k in range(len(pos))], tau=0.6, rho_rest=1.0)
    gpu = GpuSimulator(pos, f, tau=0.6, rho_rest=1.0)
    for _ in range(3):
        v1.step()
        gpu.step()

    xv = np.array([m.x for m in v1.moments])
    fv = np.stack([m.f for m in v1.moments])
    xg, fg = gpu.moments()
    rv, _, _ = bin_fields(xv, fv, -3, 3, h=1.0)
    rg, _, _ = bin_fields(xg, fg, -3, 3, h=1.0)
    # Both reconstruct a near-rest interior; their mean interior density should agree closely.
    assert abs(rv[rv > 0].mean() - rg[rg > 0].mean()) < 0.15
