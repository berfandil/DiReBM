"""GPU HexLBM must match the CPU HexLBM (within float32 tolerance) and conserve the basics."""

import numpy as np

from direbm.lbm import HexLBM
from direbm.warp.lbm import GpuHexLBM


def test_gpu_lbm_matches_cpu():
    ni = nj = 21
    cpu = HexLBM(ni, nj, tau=0.6)
    cpu.set_pulse(1.5)
    gpu = GpuHexLBM(ni, nj, tau=0.6)
    gpu.set_pulse(1.5)
    for _ in range(10):
        cpu.step()
        gpu.step()
    rc, _ = cpu.macroscopic()
    rg, _ = gpu.macroscopic()
    assert np.allclose(rc, rg, atol=1e-3, rtol=1e-3)


def test_gpu_lbm_rest_is_steady():
    gpu = GpuHexLBM(15, 15, tau=0.6, rho0=1.0)
    r0, _ = gpu.macroscopic()
    for _ in range(20):
        gpu.step()
    r1, u1 = gpu.macroscopic()
    assert np.allclose(r1, r0, atol=1e-4)
    assert np.allclose(u1, 0.0, atol=1e-5)


def test_gpu_lbm_conserves_mass():
    gpu = GpuHexLBM(25, 25, tau=0.7)
    gpu.set_pulse(2.0)
    m0 = float(gpu.f.numpy().sum())
    for _ in range(15):
        gpu.step()
    assert np.isclose(float(gpu.f.numpy().sum()), m0, rtol=1e-3)
