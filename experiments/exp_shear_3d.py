"""exp_shear_3d — 3D quantitative validation: viscous decay of a shear wave (analytic GT).

A single-Fourier-mode shear wave u_x = U cos(kz), u_y = u_z = 0 is an exact eigenmode of the
linearized incompressible Navier–Stokes: it decays as e^{-λt} with **λ = ν k²** (|K|² = k² for one
wavenumber). We initialize it on D3Q13, project the measured velocity onto the shear pattern, and
compare the decay rate to analytic — the 3D analog of exp_taylor_green. ν_phys = cs²(τ−½),
cs² = 1/5 for D3Q13. The excess ν_num = ν_eff − ν_phys is the 3D numerical viscosity (compare the
2D value ≈ 0.074).

Run: uv run python experiments/exp_shear_3d.py
Writeup: docs/results/exp_shear_3d.md
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from direbm import D3Q13, equilibrium, recover
from direbm.reference import Moment, Simulator

L = 3  # domain [-L, L]^3
K = 2.0 * np.pi / 6.0  # shear wavelength 6 (one period across z in [-3,3])
U = 0.1
TAU = 1.1
STEPS = 4
PROBE = 1.5  # measure where |x|,|y| < PROBE (avoid the x,y edges; shear is uniform in x,y)
CS2 = 1.0 / 5.0
OUT = Path(__file__).resolve().parents[1] / "docs" / "results" / "exp_shear_3d.png"


def shear(x):  # unit-amplitude shear pattern u_x = cos(k z)
    out = np.zeros_like(x)
    out[:, 0] = np.cos(K * x[:, 2])
    return out


def amplitude(x, u):
    m = (np.abs(x[:, 0]) < PROBE) & (np.abs(x[:, 1]) < PROBE)
    ref = shear(x[m])
    return float((u[m] * ref).sum() / (ref * ref).sum())


def main():
    nu_phys = CS2 * (TAU - 0.5)
    lam_analytic = nu_phys * K * K  # single-mode shear: lambda = nu * k^2
    print(f"nu_phys = {nu_phys:.4f},  analytic lambda = nu*k^2 = {lam_analytic:.5f} / step")

    rng = range(-L, L + 1)
    pts = np.array([[i, j, k] for i in rng for j in rng for k in rng], dtype=float)
    u0 = U * shear(pts)
    moms = [
        Moment(f=equilibrium(np.float64(1.0), u0[n], lattice=D3Q13), x=pts[n].copy())
        for n in range(len(pts))
    ]
    sim = Simulator(moms, tau=TAU, lattice=D3Q13, soft_mode="off", rho_rest=1.0)

    ts, amps = [0], [amplitude(pts, u0)]
    print(f"{'iter':>5} {'#mom':>7} {'A/A0':>8}")
    for it in range(1, STEPS + 1):
        sim.step()
        x = np.array([m.x for m in sim.moments])
        _, u = recover(np.stack([m.f for m in sim.moments]), D3Q13)
        ts.append(it)
        amps.append(amplitude(x, u))
        print(f"{it:>5} {len(x):>7} {amps[-1] / amps[0]:>8.4f}")
    ts, amps = np.array(ts), np.array(amps)

    lam_meas = -np.polyfit(ts, np.log(amps), 1)[0]
    nu_eff = lam_meas / (K * K)
    nu_num = nu_eff - nu_phys
    print(f"measured lambda = {lam_meas:.5f}  -> nu_eff = {nu_eff:.4f}, nu_num = {nu_num:.4f}")
    print(f"(2D nu_num was ~0.074; ratio nu_eff/nu_phys = {nu_eff / nu_phys:.2f})")

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(ts, amps / amps[0], "s", ms=7, label="DiReBM 3D (measured)")
    ax.plot(ts, np.exp(-lam_analytic * ts), "k-", lw=1.5, label=f"analytic, λ={lam_analytic:.3f}")
    ax.plot(ts, np.exp(-lam_meas * ts), "r--", lw=1, label=f"fit, λ={lam_meas:.3f}")
    ax.set(xlabel="iteration", ylabel="shear amplitude A/A₀", title="3D shear-wave decay vs analytic")
    ax.legend()
    ax.grid(True, alpha=0.3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
