"""exp_taylor_green — validate DiReBM against an ANALYTIC ground truth (not LBM).

The 2D Taylor–Green vortex is an exact decaying solution of incompressible Navier–Stokes:

    u_x = -U cos(kx) sin(ky) e^{-λt},   u_y = U sin(kx) cos(ky) e^{-λt},   λ = 2 ν k².

We initialize the field, run, and extract the TG-mode amplitude A(t) by projecting the measured
per-moment velocity onto the TG pattern (incoherent reconstruction noise averages out). The decay
rate of A(t) is compared to the analytic λ — a real ground truth, addressing the "LBM is only a
proxy" caveat (exp_lbm_vs_drbm). ν = cs²(τ − 1/2) with cs² = 1/4.

Run: uv run python experiments/exp_taylor_green.py
Writeup: docs/results/exp_taylor_green.md
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from direbm import equilibrium, recover
from direbm.reference import Moment, Simulator

L = int(os.environ.get("TG_L", "12"))  # domain [-L, L]^2 (override to gauge boundary leakage)
K = 2.0 * np.pi / 12.0  # one vortex cell per 12 units
U = 0.1  # amplitude (low Mach)
TAU = 0.9
STEPS = int(os.environ.get("TG_STEPS", "12"))
PROBE = 6  # measure the TG mode over the central |x|,|y| < PROBE (avoid edge leakage)
CS2 = 0.25
OUT = Path(__file__).resolve().parents[1] / "docs" / "results" / "exp_taylor_green.png"


def tg(x):  # unit-amplitude TG velocity pattern at positions x (N,2)
    kx, ky = K * x[:, 0], K * x[:, 1]
    return np.stack([-np.cos(kx) * np.sin(ky), np.sin(kx) * np.cos(ky)], axis=1)


def amplitude(x, u):
    m = (np.abs(x[:, 0]) < PROBE) & (np.abs(x[:, 1]) < PROBE)
    ref = tg(x[m])
    return float((u[m] * ref).sum() / (ref * ref).sum())


def main():
    nu = CS2 * (TAU - 0.5)
    lam = 2.0 * nu * K * K
    print(f"nu = {nu:.4f},  analytic lambda = 2*nu*k^2 = {lam:.5f} / step")

    pts = np.array([[float(i), float(j)] for i in range(-L, L + 1) for j in range(-L, L + 1)])
    u0 = U * tg(pts)
    moms = [Moment(f=equilibrium(np.float64(1.0), u0[n]), x=pts[n].copy()) for n in range(len(pts))]
    sim = Simulator(moms, tau=TAU, rho_rest=1.0)

    ts, amps = [0], [amplitude(pts, u0)]
    for it in range(1, STEPS + 1):
        sim.step()
        x = np.array([m.x for m in sim.moments])
        _, u = recover(np.stack([m.f for m in sim.moments]))
        ts.append(it)
        amps.append(amplitude(x, u))
    ts, amps = np.array(ts), np.array(amps)

    # fit decay rate from ln A over the clean window (before edge effects reach the probe region)
    clean = ts <= (L - PROBE) / 0.5
    slope = np.polyfit(ts[clean], np.log(amps[clean]), 1)[0]
    lam_meas = -slope
    print(f"{'iter':>5} {'A(t)':>9} {'A/A0':>8}")
    for t, a in zip(ts, amps, strict=True):
        print(f"{t:>5} {a:>9.5f} {a / amps[0]:>8.4f}")
    print(f"measured lambda = {lam_meas:.5f} / step   (analytic {lam:.5f};  ratio {lam_meas / lam:.2f})")

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(ts, amps / amps[0], "s", label="DiReBM (measured)", ms=6)
    ax.plot(ts, np.exp(-lam * ts), "k-", lw=1.5, label=f"analytic e^(−λt), λ={lam:.4f}")
    ax.plot(ts, np.exp(-lam_meas * ts), "r--", lw=1, label=f"fit, λ={lam_meas:.4f}")
    ax.set(xlabel="iteration", ylabel="TG mode amplitude A/A₀", title="Taylor–Green decay vs analytic")
    ax.legend()
    ax.grid(True, alpha=0.3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
