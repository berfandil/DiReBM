"""exp_convergence — DiReBM parameter study in the control-point density factor α.

Two questions:

  (1) Point-density inflation. The density threshold forbids control points closer than dx/α, so
      the steady sample-point density should scale ~ α². Measure interior moments/area vs α.

  (2) Convergence to LBM + stability. As α grows (finer sampling) DiReBM should approach the LBM
      macroscopic wave. Measure the L2 distance between the DiReBM and LBM radial density profiles
      vs α, and flag any blow-up (the thesis reported instability for α < 3).

Run: uv run python experiments/exp_convergence.py
Writeup: docs/results/exp_convergence.md
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from direbm import bin_fields, equilibrium
from direbm.lbm import HexLBM
from direbm.reference import Moment, Simulator

TAU = 0.6
PULSE = 1.5
PULSE_R = 2.5
ALPHAS = [2.0, 3.0, 4.0, 5.0]
OUT = Path(__file__).resolve().parents[1] / "docs" / "results" / "exp_convergence.png"


def rest_moments(half, pulse=False):
    f_rest = equilibrium(np.float64(1.0), np.zeros(2))
    out = []
    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            x = np.array([float(i), float(j)])
            f = f_rest.copy()
            if pulse and np.hypot(x[0], x[1]) < PULSE_R:
                f = equilibrium(np.float64(PULSE), np.zeros(2))
            out.append(Moment(f=f, x=x))
    return out


def radial_profile(r, v, bins):
    sums, _ = np.histogram(r, bins, weights=v)
    cnts, _ = np.histogram(r, bins)
    return sums / np.maximum(cnts, 1), cnts


def drbm_profile(sim, half, bins):
    x, _, _ = sim.macroscopic()
    f = np.stack([m.f for m in sim.moments])
    rho, _, ext = bin_fields(x, f, -half, half, h=1.0)
    cx = np.linspace(ext[0] + 0.5, ext[1] - 0.5, rho.shape[1])
    cy = np.linspace(ext[2] + 0.5, ext[3] - 0.5, rho.shape[0])
    gx, gy = np.meshgrid(cx, cy)
    r = np.sqrt(gx * gx + gy * gy).ravel()
    v = rho.ravel()
    keep = v > 0
    return radial_profile(r[keep], v[keep], bins)


def study_point_density():
    half, steps, probe = 8, 5, 4
    densities = []
    for a in ALPHAS:
        sim = Simulator(rest_moments(half), tau=TAU, alpha=a)
        for _ in range(steps):
            sim.step()
        x, _, _ = sim.macroscopic()
        interior = (np.abs(x[:, 0]) < probe) & (np.abs(x[:, 1]) < probe)
        densities.append(interior.sum() / (2 * probe) ** 2)
    return np.array(densities)


def study_convergence():
    half, steps = 10, 6
    bins = np.arange(0.0, half + 1.0, 1.0)
    rmax = half - 4
    centers = 0.5 * (bins[:-1] + bins[1:])
    win = centers < rmax

    lbm = HexLBM(ni=4 * half + 1, nj=4 * half + 1, tau=TAU, rho0=1.0)
    lbm.f[lbm.radius() < PULSE_R] = equilibrium(np.float64(PULSE), np.zeros(2))
    for _ in range(steps):
        lbm.step()
    lrho, _ = lbm.macroscopic()
    lprof, _ = radial_profile(lbm.radius().ravel(), lrho.ravel(), bins)

    errs, stable = [], []
    for a in ALPHAS:
        sim = Simulator(rest_moments(half, pulse=True), tau=TAU, alpha=a)
        ok = True
        for _ in range(steps):
            sim.step()
            if not np.isfinite(np.stack([m.f for m in sim.moments])).all():
                ok = False
                break
        if ok:
            dprof, _ = drbm_profile(sim, half, bins)
            errs.append(float(np.sqrt(np.mean((dprof[win] - lprof[win]) ** 2))))
        else:
            errs.append(float("nan"))
        stable.append(ok)
    return np.array(errs), stable, lprof, centers


def main():
    print("(1) point density vs alpha")
    dens = study_point_density()
    for a, d in zip(ALPHAS, dens, strict=True):
        print(f"   alpha={a:>3}  interior_density={d:7.3f} (moments/area)")

    print("(2) convergence to LBM + stability")
    errs, stable, _, _ = study_convergence()
    for a, e, s in zip(ALPHAS, errs, stable, strict=True):
        print(f"   alpha={a:>3}  profile_L2_err={e:7.4f}  stable={s}")

    a = np.array(ALPHAS)
    fig, (p0, p1) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    p0.plot(a, dens, "-o", label="measured")
    ref = dens[2] * (a / a[2]) ** 2  # alpha^2 reference, anchored at alpha=4
    p0.plot(a, ref, "k--", lw=1, label="∝ α² (anchored α=4)")
    p0.set(xlabel="α", ylabel="interior point density (moments/area)", title="(1) point-density inflation")
    p0.legend()

    p1.plot(a, errs, "-s", color="tab:red")
    p1.set(xlabel="α", ylabel="‖ρ_DiReBM − ρ_LBM‖₂ (interior)", title="(2) convergence to LBM vs α")
    for ai, si in zip(a, stable, strict=True):
        if not si:
            p1.annotate("unstable", (ai, 0), color="red")
    fig.suptitle("DiReBM parameter study in α")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
