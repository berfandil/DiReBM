"""exp_lbm_vs_drbm — quantitative check of DiReBM against a D2Q7 LBM baseline.

Same physics for both (D2Q7, cs²=1/4, τ=0.6), same setup: a uniform rest field (ρ=1) with the
central node/moment raised to ρ=1.5. A circular acoustic wave should radiate at the lattice sound
speed cs = √(1/4) = 0.5 cells/step. We compare the radial density profile and the front position
vs time.

NOTE on setup: unlike exp_circular_wave (a single seed in vacuum, whose *ballistic* edge moves at
1 cell/step), here both solvers sit on a rest background, so we see the genuine acoustic wave.

Run: uv run python experiments/exp_lbm_vs_drbm.py
Writeup: docs/results/exp_lbm_vs_drbm.md
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

H = 14  # domain half-width
STEPS = 8
RMAX = H - 5  # interior window for front detection (exclude the rest-block boundary rarefaction)
PULSE_R = 2.5  # pulse disk radius (broader than a single node → a detectable acoustic wave)
TAU = 0.6
PULSE = 1.5
CS = 0.5  # D2Q7 lattice sound speed √(1/4)
OUT = Path(__file__).resolve().parents[1] / "docs" / "results" / "exp_lbm_vs_drbm.png"

_BINS = np.arange(0.0, H + 1.0, 1.0)
_CENTERS = 0.5 * (_BINS[:-1] + _BINS[1:])


def radial_profile(r, v):
    sums, _ = np.histogram(r, _BINS, weights=v)
    cnts, _ = np.histogram(r, _BINS)
    prof = sums / np.maximum(cnts, 1)
    return prof, cnts


def peak_radius(prof, cnts):
    # Radius of the compression peak (max density) inside the interior window. Robust to the
    # ~1-2% reconstruction noise that makes a threshold-crossing "front" unreliable for DiReBM.
    valid = (cnts > 0) & (_CENTERS < RMAX)
    if not valid.any():
        return 0.0
    return _CENTERS[np.argmax(np.where(valid, prof, -np.inf))]


def drbm_profile(sim):
    x, _, _ = sim.macroscopic()
    f = np.stack([m.f for m in sim.moments])
    rho, _, extent = bin_fields(x, f, -H, H, h=1.0)
    xmin, xmax, ymin, ymax = extent
    nx = rho.shape[1]
    cx = np.linspace(xmin + 0.5, xmax - 0.5, nx)
    cy = np.linspace(ymin + 0.5, ymax - 0.5, rho.shape[0])
    gx, gy = np.meshgrid(cx, cy)
    r = np.sqrt(gx * gx + gy * gy).ravel()
    v = rho.ravel()
    keep = v > 0  # only populated cells contribute to the radial average
    return radial_profile(r[keep], v[keep])


def main():
    # LBM baseline.
    lbm = HexLBM(ni=4 * H + 1, nj=4 * H + 1, tau=TAU, rho0=1.0)
    lbm.f[lbm.radius() < PULSE_R] = equilibrium(np.float64(PULSE), np.zeros(2))

    # DiReBM on the same rest background + central pulse.
    f_rest = equilibrium(np.float64(1.0), np.zeros(2))
    moms = [
        Moment(f=f_rest.copy(), x=np.array([float(i), float(j)]))
        for i in range(-H, H + 1)
        for j in range(-H, H + 1)
    ]
    for m in moms:  # raise a small central disk
        if np.hypot(m.x[0], m.x[1]) < PULSE_R:
            m.f = equilibrium(np.float64(PULSE), np.zeros(2))
    drbm = Simulator(moms, tau=TAU, rho_rest=1.0)

    lbm_peak, drbm_peak = [], []
    last = {}
    print(f"{'iter':>5} {'lbm_peak':>9} {'drbm_peak':>10} {'drbm_#mom':>10}")
    for it in range(1, STEPS + 1):
        lbm.step()
        drbm.step()
        lr = lbm.radius().ravel()
        lrho, _ = lbm.macroscopic()
        lprof, lcnt = radial_profile(lr, lrho.ravel())
        dprof, dcnt = drbm_profile(drbm)
        lf, df = peak_radius(lprof, lcnt), peak_radius(dprof, dcnt)
        lbm_peak.append(lf)
        drbm_peak.append(df)
        last = {"lprof": lprof, "dprof": dprof}
        print(f"{it:>5} {lf:>9.1f} {df:>10.1f} {len(drbm.moments):>10}")

    ts = np.arange(1, STEPS + 1)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    a0.plot(_CENTERS, last["lprof"], "-o", label="LBM (D2Q7)", ms=4)
    a0.plot(_CENTERS, last["dprof"], "-s", label="DiReBM", ms=4)
    a0.axhline(1.0, color="k", lw=0.6, ls=":")
    a0.set(xlabel="radius r", ylabel="density ρ(r)", title=f"radial density profile @ iter {STEPS}")
    a0.legend()
    a1.plot(ts, lbm_peak, "-o", label="LBM peak", ms=4)
    a1.plot(ts, drbm_peak, "-s", label="DiReBM peak", ms=4)
    a1.plot(ts, PULSE_R + CS * ts, "k--", lw=1, label=f"R₀+cs·t (cs={CS})")
    a1.set(xlabel="iteration", ylabel="compression-peak radius", title="compression-peak radius vs time")
    a1.legend()
    fig.suptitle("DiReBM vs D2Q7 LBM — acoustic pressure wave (rest background + central pulse)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
