"""exp_numerical_viscosity — pin DiReBM's numerical dissipation: ν_eff(τ, k).

Taylor–Green decays at λ = 2 ν_eff k². exp_taylor_green found ν_eff ≈ 1.7 ν_phys at one (τ, k).
Here we sweep τ and wavelength, measure λ for each, and extract ν_eff = λ/(2k²) and the numerical
part ν_num = ν_eff − ν_phys (ν_phys = cs²(τ−1/2), cs²=1/4). The goal: is ν_num roughly constant
(a fixed numerical viscosity) and how does it scale with k? → a compensation rule for hitting a
target physical ν.

Run: uv run python experiments/exp_numerical_viscosity.py
Writeup: docs/results/exp_numerical_viscosity.md
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from direbm import equilibrium, recover
from direbm.reference import Moment, Simulator

CS2 = 0.25
U = 0.1
L = 10
STEPS = 9
PROBE = 5
TAUS = [0.7, 0.9, 1.1]
WAVELENGTHS = [8.0, 10.0, 12.0]
OUT = Path(__file__).resolve().parents[1] / "docs" / "results" / "exp_numerical_viscosity.png"


def measure_lambda(tau, wavelength):
    k = 2.0 * np.pi / wavelength

    def tg(x):
        kx, ky = k * x[:, 0], k * x[:, 1]
        return np.stack([-np.cos(kx) * np.sin(ky), np.sin(kx) * np.cos(ky)], axis=1)

    def amp(x, u):
        m = (np.abs(x[:, 0]) < PROBE) & (np.abs(x[:, 1]) < PROBE)
        ref = tg(x[m])
        return float((u[m] * ref).sum() / (ref * ref).sum())

    pts = np.array([[float(i), float(j)] for i in range(-L, L + 1) for j in range(-L, L + 1)])
    u0 = U * tg(pts)
    sim = Simulator([Moment(f=equilibrium(np.float64(1.0), u0[n]), x=pts[n].copy()) for n in range(len(pts))],
                    tau=tau, rho_rest=1.0)
    ts, amps = [0], [amp(pts, u0)]
    for it in range(1, STEPS + 1):
        sim.step()
        x = np.array([m.x for m in sim.moments])
        _, u = recover(np.stack([m.f for m in sim.moments]))
        ts.append(it)
        amps.append(amp(x, u))
    slope = np.polyfit(ts, np.log(amps), 1)[0]
    return -slope, k


def main():
    rows = []
    print(f"{'tau':>5} {'nu_phys':>8} {'wavelen':>8} {'nu_eff':>8} {'nu_num':>8} {'ratio':>6}")
    for tau in TAUS:
        nu_phys = CS2 * (tau - 0.5)
        for w in WAVELENGTHS:
            lam, k = measure_lambda(tau, w)
            nu_eff = lam / (2.0 * k * k)
            nu_num = nu_eff - nu_phys
            rows.append((tau, nu_phys, w, k, nu_eff, nu_num))
            ratio = nu_eff / nu_phys
            print(f"{tau:>5} {nu_phys:>8.4f} {w:>8.1f} {nu_eff:>8.4f} {nu_num:>8.4f} {ratio:>6.2f}")

    nu_num_mean = np.mean([r[5] for r in rows])
    print(f"\nnu_num: mean={nu_num_mean:.4f}, std={np.std([r[5] for r in rows]):.4f}")

    # Validate the compensation rule: to hit a target physical ν, set ν_phys = ν_target − ν_num,
    # i.e. τ = ½ + (ν_target − ν_num)/cs². Should land the measured ν_eff back on ν_target.
    nu_target = 0.15
    tau_c = 0.5 + (nu_target - nu_num_mean) / CS2
    lam_c, k_c = measure_lambda(tau_c, 12.0)
    nu_eff_c = lam_c / (2.0 * k_c * k_c)
    print(f"compensation: target nu={nu_target}, tau={tau_c:.3f} -> measured nu_eff={nu_eff_c:.4f}")

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for w in WAVELENGTHS:
        sub = [r for r in rows if r[2] == w]
        a0.plot([r[1] for r in sub], [r[4] for r in sub], "-o", ms=5, label=f"λ={w:.0f}")
    lims = [0, max(r[1] for r in rows) * 1.1]
    a0.plot(lims, lims, "k:", lw=1, label="ν_eff = ν_phys (ideal)")
    a0.set(xlabel="ν_phys = cs²(τ−½)", ylabel="ν_eff (measured)", title="(1) ν_eff vs ν_phys per wavelength")
    a0.legend()
    a0.grid(True, alpha=0.3)

    ks = np.array([2 * np.pi / w for w in WAVELENGTHS])
    nu_num_by_k = [np.mean([r[5] for r in rows if r[2] == w]) for w in WAVELENGTHS]
    a1.plot(ks**2, nu_num_by_k, "-s", ms=6)
    a1.set(xlabel="k²", ylabel="ν_num (excess, τ-averaged)", title="(2) numerical viscosity vs k²")
    a1.grid(True, alpha=0.3)
    fig.suptitle(f"DiReBM numerical dissipation — ν_num ≈ {nu_num_mean:.3f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
