"""exp_sphere — a spherical pressure wave in 3D (D3Q13 icosahedral lattice).

The 3D analog of exp_circular_wave: a point pulse in 3D rest fluid radiates a spherical wave.
Demonstrates the runnable 3D reference solver. Shows the ballistic front radius growing linearly
and a z≈0 slice of the moment cloud (a circular cross-section of the sphere).

Run: uv run python experiments/exp_sphere.py
Writeup: docs/results/exp_sphere.md
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from direbm import D3Q13, equilibrium
from direbm.reference import Moment, Simulator

PULSE = 1.6
TAU = 0.6
STEPS = 5
OUT = Path(__file__).resolve().parents[1] / "docs" / "results" / "exp_sphere.png"


def main():
    f = equilibrium(np.float64(PULSE), np.zeros(3), lattice=D3Q13)
    sim = Simulator([Moment(f=f, x=np.zeros(3))], tau=TAU, lattice=D3Q13, soft_mode="off")

    its, counts, max_r = [], [], []
    print(f"{'iter':>5} {'#moments':>9} {'max_r':>7}")
    for it in range(1, STEPS + 1):
        sim.step()
        x, _, _ = sim.macroscopic()
        r = np.linalg.norm(x, axis=1)
        its.append(it)
        counts.append(len(x))
        max_r.append(float(r.max()))
        print(f"{it:>5} {len(x):>9} {r.max():>7.2f}")

    x, rho, _ = sim.macroscopic()
    sl = np.abs(x[:, 2]) < 0.6  # z ≈ 0 slice

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    a0.plot(its, max_r, "-o", label="front radius", ms=6)
    a0.plot(its, its, "k--", lw=1, label="ballistic r = t")
    a0.set(xlabel="iteration", ylabel="max radius", title="(1) spherical front grows linearly")
    a0.legend()
    a0.grid(True, alpha=0.3)

    sc = a1.scatter(x[sl, 0], x[sl, 1], c=rho[sl], s=8, cmap="viridis")
    a1.set_aspect("equal")
    a1.set(xlabel="x", ylabel="y", title=f"(2) z≈0 slice @ iter {STEPS} (n={sl.sum()})")
    fig.colorbar(sc, ax=a1, shrink=0.8, label="ρ")
    fig.suptitle("DiReBM 3D — spherical pressure wave (D3Q13 icosahedral)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
