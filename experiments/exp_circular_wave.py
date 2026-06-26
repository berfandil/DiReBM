"""exp_circular_wave — the thesis validation demo (§5.2): a pressure pulse from rest.

Start with a single elevated-density moment at the origin (rest elsewhere) and watch the
disturbance propagate outward. This is the v1 correctness anchor: the front should spread
outward and become roughly circular despite the D2Q7 hexagonal basis.

Run: uv run python experiments/exp_circular_wave.py
Writeup: docs/results/exp_circular_wave.md
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from direbm import equilibrium
from direbm.reference import Moment, Simulator

PULSE_RHO = 1.5
TAU = 0.6
SNAPSHOTS = [4, 8, 12, 16]
OUT = Path(__file__).resolve().parents[1] / "docs" / "results" / "exp_circular_wave_spread.png"


def main():
    sim = Simulator([Moment(f=equilibrium(np.float64(PULSE_RHO), np.zeros(2)), x=np.zeros(2))], tau=TAU)

    panels = {}
    print(f"{'iter':>5} {'#moments':>9} {'max_r':>7} {'total_mass':>11}")
    for it in range(1, max(SNAPSHOTS) + 1):
        sim.step()
        x, rho, _ = sim.macroscopic()
        max_r = float(np.linalg.norm(x, axis=1).max()) if len(x) else 0.0
        print(f"{it:>5} {len(x):>9} {max_r:>7.2f} {rho.sum():>11.3f}")
        if it in SNAPSHOTS:
            panels[it] = (x.copy(), rho.copy())

    fig, axes = plt.subplots(1, len(SNAPSHOTS), figsize=(4 * len(SNAPSHOTS), 4), constrained_layout=True)
    for ax, it in zip(axes, SNAPSHOTS, strict=True):
        x, rho = panels[it]
        sc = ax.scatter(x[:, 0], x[:, 1], c=rho, s=6, cmap="viridis")
        ax.set_aspect("equal")
        ax.set_title(f"iteration {it}  (n={len(x)})")
        fig.colorbar(sc, ax=ax, shrink=0.7, label="ρ")
    fig.suptitle("DiReBM v1 — circular pressure wave from a central pulse")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
