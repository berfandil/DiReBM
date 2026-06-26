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

from direbm import bin_fields, equilibrium
from direbm.reference import Moment, Simulator

PULSE_RHO = 1.5
TAU = 0.6
SNAPSHOTS = [4, 8, 12, 16]
WIN = max(SNAPSHOTS) + 2  # binning window half-width
OUT = Path(__file__).resolve().parents[1] / "docs" / "results" / "exp_circular_wave_spread.png"


def main():
    sim = Simulator([Moment(f=equilibrium(np.float64(PULSE_RHO), np.zeros(2)), x=np.zeros(2))], tau=TAU)

    panels = {}
    print(f"{'iter':>5} {'#moments':>9} {'max_r':>7} {'total_mass':>11}")
    for it in range(1, max(SNAPSHOTS) + 1):
        sim.step()
        x, rho, _ = sim.macroscopic()
        f = np.stack([m.f for m in sim.moments])
        max_r = float(np.linalg.norm(x, axis=1).max()) if len(x) else 0.0
        print(f"{it:>5} {len(x):>9} {max_r:>7.2f} {rho.sum():>11.3f}")
        if it in SNAPSHOTS:
            panels[it] = (x.copy(), rho.copy(), f.copy())

    n = len(SNAPSHOTS)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8), constrained_layout=True)
    for col, it in enumerate(SNAPSHOTS):
        x, rho, f = panels[it]
        # Top: per-moment ρ scatter (note: this is mass-per-sample = 1/point-density, NOT field ρ).
        ax = axes[0, col]
        sc = ax.scatter(x[:, 0], x[:, 1], c=rho, s=6, cmap="viridis")
        ax.set_aspect("equal")
        ax.set_xlim(-WIN, WIN)
        ax.set_ylim(-WIN, WIN)
        ax.set_title(f"iter {it}: per-moment ρ  (n={len(x)})")
        fig.colorbar(sc, ax=ax, shrink=0.7)
        # Bottom: reconstructed macroscopic density field = mass / area.
        ax = axes[1, col]
        rho_field, _, extent = bin_fields(x, f, -WIN, WIN, h=1.0)
        im = ax.imshow(rho_field, origin="lower", extent=extent, cmap="magma")
        ax.set_title(f"iter {it}: field ρ = mass/area")
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle("DiReBM v1 — circular pressure wave: per-moment ρ (top) vs macroscopic field ρ (bottom)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
