"""exp_obstacle — a pressure wave reflecting off a circular obstacle.

Showcases DiReBM's claimed advantage over LBM: arbitrary surfaces are easy. A rest fluid surrounds
a solid disc; a pulse offset to one side radiates, strikes the obstacle, and reflects — the
component bounce (thesis §4.5) keeps fluid out of the solid and sends the wavefront back.

Run: uv run python experiments/exp_obstacle.py
Writeup: docs/results/exp_obstacle.md
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle as CirclePatch

from direbm import bin_fields, equilibrium
from direbm.reference import Circle, Moment, Simulator

HALF = 12
OBST = Circle((0.0, 0.0), 3.0)
PULSE_AT = (-7.0, 0.0)
PULSE_RHO = 1.7
TAU = 0.6
SNAPSHOTS = [3, 6, 9, 12]
OUT = Path(__file__).resolve().parents[1] / "docs" / "results" / "exp_obstacle.png"


def main():
    f_rest = equilibrium(np.float64(1.0), np.zeros(2))
    moms = []
    for i in range(-HALF, HALF + 1):
        for j in range(-HALF, HALF + 1):
            x = np.array([float(i), float(j)])
            if OBST.inside(x):
                continue
            at_pulse = np.hypot(x[0] - PULSE_AT[0], x[1] - PULSE_AT[1]) < 1.5
            fi = equilibrium(np.float64(PULSE_RHO), np.zeros(2)) if at_pulse else f_rest.copy()
            moms.append(Moment(f=fi, x=x))

    sim = Simulator(moms, tau=TAU, rho_rest=1.0, obstacle=OBST)
    panels = {}
    print(f"{'iter':>5} {'#moments':>9}")
    for it in range(1, max(SNAPSHOTS) + 1):
        sim.step()
        print(f"{it:>5} {len(sim.moments):>9}")
        if it in SNAPSHOTS:
            x = np.array([m.x for m in sim.moments])
            f = np.stack([m.f for m in sim.moments])
            panels[it] = bin_fields(x, f, -HALF, HALF, h=2.0)  # coarser bins → less reconstruction noise

    n = len(SNAPSHOTS)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.3), constrained_layout=True)
    for ax, it in zip(axes, SNAPSHOTS, strict=True):
        rho, _, ext = panels[it]
        # show the pressure perturbation ρ − 1 (diverging): compression red, rarefaction blue
        pert = np.ma.masked_where(rho <= 0.0, rho - 1.0)
        im = ax.imshow(pert, origin="lower", extent=ext, cmap="RdBu_r", vmin=-0.15, vmax=0.15)
        ax.add_patch(CirclePatch(OBST.center, OBST.radius, fill=True, color="0.4", zorder=5))
        ax.set_aspect("equal")
        ax.set_title(f"iteration {it}")
        fig.colorbar(im, ax=ax, shrink=0.7, label="ρ − 1")
    fig.suptitle("DiReBM — pressure wave reflecting off a circular obstacle (perturbation ρ − 1)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
