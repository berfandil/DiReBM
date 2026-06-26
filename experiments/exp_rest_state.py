"""exp_rest_state — is a uniform rest field a steady state of DiReBM?

Rest (ρ=ρ_rest, u=0 everywhere) must be a fixed point of any sane fluid solver. The single-seed
exp_circular_wave showed the interior diluting far below rest; the hypothesis is that this came
from pervasive empty-direction fill (vacuum surroundings), not from a fundamental flaw. Here we
initialize a dense block of rest moments and check whether the *interior* holds ρ≈ρ_rest, u≈0.

Only the interior (|x| < probe radius) is meaningful: outer moments disperse into vacuum and the
resulting rarefaction travels inward at ~1 cell/step, so the interior stays insulated for roughly
(block_half - probe) steps.

Run: uv run python experiments/exp_rest_state.py
Writeup: docs/results/exp_rest_state.md
"""

from __future__ import annotations

import numpy as np

from direbm import equilibrium
from direbm.reference import Moment, Simulator

RHO_REST = 1.0
TAU = 0.6
BLOCK_HALF = 12  # moments on integer grid [-12, 12]^2
PROBE = 5  # measure interior |x|_inf < PROBE
STEPS = 8


def rest_field():
    f_rest = equilibrium(np.float64(RHO_REST), np.zeros(2))
    moments = []
    for gx in range(-BLOCK_HALF, BLOCK_HALF + 1):
        for gy in range(-BLOCK_HALF, BLOCK_HALF + 1):
            moments.append(Moment(f=f_rest.copy(), x=np.array([float(gx), float(gy)])))
    return moments


def main():
    sim = Simulator(rest_field(), tau=TAU, rho_rest=RHO_REST)
    # Macroscopic density is a FIELD: mass per unit area, not the per-moment rho (which just
    # tracks 1 / sample-point density). rho_field = (interior mass) / (interior area).
    area = (2.0 * PROBE) ** 2
    print(f"rest field: {len(sim.moments)} moments, rho_rest={RHO_REST}, probing |x|_inf<{PROBE}")
    print(f"{'iter':>5} {'#mom':>6} {'#int':>5} {'rho_field':>10} {'|u_field|':>10} {'rho/moment':>11}")
    for it in range(1, STEPS + 1):
        sim.step()
        x, rho, u = sim.macroscopic()
        interior = (np.abs(x[:, 0]) < PROBE) & (np.abs(x[:, 1]) < PROBE)
        ri, ui = rho[interior], u[interior]
        mass = ri.sum()
        momentum = (ri[:, None] * ui).sum(axis=0)  # per-moment momentum = rho*u
        rho_field = mass / area
        u_field = float(np.linalg.norm(momentum / mass)) if mass > 0 else float("nan")
        print(
            f"{it:>5} {len(x):>6} {interior.sum():>5} "
            f"{rho_field:>10.4f} {u_field:>10.5f} {ri.mean():>11.4f}"
        )


if __name__ == "__main__":
    main()
