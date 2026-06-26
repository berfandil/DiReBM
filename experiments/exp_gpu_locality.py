"""exp_gpu_locality — where DiReBM beats LBM: a localized disturbance in a large domain.

Physics for both: a central pressure pulse on a rest (ρ=1) background, run for a fixed T steps so
the disturbance only reaches radius ~T. The "domain of interest" has width L (≫ T).

- LBM must grid the whole L×L and update every node every step → cost ∝ L², regardless of how
  little of the domain is active.
- DiReBM tracks only the active material; the quiescent far field is implicit rest (the resampling
  gap-fill supplies ρ_rest), so it costs nothing. DiReBM's cost is therefore ~constant in L.

Timing note: Warp launches are async. GpuHexLBM.step() has no host sync, so we wp.synchronize()
around the timed loop to measure real GPU execution (not just enqueue).

Run: uv run python experiments/exp_gpu_locality.py
Writeup: docs/results/exp_gpu_locality.md
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warp as wp

from direbm import equilibrium
from direbm.warp import GpuHexLBM, GpuSimulator

TAU = 0.6
PULSE = 1.6
STEPS = 8
LS = [64, 128, 256, 512, 1024, 2048, 4096]


def time_steps(sim, steps):
    sim.step()
    wp.synchronize()  # warm up (kernel compile / first touch)
    t = time.perf_counter()
    for _ in range(steps):
        sim.step()
    wp.synchronize()
    return (time.perf_counter() - t) / steps * 1e3  # ms/step


def main():
    # DiReBM: a single central pulse on an implicit rest background. Domain-size independent.
    pos = np.zeros((1, 2))
    f = equilibrium(np.float64(PULSE), np.zeros(2))[None, :]
    dre = GpuSimulator(pos, f, tau=TAU, rho_rest=1.0)
    dre_ms = time_steps(dre, STEPS)
    dre_n = dre.pos_m.shape[0]

    lbm_ms, lbm_n = [], []
    print(f"{'L':>6} {'LBM nodes':>10} {'LBM ms':>9}   (DiReBM: {dre_ms:.2f} ms, {dre_n} moments)")
    for length in LS:
        lbm = GpuHexLBM(length, length, tau=TAU)
        lbm.set_pulse(PULSE)
        lms = time_steps(lbm, STEPS)
        lbm_ms.append(lms)
        lbm_n.append(length * length)
        print(f"{length:>6} {length * length:>10} {lms:>9.3f}")

    ls = np.array(LS, dtype=float)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    a0.loglog(ls, lbm_ms, "-^", label="LBM (GPU)", ms=6)
    a0.axhline(dre_ms, color="tab:orange", ls="-", lw=2, label=f"DiReBM (GPU) ~{dre_ms:.2f} ms")
    a0.set(xlabel="domain width L", ylabel="ms / step", title="(1) step time: LBM ∝ L², DiReBM flat")
    a0.legend()
    a0.grid(True, which="both", alpha=0.3)

    a1.loglog(ls, lbm_n, "-^", label="LBM nodes (L²)", ms=6)
    a1.axhline(dre_n, color="tab:orange", ls="-", lw=2, label=f"DiReBM moments (~{dre_n})")
    a1.set(xlabel="domain width L", ylabel="elements updated", title="(2) work: LBM grids the whole domain")
    a1.legend()
    a1.grid(True, which="both", alpha=0.3)

    fig.suptitle("Localized disturbance in a large domain — DiReBM cost is domain-size-independent")
    out = Path(__file__).resolve().parents[1] / "docs" / "results" / "exp_gpu_locality.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
