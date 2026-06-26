"""exp_gpu_bench — GPU-vs-GPU step time: DiReBM (v2) vs the D2Q7 LBM, across problem sizes.

Both run on the GPU. For each domain width L: LBM on an L×L grid; DiReBM on a rest field + central
pulse over [-L/2, L/2]². Reports milliseconds per step.

IMPORTANT framing: this is a *dense, uniform* domain, which favours LBM — a regular grid with no
sort / neighbour-query / atomics, and a fixed node count. DiReBM pays for its irregular pipeline
and (here) inflates point density everywhere. DiReBM's advantages — adaptive local resolution,
sparse/empty regions tracked for free, unbounded domains — are NOT exercised by this benchmark.
It measures raw dense-uniform throughput, the case least favourable to DiReBM.

Run: uv run python experiments/exp_gpu_bench.py
Writeup: docs/results/exp_gpu_bench.md
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from direbm import equilibrium
from direbm.warp import GpuHexLBM, GpuSimulator

TAU = 0.6
PULSE = 1.6
STEPS = 5
WIDTHS = [16, 24, 32, 48]


def time_steps(sim, steps):
    sim.step()  # warm up (kernel compile / first-touch)
    t = time.perf_counter()
    for _ in range(steps):
        sim.step()
    return (time.perf_counter() - t) / steps * 1e3  # ms/step


def rest_pulse(half):
    f_rest = equilibrium(np.float64(1.0), np.zeros(2))
    pos, f = [], []
    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            x = np.array([float(i), float(j)])
            fi = equilibrium(np.float64(PULSE), np.zeros(2)) if i == 0 and j == 0 else f_rest.copy()
            pos.append(x)
            f.append(fi)
    return np.array(pos), np.stack(f)


def main():
    lbm_ms, dre_ms, lbm_n, dre_n = [], [], [], []
    print(f"{'L':>4} {'LBM nodes':>10} {'LBM ms':>8} {'DRBM mom':>9} {'DRBM ms':>8}")
    for length in WIDTHS:
        lbm = GpuHexLBM(length, length, tau=TAU)
        lbm.set_pulse(PULSE)
        lms = time_steps(lbm, STEPS)
        lbm_ms.append(lms)
        lbm_n.append(length * length)

        pos, f = rest_pulse(length // 2)
        dre = GpuSimulator(pos, f, tau=TAU, rho_rest=1.0)
        dms = time_steps(dre, STEPS)
        dre_ms.append(dms)
        dre_n.append(dre.pos_m.shape[0])

        print(f"{length:>4} {lbm_n[-1]:>10} {lms:>8.2f} {dre_n[-1]:>9} {dms:>8.2f}")

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(WIDTHS, lbm_ms, "-^", label="LBM (GPU)", ms=6)
    ax.plot(WIDTHS, dre_ms, "-s", label="DiReBM (GPU)", ms=6)
    ax.set(xlabel="domain width L", ylabel="ms / step", title="GPU step time vs domain size (dense uniform)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = Path(__file__).resolve().parents[1] / "docs" / "results" / "exp_gpu_bench.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
