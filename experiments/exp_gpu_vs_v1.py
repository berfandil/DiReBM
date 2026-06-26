"""exp_gpu_vs_v1 — the GPU (v2 Warp) solver vs the v1 reference oracle.

Same init (rest field + central pulse), same parameters. Confirms the GPU port reproduces the v1
macroscopic density field, and reports wall-clock time per step for each.

Run: uv run python experiments/exp_gpu_vs_v1.py
Writeup: docs/results/exp_gpu_vs_v1.md
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from direbm import bin_fields, equilibrium
from direbm.reference import Moment, Simulator
from direbm.warp import GpuSimulator

HALF = 8
STEPS = 5
TAU = 0.6
PULSE = 1.6
WIN = HALF + 1
OUT = Path(__file__).resolve().parents[1] / "docs" / "results" / "exp_gpu_vs_v1.png"

_BINS = np.arange(0.0, WIN + 1.0, 1.0)
_CENTERS = 0.5 * (_BINS[:-1] + _BINS[1:])


def rest_pulse():
    f_rest = equilibrium(np.float64(1.0), np.zeros(2))
    pos, f = [], []
    for i in range(-HALF, HALF + 1):
        for j in range(-HALF, HALF + 1):
            x = np.array([float(i), float(j)])
            fi = equilibrium(np.float64(PULSE), np.zeros(2)) if i == 0 and j == 0 else f_rest.copy()
            pos.append(x)
            f.append(fi)
    return np.array(pos), np.stack(f)


def radial(rho_field, extent):
    nx = rho_field.shape[1]
    cx = np.linspace(extent[0] + 0.5, extent[1] - 0.5, nx)
    cy = np.linspace(extent[2] + 0.5, extent[3] - 0.5, rho_field.shape[0])
    gx, gy = np.meshgrid(cx, cy)
    r = np.sqrt(gx * gx + gy * gy).ravel()
    v = rho_field.ravel()
    keep = v > 0
    sums, _ = np.histogram(r[keep], _BINS, weights=v[keep])
    cnts, _ = np.histogram(r[keep], _BINS)
    return sums / np.maximum(cnts, 1)


def main():
    pos, f = rest_pulse()

    v1 = Simulator([Moment(f=f[k].copy(), x=pos[k].copy()) for k in range(len(pos))], tau=TAU, rho_rest=1.0)
    t = time.perf_counter()
    for _ in range(STEPS):
        v1.step()
    v1_t = (time.perf_counter() - t) / STEPS
    xv = np.array([m.x for m in v1.moments])
    fv = np.stack([m.f for m in v1.moments])

    gpu = GpuSimulator(pos, f, tau=TAU, rho_rest=1.0)
    gpu.step()  # warm up (kernel compile) before timing
    gpu2 = GpuSimulator(pos, f, tau=TAU, rho_rest=1.0)
    t = time.perf_counter()
    for _ in range(STEPS):
        gpu2.step()
    gpu_t = (time.perf_counter() - t) / STEPS
    xg, fg = gpu2.moments()

    rv, _, ext = bin_fields(xv, fv, -WIN, WIN, h=1.0)
    rg, _, _ = bin_fields(xg, fg, -WIN, WIN, h=1.0)

    print(f"v1 (CPU) : {v1_t * 1e3:8.1f} ms/step,  {len(xv)} moments")
    print(f"gpu (v2) : {gpu_t * 1e3:8.1f} ms/step,  {len(xg)} moments")
    print(f"speedup  : {v1_t / gpu_t:6.1f}x")

    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    vmax = max(rv.max(), rg.max())
    for ax, r, name in ((a0, rv, "v1 reference (CPU)"), (a1, rg, "v2 Warp (GPU)")):
        im = ax.imshow(r, origin="lower", extent=ext, cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(f"field ρ — {name}")
        fig.colorbar(im, ax=ax, shrink=0.8)
    a2.plot(_CENTERS, radial(rv, ext), "-o", label="v1 (CPU)", ms=4)
    a2.plot(_CENTERS, radial(rg, ext), "-s", label="GPU (v2)", ms=4)
    a2.axhline(1.0, color="k", lw=0.6, ls=":")
    a2.set(xlabel="radius r", ylabel="ρ(r)", title="radial density profile")
    a2.legend()
    fig.suptitle(f"DiReBM GPU (v2) vs reference (v1) — {v1_t / gpu_t:.1f}x/step at {len(xg)} moments")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
