# Architecture — the code map

Read this first to locate code; don't re-grep the tree. Keep it current after structural
changes. Status tags: **[planned]** = not written yet, **[wip]**, **[done]**.

The method this implements is specified in `research/idea.md`. Module boundaries below mirror
the method's concepts so the paper and the code stay one-to-one.

## Stack

- **Python 3.12** + **NVIDIA Warp 1.14** (Python-authored CUDA kernels; bundled CUDA Toolkit
  12.9). Same kernels run on CPU (`cpu`) for debugging and on GPU (`cuda:0`, sm_120) for scale.
- **`wp.HashGrid`** is the spatial-hash primitive — it backs the method's "grid data structure"
  (radius-neighbor queries). This is the GPU heir of DiRe-CFD's `MultiGrid`.
- `numpy` for host-side setup/analysis; `matplotlib` for macroscopic-field plots (dev).

## Build phases

- **v1 = reference solver** (plain Python/numpy). Faithful to the thesis/C++, optimized for
  debuggability, not speed. It is the permanent **correctness oracle**. Decision +
  rationale: `research/progress.md` (2026-06-26).
- **v2 = Warp port** (CPU→GPU, same kernels). Validated against the v1 oracle. GPU is a port, not
  a rewrite.

## Package layout

```
direbm/
  __init__.py        [done]  re-exports constants + physics
  constants.py       [done]  D, Q, dx, dt, τ, α, κ_hard, κ_soft, cs²; c[] dirs, W[] weights (D2Q7)
  physics.py         [done]  f_eq(ρ,u); recover ρ,u from f; BGK collision (shared, numpy)
  reference/                 v1 reference solver (the oracle)
    types.py         [done]  Moment (µ), Component (ν), ControlPoint (p) — dataclasses/ndarrays
    grid.py          [done]  dict-of-cells spatial hash: radius-query / density-threshold
                             insert / radius-remove  (heir of MultiGrid)
    simulator.py     [done]  Simulator: step state machine + the 4 propagation sub-steps
                             (dispersion → create_control_points → refine → resampling) + collision
    boundary.py      [planned] surface bounce + direction-split (mass-conserving)
  lbm.py             [done]  D2Q7 hexagonal LBM baseline (HexLBM) for macroscopic validation
  fields.py          [done]  bin_fields(): reconstruct macroscopic ρ,u from moments (mass/area)
  warp/                      v2 GPU port — validated step-by-step against the v1 oracle
    physics.py       [done]  Warp kernels: recover + equilibrium + BGK collision (float32, GPU)
    propagation.py   [done]  dispersion; control-point cell-thinning (radix sort + runlength_encode
                             + centroid); refine (HashGrid → κ → hard-keep/inner-move, soft-spawn
                             deferred); resampling (atomic scatter + gap-fill/emit on HashGrid)
    simulator.py     [done]  GpuSimulator: full step on device (realloc per step for dynamic counts)
    lbm.py           [planned] GPU port of the HexLBM baseline (next — for a fair GPU-vs-GPU speed)
  viz.py             [planned] rendering — matplotlib (v1; currently inline in the experiment)

experiments/
  exp_circular_wave.py  [done]  thesis pressure-pulse demo → docs/results/exp_circular_wave.md
```

Open issue from the first run: the resampling rest-fill (eq 4.7) leaves quiescent regions
under-dense (macroscopic density uncalibrated). Tracked in `research/progress.md` + the result
writeup; it is the focus of the validation milestone.

## Key design notes

- **Dynamic counts**: moments/components/control points change in number each step.
  - v1 (numpy): just use Python lists / regrown arrays — simple, correct.
  - v2 (Warp): arrays are fixed-size → preallocate + **compact** (stream-compaction / atomic
    counters). The density threshold (`dx/α`) bounds control-point count → use it to size
    buffers. This is the crux of the GPU parallelization the thesis left open.
- **Spatial hash rebuilt each iteration** (points move every step). v2: `grid.build(points,
  radius=dx)`.
- **Determinism**: resampling/exp-weighted centroids are reductions. v2 atomics make float order
  non-deterministic → keep the v1 oracle as ground truth and compare macroscopic fields within a
  tolerance, not bit-exact.

## Other directories

```
research/      idea.md (canonical method) + progress.md (cross-experiment log)
docs/          this map + decisions/ (ADRs) + results/exp_<slug>.md (per-experiment writeups)
experiments/   exp_<slug>.py (repeatable) — first up: exp_circular_wave (validation anchor)
scripts/       tags.ps1 (ctags), helpers
.agent-workspace/  ephemeral scratch (gitignored)
tags           ctags index (gitignored)
```

## First milestone

`experiments/exp_circular_wave.py` — 2D circular pressure wave from rest (the thesis demo),
checked at the macroscopic level against an LBM baseline. Correctness anchor before GPU
optimization work.
