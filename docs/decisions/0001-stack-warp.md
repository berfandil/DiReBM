# ADR 0001 — Compute stack: NVIDIA Warp

- Status: accepted
- Date: 2026-06-26

## Context

DiReBM is a latticeless Boltzmann method (see `research/idea.md`). Its compute fingerprint:

- point clouds (moments / components / control points) whose counts change every step;
- a spatial hash with insert / density-threshold-insert / radius-remove / radius-query;
- per-point small dense math (7-vectors, BGK) plus neighbor reductions (weighted / exp-weighted
  centroids, f-scatter);
- dynamic create/delete of points.

This is **irregular particle + grid physics**, not the dense grid-regular workload that classic
tensor LBM libraries target. The headline goal of the revitalization is **GPU parallelization**,
which the original thesis flagged as its biggest open problem. Hardware: RTX 5080 Laptop,
Blackwell, **sm_120**, CUDA 12.8 driver.

## Options considered

1. **NVIDIA Warp** — Python→CUDA kernels, built-in `wp.HashGrid` matching the method's data
   structure, NVIDIA-maintained (good Blackwell support), CPU+GPU from one codebase, optional
   autodiff.
2. **Taichi** — native sparse spatial structures, strong dynamic-sparsity story; but slowed
   maintenance and uncertain sm_120 support.
3. **PyTorch (cu128)** — huge ecosystem + autodiff, but the irregular neighbor/hash core fights
   the tensor model and needs custom CUDA anyway.
4. **C++20 + CUDA** — max performance and reuse of the original C++ lineage, but slowest research
   loop and heavy plumbing; premature.

## Decision

**NVIDIA Warp.** It maps 1:1 onto the method's data structures (HashGrid = the paper's "grid
data structure"), keeps Python iteration speed, gives GPU from day one, and runs on CPU for
debugging.

## Validation

Smoke-tested on this machine before committing: Warp 1.14.0, bundled CUDA Toolkit 12.9, GPU
detected as `cuda:0` sm_120 (arch 120, 16 GiB, mempool enabled). A GPU saxpy kernel compiled
and produced correct results; `wp.HashGrid.build()` succeeded.

## Consequences

- Dynamic point counts need preallocation + stream compaction (Warp arrays are fixed-size). The
  `dx/α` density threshold bounds control-point count and sizes buffers.
- HashGrid is rebuilt each iteration (points move).
- Watch float non-associativity in atomic reductions when validating against the LBM baseline.
- Revisit only if a need for full autodiff over the whole solver, or a shippable C++ library,
  emerges.
