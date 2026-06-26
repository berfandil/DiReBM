# DiReBM — Progress Log

Newest first. ISO dates. Cross-experiment narrative; per-experiment detail lives in `docs/results/exp_<slug>.md`.

## 2026-06-26 — Project revitalization started

- Repo created as continuation/improvement of `../DiRe-CFD`.
- Reviewed DiRe-CFD: only substantive artifact = `MultiGrid<dim,Data>` (C++ template, particles-per-cell spatial structure with CSR-style compression). `study/` (math) was empty. Demo = Hello World.
- Direction confirmed by user:
  - Method = latticeless Boltzmann ("lattice Boltzmann, but no lattice"). Full writeup pending from user; DiRe-CFD is **not** the authoritative source.
  - Stack = undecided. To be proposed (pros/cons) once the method is understood. Not constrained to C++/Python/GPU.
- Created `research/idea.md` as the landing spot for the method writeup.
- Parsed the full source: 2020 MSc TDK thesis "The Dispersion-Resampling Boltzmann Method"
  (ELTE IK; author Léránt-Nyeste Mátyás, supervisor Bálint Csaba; 69 pp, Hungarian, draft).
  Rendered the complete method into English in `research/idea.md`: latticeless LBM on D2Q7
  (unit-length dirs); moments/components/control-points replace the lattice; propagation =
  dispersion → create control points → refine (hard/soft/inner by perceived-direction count κ)
  → resampling; collision unchanged (BGK). O(k·n). Spatial-hash grid data structure (heir of
  DiRe-CFD `MultiGrid`). Reference impl = C++11/GLM/OpenGL. Thesis quantitative comparison +
  intro + conclusions were `TODO`.
- Flagged improvement targets: GPU parallelization (author's main open problem), soft_outer
  wavefront fix, full quantitative validation, 3D, adaptive local dt/dx.
- Stack chosen: **NVIDIA Warp** (smoke-verified on sm_120). Scaffolded repo (pyproject/uv,
  `direbm/` pkg, tests, ARCHITECTURE map, ADR 0001). ruff+pytest green.

### Decisions (2026-06-26)

- **GPU not right away.** v1 = plain Python/numpy **reference solver** (the correctness oracle),
  validated on the 2D circular wave vs a small LBM baseline. v2 = port to Warp (CPU→GPU, same
  kernels). Rationale: method risk is correctness, not throughput; don't debug numerics + CUDA
  plumbing at once; numpy oracle is reusable for all future tests; Warp is device-agnostic so the
  GPU step is a port, not a rewrite.
- **Step-3 soft_outer geometry** (`2(1−√3/2)·dx` offset; imperfect for straight wavefronts):
  deferred, kept as-in-thesis for v1.
- **§7 improvements list** pulled out of `idea.md`; re-add after v1 works. Parked here:
  GPU parallelization · soft_outer straight-wavefront fix · full quantitative validation (vs LBM;
  circular wave, Taylor–Green, lid-driven cavity; convergence vs α) · 3D (D3Qm unit-length) ·
  adaptive local dt/dx · later: differentiable sim, temperature, multi-fluid.

### v1 build — started (2026-06-26)

- **Done:** `direbm/constants.py` (D2Q7 dirs/weights/params) + `direbm/physics.py` (equilibrium,
  macroscopic recovery, BGK collision). 8 tests green (conservation + equilibrium moments).
- **Findings — two bugs in the thesis equilibrium, both fixed in our oracle:**
  1. **cs² inconsistency.** Eq. 3.19 uses coefficients 3/4.5/1.5 (cs²=1/3, the D2Q9 square
     lattice), but D2Q7's weights (W₀=1/2, Wᵢ=1/12) give Σ Wᵢ cᵢαcᵢβ = ¼δ, i.e. **cs²=1/4**.
     With the 1/3 coefficients mass is not conserved. Fixed by using the cs²=1/4 form
     (coefficients 4/8/2) → mass + momentum exact. `CS2 = 1/4` in constants.
  2. **Quartic typo** in reference C++ (code 5.16): `u·u` term was squared. Correct form is
     linear in `u·u`.
- **v1 reference solver COMPLETE.** `direbm/reference/`: `types.py` (Moment/Component/ControlPoint),
  `grid.py` (dict-of-cells spatial hash), `simulator.py` (collision + 4 propagation sub-steps:
  dispersion → create control points → refine (hard/soft/inner) → resampling). 13 tests green.
  Ran `experiments/exp_circular_wave.py` — see `docs/results/exp_circular_wave.md`.
- **Finding #3 — collision was a no-op in the C++ draft** (collisionStep just cleared a grid; its
  comment wrongly assumed the moment ctor relaxed). Our oracle does the real BGK relax (thesis §4.4).
- **exp_circular_wave result:** front radius = iteration (correct), isotropic/circular spread
  (soft-outer correction works), stable, wave-like ring (compression front + rarefied interior).
### Validation milestone — started (2026-06-26)

- Committed + pushed v1 (main `080c63e`).
- **Density is sound — earlier "dilution" alarm was a measurement error.** A moment's ρ = Σf is the
  mass of one sample point = 1/(point density), NOT the macroscopic density. Macroscopic density =
  **mass/area**. Added `direbm/fields.py:bin_fields` (bin moments → ρ, u fields); always use it for
  validation/viz.
- **`exp_rest_state`** (`docs/results/exp_rest_state.md`): a uniform rest field IS preserved as a
  field — ρ_field ≈ 1.00 (0.94–1.07), |u_field| ≈ 0. Corrected `exp_circular_wave` writeup +
  artifact (now shows per-moment ρ vs reconstructed field ρ; field = rest disk + compression ring).
- **Real open signal — moment-count inflation.** Even at rest the point count climbs toward the
  α-set saturation (~1/(dx/α)² per area; bounded but wasteful in smooth regions). Improvement
  candidate: adaptive control-point density where the field is smooth (α is global+fixed now).
- 15 tests green, ruff clean.
- **D2Q7 LBM baseline DONE** (`direbm/lbm.py:HexLBM`): hexagonal lattice in skewed axial coords →
  streaming = `np.roll`; same equilibrium/τ as DiReBM. 3 tests (rest steady, mass conserved, pulse
  spreads). 18 tests total green.
- **DiReBM vs LBM comparison DONE & POSITIVE** (`docs/results/exp_lbm_vs_drbm.md`, thesis TODO
  §5.2.2): rest background + central pulse, both solvers. **Radial density profiles overlap**
  (same rarefaction core + compression peak at r≈5.5), and **compression-peak radius tracks**
  step-for-step (0.5→3.5→5.5 over 8 steps). DiReBM reproduces the LBM acoustic wave macroscopically.
  - Measurement notes: DiReBM field reconstruction is ~1–2% noisy → use compression-PEAK radius,
    not a threshold front; finite rest-block edge needs interior windowing. (The earlier ballistic
    "front=iteration" of exp_circular_wave was the vacuum-seed edge, not the acoustic wave.)
- **v1 reference solver now validated** for density (rest preserved) + acoustics (matches LBM).
- **α parameter study DONE** (`docs/results/exp_convergence.md`):
  - Point-density inflation monotonic in α (2.4 → 7.0 moments/area for α=2→5), between linear and
    quadratic at 5 steps (cap ~α²). Cost grows with α.
  - Convergence to LBM is **U-shaped**: best at **α≈3–4** (profile L2 err ≈ 0.037), worst at α=2
    (0.115). All stable (no NaN in 6 steps); thesis's "instability at small α" shows here as
    accuracy loss at α=2. → α≈4 is the sweet spot (matches thesis choice); diminishing returns past
    it motivates **adaptive local α**.
- **Caveat — LBM is not ground truth.** All "vs LBM" comparisons measure agreement with the LBM
  *result*, not with the true solution. LBM has its own errors (compressibility, lattice isotropy,
  finite dx/dt, BGK). A DiReBM–LBM gap could be either method's fault. LBM is a fine proxy for now;
  true accuracy may later need an analytic case / DNS / Richardson extrapolation. (Note in
  `exp_lbm_vs_drbm.md`.)
- **Robust α study** (`docs/results/exp_alpha_robust.md`, seed-averaged + sound-speed fit):
  - **Sound speed:** LBM peak speed = 0.54 ≈ cs=0.5 (metric validated); DiReBM (α≥3) peak overlaps
    LBM in the propagation phase → wave travels at ≈cs. α=2 lags. Per-α speed fits noisy (coarse
    bins) but bracket cs.
  - **Profile error:** seed-averaged bars overlap; **α≈4 mild optimum**, fairly insensitive for
    α≥3. Tempers exp_convergence's single-run U-shape (α=2 penalty was a short-run transient; α=5
    uptick within noise).
- **Milestone 1 (validation) COMPLETE.** v1 reference solver validated: rest preserved, acoustics
  match LBM at ≈cs (LBM = proxy, per caveat above), α≈4 sweet spot.

### v2 Warp GPU port — started (2026-06-26)

- `direbm/warp/physics.py`: Warp kernels for recover + equilibrium + BGK collision (float32, GPU).
  Validated against the v1 float64 oracle within ~1e-4 (`tests/test_warp_physics.py`, 4 tests, run
  on `cuda:0` sm_120). 22 tests total green. Establishes the Warp idioms (2D `f` array, vec2
  positions, device handling, float32 tolerance).
- **Design decision for the hard step (control-point creation):** v1's density-threshold creation
  is greedy/sequential (a point is created only if none exists within dx/α — order-dependent), bad
  for GPU. v2 will use **cell-based thinning**: a sub-grid of cell size dx/α, one control point per
  occupied cell (placed at its components' centroid). Deterministic + parallel; approximates the
  greedy threshold. Will validate the resulting macroscopic fields against v1 (not bit-exact).
- **v2 increment 2 DONE** (`direbm/warp/propagation.py`): dispersion kernel (moment→7 components)
  + **cell-thinning control-point creation** on GPU (radix_sort_pairs → runlength_encode → exclusive
  array_scan → centroid kernel; one control point per dx/α cell at its components' centroid).
  Validated against a numpy reference of the same thinning (`tests/test_warp_propagation.py`, 3
  tests). 25 tests total green.
- **v2 increment 3 DONE** — refine (`direbm/warp/propagation.py:refine_control_points`): builds a
  `wp.HashGrid` over components, per control point queries components within dx → κ (direction
  bitmask) → hard_outer (κ≤kappa_hard) keep / else exp(f)-weighted centroid move. Validated with
  controlled κ-regime tests (`tests/test_warp_propagation.py`). 27 tests total green.
  - **Simplification:** soft_outer treated as inner (no anti-anisotropy spawn) — that is the
    deferred step-3 issue; revisit later. Confirms the HashGrid radius-query primitive works on GPU.
- **v2 CORE COMPLETE — full GPU DiReBM solver works.**
  - Resampling (`resample`): phase-1 atomic scatter (component f → nearby control points, HashGrid)
    + phase-2 gap-fill from rest-eq (read-only snapshot, no race) + emit moments.
  - `GpuSimulator` (`direbm/warp/simulator.py`): full step on device (collide → disperse → cell-thin
    → refine → resample), moments persist on device, realloc per step for dynamic counts.
  - Validated: rest preserved, pulse spreads, **matches v1 macroscopically** (3 tests). 30 tests total.
  - **`exp_gpu_vs_v1`** (`docs/results/exp_gpu_vs_v1.md`): radial profiles overlap within ~0.05;
    GPU field is *smoother* (cell-thinning more uniform than greedy). Timing: GPU **~1.4 ms/step**
    vs v1 **~2.9 s/step** at ~5–8k moments. (The ~2000× is vs unoptimized Python — overstates GPU
    vs good CPU; honest point: full irregular pipeline in ~1 ms, scales v1 can't reach.)
  - **The thesis's open problem — parallelizing this latticeless method — now has a working,
    validated GPU implementation.**
- **Next:** GPU LBM baseline (`direbm/warp/lbm.py`, user request) for a fair GPU-vs-GPU speed
  comparison; then scaling studies + deferred refinements (soft_outer step-3, adaptive α,
  on-device compaction instead of per-step realloc).
- **TODO after the DiReBM GPU solver is ready (user request):** GPU port of the LBM baseline
  (`direbm/warp/lbm.py`) — streaming via shifts + the existing collision kernel — for a fair
  GPU-vs-GPU speed comparison.
- **Next:** begin the **Warp (v2) GPU port** against the trusted v1 oracle. (Optional polish:
  seed-averaged α error bars, sound-speed fit.)
