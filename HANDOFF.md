# DiReBM — Handoff

Pick-up document for the next agent. **Follow-up direction: the 3D GPU port (§7).** Read §1 first,
then the "re-read" list in §2 before touching code.

---

## 1. What this project is

**DiReBM = Dispersion-Resampling Boltzmann Method** — a *latticeless* Lattice-Boltzmann fluid
solver. Keep LBM's BGK collision + a discrete unit-length velocity set, but drop the fixed spatial
lattice: distribution values live at points chosen dynamically in space, so resolution can vary
locally at O(k·n) cost.

Origin: a revitalization of (a) the prior C++ project `../DiRe-CFD` (only artifact was a `MultiGrid`
spatial structure) and (b) a **2020 MSc TDK thesis** (Hungarian) "The Dispersion-Resampling
Boltzmann Method" by Léránt-Nyeste Mátyás (ELTE). The thesis is the method source; it named the
method **DRBM** (same thing). The English reconstruction of the method lives in
`research/idea.md` (canonical) — the thesis itself is NOT in the repo.

Method in one breath: each **moment** (a full distribution at a point) **disperses** into 7/13
**components** (one per unit direction, shifted dx); **control points** are chosen among them
(density ~dx/α); positions are **refined** (κ = perceived-direction count → hard-keep / inner-move);
then **resampling** scatters component f into control points and emits new moments. Collision = BGK.
Empty directions are filled from the rest equilibrium (so untracked space = implicit rest).

---

## 2. Re-read these before starting (in order)

1. **`research/idea.md`** — the canonical method (problem → LBM background → the 4 sub-steps →
   complexity → §7 improvement board with done/open status). This is the source of truth for *what*
   and *why*.
2. **`research/progress.md`** — the full running log (newest first): every decision, finding, and
   negative result, dated. Read the top ~15 entries.
3. **`docs/ARCHITECTURE.md`** — the code map (which file does what; status tags).
4. **`docs/decisions/`** — ADR 0001 (stack = Warp) and **ADR 0002 (3D velocity set = icosahedral,
   not FCC)** — read 0002 before the 3D GPU port.
5. **`CLAUDE.md`** — project instructions: hardware (RTX 5080, sm_120), toolchain (`uv`), commands,
   conventions, **caveman comms voice** (terse; code/commits/PRs/security in normal English).
6. **Result writeups** relevant to your task under `docs/results/` (see the map). For the GPU port:
   `exp_gpu_vs_v1.md`, `exp_gpu_bench.md`, `exp_gpu_locality.md`, `exp_numerical_viscosity.md`.
7. **The thesis** (optional, for method detail beyond idea.md): Hungarian PDF, 69 pages, at
   `https://drive.google.com/file/d/1AzDZC9AKV52JJ0f7H3HmjyAbuWAbT-9D/view`. Not in the repo; the
   session scratchpad copy is gone. Re-download to the scratchpad and parse with pymupdf if needed
   (`uv run --with pymupdf python` — extract text + render pages; it is math/figure-heavy).

---

## 3. Environment & commands

- Python 3.12 via **`uv`** (winget path: `%LOCALAPPDATA%\Microsoft\WinGet\Packages\astral-sh.uv_*\uv.exe`).
  Corp TLS intercept → `[tool.uv] system-certs = true` is set in `pyproject.toml` (do not remove).
- GPU: **NVIDIA RTX 5080 Laptop, Blackwell, sm_120**, 16 GB. Warp 1.14 (bundled CUDA 12.9) verified.
- Commands:
  ```pwsh
  uv sync --extra dev        # install
  uv run pytest -q           # 51 tests, ~90s (3D tests are heavy)
  uv run ruff check .        # lint (line length 110)
  uv run python experiments/exp_<slug>.py
  ```
- Git: on `main`, remote `github.com/berfandil/DiReBM`. Commit messages end with the Co-Authored-By
  + Claude-Session trailers (see CLAUDE.md / git config). All work so far is committed + pushed.

---

## 4. What exists (state at handoff)

- **v1 reference solver (numpy, the correctness oracle) — 2D + 3D.**
  `direbm/{constants,lattices,physics,fields,lbm}.py` + `direbm/reference/{types,grid,simulator,boundary}.py`.
  Dimension-generic: `physics`/`grid`/`Simulator` take a `Lattice` (default `D2Q7`; pass `D3Q13`).
  Obstacles (`Circle` 2D / `Sphere` 3D) with specular bounce + direction-split.
- **v2 GPU solver (NVIDIA Warp) — 2D ONLY.**
  `direbm/warp/{physics,propagation,simulator,lbm}.py`. Full DiReBM step on GPU (collide → disperse
  → cell-thinning control points → refine via `wp.HashGrid` → atomic-scatter resampling), plus a
  GPU D2Q7 LBM. Validated against the oracle. **This is what the follow-up extends to 3D.**
- **LBM baselines:** `direbm/lbm.py` (CPU HexLBM) + `direbm/warp/lbm.py` (GPU).
- **15 experiments, 51 tests, 2 ADRs.** All green, ruff clean.

### Key numbers / facts to keep in mind
- Lattices: **D2Q7** (2D, cs²=1/4, W₀=1/2 Wᵢ=1/12) · **D3Q13-ico** (3D, cs²=1/5, W₀=2/5 Wᵢ=1/20).
- **Per-moment ρ is diluted** (≈ ρ/point-density) — the macroscopic field is **mass/area**, via
  `direbm.bin_fields` (2D; a 3D version is missing). Never read per-moment ρ as the density.
- **Numerical viscosity ν_num ≈ 0.074 (2D) ≈ 0.069 (3D)** — a *fixed additive* dissipation,
  ~independent of τ and k (dimension-robust ~0.07). ν_eff = ν_phys + ν_num. Compensation:
  `τ = ½ + (ν_target − ν_num)/cs²`. Minimum achievable ν ≈ ν_num. Root cause = over-sampling /
  resampling smoothing.
- **GPU cost scales with active material, not domain size** (`exp_gpu_locality`): DiReBM flat vs
  LBM ∝ L²; crossover ~L=2000, overhead-bound (the per-step floor is HashGrid builds + kernels,
  NOT allocations — profiled).

### Fixed thesis bugs (already corrected in the oracle)
1. Equilibrium used cs²=1/3 coefficients (D2Q9) with D2Q7 weights → leaked mass. Fixed to cs²=1/4.
2. Reference C++ squared the (u·u) term (quartic) — a typo. Fixed.
3. C++ `collisionStep` was a no-op — the oracle does the real BGK relax.

### Negative results (do NOT re-attempt naively — read the writeups)
- **Adaptive rest-pruning** (drop moments matching f_rest): blocked by the ρ-dilution — "rest" is a
  field property, not per-moment detectable. Reverted (progress.md 2026-06-27). A real fix needs a
  de-diluting *consolidation* step first.
- **soft_outer step-3 "fix"** (curvature-gating the spawn): failed (lost the circular benefit).
  Reverted. Finding: the spawn is actually good (16× less circular hex ripple); the thesis's
  straight-wavefront worry is largely unfounded. See `docs/results/exp_soft_outer.md`. Priority
  downgraded, not open.
- **GPU sync-removal**: only ~13% (the floor is HashGrid builds, not syncs — profiled).

---

## 5. Conventions & gotchas

- **Caveman comms** (SessionStart hook): terse voice; code/commits/PRs/security in normal English.
- **3D is slow**: Python reference + severe 3D over-sampling (a pulse inflates 13 → ~35k moments in
  4 steps). Keep 3D domains/steps tiny; run long experiments in the background. This slowness is the
  motivation for the 3D GPU port.
- Console is cp1252 → **no Greek/unicode in `print()`** (use ASCII: `nu`, `lambda`); matplotlib
  labels can keep unicode.
- Warp kernels can't be defined in `python -c`/stdin (source extraction fails) — put them in files.
- `uv.lock` is tracked; don't hand-edit. `.agent-workspace/` is gitignored scratch (delete when done).
- Full `pytest` is ~90s; when iterating, run a subset (`uv run pytest -q tests/test_x.py`).

---

## 6. §7 improvement board (from idea.md) — status

GPU (2D) ✅ · boundaries (2D+3D) ✅ · quantitative validation incl. **analytic GT** (2D Taylor–Green,
3D shear) ✅ · soft_outer characterized ✅ · **numerical viscosity pinned** ✅ · **3D runnable +
validated + obstacles** ✅.

Open: **3D GPU port (next)** · GPU performance / over-sampling reduction · differentiable sim,
temperature, multi-fluid (later) · D2Q9 lattice (future task — see progress.md; note D2Q9 diagonals
aren't unit-length, decide purpose first) · complete the thesis prose (intro/conclusions were TODO).

---

## 7. FOLLOW-UP DIRECTION — 3D GPU port

Goal: extend the **v2 Warp GPU solver** (`direbm/warp/`) from 2D-only to 3D (D3Q13), validated
against the 3D reference oracle. This makes 3D runs fast (the reference is painfully slow in 3D).

### What is 2D-hardcoded in `direbm/warp/` (the work)
- `physics.py`: `_consts` uploads the 2D `C`/`W`; `_equilibrium_row` bakes `_INV = 1/CS2` with the
  **2D** cs²=0.25; the collide/equilibrium kernels loop `range(7)`. For 3D: upload `D3Q13.C`
  (as `wp.vec3`) / `W` (13), use cs²=1/5 (`_INV=5`), loop `range(13)`.
- `propagation.py`: positions are `wp.vec2`; `_disperse` loops `range(7)`; `_cell_key` packs a **2D**
  integer key (`ix*STRIDE + iy`) → needs a 3D pack (add iz; widen or use int64 keys — radix_sort
  supports int64); `_to_vec3` converts vec2→vec3 (for real 3D, positions are already vec3). Refine /
  scatter / gapfill already use `wp.vec3` for the `wp.HashGrid` (currently with z=0) — the HashGrid
  is genuinely 3D, so real z "just works" once positions are vec3.
- `simulator.py` (`GpuSimulator`): `wp.vec2`, D2Q7 constants, `_collide` with 7.

### Approach
- Warp kernels can't take a runtime `Q` (loop bounds are static / codegen-time). Options: (a)
  separate 3D kernels (`_disperse3`, etc.) with `range(13)` and `vec3`; or (b) parameterize via
  Warp generics/constants. (a) is simplest and lowest-risk — mirror the 2D kernels.
- Reuse the existing pipeline structure (dispersion → cell-thin via radix_sort/runlength/scan →
  refine on HashGrid → atomic-scatter resample). The `wp.HashGrid(dim,dim,dim)` already supports 3D.
- Keep the 2D path untouched (all 51 tests must stay green).

### Validate against
- `direbm.reference.Simulator(lattice=D3Q13)` — the 3D oracle.
- Reproduce `test_reference_3d.py` behaviour on GPU (spherical pulse spreads, rest density
  preserved), and ideally re-run `exp_shear_3d`'s ν_num measurement on GPU (should match ~0.069 and
  run far faster).
- Then a GPU-vs-oracle macroscopic comparison in 3D (mirror `exp_gpu_vs_v1`).

### Watch out for
- Note: `bin_fields` is 2D-only — you'll need 3D binning (mass/volume) for macroscopic 3D fields.
- 3D over-sampling means far more components/control-points per moment than 2D → the HashGrid /
  sort buffers must be sized for it; expect higher `nc = 13*M`.
- Float32 vs the float64 oracle: validate within ~1e-4 (as the 2D GPU tests do).
- soft_outer spawn stays "off" in 3D (2D hex geometry only); obstacles need a 3D split on GPU if you
  port boundaries too (the reference `split_direction_nd` is the reference behaviour to match).

Good luck. The reference oracle is trustworthy — validate every GPU stage against it, and keep the
honest-negative-results discipline (measure before "fixing").
