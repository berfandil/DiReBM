# DiReBM — Dispersion-Resampling Boltzmann Method

> Canonical research vision. Source of truth for *what* we build and *why*.
> This is an English reconstruction of the 2020 MSc TDK thesis "The Dispersion-Resampling
> Boltzmann Method" (ELTE IK, author Léránt-Nyeste Mátyás, supervisor Bálint Csaba). The
> thesis names the method **DRBM**; this project names it **DiReBM** — same method.
> The thesis was a *draft*: the method + a C++ reference implementation are complete; the
> introduction, quantitative LBM-vs-DRBM comparison, α-study, and conclusions were left
> `TODO`. Completing and improving those is part of this project.

## One-line

A **latticeless** Lattice-Boltzmann method: keep LBM's discrete velocity set, throw away the
fixed spatial lattice. Distribution-function values live at points chosen dynamically in
(possibly unbounded) space, so resolution can vary locally in space and time at O(k·n) cost.

## 1. Problem

Simulate fluid flow (density, velocity, pressure, turbulence). Two views:

- **Macroscopic** — continuous fields (mass, density ρ, velocity u). Classically solved with
  Navier–Stokes via finite-difference / finite-volume / finite-element methods.
- **Microscopic / mesoscopic** — particle distribution. Solved with the **Lattice Boltzmann
  Method (LBM)**. DiReBM lives here.

Scope (inherited from thesis, kept for now): incompressible/ideal fluid only; single fluid,
unit-mass particles; no temperature; perfectly elastic collisions. **Compressible fluids are
out of scope** (same math basis as LBM).

## 2. LBM background (what we keep)

Boltzmann transport equation:

    ∂f/∂t + v·∇f = Q(f, f)

with `f(x, v, t)` the Maxwell–Boltzmann distribution. **BGK** approximation replaces the
collision integral with single-relaxation-time relaxation toward equilibrium:

    Q(f,f) ≈ (1/τ) [ f_eq − f ]

Discrete-velocity form on a `DnQm` lattice (directions `c_i`):

- Macroscopic recovery:   ρ = Σ_i f_i        u = (1/ρ) Σ_i f_i c_i
- Equilibrium:            f_eq_i(ρ,u) = ρ W_i [ 1 + 3(c_i·u) + (9/2)(c_i·u)² − (3/2)(u·u) ]
- Collision:              f*_i = f_i + (1/τ)(f_eq_i − f_i) = (1/τ) f_eq_i + (1 − 1/τ) f_i
- Streaming:              f_i(x + c_i Δt, t + Δt) = f*_i
- Viscosity:              ν = (1/3)(τ − 1/2),  so τ > 1/2
- Relation:               dx/dt = |c_i| (lattice speed) → finer space ⇒ finer time

DiReBM **keeps the collision math and the BGK equilibrium unchanged**. It only replaces
*streaming on a fixed lattice* with a dispersion→resampling propagation in free space.

## 3. The DiReBM method

### 3.1 Why D2Q7

DiReBM is built on the **D2Q7** velocity set (hexagonal): `c_0 = (0,0)`, and
`c_i = (cos 2πi/6, sin 2πi/6)` for `i ∈ {1..6}`. Crucial property: **every direction is unit
length**. The method requires this (unlike LBM's D2Q9 where diagonal speeds are √2). Unit
lengths make the dispersed components land on a hexagonal graph where every neighbour is at
unit distance — after two iterations this covers space more uniformly than D2Q9 and keeps a
spreading wave more isotropic.

### 3.2 Three object classes replace the lattice

- **Moment `µ`** — a full distribution at a point. 7 reals `µ.f0…µ.f6 ∈ ℝ` + position `µ.x ∈ ℝ^d`.
  `µ_all` = all moments. Moments are what collision acts on and what macroscopic values come from.
- **Component `ν`** — one `f_i` of a moment plus a position. `ν.f ∈ ℝ`, direction index `ν.i`,
  position `ν.x`. `ν_all` = all components. A component = a group of fluid particles moving in
  one direction.
- **Control point `p`** — a distinguished location. `p.ν_near^dx` = components within radius
  `dx`. `p.C` = the set of *distinct* directions among those components; **`p.κ = |p.C|`** =
  "perceived direction count". `p_all` = all control points. Control points are where new
  moments get created.

### 3.3 Grid data structure (spatial hash)

A fixed-cell spatial bucket for points, supporting fast radius-neighbour queries. (This is the
heir of DiRe-CFD's `MultiGrid`.)

- **Subgrids**: tiled, non-overlapping, **dynamically created/dropped** as the occupied region
  grows/shrinks → no fixed domain bounds.
- **Cells**: fixed count per subgrid, each an unordered list. Optimal cell size = `dx` (matches
  the `dx`-radius neighbour search).
- Operations & cost (`δ` = expected elems per cell, a constant set by params):
  - insert: **O(1)**
  - insert-with-density-threshold (reject if another elem within a radius): **O(δ)**
  - remove-in-radius: **O(δ)**
  - query-all: **O(n)**
  - query-in-radius: **O(δ)**
- `δ_avg_ν`, `δ_avg_p` = average components / control points per cell. Constant in the method's
  parameters.

### 3.4 One iteration = collision + propagation

**Collision** (per moment, unchanged from LBM):

    µ.f_i := (1/τ) f_eq_i + (1 − 1/τ) µ.f_i      i ∈ {0..6}        cost O(|µ_all|)

(In the reference code, collision is folded into the Moment constructor, which recovers ρ,u
from f and re-equilibrates.)

**Propagation** splits into four sub-steps:

**(1) Dispersion ("explosion")** — each moment is destroyed and replaced by 7 components, each
shifted `dx` along its direction:

    ν_i.f := µ.f_i,   ν_i.x := µ.x + c_i · dx      (c_0 stays put)   cost O(|µ_all|)

**(2) Create control points** — walk the components; place a control point at each component's
location **unless** one already exists within `dx/α` (density threshold; `α` = density factor).
For each control point compute `p.κ` (stencil over the 7 directions of near components) and
classify:

    κ ≤ κ_hard  (=4)            → hard_outer
    κ_hard < κ ≤ κ_soft (=5)    → soft_outer
    κ > κ_soft                  → inner
                                                          cost O(|ν_all| · δ_avg_p)

Intuition: dispersed components move *away* from where material was, so boundary control points
see fewer distinct directions (low κ) → they are "outer".

**(3) Refine control-point positions** — per type:

- **hard_outer**: don't move. Preserves the fluid's free surface / boundary.
- **soft_outer**: create a *new* control point `p′`, then treat the original `p` as inner.
  With `c_sum = Σ_{ν∈p.ν_near} c_{ν.i}`:

      p′.x = p.x + (c_sum / ||c_sum||) · 2·(1 − √3/2) · dx

  The `2(1−√3/2)·dx` offset is the hexagonal gap geometry; it counteracts the every-other-step
  hexagonal anisotropy so a circular wave stays circular. (Author flags this as imperfect for
  *straight* wavefronts → open problem.)
- **inner**: move `p` to the **exponentially f-weighted average** of its near components'
  positions:

      p.x := ( Σ e^{ν.f} )^{-1} · Σ e^{ν.f} · ν.x

  → control points migrate toward high-f (denser) regions, putting resolution where the
  material is.
                                                  cost O(|p_all| · (δ_avg_ν + δ_avg_p))

**(4) Resampling** — rebuild moments at control points, conserving mass. Two phases:

- *(a) scatter component f into control points*: for each component `ν`, find control points
  within `dx`; weight `w_p := dx − ||ν.x − p.x||` (closer ⇒ larger). If none, create one
  (weight `dx`). Distribute:

      p.f_{ν.i} += (w_p / Σw) · ν.f

- *(b) fill gaps & emit moments*: for each control point, any direction `i` with `p.f_i == 0`
  (no component contributed it) is filled from equilibrium, mass-scaled across nearby
  also-empty control points so no extra mass enters:

      p.f_i := (dx / Σw) · f_eq_i,   Σw over near p′ that also need direction i

  Then emit a moment at `p.x` from its 7 f-values.
                                                  cost O((|ν_all| + |p_all|) · δ_avg_p)

During propagation, components and control points are transient; only the emitted moments
persist to the next collision.

### 3.5 Boundaries

Any surface with a representation. Components that strike a surface during dispersion **bounce**
(reflected direction from incidence + surface normal). If the reflected direction is not one of
the `c_i`, the component is **split into two valid-direction components**, sharing its `f` by
linear interpolation → mass conserved. Cost O(|ν_all|) (less with a surface acceleration
structure).

### 3.6 The α parameter

Density threshold for control-point creation is `dx/α`. Larger α ⇒ more control points ⇒ more
precision for more compute. Empirically: α ≥ 3 onward small instability appears and grows as α
shrinks; tested at **α = 4 and α = 5** (sufficient precision).

## 4. Complexity & comparison to LBM

Collision is identical. Propagation total:

    O( (15 + 2·δ_avg_p)·|µ_all| + (δ_avg_ν + 2·δ_avg_p)·|p_all| )

`|µ_all|, |p_all|` scale with the **non-resting** material; `δ` scale with α (precision at
given `dx`). This is **O(k·n)** — only a constant factor over LBM's O(n), no order-of-magnitude
loss — while gaining:

- **Local refinement**: `dt`, `dx` variable in space and time → precision where it matters.
- **Arbitrary surfaces**: trivial vs LBM (which struggles off-axis).
- **Unbounded domain**: no fixed grid extent.
- **Adaptive turbulence resolution**: control points track the fluid; α raises count.

## 5. Reference implementation (old, for porting)

C++11 + GLM, OpenGL viz. `float_t = double`, `D=2`, `Q=7`. Constants: `dt=1e-3`, `dx=1.0`,
`NU=8.9e-4`, `g=0`, `rho_0=1`, `u_0=0`, `tau=(6·NU·dt/dx²+1)/2`, `kappa_hard=4`,
`kappa_soft=5`, `alpha=4`. Simulator state machine: collision → dispersion →
creating_control_points → refining_control_point_positions → resampling → (repeat). Grid 100²
cells. Validation demo: rest state, raise pressure at center, watch a circular wave spread
(`τ ≈ 0.50000801`).

**Known issues in the reference code (fix on reimplementation):**

- `equilibriumDistribution` (code 5.16) has `u_dot_u * u_dot_u` — a quartic where the formula
  wants `u·u` (quadratic). Likely a typo bug.
- `refiningControlPointPositionsStep` (code 5.12): the `soft_outer` case deliberately falls
  through (no `break`) into `inner` — this matches the spec ("then treat it as inner"), but is
  fragile; make it explicit.
- Collision hidden inside the Moment constructor — surprising; make it an explicit step.

## 6. Status from the thesis

- **Done**: full method spec, C++ reference impl, mesoscopic 4-iteration walkthrough figures.
- **Left TODO (we complete)**: introduction; quantitative DRBM-vs-LBM macroscopic comparison
  (§5.2.2); α-variation study (§5.2.3); conclusions; appendix.

## 7. Improvement directions (status)

Re-added now that a working version exists (parked in `research/progress.md` during v1 bring-up).

1. **GPU parallelization** — **DONE** (v2 Warp, `direbm/warp/`). Full solver + LBM on GPU, validated
   against the v1 oracle (`exp_gpu_vs_v1`). The thesis's main open problem.
2. **Boundaries / arbitrary surfaces** — **DONE** (reference; `boundary.py`, `exp_obstacle`):
   specular bounce + mass-conserving direction-split. GPU port pending.
3. **Quantitative validation** — **DONE**, incl. an **analytic ground truth**: LBM baseline
   (acoustics ≈ cs), α study, and the **Taylor–Green vortex** (`exp_taylor_green`). TG finding:
   DiReBM reproduces the decay *form* but is **~1.65–1.72× over-dissipative** — its effective
   viscosity exceeds the physical ν (numerical dissipation from dispersion/resampling; confirmed
   intrinsic, not a boundary artifact). **Pinned** (`exp_numerical_viscosity`): the excess is a
   fixed additive **ν_num ≈ 0.074**, ~independent of τ and k → effective ν = ν_phys + 0.074. Gives a
   validated compensation rule `τ = ½ + 4(ν_target − 0.074)` and a **minimum-viscosity floor ≈
   0.074**. New open item: **reduce ν_num** (ties to #5, over-sampling) to lower the floor.
4. **soft_outer step-3 correction** — CHARACTERIZED (priority down). The spawn is actually
   effective — 16× less circular hex ripple, ~1.6× lower straight-front roughness, helping most
   orientations; the thesis's straight-front worry is largely unfounded. Residual: a small, noisy
   degradation near 30° off-axis. A curvature-gated fix was tried and failed (lost the circular
   benefit). See `docs/results/exp_soft_outer.md`.
5. **Adaptive local resolution (α / dt / dx)** — OPEN. Reduce over-sampling so cost tracks active
   material. Naïve rest-pruning is blocked by per-moment density dilution (see progress.md); needs a
   de-diluting consolidation or variable-cell thinning.
6. **3D** — **RUNNABLE**. Velocity set = **D3Q13 icosahedral** (12 unit dirs + rest), NOT FCC — FCC's
   12 neighbours aren't 4th-order isotropic; icosahedral is (ADR 0002). Done: `Lattice` abstraction
   + dimension-generic physics/grid/simulator. A spherical pulse propagates isotropically
   (`exp_sphere`); validated (isotropy, 3D conservation, spread, rest-density). Open: quantitative
   3D validation (sphere speed / 3D Taylor–Green); 3D obstacles (need a `Sphere` + 3D direction
   split); 3D GPU port. Note: 3D over-sampling is heavier (slower) — same over-sampling lever.
7. **GPU performance** — OPEN. The cost is in the neighbour-reduction kernels; over-sampling is the
   lever (ties to #5). Roadmap in progress.md.
8. Later: differentiable simulation, temperature, multi-fluid.

## 8. Smallest validation case (first milestone)

2D circular pressure wave from rest (the thesis demo), reproduced and checked against an LBM
baseline at the macroscopic level. This is the correctness anchor before any GPU work.

## Non-goals (for now)

- Compressible fluids. Temperature. Multi-fluid mixing. (All deferred, as in the thesis.)
