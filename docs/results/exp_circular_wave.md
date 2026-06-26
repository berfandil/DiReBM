# exp_circular_wave — pressure pulse from rest (v1 reference)

Date: 2026-06-26 · Code: `experiments/exp_circular_wave.py` · Solver: `direbm.reference.Simulator`

## Setup

Single elevated-density moment (ρ=1.5, u=0) at the origin; everywhere else implicit rest. τ=0.6,
α=4, κ_hard=4, κ_soft=5 (lattice units). Run 16 iterations; snapshot moment positions + density
at iterations 4/8/12/16. This is the thesis demo (§5.2).

## Result

![spread](exp_circular_wave_spread.png)

```
 iter  #moments   max_r  total_mass
    1         7    1.00       4.500
    4       375    4.00      47.259
    8      1784    8.00     165.114
   12      4331   12.00     347.656
   16      7868   16.00     602.976
```

- **Front radius = iteration exactly** — unit dispersion per step, as designed.
- **Isotropic / circular** spread, not hexagonal → the soft-outer correction (eq 4.3) works.
- **Stable**, all-finite, ρ>0 throughout.
- Structure is **wave-like**: a bright high-density ring at the expanding front, rarefied
  interior (compression front + rarefaction behind a point pulse).

## Open issue (for the validation milestone)

Absolute interior density decays far below rest (mean ρ ≈ 0.08 by it16; total mass grows as the
active region expands). Cause: the resampling rest-fill (thesis eq 4.7) scales `f_eq_i` by
`dx/Σw` over nearby empty control points → from a single seed the fill is diluted by the
neighbour count. Faithful to the spec, but the macroscopic density is **not yet calibrated**.

The thesis itself left the macroscopic DRBM-vs-LBM comparison (§5.2.2) as TODO, so this is
expected territory. Next: a small LBM baseline + a fully-populated rest field (perturb-centre)
initial condition, and check whether quiescent regions hold ρ≈ρ_rest. If not, the fill rule
needs revisiting (an improvement-list item).

## Status

v1 reference solver runs end to end (collision + dispersion + create/refine control points +
resampling) and reproduces the qualitative thesis demo. Correctness anchor established; absolute
macroscopic calibration deferred to the validation milestone.
