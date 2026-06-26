# exp_rest_state — is rest a steady state? (v1 reference)

Date: 2026-06-26 · Code: `experiments/exp_rest_state.py` · Solver: `direbm.reference.Simulator`

## Question

Rest (ρ = ρ_rest, u = 0 everywhere) must be a fixed point of any sane fluid solver. The
single-seed `exp_circular_wave` *appeared* to dilute the interior density; this experiment
isolates the question with a dense block of rest moments and measures the macroscopic field.

## Setup

25×25 = 625 moments on the integer grid [−12, 12]², each at f_eq(ρ=1, u=0). τ=0.6, α=4. Run 8
steps. Probe only the interior (|x|_∞ < 5), which stays insulated from the outward-dispersing
boundary for ~(12−5) steps. Density is measured as a **field**: mass / area, not per-moment ρ.

## Result

```
 iter   #mom  #int  rho_field  |u_field|  rho/moment
    1   2119   321     1.0655    0.00000      0.3319
    2   4064   441     0.9385    0.00004      0.2128
    4   6731   555     1.0023    0.00021      0.1806
    6   9504   601     1.0065    0.00104      0.1675
    8  13185   765     1.0012    0.00138      0.1309
```

- **ρ_field ≈ 1.00** throughout (range 0.94–1.07, settling to ≈1.00). Rest **is** preserved as a
  macroscopic field.
- **|u_field| ≈ 0** (≤ 1.4e-3). No spurious flow.
- **Per-moment ρ falls** (0.33 → 0.13) only because the **number of sample points grows**: the
  per-moment ρ is mass/sample-point = 1/(point density), not the physical density.

## Findings

1. **Density is sound.** The earlier "dilution" alarm (`exp_circular_wave` draft) was a
   measurement error — per-moment ρ ≠ macroscopic density. Always reconstruct fields with
   `direbm.bin_fields` (mass/area). This is the key lesson for all future validation and viz.
2. **Moment-count inflation.** Even at rest the moment count climbs (625 → 13185 over 8 steps;
   interior point density 3.2 → 7.7 per unit area). It is bounded — the density-threshold insert
   forbids control points closer than dx/α, capping point density at ~1/(dx/α)² (≈16/area for
   α=4) — but it inflates toward that cap rather than holding the input sampling. In a featureless
   rest region this is wasted work/memory. Candidate improvement: adaptively coarsen the
   control-point density where the field is smooth (currently α is global and fixed).

## Status

Rest-state preservation confirmed at the field level. Density correctness established for v1.
Remaining: LBM-baseline comparison for wave dynamics, and characterizing/curbing point-density
inflation.
