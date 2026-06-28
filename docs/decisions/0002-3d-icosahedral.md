# ADR 0002 — 3D velocity set: D3Q13 icosahedral

- Status: accepted
- Date: 2026-06-28

## Context

DiReBM requires **unit-length** velocity directions: dispersion moves every component by the same
dx, which is what gives the method its uniform space coverage (the thesis chose the D2Q7 hexagon in
2D precisely because all six directions are unit length). The equilibrium additionally needs the
velocity moments to be isotropic — the 2nd moment for mass/momentum, and the **4th moment** for
Navier–Stokes — and the 4th-moment constant must equal cs⁴.

The obvious cubic 3D analog (FCC: the 12 nearest-neighbour directions (±1,±1,0)/√2 etc., all unit
length) **fails**: with a single weight, the c_x⁴ isotropy condition wants w=1/24 while c_x²c_y²
wants w=1/16 — they cannot both hold, so FCC D3Q13 is 4th-order anisotropic.

## Decision

Use the **twelve icosahedron-vertex directions** (unit length) + rest = **D3Q13-ico**. Icosahedral
symmetry has no 2nd- or 4th-rank anisotropic invariants (it is isotropic to 5th order), so a single
weight gives isotropic 2nd and 4th moments.

Choosing the weight: cs² = 4w (2nd moment), and the (isotropic) 4th moment is Σ W c_x²c_y² = 0.8w.
Matching it to cs⁴ for correct NS: 0.8w = (4w)² ⟹ **w = 1/20, cs² = 1/5, W₀ = 2/5**.

## Validation

`tests/test_lattices_3d.py`: directions are unit length, Σ W = 1, the 2nd moment = cs²·I, the 4th
moment is isotropic and equals cs⁴ (c_xxxx = 3 cs⁴, c_xxyy = cs⁴), and equilibrium/collision
conserve mass and momentum in 3D.

## Consequences

- The velocity directions do **not** form a lattice (icosahedral has no translational tiling) — but
  DiReBM is latticeless, so this is fine; dispersed components form a point cloud handled by the
  existing cell-thinning / control-point machinery.
- Physics (`equilibrium`/`recover`/`collide`) was made dimension-generic via a `Lattice` bundle
  (default `D2Q7`), so 2D code is unchanged and 3D just passes `D3Q13`.
- The hex-specific soft_outer offset does not carry over; 3D will use `soft_mode="off"` initially
  (the spawn is a minor correction — see exp_soft_outer).
