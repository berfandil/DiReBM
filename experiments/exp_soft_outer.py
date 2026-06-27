"""exp_soft_outer — diagnose wavefront anisotropy and the soft_outer step-3 correction.

The thesis's soft_outer rule spawns an extra control point a fixed 2(1−√3/2)·dx ahead of the
front to counteract the D2Q7 hexagonal anisotropy. It works for circular (point-source) fronts but
the author flagged it as wrong for straight fronts. This measures it.

Two probes:
  - circular: a point pulse → front radius r(θ); hexagonal bias shows as a 6-fold ripple.
  - straight: a vertical sheet source → planar front; bias shows as roughness in the front x(y).

For each soft_mode we report an anisotropy/roughness metric. Lower = more isotropic.

Run: uv run python experiments/exp_soft_outer.py
"""

from __future__ import annotations

import numpy as np

from direbm import equilibrium
from direbm.reference import Moment, Simulator

TAU = 0.6
PULSE = 1.6
MODES = ["off", "spawn"]


def _pulse_moments():
    return [Moment(f=equilibrium(np.float64(PULSE), np.zeros(2)), x=np.zeros(2))]


def circular_anisotropy(mode, steps=12, sectors=36):
    sim = Simulator(_pulse_moments(), tau=TAU, soft_mode=mode)
    for _ in range(steps):
        sim.step()
    x = np.array([m.x for m in sim.moments])
    r = np.linalg.norm(x, axis=1)
    th = np.arctan2(x[:, 1], x[:, 0]) % (2 * np.pi)
    edges = np.linspace(0, 2 * np.pi, sectors + 1)
    front = np.array([r[(th >= edges[k]) & (th < edges[k + 1])].max(initial=0.0) for k in range(sectors)])
    rel_std = front.std() / front.mean()
    sixfold = np.abs(np.fft.rfft(front - front.mean())[6]) / (front.mean() * sectors / 2)
    return rel_std, sixfold, front


def angled_roughness(mode, phi_deg, half=8, steps=6):
    """Roughness of a planar front propagating along angle phi (a sheet source ⟂ to it)."""
    phi = np.radians(phi_deg)
    d = np.array([np.cos(phi), np.sin(phi)])  # propagation direction
    dp = np.array([-np.sin(phi), np.cos(phi)])  # along the front
    f_rest = equilibrium(np.float64(1.0), np.zeros(2))
    moms = []
    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            x = np.array([float(i), float(j)])
            fi = equilibrium(np.float64(PULSE), np.zeros(2)) if abs(x @ d) < 0.7 else f_rest.copy()
            moms.append(Moment(f=fi, x=x))
    sim = Simulator(moms, tau=TAU, rho_rest=1.0, soft_mode=mode)
    for _ in range(steps):
        sim.step()
    x = np.array([m.x for m in sim.moments])
    s, t = x @ d, x @ dp  # along / across propagation
    band = np.abs(t) < half - 4
    sb, tb = s[band], t[band]
    fronts = []
    for tt in np.arange(-(half - 4), half - 3, 1.0):
        sel = np.abs(tb - tt) < 0.5
        if sel.any():
            fronts.append(sb[sel].max())
    return float(np.std(fronts)) if len(fronts) > 2 else float("nan")


def main():
    print("circular wavefront (point pulse) — anisotropy (lower = more isotropic):")
    print(f"  {'mode':>8} {'rel_std':>9} {'6-fold':>9}")
    for mode in MODES:
        rel_std, sixfold, _ = circular_anisotropy(mode)
        print(f"  {mode:>8} {rel_std:>9.4f} {sixfold:>9.4f}")

    angles = [0, 15, 30, 45, 60]
    print("\nstraight wavefront roughness vs propagation angle:")
    print(f"  {'mode':>8} " + " ".join(f"{a:>6}°" for a in angles) + f" {'mean':>7}")
    for mode in MODES:
        rough = [angled_roughness(mode, a) for a in angles]
        print(f"  {mode:>8} " + " ".join(f"{r:>7.3f}" for r in rough) + f" {np.nanmean(rough):>7.3f}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
