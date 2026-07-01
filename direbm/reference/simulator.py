"""The DiReBM reference simulator: collision + the four-part propagation (thesis §4.3–4.4).

One full iteration, in order (matching the C++ state machine):

    collision → dispersion → create control points → refine positions → resampling

Dimension-generic: pass a `lattice` (default `D2Q7` for 2D, `D3Q13` for 3D). Obstacles work in both
(Circle / Sphere, with a generic reflected-direction split in 3D). The soft_outer spawn is 2D-only
(hex geometry); 3D uses `soft_mode="off"`. Collision is a real BGK relaxation (the C++ draft left it
a no-op; thesis §4.4 specifies the relax).
"""

from __future__ import annotations

import numpy as np

from ..constants import ALPHA, DX, KAPPA_HARD, KAPPA_SOFT
from ..lattices import D2Q7
from ..physics import collide, equilibrium, recover
from .boundary import reflect, split_direction, split_direction_nd
from .grid import Grid
from .types import Component, ControlPoint, Moment

_EPS = 1e-8
# Geometric offset for new soft-outer control points (thesis §4.3.3, eq. 4.3): the CD gap in the
# hexagon, 2·(1 − √3/2)·dx. Counteracts every-other-step hexagonal anisotropy (2D D2Q7 only).
_SOFT_OFFSET = 2.0 * (1.0 - np.sqrt(3.0) / 2.0)


class Simulator:
    def __init__(
        self,
        moments: list[Moment],
        tau: float,
        *,
        lattice=D2Q7,
        dx: float = DX,
        alpha: float = ALPHA,
        kappa_hard: int = KAPPA_HARD,
        kappa_soft: int = KAPPA_SOFT,
        rho_rest: float = 1.0,
        u_rest=None,
        obstacle=None,
        soft_mode: str = "spawn",
    ):
        self.moments = list(moments)
        self.lattice = lattice
        self.C = lattice.C
        self.Q = lattice.Q
        self.D = lattice.D
        # Optional solid obstacle (fluid outside): Circle (2D) or Sphere (3D). inside()/ray_hit().
        self.obstacle = obstacle
        # soft_outer step-3 placement (2D only): "spawn" = thesis fixed offset, "off" = none.
        self.soft_mode = soft_mode
        self.tau = float(tau)
        self.dx = float(dx)
        self.alpha = float(alpha)
        self.kappa_hard = int(kappa_hard)
        self.kappa_soft = int(kappa_soft)
        # Density-threshold radius for control-point creation: dx/α (thesis §4.3.2).
        self.r_thresh = self.dx / self.alpha - _EPS
        # Rest-state equilibrium (fills directions no component delivered, thesis §4.3.4).
        u_rest_vec = np.zeros(self.D) if u_rest is None else np.asarray(u_rest, dtype=np.float64)
        self.f_rest = equilibrium(np.float64(rho_rest), u_rest_vec, self.lattice)
        self.nu_grid = Grid(self.dx)
        self.p_grid = Grid(self.dx)
        self.iteration = 0

    def _cp(self, x):
        return ControlPoint(x=x, f=np.zeros(self.Q))

    # -- full step ---------------------------------------------------------------------------
    def step(self):
        self.collision()
        self.dispersion()
        self.create_control_points()
        self.refine_control_points()
        self.resampling()
        if self.obstacle is not None:
            self.moments = [m for m in self.moments if not self.obstacle.inside(m.x)]
        self.iteration += 1

    # -- collision (thesis §4.4) -------------------------------------------------------------
    def collision(self):
        if not self.moments:
            return
        f = collide(np.stack([m.f for m in self.moments]), self.tau, self.lattice)
        for m, fi in zip(self.moments, f, strict=True):
            m.f = fi

    # -- (1) dispersion (thesis §4.3.1) ------------------------------------------------------
    def dispersion(self):
        """Each moment explodes into Q components, each shifted dx along its direction (c_0 stays).
        Components that would enter the obstacle bounce specularly (thesis §4.5, 2D)."""
        self.nu_grid.clear()
        for m in self.moments:
            for i in range(self.Q):
                end = m.x + self.C[i] * self.dx
                if self.obstacle is not None and i != 0 and self.obstacle.inside(end):
                    self._bounce(m.x, i, float(m.f[i]))
                else:
                    self.nu_grid.insert(Component(f=float(m.f[i]), i=i, x=end))
        self.moments = []

    def _bounce(self, x0, i, fval):
        """Reflect a component off the obstacle and split it into valid lattice directions,
        conserving its mass (thesis §4.5). 2D uses the exact hex split; 3D uses the generic split."""
        hit, _, n = self.obstacle.ray_hit(x0, x0 + self.C[i] * self.dx)
        if not hit:
            return
        d = reflect(self.C[i], n)
        parts = split_direction(d) if self.D == 2 else split_direction_nd(d, self.C)
        for idx, w in parts:
            pos = x0 + self.C[idx] * self.dx
            if self.obstacle.inside(pos):
                pos = x0.copy()
            self.nu_grid.insert(Component(f=fval * w, i=idx, x=pos))

    # -- (2) create control points (thesis §4.3.2) -------------------------------------------
    def create_control_points(self):
        self.p_grid.clear()
        for nu in self.nu_grid.all():
            p = self.p_grid.insert_with_density_threshold(self._cp(nu.x.copy()), self.r_thresh)
            if p is None:  # too close to an existing control point → skip duplicate
                continue
            p.nu_near = self.nu_grid.query_radius(p.x, self.dx + _EPS)
            p.kappa = len({n.i for n in p.nu_near})  # perceived-direction count κ
            if p.kappa <= self.kappa_hard:
                p.type = "hard_outer"
            elif p.kappa <= self.kappa_soft:
                p.type = "soft_outer"
            else:
                p.type = "inner"

    # -- (3) refine control-point positions (thesis §4.3.3) ----------------------------------
    def refine_control_points(self):
        # Iterate a snapshot: new soft-outer / repositioned inner points are not re-processed.
        for p in self.p_grid.all():
            if p.type == "hard_outer":
                continue  # leave on the free surface
            if p.type == "soft_outer":
                new_x = self._soft_new_point(p)
                if new_x is not None:
                    self.p_grid.insert(self._cp(new_x))
            # inner (and fallen-through soft_outer): move to the exp(f)-weighted mean of the
            # nearby component positions → resolution follows denser material.
            self.p_grid.remove_near(p.x, _EPS)
            w = np.array([np.exp(n.f) for n in p.nu_near])
            xs = np.array([n.x for n in p.nu_near])
            p.x = (w[:, None] * xs).sum(axis=0) / w.sum()
            self.p_grid.insert(p)

    def _soft_new_point(self, p):
        """Spawn position for the soft_outer extra control point (thesis §4.3.3, 2D D2Q7 only)."""
        if self.soft_mode == "off":
            return None
        c_sum = np.zeros(self.D)
        for n in p.nu_near:
            c_sum += self.C[n.i]
        norm = np.linalg.norm(c_sum)
        if norm <= _EPS:  # balanced directions → interior-like, no front to fill
            return None
        normal = c_sum / norm
        if self.soft_mode == "spawn":
            return p.x + normal * _SOFT_OFFSET * self.dx
        raise ValueError(f"unknown soft_mode {self.soft_mode!r}")

    # -- (4) resampling (thesis §4.3.4) ------------------------------------------------------
    def resampling(self):
        # Phase 1: scatter each component's f into nearby control points, distance-weighted.
        for nu in self.nu_grid.all():
            near = self.p_grid.query_radius(nu.x, self.dx + _EPS)
            if not near:
                near = [self.p_grid.insert(self._cp(nu.x.copy()))]
            weights = [self.dx + 2.0 * _EPS - float(np.linalg.norm(p.x - nu.x)) for p in near]
            sw = sum(weights)
            for p, wgt in zip(near, weights, strict=True):
                p.f[nu.i] += (wgt / sw) * nu.f

        # Phase 2: fill directions no component delivered with the rest equilibrium, mass-scaled
        # over nearby also-empty control points so no spurious mass enters; emit a moment per CP.
        new_moments = []
        for p in self.p_grid.all():
            f = p.f.copy()
            near = None
            for i in range(self.Q):
                if f[i] < _EPS:
                    if near is None:
                        near = self.p_grid.query_radius(p.x, self.dx + _EPS)
                    sw = sum(
                        self.dx + 2.0 * _EPS - float(np.linalg.norm(p.x - q.x))
                        for q in near
                        if q.f[i] < _EPS
                    )
                    f[i] = (self.dx / sw) * self.f_rest[i]
            new_moments.append(Moment(f=f, x=p.x.copy()))
        self.moments = new_moments
        self.nu_grid.clear()

    # -- diagnostics -------------------------------------------------------------------------
    def macroscopic(self):
        """Return (positions (N,D), density (N,), velocity (N,D)) of the current moments."""
        if not self.moments:
            return np.zeros((0, self.D)), np.zeros((0,)), np.zeros((0, self.D))
        x = np.array([m.x for m in self.moments])
        f = np.stack([m.f for m in self.moments])
        rho, u = recover(f, self.lattice)
        return x, rho, u


__all__ = ["Simulator"]
