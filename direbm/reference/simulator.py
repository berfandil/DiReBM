"""The DiReBM reference simulator: collision + the four-part propagation (thesis §4.3–4.4).

One full iteration, in order (matching the C++ state machine):

    collision → dispersion → create control points → refine positions → resampling

Collision is a real BGK relaxation here. (The C++ draft left collisionStep as a no-op — its
comment claimed the moment constructor did it, but that constructor only recovers ρ,u. Thesis
§4.4 specifies the BGK relax, so the oracle does it.)
"""

from __future__ import annotations

import numpy as np

from ..constants import ALPHA, DX, KAPPA_HARD, KAPPA_SOFT, C, Q
from ..physics import collide, equilibrium
from .grid import Grid
from .types import Component, ControlPoint, Moment

_EPS = 1e-8
# Geometric offset for new soft-outer control points (thesis §4.3.3, eq. 4.3): the CD gap in the
# hexagon, 2·(1 − √3/2)·dx. Counteracts every-other-step hexagonal anisotropy of a circular wave.
_SOFT_OFFSET = 2.0 * (1.0 - np.sqrt(3.0) / 2.0)


class Simulator:
    def __init__(
        self,
        moments: list[Moment],
        tau: float,
        *,
        dx: float = DX,
        alpha: float = ALPHA,
        kappa_hard: int = KAPPA_HARD,
        kappa_soft: int = KAPPA_SOFT,
        rho_rest: float = 1.0,
        u_rest=(0.0, 0.0),
    ):
        self.moments = list(moments)
        self.tau = float(tau)
        self.dx = float(dx)
        self.alpha = float(alpha)
        self.kappa_hard = int(kappa_hard)
        self.kappa_soft = int(kappa_soft)
        # Density-threshold radius for control-point creation: dx/α (thesis §4.3.2).
        self.r_thresh = self.dx / self.alpha - _EPS
        # Rest-state equilibrium used to fill directions no component delivered (thesis §4.3.4):
        # this is the surrounding quiescent fluid, f_eq_i(ρ_rest, u_rest) = ρ_rest·W_i.
        self.f_rest = equilibrium(np.float64(rho_rest), np.asarray(u_rest, dtype=np.float64))
        self.nu_grid = Grid(self.dx)
        self.p_grid = Grid(self.dx)
        self.iteration = 0

    # -- full step ---------------------------------------------------------------------------
    def step(self):
        self.collision()
        self.dispersion()
        self.create_control_points()
        self.refine_control_points()
        self.resampling()
        self.iteration += 1

    # -- collision (thesis §4.4) -------------------------------------------------------------
    def collision(self):
        if not self.moments:
            return
        f = np.stack([m.f for m in self.moments])
        f = collide(f, self.tau)
        for m, fi in zip(self.moments, f, strict=True):
            m.f = fi

    # -- (1) dispersion (thesis §4.3.1) ------------------------------------------------------
    def dispersion(self):
        """Each moment explodes into 7 components, each shifted dx along its direction (c_0 stays)."""
        self.nu_grid.clear()
        for m in self.moments:
            for i in range(Q):
                self.nu_grid.insert(Component(f=float(m.f[i]), i=i, x=m.x + C[i] * self.dx))
        self.moments = []

    # -- (2) create control points (thesis §4.3.2) -------------------------------------------
    def create_control_points(self):
        self.p_grid.clear()
        for nu in self.nu_grid.all():
            p = self.p_grid.insert_with_density_threshold(ControlPoint(x=nu.x.copy()), self.r_thresh)
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
        # Iterate a snapshot: new soft-outer points and repositioned inner points must not be
        # re-processed this step (matches the C++ p_all_copy).
        for p in self.p_grid.all():
            if p.type == "hard_outer":
                continue  # leave on the free surface
            if p.type == "soft_outer":
                # Spawn a new control point along the summed incoming directions, then treat the
                # original as inner (the C++ switch deliberately falls through).
                c_sum = np.zeros(2)
                for n in p.nu_near:
                    c_sum += C[n.i]
                norm = np.linalg.norm(c_sum)
                if norm > _EPS:
                    new_x = p.x + (c_sum / norm) * _SOFT_OFFSET * self.dx
                    self.p_grid.insert(ControlPoint(x=new_x))
            # inner (and fallen-through soft_outer): move to the exp(f)-weighted mean of the
            # nearby component positions → resolution follows denser material.
            self.p_grid.remove_near(p.x, _EPS)
            w = np.array([np.exp(n.f) for n in p.nu_near])
            xs = np.array([n.x for n in p.nu_near])
            p.x = (w[:, None] * xs).sum(axis=0) / w.sum()
            self.p_grid.insert(p)

    # -- (4) resampling (thesis §4.3.4) ------------------------------------------------------
    def resampling(self):
        # Phase 1: scatter each component's f into nearby control points, distance-weighted.
        for nu in self.nu_grid.all():
            near = self.p_grid.query_radius(nu.x, self.dx + _EPS)
            if not near:
                near = [self.p_grid.insert(ControlPoint(x=nu.x.copy()))]
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
            for i in range(Q):
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
        """Return (positions (N,2), density (N,), velocity (N,2)) of the current moments."""
        if not self.moments:
            return np.zeros((0, 2)), np.zeros((0,)), np.zeros((0, 2))
        from ..physics import recover

        x = np.array([m.x for m in self.moments])
        f = np.stack([m.f for m in self.moments])
        rho, u = recover(f)
        return x, rho, u


__all__ = ["Simulator"]
