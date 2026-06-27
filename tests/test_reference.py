"""Reference solver: grid behaviour + a single-moment spread sanity run."""

import numpy as np

from direbm import equilibrium
from direbm.reference import Grid, Moment, Simulator
from direbm.reference.types import Component


def _comp(x, y):
    return Component(f=1.0, i=0, x=np.array([float(x), float(y)]))


def test_grid_query_radius():
    g = Grid(cell_size=1.0)
    g.insert(_comp(0.0, 0.0))
    g.insert(_comp(0.5, 0.0))
    g.insert(_comp(5.0, 5.0))
    near = g.query_radius(np.array([0.0, 0.0]), 1.0)
    assert len(near) == 2  # the two near the origin, not the far one


def test_grid_density_threshold():
    g = Grid(cell_size=1.0)
    assert g.insert_with_density_threshold(_comp(0.0, 0.0), radius=0.25) is not None
    # within threshold → rejected
    assert g.insert_with_density_threshold(_comp(0.1, 0.0), radius=0.25) is None
    # outside threshold → accepted
    assert g.insert_with_density_threshold(_comp(0.5, 0.0), radius=0.25) is not None
    assert len(g) == 2


def test_grid_remove_near():
    g = Grid(cell_size=1.0)
    g.insert(_comp(0.0, 0.0))
    g.insert(_comp(3.0, 0.0))
    removed = g.remove_near(np.array([0.0, 0.0]), 1e-6)
    assert len(removed) == 1
    assert len(g) == 1


def _central_pulse(rho=1.5):
    return [Moment(f=equilibrium(np.float64(rho), np.zeros(2)), x=np.zeros(2))]


def test_single_moment_spreads_and_stays_finite():
    sim = Simulator(_central_pulse(), tau=0.6)
    for _ in range(8):
        sim.step()
    x, rho, _ = sim.macroscopic()
    assert sim.iteration == 8
    assert len(x) > 1  # it spread into many moments
    assert np.isfinite(rho).all()
    assert (rho > 0).all()
    # spread radius grew to roughly the number of iterations (unit dispersion per step)
    assert x.shape[0] > 0
    assert np.linalg.norm(x, axis=1).max() > 3.0


def test_spread_is_roughly_centered():
    # From a single central pulse, the moment cloud's centroid should stay near the origin.
    sim = Simulator(_central_pulse(), tau=0.6)
    for _ in range(8):
        sim.step()
    x, _, _ = sim.macroscopic()
    centroid = x.mean(axis=0)
    assert np.linalg.norm(centroid) < 1.0


def _circular_anisotropy(mode, steps=10, sectors=36):
    sim = Simulator(_central_pulse(1.6), tau=0.6, soft_mode=mode)
    for _ in range(steps):
        sim.step()
    x, _, _ = sim.macroscopic()
    r = np.linalg.norm(x, axis=1)
    th = np.arctan2(x[:, 1], x[:, 0]) % (2 * np.pi)
    edges = np.linspace(0, 2 * np.pi, sectors + 1)
    front = np.array([r[(th >= edges[k]) & (th < edges[k + 1])].max(initial=0.0) for k in range(sectors)])
    return front.std() / front.mean()


def test_soft_spawn_reduces_circular_anisotropy():
    # The soft_outer step-3 spawn exists to counter D2Q7 hexagonal anisotropy: a point-pulse front
    # must be more isotropic with the spawn than without it (see exp_soft_outer).
    assert _circular_anisotropy("spawn") < _circular_anisotropy("off")
