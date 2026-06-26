"""Boundary geometry: circle hit-test, specular reflection, direction split."""

import numpy as np

from direbm import equilibrium
from direbm.constants import C
from direbm.reference import Moment, Simulator
from direbm.reference.boundary import Circle, reflect, split_direction


def test_circle_inside():
    c = Circle((0.0, 0.0), 1.0)
    assert c.inside((0.0, 0.0))
    assert not c.inside((2.0, 0.0))


def test_ray_hit_enters_circle():
    c = Circle((0.0, 0.0), 1.0)
    hit, p, n = c.ray_hit((2.0, 0.0), (0.0, 0.0))
    assert hit
    assert np.allclose(p, [1.0, 0.0])
    assert np.allclose(n, [1.0, 0.0])  # outward normal at the entry point


def test_ray_miss():
    c = Circle((0.0, 0.0), 1.0)
    hit, _, _ = c.ray_hit((2.0, 2.0), (2.0, -2.0))
    assert not hit


def test_reflect_specular():
    # head-on into a wall with outward normal +x reverses the x-component
    assert np.allclose(reflect((-1.0, 0.0), (1.0, 0.0)), [1.0, 0.0])
    # grazing: normal +y reflects the y-component only
    assert np.allclose(reflect((1.0, -1.0), (0.0, 1.0)), [1.0, 1.0])


def test_split_direction_weights_sum_to_one():
    for ang in np.linspace(0, 2 * np.pi, 17):
        d = np.array([np.cos(ang), np.sin(ang)])
        parts = split_direction(d)
        assert abs(sum(w for _, w in parts) - 1.0) < 1e-12
        for idx, _ in parts:
            assert 1 <= idx <= 6


def test_split_direction_aligned_is_pure():
    # exactly along c_1 (0°) → all weight on direction 1
    parts = dict((i, w) for i, w in split_direction(C[1]))
    assert abs(parts.get(1, 0.0) - 1.0) < 1e-9
    # exactly along c_2 (60°) → all weight on direction 2
    parts = dict((i, w) for i, w in split_direction(C[2]))
    assert abs(parts.get(2, 0.0) - 1.0) < 1e-9


def test_obstacle_keeps_fluid_out():
    obs = Circle((0.0, 0.0), 2.0)
    f_rest = equilibrium(np.float64(1.0), np.zeros(2))
    moms = []
    for i in range(-6, 7):
        for j in range(-6, 7):
            x = np.array([float(i), float(j)])
            if obs.inside(x):
                continue  # fluid lives only outside the solid
            fi = equilibrium(np.float64(1.5), np.zeros(2)) if (i == 4 and j == 0) else f_rest.copy()
            moms.append(Moment(f=fi, x=x))
    sim = Simulator(moms, tau=0.6, obstacle=obs, rho_rest=1.0)
    for _ in range(5):
        sim.step()
    xs = [m.x for m in sim.moments]
    assert len(xs) > 0
    assert not any(obs.inside(x) for x in xs)  # nothing penetrates the solid
    assert np.isfinite(np.stack([m.f for m in sim.moments])).all()

