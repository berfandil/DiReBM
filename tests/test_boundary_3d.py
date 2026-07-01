"""3D obstacles: sphere hit-test, generic direction split, and fluid kept out of a sphere."""

import numpy as np

from direbm import D3Q13, equilibrium
from direbm.reference import Moment, Simulator, Sphere, split_direction_nd


def test_sphere_hit_and_normal():
    s = Sphere((0.0, 0.0, 0.0), 1.0)
    assert s.inside((0.0, 0.0, 0.0))
    assert not s.inside((2.0, 0.0, 0.0))
    hit, p, n = s.ray_hit((2.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert hit
    assert np.allclose(p, [1.0, 0.0, 0.0])
    assert np.allclose(n, [1.0, 0.0, 0.0])


def test_split_direction_nd_conserves_and_points_forward():
    d = np.array([0.3, -0.7, 0.6])
    d /= np.linalg.norm(d)
    parts = split_direction_nd(d, D3Q13.C)
    assert abs(sum(w for _, w in parts) - 1.0) < 1e-9  # mass conserved
    # the weighted direction points roughly along d (forward), not backward
    v = sum(w * D3Q13.C[i] for i, w in parts)
    assert float(v @ d) > 0.0


def test_sphere_keeps_fluid_out_3d():
    obs = Sphere((0.0, 0.0, 0.0), 1.0)
    f_rest = equilibrium(np.float64(1.0), np.zeros(3), lattice=D3Q13)
    pulse = equilibrium(np.float64(1.6), np.zeros(3), lattice=D3Q13)
    moms = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                x = np.array([float(i), float(j), float(k)])
                if obs.inside(x):
                    continue  # fluid only outside the solid
                moms.append(Moment(f=pulse if (i, j, k) == (2, 0, 0) else f_rest.copy(), x=x))
    sim = Simulator(moms, tau=0.6, lattice=D3Q13, soft_mode="off", rho_rest=1.0, obstacle=obs)
    sim.step()
    xs = [m.x for m in sim.moments]
    assert len(xs) > 0
    assert not any(obs.inside(x) for x in xs)  # nothing penetrates the sphere
    assert np.isfinite(np.stack([m.f for m in sim.moments])).all()
