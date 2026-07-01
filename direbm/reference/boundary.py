"""Boundary handling (thesis §4.5): components that strike a surface bounce specularly.

DiReBM's claimed advantage over LBM is easy arbitrary surfaces. A component travelling in
direction c_i that would cross a surface is reflected about the surface normal; the reflected
direction d' is generally not one of the seven c_i, so it is split into the two adjacent lattice
directions with linear-interpolation weights — conserving the component's mass.

Surfaces expose `inside(p)` and `ray_hit(x0, x1) -> (hit, point, normal)` (outward normal). A
`Circle` obstacle (fluid outside) is provided; other shapes just implement the same two methods.
"""

from __future__ import annotations

import numpy as np

# Lattice direction angles: c_1..c_6 sit at 0°,60°,…,300°; angle a·60° → direction index a+1.
_SECTOR = np.pi / 3.0


class Circle:
    """A circular obstacle; fluid lives outside it. The math is dimension-generic (see `Sphere`)."""

    def __init__(self, center, radius):
        self.center = np.asarray(center, dtype=np.float64)
        self.radius = float(radius)

    def inside(self, p):
        d = np.asarray(p, dtype=np.float64) - self.center
        return float(d @ d) < self.radius * self.radius

    def ray_hit(self, x0, x1):
        """First crossing of segment x0→x1 with the circle. Returns (hit, point, outward normal)."""
        x0 = np.asarray(x0, dtype=np.float64)
        d = np.asarray(x1, dtype=np.float64) - x0
        f = x0 - self.center
        a = float(d @ d)
        b = 2.0 * float(f @ d)
        c = float(f @ f) - self.radius * self.radius
        disc = b * b - 4.0 * a * c
        if a == 0.0 or disc < 0.0:
            return False, None, None
        sq = np.sqrt(disc)
        for t in ((-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)):
            if 0.0 <= t <= 1.0:
                p = x0 + t * d
                return True, p, (p - self.center) / self.radius
        return False, None, None


class Sphere(Circle):
    """A spherical obstacle in 3D; fluid outside. Reuses Circle's dimension-generic hit-test."""


def reflect(c, n):
    """Specular reflection of direction c about a surface with outward normal n (any dimension)."""
    c = np.asarray(c, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    return c - 2.0 * (c @ n) * n


def split_direction(d):
    """Split an arbitrary 2D direction into the two adjacent D2Q7 directions (indices 1..6) with
    linear-interpolation weights summing to 1."""
    theta = np.arctan2(d[1], d[0]) % (2.0 * np.pi)
    seg = theta / _SECTOR
    k = int(np.floor(seg))
    frac = seg - k
    a = k % 6
    b = (k + 1) % 6
    return [(a + 1, 1.0 - frac), (b + 1, frac)]


def split_direction_nd(d, C):
    """Split an arbitrary reflected direction d among the lattice's non-rest directions, for any
    dimension. Weight each direction by its forward alignment (max(d·c_i, 0))², normalized to sum
    to 1 → the bounced mass moves roughly along d (away from the surface), conserving mass. Returns
    a list of (direction index into C, weight)."""
    d = np.asarray(d, dtype=np.float64)
    dots = C[1:] @ d  # alignment with each non-rest direction
    w = np.maximum(dots, 0.0) ** 2
    s = w.sum()
    if s <= 0.0:  # d opposes every direction (shouldn't happen) → send along the least-opposed one
        return [(int(np.argmax(dots)) + 1, 1.0)]
    w = w / s
    return [(i + 1, float(wi)) for i, wi in enumerate(w) if wi > 1e-9]


__all__ = ["Circle", "Sphere", "reflect", "split_direction", "split_direction_nd"]
