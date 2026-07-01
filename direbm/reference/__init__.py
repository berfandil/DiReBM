"""v1 reference solver — plain Python/numpy, optimized for debuggability. The correctness oracle.

Faithful to the thesis method (research/idea.md §3), with corrections where the thesis/C++ were
inconsistent (equilibrium cs², real BGK collision). The Warp GPU port (v2) validates against this.
"""

from .boundary import Circle, Sphere, reflect, split_direction, split_direction_nd
from .grid import Grid
from .simulator import Simulator
from .types import Component, ControlPoint, Moment

__all__ = [
    "Grid",
    "Simulator",
    "Moment",
    "Component",
    "ControlPoint",
    "Circle",
    "Sphere",
    "reflect",
    "split_direction",
    "split_direction_nd",
]
