"""The three object classes that replace the lattice (thesis §4.1): Moment, Component, ControlPoint."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..constants import Q


@dataclass
class Moment:
    """A full distribution at a point (µ): 7 f-values + position."""

    f: np.ndarray  # shape (Q,)
    x: np.ndarray  # shape (D,)


@dataclass
class Component:
    """One f-value of a moment plus a position (ν): a group of particles moving in direction i."""

    f: float
    i: int  # direction index into C
    x: np.ndarray  # shape (D,)


@dataclass
class ControlPoint:
    """A distinguished location (p) where a new moment will be born (thesis §4.1)."""

    x: np.ndarray  # shape (D,)
    nu_near: list[Component] = field(default_factory=list)  # components within dx
    kappa: int = 0  # perceived-direction count |distinct ν.i in nu_near|
    type: str = "unknown"  # "hard_outer" | "soft_outer" | "inner"
    f: np.ndarray = field(default_factory=lambda: np.zeros(Q, dtype=np.float64))


__all__ = ["Moment", "Component", "ControlPoint"]
