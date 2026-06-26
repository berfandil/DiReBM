"""Spatial-hash grid: the method's 'rács adatszerkezet' (thesis §4.2), heir of DiRe-CFD MultiGrid.

A dict of fixed-size cells keyed by integer coordinates → no fixed domain bounds (cells spring
into existence as points land in them). Stores any item exposing an `.x` position array. Radius
neighbour queries scan the cells overlapping the query disk. This v1 is the simple, correct
oracle; v2 replaces it with wp.HashGrid.
"""

from __future__ import annotations

import math

import numpy as np


class Grid:
    def __init__(self, cell_size: float):
        # Optimal cell size = dx, since neighbour searches use a dx radius (thesis §4.2.1).
        self.cell_size = float(cell_size)
        self.cells: dict[tuple[int, int], list] = {}

    def _key(self, x) -> tuple[int, int]:
        cs = self.cell_size
        return (int(math.floor(x[0] / cs)), int(math.floor(x[1] / cs)))

    def _neighbour_keys(self, x, radius):
        r = int(math.ceil(radius / self.cell_size))
        kx, ky = self._key(x)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                yield (kx + dx, ky + dy)

    def insert(self, item):
        self.cells.setdefault(self._key(item.x), []).append(item)
        return item

    def insert_with_density_threshold(self, item, radius):
        """Insert unless another item already sits within `radius`; then return None (rejected)."""
        if self.query_radius(item.x, radius):
            return None
        return self.insert(item)

    def query_radius(self, x, radius):
        x = np.asarray(x, dtype=np.float64)
        r2 = radius * radius
        out = []
        for key in self._neighbour_keys(x, radius):
            bucket = self.cells.get(key)
            if not bucket:
                continue
            for it in bucket:
                d = it.x - x
                if d[0] * d[0] + d[1] * d[1] <= r2:
                    out.append(it)
        return out

    def remove_near(self, x, radius):
        """Remove and return every item within `radius` of x."""
        x = np.asarray(x, dtype=np.float64)
        r2 = radius * radius
        removed = []
        for key in self._neighbour_keys(x, radius):
            bucket = self.cells.get(key)
            if not bucket:
                continue
            keep = []
            for it in bucket:
                d = it.x - x
                if d[0] * d[0] + d[1] * d[1] <= r2:
                    removed.append(it)
                else:
                    keep.append(it)
            if keep:
                self.cells[key] = keep
            else:
                del self.cells[key]
        return removed

    def all(self):
        out = []
        for bucket in self.cells.values():
            out.extend(bucket)
        return out

    def __len__(self):
        return sum(len(b) for b in self.cells.values())

    def clear(self):
        self.cells.clear()


__all__ = ["Grid"]
