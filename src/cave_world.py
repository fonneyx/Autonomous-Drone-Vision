"""Procedural underground worlds: winding tunnels, branches and chambers on a 2D grid.

The cave is carved out of solid rock by random "worm" walkers. Each walker moves
forward with a slowly drifting heading, sometimes turns sharply, sometimes splits
into a side branch and sometimes opens a chamber. Everything is carved from the
home cell outwards, so every open cell is reachable from home by construction.

Coordinates: the grid is indexed free[ix, iy]; cell (ix, iy) covers world x in
[ix, ix + 1) * cell_size and y in [iy, iy + 1) * cell_size. Pure NumPy, no MuJoCo,
so layouts are cheap to generate and easy to test.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

CELL_SIZE = 1.0  # metres per layout cell
TUNNEL_HEIGHT = 3.0  # floor to ceiling, metres


@dataclass(frozen=True)
class CaveLayout:
    """Top-down cave map: True = open space, False = rock."""

    free: np.ndarray
    home: tuple[int, int]
    cell_size: float = CELL_SIZE

    @property
    def shape(self) -> tuple[int, int]:
        return self.free.shape  # type: ignore[return-value]

    @property
    def size_m(self) -> tuple[float, float]:
        return self.shape[0] * self.cell_size, self.shape[1] * self.cell_size

    def cell_centre(self, ix: int, iy: int) -> tuple[float, float]:
        return (ix + 0.5) * self.cell_size, (iy + 0.5) * self.cell_size

    @property
    def home_xy(self) -> tuple[float, float]:
        return self.cell_centre(*self.home)

    def is_free_world(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Vectorised lookup: is world point (x, y) in open space? Outside the grid is rock."""
        ix = np.floor(np.asarray(x) / self.cell_size).astype(int)
        iy = np.floor(np.asarray(y) / self.cell_size).astype(int)
        inside = (ix >= 0) & (ix < self.shape[0]) & (iy >= 0) & (iy < self.shape[1])
        result = np.zeros(ix.shape, dtype=bool)
        result[inside] = self.free[ix[inside], iy[inside]]
        return result

    def boundary_rock(self) -> np.ndarray:
        """Rock cells touching open space (8-neighbourhood): the only walls that need geometry."""
        padded = np.pad(self.free, 1, constant_values=False)
        near_free = np.zeros_like(self.free)
        nx, ny = self.shape
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                near_free |= padded[1 + dx : 1 + dx + nx, 1 + dy : 1 + dy + ny]
        return near_free & ~self.free

    def wall_boxes(self) -> list[tuple[float, float, float, float]]:
        """Boundary rock merged into few rectangles: (centre x, centre y, half x, half y) in metres.

        Greedy merge: grow each rectangle along x, then along y while the whole row
        is still unclaimed wall. Fewer boxes = faster depth ray casting.
        """
        remaining = self.boundary_rock().copy()
        nx, ny = self.shape
        boxes = []
        for iy in range(ny):
            for ix in range(nx):
                if not remaining[ix, iy]:
                    continue
                width = 1
                while ix + width < nx and remaining[ix + width, iy]:
                    width += 1
                height = 1
                while iy + height < ny and remaining[ix : ix + width, iy + height].all():
                    height += 1
                remaining[ix : ix + width, iy : iy + height] = False
                c = self.cell_size
                boxes.append(((ix + width / 2) * c, (iy + height / 2) * c, width * c / 2, height * c / 2))
        return boxes

    def reachable(self) -> np.ndarray:
        """Open cells connected to home (4-neighbourhood flood fill)."""
        seen = np.zeros_like(self.free)
        queue = deque([self.home])
        seen[self.home] = True
        nx, ny = self.shape
        while queue:
            ix, iy = queue.popleft()
            for jx, jy in ((ix + 1, iy), (ix - 1, iy), (ix, iy + 1), (ix, iy - 1)):
                if 0 <= jx < nx and 0 <= jy < ny and self.free[jx, jy] and not seen[jx, jy]:
                    seen[jx, jy] = True
                    queue.append((jx, jy))
        return seen


def _disk_offsets(radius: float) -> np.ndarray:
    r = int(math.ceil(radius))
    dx, dy = np.mgrid[-r : r + 1, -r : r + 1]
    inside = dx**2 + dy**2 <= radius**2
    return np.stack([dx[inside], dy[inside]], axis=1)


def generate_cave(
    rng: np.random.Generator,
    size: tuple[int, int] = (48, 48),
    target_free: float = 0.22,
    tunnel_radius: tuple[float, float] = (1.2, 1.9),
    chamber_radius: tuple[float, float] = (2.5, 4.0),
    turn_noise: float = 0.25,
    sharp_turn_chance: float = 0.06,
    branch_chance: float = 0.03,
    chamber_chance: float = 0.012,
    max_steps: int = 20_000,
) -> CaveLayout:
    """Carve a random tunnel network until about target_free of the grid is open.

    tunnel_radius is in cells: 1.2 gives 3 m wide tunnels, 1.9 gives ~4 m. Home is
    on the left (west) edge, with the first tunnel heading east into the rock.
    """
    nx, ny = size
    free = np.zeros(size, dtype=bool)
    margin = int(math.ceil(max(tunnel_radius[1], chamber_radius[1]))) + 1
    if nx <= 2 * margin + 2 or ny <= 2 * margin + 2:
        raise ValueError(f"Cave size {size} too small for the tunnel/chamber radius.")

    def carve(cx: float, cy: float, radius: float) -> None:
        cells = _disk_offsets(radius) + [int(round(cx)), int(round(cy))]
        cells[:, 0] = np.clip(cells[:, 0], 1, nx - 2)
        cells[:, 1] = np.clip(cells[:, 1], 1, ny - 2)
        free[cells[:, 0], cells[:, 1]] = True

    home = (margin, ny // 2)
    # walker: [x, y, heading, radius, steps left]
    walkers: list[list[float]] = [[float(home[0]), float(home[1]), 0.0, rng.uniform(*tunnel_radius), 80.0]]
    carve(home[0], home[1], walkers[0][3])

    for _ in range(max_steps):
        if free.mean() >= target_free:
            break
        if not walkers:
            # Restart from a random open cell so the network keeps growing connected
            open_cells = np.argwhere(free)
            sx, sy = open_cells[rng.integers(len(open_cells))]
            walkers.append(
                [float(sx), float(sy), rng.uniform(-math.pi, math.pi), rng.uniform(*tunnel_radius), rng.uniform(30, 120)]
            )
        for walker in list(walkers):
            x, y, heading, radius, life = walker
            heading += rng.normal(0.0, turn_noise)
            if rng.random() < sharp_turn_chance:
                heading += rng.choice([-1.0, 1.0]) * math.pi / 2
            nx_, ny_ = x + math.cos(heading), y + math.sin(heading)
            if not (margin <= nx_ <= nx - 1 - margin and margin <= ny_ <= ny - 1 - margin):
                # Bounce off the edge of the world instead of carving into the border
                heading += math.pi + rng.normal(0.0, 0.5)
                nx_, ny_ = x, y
            walker[:] = [nx_, ny_, heading, radius, life - 1]
            carve(nx_, ny_, radius)
            if rng.random() < chamber_chance:
                carve(nx_, ny_, rng.uniform(*chamber_radius))
            if rng.random() < branch_chance:
                side = heading + rng.choice([-1.0, 1.0]) * math.pi / 2
                walkers.append([nx_, ny_, side, rng.uniform(*tunnel_radius), rng.uniform(30, 120)])
            if walker[4] <= 0:
                walkers.remove(walker)

    return CaveLayout(free=free, home=home)
