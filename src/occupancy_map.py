"""2D occupancy grid map built live from range scans (depth camera row, ToF or 2D lidar).

Every cell holds a log-odds value: negative = probably free, positive = probably
wall, zero = never seen. Each scan ray lowers the cells it passes through and
raises the cell where it hits something. Log-odds makes the map robust to sensor
noise: one bad reading never erases a wall that was seen many times.

This is the same map the real drone would build, so it is independent of MuJoCo:
it only needs the sensor position, ray angles and measured ranges.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

L_FREE = -0.4  # added to cells a ray passes through
L_OCC = 0.85  # added to the cell a ray ends in
L_MIN, L_MAX = -4.0, 4.0  # clamping keeps the map able to change its mind
FREE_THRESHOLD = -0.5  # needs two "free" observations: a single dropout cannot open a wall
OCC_THRESHOLD = 0.5
SOLID_THRESHOLD = 2.5  # wall seen 3+ times: rays stop here, so dropouts cannot "see through" it

UNKNOWN_BGR = (110, 110, 110)
FREE_BGR = (235, 235, 235)
WALL_BGR = (40, 40, 40)
FRONTIER_BGR = (230, 190, 40)


class OccupancyMap:
    """Grid over [origin, origin + size_m]; cells indexed [ix, iy] like the cave layout."""

    def __init__(
        self,
        size_m: tuple[float, float],
        resolution: float = 0.25,
        origin: tuple[float, float] = (0.0, 0.0),
    ):
        self.resolution = resolution
        self.origin = np.asarray(origin, dtype=np.float64)
        self.shape = (int(math.ceil(size_m[0] / resolution)), int(math.ceil(size_m[1] / resolution)))
        self.log_odds = np.zeros(self.shape, dtype=np.float32)

    # ---------------------------------------------------------------- geometry
    def world_to_cell(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ix = np.floor((np.asarray(x) - self.origin[0]) / self.resolution).astype(int)
        iy = np.floor((np.asarray(y) - self.origin[1]) / self.resolution).astype(int)
        return ix, iy

    def cell_centres(self) -> tuple[np.ndarray, np.ndarray]:
        """World (x, y) of every cell centre, each shaped like the grid."""
        xs = self.origin[0] + (np.arange(self.shape[0]) + 0.5) * self.resolution
        ys = self.origin[1] + (np.arange(self.shape[1]) + 0.5) * self.resolution
        return np.meshgrid(xs, ys, indexing="ij")

    def _inside(self, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        return (ix >= 0) & (ix < self.shape[0]) & (iy >= 0) & (iy < self.shape[1])

    # ----------------------------------------------------------------- update
    def integrate_scan(
        self,
        sensor_xy: np.ndarray,
        angles: np.ndarray,
        ranges: np.ndarray,
        max_range: float,
    ) -> int:
        """Add one scan. angles are world-frame headings (rad), ranges horizontal metres.

        A range >= max_range means "nothing hit": the ray only clears space.
        Returns how many cells became known-free for the first time (exploration progress).
        """
        was_free = self.free_mask()
        sensor_xy = np.asarray(sensor_xy, dtype=np.float64)
        ranges = np.minimum(np.asarray(ranges, dtype=np.float64), max_range)
        directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)

        # Free space: sample every half cell along each ray, stopping short of the hit
        step = self.resolution * 0.5
        t = np.arange(int(math.ceil(max_range / step)) + 1) * step
        hit = ranges < max_range - 1e-6
        stop = np.where(hit, ranges - self.resolution * 0.5, ranges)
        along = t[None, :] < stop[:, None]
        points = sensor_xy + directions[:, None, :] * t[None, :, None]
        px, py = self.world_to_cell(points[..., 0], points[..., 1])
        inside = self._inside(px, py)
        solid = np.zeros_like(along)
        solid[inside] = self.log_odds[px[inside], py[inside]] > SOLID_THRESHOLD
        along &= np.cumsum(solid, axis=1) == 0  # nothing behind a known wall gets cleared
        fx, fy = px[along], py[along]

        # Hit cells: nudge the end point a little into the surface so it lands in the wall cell
        ends = sensor_xy + directions[hit] * (ranges[hit, None] + self.resolution * 0.25)
        ox, oy = self.world_to_cell(ends[:, 0], ends[:, 1])

        free_flat = self._unique_flat(fx, fy)
        occ_flat = self._unique_flat(ox, oy)
        free_flat = np.setdiff1d(free_flat, occ_flat, assume_unique=True)
        flat = self.log_odds.reshape(-1)
        flat[free_flat] += L_FREE
        flat[occ_flat] += L_OCC
        np.clip(self.log_odds, L_MIN, L_MAX, out=self.log_odds)
        return int((self.free_mask() & ~was_free).sum())

    def _unique_flat(self, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        inside = self._inside(ix, iy)
        return np.unique(np.ravel_multi_index((ix[inside], iy[inside]), self.shape))

    # ------------------------------------------------------------------ queries
    def free_mask(self) -> np.ndarray:
        return self.log_odds < FREE_THRESHOLD

    def occupied_mask(self) -> np.ndarray:
        return self.log_odds > OCC_THRESHOLD

    def unknown_mask(self) -> np.ndarray:
        return ~self.free_mask() & ~self.occupied_mask()

    def frontier_mask(self) -> np.ndarray:
        """Free cells next to unknown cells: the edge of explored space, where exploring continues."""
        unknown = np.pad(self.unknown_mask(), 1, constant_values=False)
        next_to_unknown = (
            unknown[2:, 1:-1] | unknown[:-2, 1:-1] | unknown[1:-1, 2:] | unknown[1:-1, :-2]
        )
        return self.free_mask() & next_to_unknown

    # ------------------------------------------------------------------ drawing
    def to_bgr(self, scale: int = 3, show_frontiers: bool = True) -> np.ndarray:
        """Top-down image, north (+y) up, east (+x) right, scale pixels per cell."""
        image = np.empty(self.shape + (3,), dtype=np.uint8)
        image[:] = UNKNOWN_BGR
        image[self.free_mask()] = FREE_BGR
        image[self.occupied_mask()] = WALL_BGR
        if show_frontiers:
            image[self.frontier_mask()] = FRONTIER_BGR
        image = np.flipud(image.transpose(1, 0, 2))  # [ix, iy] -> rows = -y, columns = x
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    def world_to_pixel(self, x: float, y: float, scale: int = 3) -> tuple[int, int]:
        """Pixel (column, row) of a world point in the to_bgr image."""
        column = (x - self.origin[0]) / self.resolution * scale
        row = (self.shape[1] - (y - self.origin[1]) / self.resolution) * scale
        return int(round(column)), int(round(row))
