"""Frontier-based exploration: decide where to fly next using only the live map.

A frontier is a known-free cell next to unknown space (the edge of what has been
mapped). Flying to frontiers keeps revealing new space until none are left
(Yamauchi, 1997, "A frontier-based approach for autonomous exploration").

Every few control steps the explorer:
1. builds the "safe" grid: known-free cells at least `clearance` away from walls,
2. computes the path distance from the drone to every safe cell (grid wavefront),
3. groups frontier cells into connected clusters and drops clusters narrower than
   the drone (those are gaps in a scanned wall, not openings into new space),
4. scores safe cells next to a cluster: path distance minus a bonus for the
   cluster's width (wide frontier = big opening = much to discover),
5. walks the path back from the best target and hands the pilot a waypoint a few
   metres ahead that is in direct line of sight.

Targets the drone reaches, or cannot reach within a time limit, are blacklisted so
it never circles one spot forever. When no target is left, the drone first turns a
full circle on the spot (the camera only looks forward) and the cave only counts as
explored if that look-around reveals nothing new. The explorer only reads the OccupancyMap and the
drone position, so the same code runs in simulation and on the real drone's laptop.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from src.occupancy_map import OccupancyMap

# 8-connected neighbour offsets
_NEIGHBOURS = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]


def _shift(grid: np.ndarray, dx: int, dy: int, fill: float | bool) -> np.ndarray:
    """grid shifted so result[x, y] = grid[x - dx, y - dy] (out of range -> fill)."""
    out = np.full_like(grid, fill)
    nx, ny = grid.shape
    out[max(0, dx) : nx + min(0, dx), max(0, dy) : ny + min(0, dy)] = grid[
        max(0, -dx) : nx + min(0, -dx), max(0, -dy) : ny + min(0, -dy)
    ]
    return out


def _disk(radius_cells: int) -> list[tuple[int, int]]:
    r = radius_cells
    return [(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1) if dx * dx + dy * dy <= r * r]


def dilate(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    """Grow True regions by a disk of radius_cells."""
    out = mask.copy()
    for dx, dy in _disk(radius_cells):
        out |= _shift(mask, dx, dy, False)
    return out


def max_filter(values: np.ndarray, radius_cells: int) -> np.ndarray:
    """Largest value within a disk of radius_cells around each cell."""
    out = values.copy()
    for dx, dy in _disk(radius_cells):
        np.maximum(out, _shift(values, dx, dy, 0), out=out)
    return out


def cluster_sizes(mask: np.ndarray) -> np.ndarray:
    """For every True cell, the number of cells in its 8-connected cluster (0 elsewhere)."""
    sizes = np.zeros(mask.shape, dtype=np.int32)
    seen = np.zeros_like(mask)
    nx, ny = mask.shape
    for start in map(tuple, np.argwhere(mask)):
        if seen[start]:
            continue
        seen[start] = True
        queue, members = deque([start]), [start]
        while queue:
            x, y = queue.popleft()
            for dx, dy in _NEIGHBOURS:
                cell = (x + dx, y + dy)
                if 0 <= cell[0] < nx and 0 <= cell[1] < ny and mask[cell] and not seen[cell]:
                    seen[cell] = True
                    queue.append(cell)
                    members.append(cell)
        xs, ys = zip(*members)
        sizes[list(xs), list(ys)] = len(members)
    return sizes


def wavefront(
    passable: np.ndarray, start: tuple[int, int], cost: np.ndarray | None = None, max_iterations: int = 4000
) -> np.ndarray:
    """Path cost (in cells) from start to every passable cell; inf where unreachable.

    Vectorised grid Dijkstra with 8-connected moves (diagonals cost sqrt 2). Each
    sweep relaxes every cell from its neighbours; it stops when nothing changes.
    cost: optional per-cell multiplier for entering a cell (1 = normal), used to make
    paths avoid areas without forbidding them.
    """
    distance = np.full(passable.shape, np.inf)
    distance[start] = 0.0
    weight = np.ones(passable.shape) if cost is None else cost
    for _ in range(max_iterations):
        best = distance.copy()
        for dx, dy in _NEIGHBOURS:
            step = math.sqrt(2.0) if dx and dy else 1.0
            np.minimum(best, _shift(distance, dx, dy, np.inf) + step * weight, out=best)
        best[~passable] = np.inf
        best[start] = 0.0
        if np.array_equal(best, distance):
            break
        distance = best
    return distance


class FrontierExplorer:
    """Pick frontier targets on an OccupancyMap and produce waypoints for the pilot."""

    def __init__(
        self,
        grid: OccupancyMap,
        clearance: float = 0.5,
        min_frontier_width: float = 0.75,
        width_weight: float = 1.0,
        min_target_distance: float = 1.5,
        lookahead: float = 2.5,
        replan_every: int = 5,
        reached_distance: float = 1.0,
        give_up_after: int = 200,
        look_around_steps: int = 75,
        home: tuple[float, float] | None = None,
    ):
        """Distances in metres.

        clearance: min distance from walls for planned paths. min_frontier_width: narrower
        frontier clusters are wall-scan gaps, not openings. width_weight: metres of extra
        path accepted per metre of frontier width. min_target_distance: closer targets are
        skipped (with a forward-only camera the unknown space right behind the drone would
        otherwise always win and the drone would never move). give_up_after: control steps
        before an unreached target is blacklisted. look_around_steps: control steps of
        turning on the spot before giving up (75 steps at the pilot's turn rate is ~360 deg).
        home: world (x, y) of the take-off point; every plan then also measures the flight
        distance home (`distance_home`), and go_home() switches from exploring to returning.
        """
        self.map = grid
        r = grid.resolution
        self.clearance_cells = max(1, round(clearance / r))
        self.min_frontier_cells = max(1, round(min_frontier_width / r))
        self.width_weight = width_weight
        self.min_target_cells = min_target_distance / r
        self.lookahead = lookahead
        self.replan_every = replan_every
        self.reached_distance = reached_distance
        self.give_up_after = give_up_after
        self.look_around_steps = look_around_steps

        self.target: tuple[int, int] | None = None  # chosen target cell
        self.path: list[tuple[float, float]] = []  # world points, drone -> target
        self.waypoint: np.ndarray | None = None
        self.done = False  # no reachable frontier left: exploration finished
        self.blacklist = np.zeros(grid.shape, dtype=bool)
        # Soft keep-out zones (e.g. around people): paths avoid them when any other way
        # exists, and pass through only when they block the only passage
        self.keep_out = np.zeros(grid.shape, dtype=bool)
        self.keep_out_cost = 6.0
        self.home = None if home is None else np.asarray(home, dtype=np.float64)
        self.distance_home = math.inf  # path length home (m) from the latest plan
        self.returning = False  # True after go_home(): plan to home instead of frontiers
        self._safe: np.ndarray | None = None  # safe grid from the latest plan
        self._steps_since_plan = replan_every
        self._steps_on_target = 0
        self._look_around = 0  # steps spent turning on the spot since the last target
        self._last_xy: np.ndarray | None = None

    # ------------------------------------------------------------------ planning
    def safe_mask(self) -> np.ndarray:
        """Known-free cells with at least `clearance` to the nearest mapped wall."""
        return self.map.free_mask() & ~dilate(self.map.occupied_mask(), self.clearance_cells)

    def add_keep_out(self, xy: np.ndarray, radius: float) -> None:
        """Avoid the disk of `radius` metres around world point xy (soft: see keep_out_cost)."""
        cx, cy = self.map.cell_centres()
        self.keep_out |= np.hypot(cx - xy[0], cy - xy[1]) < radius

    def _start_cell(self, xy: np.ndarray, safe: np.ndarray) -> tuple[int, int] | None:
        """Drone cell, or the nearest safe cell within 2 m if the drone hugs a wall or keep-out zone."""
        ix, iy = (int(v) for v in self.map.world_to_cell(xy[0], xy[1]))
        if 0 <= ix < safe.shape[0] and 0 <= iy < safe.shape[1] and safe[ix, iy]:
            return ix, iy
        reach = round(2.0 / self.map.resolution)
        x0, y0 = max(0, ix - reach), max(0, iy - reach)
        window = safe[x0 : ix + reach + 1, y0 : iy + reach + 1]
        cells = np.argwhere(window)
        if len(cells) == 0:
            return None
        nearest = cells[np.argmin(((cells + [x0, y0] - [ix, iy]) ** 2).sum(1))]
        return int(nearest[0] + x0), int(nearest[1] + y0)

    def frontier_widths(self) -> np.ndarray:
        """Frontier cells labelled with their cluster size in cells; small clusters removed."""
        sizes = cluster_sizes(self.map.frontier_mask())
        sizes[sizes < self.min_frontier_cells] = 0
        return sizes

    def plan(self, xy: np.ndarray) -> None:
        """Choose a frontier target and the path to it from position xy."""
        safe = self.safe_mask()
        self._safe = safe
        start = self._start_cell(xy, safe)
        if start is None:
            self.path, self.waypoint = [], None
            return
        passable = safe.copy()
        passable[start] = True
        distance = wavefront(passable, start, np.where(self.keep_out, self.keep_out_cost, 1.0))
        if self.home is not None:
            home_cell = self._nearest_reachable(distance, self.home)
            self.distance_home = math.inf if home_cell is None else distance[home_cell] * self.map.resolution
            if self.returning:
                if home_cell is not None:
                    self.target = home_cell
                    self.path = self._trace_path(distance, home_cell) + [tuple(self.home)]
                return

        # Targets: reachable safe cells next to a wide-enough frontier, not blacklisted
        width = max_filter(self.frontier_widths(), self.clearance_cells + 1)
        candidates = (width > 0) & np.isfinite(distance) & ~self.blacklist
        # Frontiers inside a keep-out zone (e.g. the blind spot behind a person) only as a last resort
        if (candidates & ~self.keep_out).any():
            candidates &= ~self.keep_out
        if (candidates & (distance >= self.min_target_cells)).any():
            candidates &= distance >= self.min_target_cells
        if not candidates.any():
            self.target, self.path, self.waypoint = None, [], None
            self.done = self._look_around >= self.look_around_steps
            return
        self._look_around = 0
        r = self.map.resolution
        cost = np.where(candidates, (distance - self.width_weight * width) * r, np.inf)
        # Hysteresis: keep the current target unless another is clearly better
        if self.target is not None and candidates[self.target]:
            cost[self.target] -= 1.5
        target = np.unravel_index(int(np.argmin(cost)), cost.shape)
        target = (int(target[0]), int(target[1]))
        if target != self.target:
            self._steps_on_target = 0
        self.target = target
        self.path = self._trace_path(distance, target)
        self.done = False

    def _nearest_reachable(self, distance: np.ndarray, xy: np.ndarray) -> tuple[int, int] | None:
        """Reachable cell closest (straight line) to world point xy, within 1.5 m."""
        reachable = np.argwhere(np.isfinite(distance))
        if len(reachable) == 0:
            return None
        cx, cy = self.map.world_to_cell(xy[0], xy[1])
        offsets = ((reachable - [int(cx), int(cy)]) ** 2).sum(1)
        best = int(np.argmin(offsets))
        if math.sqrt(offsets[best]) * self.map.resolution > 1.5:
            return None
        return int(reachable[best][0]), int(reachable[best][1])

    def go_home(self) -> None:
        """Stop exploring and fly back to `home` (battery low or cave explored)."""
        if self.home is None:
            raise ValueError("FrontierExplorer was created without a home position.")
        self.returning = True
        self.done = False
        self.path = []
        self._steps_since_plan = self.replan_every

    @property
    def at_home(self) -> bool:
        return self.returning and bool(self.path) and self._last_xy is not None and (
            np.linalg.norm(self._last_xy - self.home) < self.reached_distance
        )

    def _trace_path(self, distance: np.ndarray, target: tuple[int, int]) -> list[tuple[float, float]]:
        """Follow the distance field downhill from target to the drone; return world points."""
        cells = [target]
        current = target
        nx, ny = distance.shape
        for _ in range(nx * ny):
            if distance[current] == 0.0:
                break
            options = [
                (current[0] + dx, current[1] + dy)
                for dx, dy in _NEIGHBOURS
                if 0 <= current[0] + dx < nx and 0 <= current[1] + dy < ny
            ]
            current = min(options, key=lambda cell: distance[cell])
            cells.append(current)
        cells.reverse()
        r = self.map.resolution
        return [(self.map.origin[0] + (cx + 0.5) * r, self.map.origin[1] + (cy + 0.5) * r) for cx, cy in cells]

    def _line_of_sight(self, a: np.ndarray, b: np.ndarray, safe: np.ndarray) -> bool:
        samples = max(2, int(np.linalg.norm(b - a) / (self.map.resolution * 0.5)))
        points = a + (b - a) * np.linspace(0.0, 1.0, samples)[:, None]
        ix, iy = self.map.world_to_cell(points[:, 0], points[:, 1])
        inside = (ix >= 0) & (ix < safe.shape[0]) & (iy >= 0) & (iy < safe.shape[1])
        return bool(inside.all() and safe[ix, iy].all())

    def _choose_waypoint(self, xy: np.ndarray) -> None:
        """Farthest path point within lookahead that the drone can see in a straight line."""
        if not self.path:
            self.waypoint = None
            return
        points = np.asarray(self.path)
        # Drop the part of the path already behind the drone
        nearest = int(np.argmin(np.linalg.norm(points - xy, axis=1)))
        ahead = points[nearest:]
        safe = self._safe if self._safe is not None else self.safe_mask()
        if self.keep_out.any():
            # Shortcuts must not cut through a keep-out zone the path itself goes around
            on_path = np.zeros_like(safe)
            px, py = self.map.world_to_cell(points[:, 0], points[:, 1])
            inside = (px >= 0) & (px < safe.shape[0]) & (py >= 0) & (py < safe.shape[1])
            on_path[px[inside], py[inside]] = True
            safe = safe & ~(self.keep_out & ~dilate(on_path, 1))
        waypoint = ahead[0]
        for point in ahead[1:]:
            if np.linalg.norm(point - xy) > self.lookahead:
                break
            if self._line_of_sight(xy, point, safe):
                waypoint = point
        if len(ahead) > 1 and np.linalg.norm(waypoint - xy) < 0.3:
            waypoint = ahead[min(len(ahead) - 1, 2)]  # never stall on a point we are already at
        self.waypoint = np.asarray(waypoint, dtype=np.float64)

    def _retire_target(self) -> None:
        """Blacklist the area around the current target (reached or unreachable)."""
        if self.target is None:
            return
        mark = np.zeros_like(self.blacklist)
        mark[self.target] = True
        self.blacklist |= dilate(mark, round(self.reached_distance / self.map.resolution))
        self.target, self.path = None, []

    # ------------------------------------------------------------------ main API
    def update(self, xy: np.ndarray) -> np.ndarray | None:
        """Call every control step with the drone (x, y). Returns the waypoint or None when done."""
        xy = np.asarray(xy, dtype=np.float64)[:2]
        self._last_xy = xy
        self._steps_since_plan += 1
        self._steps_on_target += 1
        reached = bool(self.path) and np.linalg.norm(np.asarray(self.path[-1]) - xy) < self.reached_distance
        if not self.returning and (reached or self._steps_on_target > self.give_up_after):
            self._retire_target()
            self._steps_on_target = 0
        if self._steps_since_plan >= self.replan_every or not self.path:
            self.plan(xy)
            self._steps_since_plan = 0
        if not self.path and not self.done:
            # Nothing to fly to yet: look around. A waypoint 1 m away that keeps
            # circling the drone makes any pilot turn on the spot.
            angle = 0.12 * self._look_around
            self._look_around += 1
            self.waypoint = xy + [math.cos(angle), math.sin(angle)]
            return self.waypoint
        self._choose_waypoint(xy)
        return self.waypoint
