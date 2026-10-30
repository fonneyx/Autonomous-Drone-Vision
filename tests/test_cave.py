"""Tests for Stage 1: cave generator, occupancy map, cave env and the follow-the-gap baseline.

Run: python -m pytest tests -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.baselines import follow_the_gap  # noqa: E402
from src.cave_env import CaveExploreEnv  # noqa: E402
from src.cave_world import generate_cave  # noqa: E402
from src.occupancy_map import OccupancyMap  # noqa: E402
from src.sim_env import FLIGHT_ALTITUDE, MAX_SPEED, MAX_YAW_RATE  # noqa: E402


# ------------------------------------------------------------------ cave generator
@pytest.mark.parametrize("seed", range(5))
def test_cave_is_connected_and_closed(seed: int) -> None:
    cave = generate_cave(np.random.default_rng(seed))
    assert cave.free[cave.home], "home must be open"
    assert np.array_equal(cave.reachable(), cave.free), "every open cell must be reachable from home"
    border = np.concatenate([cave.free[0], cave.free[-1], cave.free[:, 0], cave.free[:, -1]])
    assert not border.any(), "the cave must be enclosed by rock"
    assert 0.2 <= cave.free.mean() <= 0.3


def test_cave_is_deterministic_per_seed() -> None:
    a = generate_cave(np.random.default_rng(3))
    b = generate_cave(np.random.default_rng(3))
    c = generate_cave(np.random.default_rng(4))
    assert np.array_equal(a.free, b.free)
    assert not np.array_equal(a.free, c.free)


def test_wall_boxes_cover_exactly_the_boundary_rock() -> None:
    cave = generate_cave(np.random.default_rng(1))
    painted = np.zeros(cave.shape, dtype=int)
    for cx, cy, hx, hy in cave.wall_boxes():
        x0, y0 = round(cx - hx), round(cy - hy)
        painted[x0 : round(cx + hx), y0 : round(cy + hy)] += 1
    assert np.array_equal(painted == 1, cave.boundary_rock()), "boxes must tile the walls, no gaps or overlaps"
    assert painted.max() == 1


# ------------------------------------------------------------------ occupancy map
def test_single_ray_marks_free_path_and_wall() -> None:
    grid = OccupancyMap((10.0, 10.0), resolution=0.25)
    for _ in range(2):  # free needs two observations
        grid.integrate_scan(np.array([1.0, 5.0]), np.array([0.0]), np.array([4.0]), max_range=8.0)
    ix, iy = grid.world_to_cell(np.array([2.0, 4.6, 5.1]), np.array([5.0, 5.0, 5.0]))
    assert grid.free_mask()[ix[0], iy[0]], "cell along the ray should be free"
    assert grid.free_mask()[ix[1], iy[1]], "cell just before the hit should be free"
    assert grid.occupied_mask()[ix[2], iy[2]], "cell at the hit should be a wall"
    assert grid.unknown_mask()[grid.world_to_cell(np.array([7.0]), np.array([5.0]))].all(), "behind the wall is unknown"
    assert grid.frontier_mask().any()


def test_max_range_reading_is_not_a_wall() -> None:
    grid = OccupancyMap((20.0, 20.0))
    grid.integrate_scan(np.array([1.0, 10.0]), np.array([0.0]), np.array([10.0]), max_range=10.0)
    assert not grid.occupied_mask().any()


def test_dropout_cannot_see_through_a_known_wall() -> None:
    grid = OccupancyMap((20.0, 20.0))
    sensor, angle = np.array([1.0, 10.0]), np.array([0.0])
    for _ in range(5):
        grid.integrate_scan(sensor, angle, np.array([4.0]), max_range=10.0)
    for _ in range(5):  # sensor dropout: reads "nothing there"
        grid.integrate_scan(sensor, angle, np.array([10.0]), max_range=10.0)
    behind = grid.world_to_cell(np.array([7.0]), np.array([10.0]))
    assert grid.unknown_mask()[behind].all()


def test_new_cell_count_reports_progress() -> None:
    grid = OccupancyMap((10.0, 10.0))
    sensor, angles, ranges = np.array([5.0, 5.0]), np.linspace(-1, 1, 20), np.full(20, 3.0)
    assert grid.integrate_scan(sensor, angles, ranges, 8.0) == 0  # first look: not yet sure
    assert grid.integrate_scan(sensor, angles, ranges, 8.0) > 0
    assert grid.integrate_scan(sensor, angles, ranges, 8.0) == 0  # nothing new the third time


# --------------------------------------------------------------------- baseline
def test_gap_pilot_turns_towards_the_opening() -> None:
    angles = np.linspace(0.78, -0.78, 64)  # left to right
    ranges = np.full(64, 1.5)
    ranges[:16] = 10.0  # open on the left
    action = follow_the_gap(angles, ranges, MAX_SPEED, MAX_YAW_RATE)
    assert action[1] > 0.2, "should turn left"


def test_gap_pilot_stops_and_turns_in_a_dead_end() -> None:
    angles = np.linspace(0.78, -0.78, 64)
    action = follow_the_gap(angles, np.full(64, 0.8), MAX_SPEED, MAX_YAW_RATE)
    assert action[0] == pytest.approx(-1.0) and abs(action[1]) == pytest.approx(1.0)


# --------------------------------------------------------------------- cave env
@pytest.fixture(scope="module")
def env() -> CaveExploreEnv:
    return CaveExploreEnv(world_seed=7)


def test_env_passes_gymnasium_checks(env: CaveExploreEnv) -> None:
    check_env(env, skip_render_check=True)


def test_env_starts_at_home_with_some_map(env: CaveExploreEnv) -> None:
    obs, info = env.reset(seed=0)
    assert np.allclose(env.position[:2], env.layout.home_xy)
    assert env.position[2] == pytest.approx(FLIGHT_ALTITUDE)
    assert obs["depth"].shape == (1, 48, 64) and obs["state"].shape == (6,)
    assert env.map.occupied_mask().any(), "the first scan should already show walls"


def test_flying_into_a_wall_crashes(env: CaveExploreEnv) -> None:
    env.reset(seed=0)
    env.data.qpos[3] = math.pi  # face the rock behind the home pad (home is on the west edge)
    info: dict = {}
    for _ in range(100):
        _, reward, terminated, _, info = env.step(np.array([1.0, 0.0]))
        if terminated:
            break
    assert info["collision"] and reward < -10


def test_gap_pilot_explores_without_crashing(env: CaveExploreEnv) -> None:
    env.reset(seed=0)
    info: dict = {}
    for _ in range(400):
        _, _, terminated, truncated, info = env.step(follow_the_gap(*env.scan(), MAX_SPEED, MAX_YAW_RATE))
        if terminated or truncated:
            break
    assert not info["collision"]
    assert info["coverage"] > 0.3
    assert env.wall_precision() > 0.95, "mapped walls must match the real cave"
