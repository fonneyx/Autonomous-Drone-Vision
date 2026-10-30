"""Tests for Stage 2: frontier explorer, look-around, safety shield and a full run.

Run: python -m pytest tests -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.baselines import follow_the_gap  # noqa: E402
from src.cave_env import CaveExploreEnv  # noqa: E402
from src.explorer import FrontierExplorer, cluster_sizes, dilate, max_filter, wavefront  # noqa: E402
from src.occupancy_map import L_MAX, L_MIN, OccupancyMap  # noqa: E402
from src.safety import SafetyShield  # noqa: E402
from src.sim_env import MAX_SPEED, MAX_YAW_RATE  # noqa: E402


# ------------------------------------------------------------------ grid helpers
def test_wavefront_goes_around_a_wall() -> None:
    passable = np.ones((10, 10), dtype=bool)
    passable[5, :8] = False  # wall with a gap at the top
    distance = wavefront(passable, (2, 2))
    assert distance[2, 2] == 0
    assert np.isinf(distance[5, 3])
    straight = 6  # (2,2) -> (8,2) through the wall would be 6 cells
    assert distance[8, 2] > straight + 6, "path must detour through the gap"
    assert np.isfinite(distance[8, 2])


def test_wavefront_marks_unreachable_cells() -> None:
    passable = np.ones((8, 8), dtype=bool)
    passable[4, :] = False
    assert np.isinf(wavefront(passable, (1, 1))[6, 6])


def test_cluster_sizes_and_filters() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[1, 1:5] = True  # cluster of 4
    mask[7, 7] = True  # lone cell
    sizes = cluster_sizes(mask)
    assert sizes[1, 2] == 4 and sizes[7, 7] == 1 and sizes[0, 0] == 0
    assert max_filter(sizes, 1)[2, 2] == 4
    assert dilate(mask, 1)[8, 7] and not dilate(mask, 1)[8, 8], "radius-1 disk has no diagonals"


# --------------------------------------------------------------- target choice
def corridor_map() -> OccupancyMap:
    """5 m x 20 m grid: a known-free corridor x 0..6 m that continues into unknown space."""
    grid = OccupancyMap((20.0, 5.0), resolution=0.25)
    grid.log_odds[:, :] = 0.0
    grid.log_odds[: round(6 / 0.25), 4:16] = L_MIN  # free corridor, 3 m wide
    grid.log_odds[: round(6 / 0.25), 2:4] = L_MAX  # walls either side
    grid.log_odds[: round(6 / 0.25), 16:18] = L_MAX
    return grid


def test_explorer_heads_for_the_open_end() -> None:
    grid = corridor_map()
    explorer = FrontierExplorer(grid)
    waypoint = explorer.update(np.array([1.0, 2.5]))
    assert not explorer.done
    assert explorer.path, "a path to the frontier is expected"
    assert explorer.path[-1][0] > 4.5, "target should be at the corridor's open end (x = 6 m)"
    assert waypoint is not None and waypoint[0] > 1.5, "waypoint should lead along the corridor"


def test_tiny_frontier_is_ignored() -> None:
    grid = corridor_map()
    grid.log_odds[round(6 / 0.25) :, :] = L_MAX  # close the end: corridor is fully known
    grid.log_odds[10, 3] = 0.0  # one unseen wall cell = a 1-cell frontier (scan gap)
    explorer = FrontierExplorer(grid)
    explorer.update(np.array([1.0, 2.5]))
    assert explorer.target is None, "a one-cell scan gap is not an opening"


def test_look_around_before_giving_up() -> None:
    grid = corridor_map()
    grid.log_odds[round(6 / 0.25) :, :] = L_MAX  # nothing left to explore
    explorer = FrontierExplorer(grid, look_around_steps=10)
    xy = np.array([1.0, 2.5])
    angles = []
    for _ in range(10):
        waypoint = explorer.update(xy)
        assert waypoint is not None and not explorer.done, "should turn on the spot first"
        angles.append(math.atan2(*(waypoint - xy)[::-1]))
    assert len(set(np.round(angles, 3))) == 10, "look-around waypoint must circle the drone"
    for _ in range(explorer.replan_every + 1):
        waypoint = explorer.update(xy)
    assert explorer.done and waypoint is None


def test_unreachable_target_gets_blacklisted() -> None:
    grid = corridor_map()
    explorer = FrontierExplorer(grid, give_up_after=20)
    xy = np.array([1.0, 2.5])
    explorer.update(xy)
    first = explorer.target
    for _ in range(25):  # drone never moves
        explorer.update(xy)
    assert explorer.blacklist[first], "a target not reached in time must be blacklisted"


# --------------------------------------------------------------- pilot + target
def test_gap_pilot_turns_to_a_waypoint_behind() -> None:
    angles = np.linspace(0.78, -0.78, 64)
    action = follow_the_gap(angles, np.full(64, 8.0), MAX_SPEED, MAX_YAW_RATE, target_angle=-2.5)
    assert action[0] == pytest.approx(-1.0), "stop before turning around"
    assert action[1] < 0, "turn right towards the waypoint"


def test_gap_pilot_prefers_gap_towards_waypoint() -> None:
    angles = np.linspace(0.78, -0.78, 64)
    ranges = np.full(64, 8.0)  # wide open: the waypoint direction should win
    left = follow_the_gap(angles, ranges, MAX_SPEED, MAX_YAW_RATE, target_angle=0.5)
    right = follow_the_gap(angles, ranges, MAX_SPEED, MAX_YAW_RATE, target_angle=-0.5)
    assert left[1] > 0.2 and right[1] < -0.2


# ----------------------------------------------------------------- safety shield
def test_shield_passes_safe_actions_through() -> None:
    shield = SafetyShield(MAX_SPEED, MAX_YAW_RATE)
    angles = np.linspace(0.78, -0.78, 64)
    action = np.array([0.8, -0.3], dtype=np.float32)
    assert np.allclose(shield.filter(action, angles, np.full(64, 5.0)), action)
    assert shield.override_rate == 0.0


def test_shield_overrides_when_a_wall_is_close() -> None:
    shield = SafetyShield(MAX_SPEED, MAX_YAW_RATE)
    angles = np.linspace(0.78, -0.78, 64)
    ranges = np.full(64, 5.0)
    ranges[30:34] = 0.7  # wall dead ahead
    full_speed_ahead = np.array([1.0, 0.0], dtype=np.float32)
    safe_action = shield.filter(full_speed_ahead, angles, ranges)
    assert safe_action[0] < 0.0, "shield must slow the drone down"
    assert shield.override_rate == 1.0


def test_shield_speed_limit() -> None:
    shield = SafetyShield(MAX_SPEED, MAX_YAW_RATE, speed_limit=1.0)
    action = shield.filter(np.array([1.0, 0.0]), np.linspace(0.78, -0.78, 64), np.full(64, 9.0))
    assert (action[0] + 1.0) / 2.0 * MAX_SPEED == pytest.approx(1.0)


# ------------------------------------------------------------------- full run
def test_explorer_beats_wandering_in_a_cave() -> None:
    """Same caves, same 60 s: frontier exploration must map clearly more than wandering on average."""

    def run(seed: int, explore: bool) -> tuple[float, bool]:
        env = CaveExploreEnv(world_seed=seed, max_steps=600)
        env.reset(seed=seed)
        explorer = FrontierExplorer(env.map) if explore else None
        info = {"collision": False}
        for _ in range(env.max_steps):
            target_angle = 0.0
            if explorer is not None:
                env.goal = explorer.update(env.position[:2])
                if explorer.done:
                    break
                if env.goal is not None:
                    dx, dy = env.goal - env.position[:2]
                    target_angle = math.atan2(dy, dx) - env.yaw
            action = follow_the_gap(*env.scan(), MAX_SPEED, MAX_YAW_RATE, target_angle=target_angle)
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        return env.coverage(), info["collision"]

    wander = [run(seed, explore=False) for seed in (0, 1)]
    explore = [run(seed, explore=True) for seed in (0, 1)]
    assert not any(crashed for _, crashed in explore)
    wander_coverage = np.mean([coverage for coverage, _ in wander])
    explore_coverage = np.mean([coverage for coverage, _ in explore])
    assert explore_coverage > wander_coverage + 0.1, (wander_coverage, explore_coverage)
