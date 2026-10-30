"""Tests for the MuJoCo obstacle course: API checks, depth sensing, crashes, altitude hold, speed.

Run: python -m pytest tests -q
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.sim.avoidance_env import DEPTH_RANGE, FLIGHT_ALTITUDE, DroneAvoidanceEnv  # noqa: E402


@pytest.fixture(scope="module")
def env() -> DroneAvoidanceEnv:
    return DroneAvoidanceEnv(world_seed=0)


def test_gymnasium_api(env: DroneAvoidanceEnv) -> None:
    check_env(env, skip_render_check=True)


def test_empty_course_flown_straight_holds_altitude(env: DroneAvoidanceEnv) -> None:
    env.reset(seed=1, options={"obstacles": 0})
    env.data.qpos[1] = 0.0
    env.data.qpos[3] = 0.0
    info: dict = {}
    for _ in range(env.max_steps):
        _, _, terminated, truncated, info = env.step(np.array([1.0, 0.0]))
        if terminated or truncated:
            break
    assert info["is_success"], info
    assert abs(env.position[2] - FLIGHT_ALTITUDE) < 0.1, "altitude hold drifted"


def test_depth_camera_sees_pole_and_crash_is_detected(env: DroneAvoidanceEnv) -> None:
    env.reset(seed=2, options={"obstacles": 0})
    env.data.qpos[:4] = [0.0, 0.0, FLIGHT_ALTITUDE, 0.0]
    env.data.mocap_pos[env._obstacle_mocap[0]] = [6.0, 0.0, env._half_heights[0]]
    env.data.mocap_quat[env._obstacle_mocap[0]] = [1.0, 0.0, 0.0, 0.0]
    env.data.qvel[:] = 0.0
    env.step(np.array([-1.0, 0.0]))
    assert env.nearest_obstacle_distance() < 6.0, "pole ahead not seen"
    assert env.nearest_obstacle_distance() < DEPTH_RANGE
    info: dict = {}
    reward = 0.0
    for _ in range(100):
        _, reward, terminated, _, info = env.step(np.array([1.0, 0.0]))
        if terminated:
            break
    assert info["collision"] and reward < -10, info


def test_simulation_is_fast_enough_for_training(env: DroneAvoidanceEnv) -> None:
    env.reset(seed=3)
    steps, start = 0, time.perf_counter()
    while time.perf_counter() - start < 1.0:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        steps += 1
        if terminated or truncated:
            env.reset()
    assert steps > 50, f"only {steps} steps/s"
