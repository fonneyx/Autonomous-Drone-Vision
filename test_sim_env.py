"""Smoke test for the MuJoCo obstacle course: API checks, sensing, crashes, speed."""

import time

import numpy as np
from gymnasium.utils.env_checker import check_env

from src.sim_env import DEPTH_RANGE, FLIGHT_ALTITUDE, DroneAvoidanceEnv

env = DroneAvoidanceEnv(world_seed=0)
check_env(env, skip_render_check=True)
print("[OK] Gymnasium API check passed")

# Empty course: flying straight at full speed must reach the finish line
obs, _ = env.reset(seed=1, options={"obstacles": 0})
env.data.qpos[1] = 0.0
env.data.qpos[3] = 0.0
info = {}
for _ in range(env.max_steps):
    obs, reward, terminated, truncated, info = env.step(np.array([1.0, 0.0]))
    if terminated or truncated:
        break
assert info["is_success"], info
assert abs(env.position[2] - FLIGHT_ALTITUDE) < 0.1, "altitude hold drifted"
print(f"[OK] Empty course flown straight: success at step {env._step_count}")

# Pole placed dead ahead: depth camera must see it, flying into it must crash
env.reset(seed=2, options={"obstacles": 0})
env.data.qpos[:4] = [0.0, 0.0, FLIGHT_ALTITUDE, 0.0]
env.data.mocap_pos[env._obstacle_mocap[0]] = [6.0, 0.0, env._half_heights[0]]
env.data.mocap_quat[env._obstacle_mocap[0]] = [1.0, 0.0, 0.0, 0.0]
env.data.qvel[:] = 0.0
env.step(np.array([-1.0, 0.0]))
nearest = env.nearest_obstacle_distance()
assert nearest < 6.0, f"pole not seen, nearest reading {nearest:.2f} m"
print(f"[OK] Depth camera sees pole ahead at {nearest:.2f} m (range {DEPTH_RANGE} m)")
for _ in range(100):
    _, reward, terminated, _, info = env.step(np.array([1.0, 0.0]))
    if terminated:
        break
assert info["collision"] and reward < -10, info
print(f"[OK] Collision detected at x={env.position[0]:.2f} m, reward {reward:.1f}")

# Throughput with random actions on a full course
env.reset(seed=3)
steps, start = 0, time.perf_counter()
while time.perf_counter() - start < 3.0:
    _, _, terminated, truncated, _ = env.step(env.action_space.sample())
    steps += 1
    if terminated or truncated:
        env.reset()
print(f"[OK] Simulation speed: {steps / (time.perf_counter() - start):.0f} steps/s per process")
