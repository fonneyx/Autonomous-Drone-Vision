#!/usr/bin/env python3
"""Test script for DroneAvoidanceEnv"""

from src.rl_environment import DroneAvoidanceEnv
import numpy as np

print("[TEST] Initializing DroneAvoidanceEnv...")
env = DroneAvoidanceEnv(max_obstacles=5, max_steps=100)

print("[INFO] Observation Space:", env.observation_space)
print("[INFO] Action Space:", env.action_space)

print("\n[TEST] Resetting environment...")
obs, info = env.reset()
print("[INFO] Initial observation shape:", obs.shape)
print("[INFO] Initial info:", info)

print("\n[TEST] Running 5 environment steps...")
for step in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  Step {step+1}: Reward={reward:.2f}, Terminated={terminated}, Truncated={truncated}")
    print(f"           Obstacles={info['num_obstacles']}, Velocity={info['drone_velocity']:.2f} m/s")

print("\n[SUCCESS] RL Environment test passed!")
