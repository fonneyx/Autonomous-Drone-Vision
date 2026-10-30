"""Fly a drone through an unknown cave and watch the map being built live.

Examples:
    python scripts/explore_cave.py                      # follow-the-gap pilot, 3D viewer + map
    python scripts/explore_cave.py --pilot rl           # trained avoidance pilot ("wander" mode)
    python scripts/explore_cave.py --pilot keys         # fly yourself: W/S speed, A/D turn
    python scripts/explore_cave.py --seed 7 --no-viewer # only the map window
    python scripts/explore_cave.py --headless --episodes 20   # score: coverage and crashes
    python scripts/explore_cave.py --record cave.mp4    # top-down video + map side by side

Map colours: grey = unknown, white = free, black = wall, orange = frontier
(edge of explored space), red = drone, blue = flown path, green = home.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.baselines import follow_the_gap  # noqa: E402
from src.cave_env import CaveExploreEnv  # noqa: E402
from src.sim_env import CONTROL_DT, MAX_SPEED, MAX_YAW_RATE  # noqa: E402

MAP_SCALE = 3  # pixels per map cell


def draw_map(env: CaveExploreEnv, info: dict) -> np.ndarray:
    image = env.map.to_bgr(MAP_SCALE)
    points = [env.map.world_to_pixel(x, y, MAP_SCALE) for x, y in env.trail]
    if len(points) > 1:
        cv2.polylines(image, [np.array(points, dtype=np.int32)], False, (200, 120, 30), 1, cv2.LINE_AA)
    cv2.circle(image, env.map.world_to_pixel(*env.layout.home_xy, MAP_SCALE), 6, (60, 180, 60), 2)
    x, y = env.position[:2]
    nose = (x + 0.8 * np.cos(env.yaw), y + 0.8 * np.sin(env.yaw))
    cv2.arrowedLine(
        image, env.map.world_to_pixel(x, y, MAP_SCALE), env.map.world_to_pixel(*nose, MAP_SCALE),
        (30, 30, 230), 2, cv2.LINE_AA, tipLength=0.5,
    )
    text = f"coverage {info.get('coverage', 0.0) * 100:4.1f}%   t {env._step_count * CONTROL_DT:5.1f}s"
    cv2.rectangle(image, (0, 0), (250, 28), (30, 30, 30), -1)
    cv2.putText(image, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def newest_avoidance_model() -> Path:
    candidates = sorted((ROOT / "runs" / "avoidance").glob("*/best_model.zip"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("No trained avoidance pilot in runs/avoidance; use --pilot gap.")
    return candidates[-1]


class KeyboardPilot:
    """W/S change speed, A/D turn while held (keys read from the OpenCV map window)."""

    def __init__(self) -> None:
        self.speed = 0.0

    def __call__(self, key: int) -> np.ndarray:
        yaw = 0.0
        if key in (ord("w"), ord("W")):
            self.speed = min(MAX_SPEED, self.speed + 0.25)
        elif key in (ord("s"), ord("S")):
            self.speed = max(0.0, self.speed - 0.5)
        elif key in (ord("a"), ord("A")):
            yaw = 0.6
        elif key in (ord("d"), ord("D")):
            yaw = -0.6
        return np.array([self.speed / MAX_SPEED * 2.0 - 1.0, yaw], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot", choices=["gap", "rl", "keys"], default="gap")
    parser.add_argument("--model", type=Path, help="RL pilot .zip (default: newest avoidance best_model).")
    parser.add_argument("--seed", type=int, default=0, help="Cave seed (each episode adds 1).")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--headless", action="store_true", help="No windows; print scores only.")
    parser.add_argument("--no-viewer", action="store_true", help="Map window only, no MuJoCo 3D viewer.")
    parser.add_argument("--record", type=Path, help="Save top-down view + map as .mp4.")
    parser.add_argument("--save-map", type=Path, help="Save the final map of the last episode as .png.")
    args = parser.parse_args()
    if args.pilot == "keys" and args.headless:
        parser.error("--pilot keys needs the map window.")

    policy = None
    if args.pilot == "rl":
        from stable_baselines3 import PPO

        model_path = args.model or newest_avoidance_model()
        policy = PPO.load(model_path, device="cpu")
        print(f"RL pilot: {model_path}")
    keyboard = KeyboardPilot() if args.pilot == "keys" else None

    writer = None
    results = []
    for episode in range(args.episodes):
        seed = args.seed + episode
        env = CaveExploreEnv(world_seed=seed, render_mode="rgb_array" if args.record else None)
        obs, info = env.reset(seed=seed)

        viewer = None
        if not (args.headless or args.no_viewer):
            import mujoco
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(env.model, env.data)
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = env._drone_body
            viewer.cam.distance, viewer.cam.elevation = 14.0, -65.0

        key = -1
        total_reward = 0.0
        for _ in range(env.max_steps):
            tick = time.perf_counter()
            if keyboard is not None:
                action = keyboard(key)
            elif policy is not None:
                action, _ = policy.predict(obs, deterministic=True)
            else:
                action = follow_the_gap(*env.scan(), MAX_SPEED, MAX_YAW_RATE)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if args.record:
                # 640x480 top-down view next to the map scaled to the same height
                frame = np.hstack(
                    [cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR), cv2.resize(draw_map(env, info), (480, 480))]
                )
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(args.record), cv2.VideoWriter_fourcc(*"mp4v"), 1 / CONTROL_DT, (width, height)
                    )
                writer.write(frame)
            if not args.headless:
                cv2.imshow("Cave map (live)", draw_map(env, info))
                key = cv2.waitKey(1)
                if key == 27:  # Esc
                    terminated = True
                if viewer is not None:
                    if not viewer.is_running():
                        terminated = True
                    viewer.sync()
                time.sleep(max(0.0, CONTROL_DT - (time.perf_counter() - tick)))
            if terminated or truncated:
                break

        results.append(info)
        print(
            f"cave {seed:3d}: coverage {info['coverage'] * 100:5.1f}%  "
            f"walls correct {env.wall_precision() * 100:5.1f}%  "
            f"{'CRASH' if info['collision'] else 'ok   '}  time {env._step_count * CONTROL_DT:5.1f}s  "
            f"reward {total_reward:6.1f}"
        )
        if args.save_map:
            cv2.imwrite(str(args.save_map), draw_map(env, info))
        if viewer is not None:
            viewer.close()
        if not args.headless and episode == args.episodes - 1 and not args.record:
            print("Press any key in the map window to close.")
            cv2.waitKey(0)
        env.close()

    if writer is not None:
        writer.release()
        print(f"Video saved: {args.record}")
    cv2.destroyAllWindows()
    if len(results) > 1:
        crashes = np.mean([r["collision"] for r in results])
        coverage = np.mean([r["coverage"] for r in results])
        print(f"\n{len(results)} caves: mean coverage {coverage * 100:.1f}%, crash rate {crashes * 100:.0f}%")


if __name__ == "__main__":
    main()
