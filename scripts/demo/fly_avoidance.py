"""Watch a trained avoidance policy fly the MuJoCo obstacle course, or score it.

Examples:
    python scripts/demo/fly_avoidance.py                         # newest best_model, 3D viewer
    python scripts/demo/fly_avoidance.py --obstacles 45          # harder course
    python scripts/demo/fly_avoidance.py --random                # untrained drone, for comparison
    python scripts/demo/fly_avoidance.py --headless --episodes 100   # success-rate test
    python scripts/demo/fly_avoidance.py --record flight.mp4 --episodes 3

Viewer: drag to orbit, scroll to zoom, close the window to quit.
The small window shows what the drone's depth camera sees (bright = close).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.sim.avoidance_env import CONTROL_DT, DEPTH_RANGE, DroneAvoidanceEnv  # noqa: E402


def newest_model() -> Path:
    candidates = sorted(
        (ROOT / "runs" / "avoidance").glob("*/best_model.zip"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No trained model yet. Run scripts/train/train_pilot.py or pass --random.")
    return candidates[0]


def depth_preview(env: DroneAvoidanceEnv) -> np.ndarray:
    closeness = 1.0 - env.depth_image_metres() / DEPTH_RANGE
    image = cv2.applyColorMap((closeness * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.resize(image, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, help="Policy .zip (default: newest best_model.zip).")
    parser.add_argument("--random", action="store_true", help="Fly with random actions instead.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--obstacles", type=int, help="Fixed obstacle count (default: random 10-40).")
    parser.add_argument("--headless", action="store_true", help="No windows; just print the score.")
    parser.add_argument("--record", type=Path, help="Save a chase-camera video (.mp4).")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    policy = None
    if not args.random:
        from stable_baselines3 import PPO

        model_path = args.model or newest_model()
        policy = PPO.load(model_path, device="cpu")
        print(f"Policy: {model_path}")

    env = DroneAvoidanceEnv(render_mode="rgb_array" if args.record else None, world_seed=args.seed)
    options = {"obstacles": args.obstacles} if args.obstacles is not None else None
    writer = None
    if args.record:
        writer = cv2.VideoWriter(str(args.record), cv2.VideoWriter_fourcc(*"mp4v"), 1 / CONTROL_DT, (640, 480))

    viewer = None
    if not args.headless:
        import mujoco
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(env.model, env.data)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "drone")
        viewer.cam.distance = 6.0
        viewer.cam.elevation = -25.0
        viewer.cam.azimuth = 180.0

    results = {"success": 0, "crash": 0, "timeout": 0}
    try:
        for episode in range(args.episodes):
            obs, reset_info = env.reset(seed=args.seed + episode, options=options)
            steps, distance_flown = 0, 0.0
            while True:
                tick = time.perf_counter()
                if policy is None:
                    action = env.action_space.sample()
                else:
                    action, _ = policy.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                steps += 1
                distance_flown = info["progress"]

                if writer is not None:
                    writer.write(cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR))
                if viewer is not None:
                    if not viewer.is_running():
                        return
                    viewer.sync()
                    cv2.imshow("Depth camera", depth_preview(env))
                    cv2.waitKey(1)
                    time.sleep(max(0.0, CONTROL_DT - (time.perf_counter() - tick)))
                if terminated or truncated:
                    break

            outcome = "success" if info["is_success"] else "crash" if info["collision"] else "timeout"
            results[outcome] += 1
            print(
                f"Episode {episode + 1}: {outcome:<8} obstacles={reset_info['obstacles']:<3} "
                f"progress={distance_flown:5.0%}  time={steps * CONTROL_DT:5.1f}s"
            )
            if viewer is not None:
                time.sleep(1.0)
    finally:
        if viewer is not None:
            viewer.close()
            cv2.destroyAllWindows()
        if writer is not None:
            writer.release()
            print(f"Video saved: {args.record}")
        env.close()

    total = sum(results.values())
    print(
        f"\nSuccess rate: {results['success'] / total:.0%}  "
        f"(success {results['success']}, crash {results['crash']}, timeout {results['timeout']})"
    )


if __name__ == "__main__":
    main()
