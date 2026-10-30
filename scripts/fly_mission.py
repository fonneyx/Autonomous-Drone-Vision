"""Watch the drone find a person / vehicle / drone with YOLO and fly to it, or score it.

The full loop runs here: the colour camera image goes to YOLO, YOLO's box for the
target class goes to the pilot policy, and the pilot steers using that plus
its depth camera.

Examples:
    python scripts/fly_mission.py                          # random target each flight, 3D view
    python scripts/fly_mission.py --target person          # always look for a person
    python scripts/fly_mission.py --headless --episodes 50 # score: success rate + detection accuracy
    python scripts/fly_mission.py --sim-detector           # perfect-ish simulated detector instead of YOLO
    python scripts/fly_mission.py --record mission.mp4 --episodes 3

Windows: "Drone camera" shows what the drone sees with YOLO boxes (green =
target, yellow = other objects) and its depth image in the corner. The 3D
viewer follows the drone; drag to orbit, scroll to zoom, close it to quit.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.mission_env import CLASSES, IMAGE_HEIGHT, IMAGE_WIDTH, DroneMissionEnv  # noqa: E402
from src.sim_env import CONTROL_DT, DEPTH_RANGE  # noqa: E402

GREEN, YELLOW, WHITE = (60, 220, 60), (0, 210, 255), (255, 255, 255)


def newest(pattern: str, what: str, hint: str) -> Path:
    candidates = sorted(ROOT.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No {what} found ({pattern}). {hint}")
    return candidates[0]


class YoloDetector:
    """Runs YOLO on the camera image and returns boxes in the simulator's format."""

    def __init__(self, weights: Path, confidence: float):
        import torch
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.confidence = confidence
        self.device = 0 if torch.cuda.is_available() else "cpu"
        names = {int(k): v for k, v in self.model.names.items()}
        missing = [c for c in CLASSES if c not in names.values()]
        if missing:
            raise ValueError(f"{weights} does not know classes {missing}; train it on SimVision_Dataset.")
        self.class_index = {name: index for index, name in names.items()}

    def detect(self, image_bgr: np.ndarray) -> list[tuple[str, float, np.ndarray]]:
        result = self.model.predict(image_bgr, imgsz=640, conf=self.confidence, device=self.device,
                                    verbose=False)[0]
        names = {index: name for name, index in self.class_index.items()}
        return [
            (names[int(c)], float(p), xyxy)
            for xyxy, p, c in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                                  result.boxes.cls.cpu().numpy())
        ]


def to_env_box(xyxy: np.ndarray) -> np.ndarray:
    """Pixel corners -> [cx, cy] in -1..1 and [w, h] as image fractions."""
    x0, y0, x1, y1 = xyxy
    return np.array([(x0 + x1) / IMAGE_WIDTH - 1, (y0 + y1) / IMAGE_HEIGHT - 1,
                     (x1 - x0) / IMAGE_WIDTH, (y1 - y0) / IMAGE_HEIGHT])


def to_pixels(box: np.ndarray) -> tuple[int, int, int, int]:
    cx, cy = (box[0] + 1) / 2 * IMAGE_WIDTH, (box[1] + 1) / 2 * IMAGE_HEIGHT
    w, h = box[2] * IMAGE_WIDTH, box[3] * IMAGE_HEIGHT
    return int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)


def draw_view(image: np.ndarray, detections, target: str, env: DroneMissionEnv, status: str) -> np.ndarray:
    view = image.copy()
    best = max((d for d in detections if d[0] == target), key=lambda d: d[1], default=None)
    for name, confidence, xyxy in detections:
        is_target = best is not None and xyxy is best[2]
        colour = GREEN if is_target else YELLOW
        x0, y0, x1, y1 = (int(v) for v in xyxy)
        cv2.rectangle(view, (x0, y0), (x1, y1), colour, 3 if is_target else 1)
        label = f"{'TARGET ' if is_target else ''}{name} {confidence:.2f}"
        cv2.putText(view, label, (x0, max(14, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2 if is_target else 1)

    speed = float(np.hypot(*env.data.qvel[:2]))
    lines = [f"Looking for: {target.upper()}", f"Distance: {env.target_distance():5.1f} m",
             f"Speed: {speed:4.1f} m/s", status]
    cv2.rectangle(view, (0, 0), (250, 22 * len(lines) + 8), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(view, line, (8, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)

    closeness = 1.0 - env.depth_image_metres() / DEPTH_RANGE
    depth = cv2.applyColorMap((closeness * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    depth = cv2.resize(depth, (160, 120), interpolation=cv2.INTER_NEAREST)
    view[-124:-4, -164:-4] = depth
    cv2.putText(view, "depth", (IMAGE_WIDTH - 160, IMAGE_HEIGHT - 128), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
    return view


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=CLASSES, help="Object to find (default: random each flight).")
    parser.add_argument("--policy", type=Path, help="Pilot .zip (default: newest runs/mission/*/best_model.zip).")
    parser.add_argument("--yolo", type=Path, help="YOLO weights (default: newest runs/train/sim-vision*/weights/best.pt).")
    parser.add_argument("--sim-detector", action="store_true", help="Use the simulated detector instead of YOLO.")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--obstacles", type=int, help="Fixed obstacle count (default: random 8-30).")
    parser.add_argument("--headless", action="store_true", help="No windows; just print the score.")
    parser.add_argument("--record", type=Path, help="Save the drone-camera view as a video (.mp4).")
    parser.add_argument("--seed", type=int, default=321)
    args = parser.parse_args()

    from stable_baselines3 import PPO

    policy_path = args.policy or newest("runs/mission/*/best_model.zip", "mission pilot",
                                        "Run: python scripts/train_pilot.py --task mission")
    policy = PPO.load(policy_path, device="cpu")
    print(f"Pilot: {policy_path}")
    detector = None
    if not args.sim_detector:
        yolo_path = args.yolo or newest("runs/train/sim-vision*/weights/best.pt", "vision model",
                                        "Run: python scripts/train_local.py --data ../SimVision_Dataset/data.yaml "
                                        "--name sim-vision")
        detector = YoloDetector(yolo_path, args.conf)
        print(f"Vision: {yolo_path}")

    env = DroneMissionEnv(target_class=args.target, world_seed=args.seed,
                          detector="simulated" if args.sim_detector else "external")
    camera = mujoco.Renderer(env.model, IMAGE_HEIGHT, IMAGE_WIDTH)
    show = not args.headless
    writer = None
    if args.record:
        writer = cv2.VideoWriter(str(args.record), cv2.VideoWriter_fourcc(*"mp4v"), 1 / CONTROL_DT,
                                 (IMAGE_WIDTH, IMAGE_HEIGHT))

    viewer = None
    if show:
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "drone")
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 7.0, -25.0, 180.0

    def look(target: str, status: str):
        """Render the camera, run YOLO, feed the pilot; return the observation."""
        need_image = detector is not None or show or writer is not None
        image, detections = None, []
        if need_image:
            camera.update_scene(env.data, camera="fpv")
            image = cv2.cvtColor(camera.render(), cv2.COLOR_RGB2BGR)
        if detector is not None:
            detections = detector.detect(image)
            best = max((d for d in detections if d[0] == target), key=lambda d: d[1], default=None)
            env.set_detection(None if best is None else to_env_box(best[2]))
        elif image is not None and env._detection is not None:
            detections = [(target, 1.0, np.array(to_pixels(env._detection), dtype=float))]
        if image is not None:
            view = draw_view(image, detections, target, env, status)
            if writer is not None:
                writer.write(view)
            if show:
                cv2.imshow("Drone camera", view)
                cv2.waitKey(1)
        return env.observe()

    results = {"success": 0, "crash": 0, "timeout": 0}
    seen_steps = found_steps = phantom_steps = unseen_steps = 0
    try:
        for episode in range(args.episodes):
            _, info = env.reset(seed=args.seed + episode, options={"obstacles": args.obstacles}
                                if args.obstacles is not None else None)
            target = info["target_class"]
            observation = look(target, "searching")
            steps = 0
            while True:
                tick = time.perf_counter()
                action, _ = policy.predict(observation, deterministic=True)
                _, _, terminated, truncated, info = env.step(action)
                steps += 1
                status = "TARGET IN SIGHT" if env._detection is not None else "searching"
                observation = look(target, status)
                # Detection accuracy against the simulator's ground truth
                if info["target_visible"]:
                    seen_steps += 1
                    found_steps += env._detection is not None
                else:
                    unseen_steps += 1
                    phantom_steps += env._detection is not None
                if viewer is not None:
                    if not viewer.is_running():
                        return
                    viewer.sync()
                    time.sleep(max(0.0, CONTROL_DT - (time.perf_counter() - tick)))
                if terminated or truncated:
                    break
            outcome = "success" if info["is_success"] else "crash" if info["collision"] else "timeout"
            results[outcome] += 1
            print(f"Flight {episode + 1}: {outcome:<8} target={target:<8} time={steps * CONTROL_DT:5.1f}s "
                  f"final distance={info['target_distance']:5.1f} m")
            if show:
                time.sleep(1.0)
    finally:
        if viewer is not None:
            viewer.close()
        if show:
            cv2.destroyAllWindows()
        if writer is not None:
            writer.release()
            print(f"Video saved: {args.record}")
        camera.close()
        env.close()

    total = sum(results.values())
    print(f"\nMission success: {results['success'] / total:.0%}  "
          f"(success {results['success']}, crash {results['crash']}, timeout {results['timeout']})")
    if seen_steps:
        print(f"Detection: found the target in {found_steps / seen_steps:.0%} of frames where it was visible; "
              f"false target in {phantom_steps / max(unseen_steps, 1):.1%} of frames where it was not")


if __name__ == "__main__":
    main()
