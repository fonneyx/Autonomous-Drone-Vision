"""Search-and-rescue mission in an unknown cave: explore, mark people, return home.

Examples:
    python scripts/rescue_mission.py                          # classical pilot, simulated detector
    python scripts/rescue_mission.py --detector yolo          # real YOLO on the drone camera
    python scripts/rescue_mission.py --pilot rl --shield      # RL pilot + safety shield
    python scripts/rescue_mission.py --headless --episodes 10 # score 10 caves
    python scripts/rescue_mission.py --record rescue.mp4      # top-down view + map video

Map: red crosses = people the drone marked; grey rings = where people really are
(simulation only, to check the markers). With --detector yolo a third window
shows the drone camera with YOLO's boxes.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.cave_env import CaveExploreEnv  # noqa: E402
from src.map_view import draw_map  # noqa: E402
from src.rescue import SimulatedPersonDetector, YoloPersonDetector  # noqa: E402
from src.rescue_mission import RescueMission  # noqa: E402
from src.safety import SafetyShield  # noqa: E402
from src.sim_env import CONTROL_DT, MAX_SPEED, MAX_YAW_RATE  # noqa: E402

DEFAULT_RL_PILOT = ROOT / "runs" / "avoidance" / "avoidance-2" / "best_model.zip"


def camera_view(detector: YoloPersonDetector) -> np.ndarray | None:
    if detector.last_frame is None:
        return None
    frame = cv2.cvtColor(detector.last_frame, cv2.COLOR_RGB2BGR)
    for x1, y1, x2, y2, confidence in detector.last_boxes:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (40, 220, 40), 2)
        cv2.putText(frame, f"person {confidence:.2f}", (int(x1), max(12, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 220, 40), 1, cv2.LINE_AA)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot", choices=["gap", "rl"], default="gap")
    parser.add_argument("--model", type=Path, default=DEFAULT_RL_PILOT, help="RL pilot .zip.")
    parser.add_argument("--shield", action="store_true", help="Safety shield overrides the pilot near walls.")
    parser.add_argument("--detector", choices=["sim", "yolo"], default="sim")
    parser.add_argument("--people", type=int, default=3)
    parser.add_argument("--battery", type=float, default=240.0, help="Flight time available (s).")
    parser.add_argument("--seed", type=int, default=0, help="Cave seed (each episode adds 1).")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--record", type=Path, help="Save top-down view + map as .mp4.")
    parser.add_argument("--save-map", type=Path, help="Save the final map of the last episode as .png.")
    args = parser.parse_args()

    policy = None
    if args.pilot == "rl":
        from stable_baselines3 import PPO

        policy = PPO.load(args.model, device="cpu")
    detector = YoloPersonDetector() if args.detector == "yolo" else None

    writer = None
    results = []
    for episode in range(args.episodes):
        seed = args.seed + episode
        env = CaveExploreEnv(world_seed=seed, people=args.people, render_mode="rgb_array" if args.record else None)
        shield = SafetyShield(MAX_SPEED, MAX_YAW_RATE) if args.shield else None
        mission = RescueMission(
            env, detector or SimulatedPersonDetector(seed=seed), policy=policy, shield=shield,
            battery_seconds=args.battery,
        )
        mission.reset(seed=seed)

        while True:
            tick = time.perf_counter()
            over = mission.step()
            battery = max(0.0, 1.0 - mission.elapsed / args.battery)
            status = f"{mission.phase}  people {len(mission.markers.confirmed())}  battery {battery * 100:.0f}%"
            if args.record or not args.headless:
                map_image = draw_map(env, mission.info, mission.explorer, mission.markers.confirmed(), True, status)
            if args.record:
                frame = np.hstack([cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR), cv2.resize(map_image, (480, 480))])
                if writer is None:
                    writer = cv2.VideoWriter(str(args.record), cv2.VideoWriter_fourcc(*"mp4v"), 1 / CONTROL_DT,
                                             (frame.shape[1], frame.shape[0]))
                writer.write(frame)
            if not args.headless:
                cv2.imshow("Rescue map (live)", map_image)
                if detector is not None and (view := camera_view(detector)) is not None:
                    cv2.imshow("Drone camera + YOLO", view)
                if cv2.waitKey(1) == 27:  # Esc
                    break
                time.sleep(max(0.0, CONTROL_DT - (time.perf_counter() - tick)))
            if over:
                break

        result = mission.result()
        results.append(result)
        error = "  -  " if math.isnan(result["marker_error_m"]) else f"{result['marker_error_m']:.2f}m"
        shield_text = "" if math.isnan(result["shield_rate"]) else f"  shield {result['shield_rate'] * 100:.0f}%"
        print(
            f"cave {seed:3d}: found {result['found']}/{result['people']}  error {error}  "
            f"false {result['false_markers']}  coverage {result['coverage'] * 100:5.1f}%  "
            f"{result['outcome']:13s} t {result['time']:5.1f}s  "
            f"(home called at {result['return_time']:.0f}s: {result['return_reason'] or '-'}){shield_text}"
        )
        if args.save_map:
            cv2.imwrite(str(args.save_map), draw_map(env, mission.info, mission.explorer,
                                                     mission.markers.confirmed(), True, mission.phase))
        if not args.headless and episode == args.episodes - 1 and not args.record:
            print("Press any key in the map window to close.")
            cv2.waitKey(0)
        env.close()

    if writer is not None:
        writer.release()
        print(f"Video saved: {args.record}")
    cv2.destroyAllWindows()
    if len(results) > 1:
        people = sum(r["people"] for r in results)
        found = sum(r["found"] for r in results)
        errors = [r["marker_error_m"] for r in results if not math.isnan(r["marker_error_m"])]
        print(
            f"\n{len(results)} caves: people found {found}/{people} ({found / max(1, people) * 100:.0f}%), "
            f"marker error {np.mean(errors):.2f} m, false markers {sum(r['false_markers'] for r in results)}, "
            f"home {sum(r['home'] for r in results)}/{len(results)}, "
            f"crashes {sum(r['outcome'] == 'crashed' for r in results)}, "
            f"coverage {np.mean([r['coverage'] for r in results]) * 100:.1f}%, "
            f"time {np.mean([r['time'] for r in results]):.0f}s"
        )


if __name__ == "__main__":
    main()
