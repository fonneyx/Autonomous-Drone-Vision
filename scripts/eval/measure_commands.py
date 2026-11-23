"""Measure gesture and voice command accuracy and latency (thesis metrics).

The screen tells you what to do; every trial is saved to a CSV file straight away.

    python scripts/eval/measure_commands.py --source gesture                    # 1, 2, 3, 5 m
    python scripts/eval/measure_commands.py --source voice                      # quiet, noisy
    python scripts/eval/measure_commands.py --source gesture --conditions 2m 4m --repeats 10
    python scripts/eval/report_commands.py                                      # tables + charts

Gesture session: stand alone in front of the webcam (the biggest person in view is the
tester). Before each block, walk to the distance on screen (mark 1/2/3/5 m on the floor
with tape) and press SPACE. Each trial: 3 s to get ready with your hands DOWN (the countdown
waits while a hand is raised), then 6 s to do the gesture with the hand raised above
mid-chest (raise it only at NOW, lower it again after the result).

Voice session: use the phone/headset microphone the drone will use. Do one block in a
quiet room, then one with noise (music or people talking, or a fan) - press SPACE when
ready. Speak normally, once per trial.

Keys: SPACE start block   K skip this trial (you messed up - it is left out of the report)
      Q/Esc stop (results so far are already saved)
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.commands import Command, CommandEvent  # noqa: E402
from src.evaluation.command_trials import (  # noqa: E402
    CORRECT,
    NO_TRIGGER,
    SKIPPED,
    TrialRunner,
    append_csv,
    make_plan,
)

OUT_DIR = ROOT / "runs" / "measurements"
DEFAULT_CONDITIONS = {"gesture": ["1m", "2m", "3m", "5m"], "voice": ["quiet", "noisy"]}
WHITE, GREY, YELLOW, GREEN, RED, CYAN = (255, 255, 255), (150, 150, 150), (0, 220, 255), (60, 200, 60), (40, 40, 230), (230, 200, 40)


def text(image: np.ndarray, line: str, origin: tuple[int, int], colour=WHITE, scale: float = 0.6,
         thickness: int = 1) -> None:
    cv2.putText(image, line, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, line, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)


def expected_shape(command: Command | None) -> str | None:
    from src.perception.gestures import FIST, OPEN_PALM

    return {Command.STOP: OPEN_PALM, Command.COME_BACK: OPEN_PALM, Command.LAND: FIST}.get(command)


def draw(view: np.ndarray, runner: TrialRunner, now: float, args: argparse.Namespace) -> None:
    height, width = view.shape[:2]
    trial = runner.trial
    panel = view[0:150, :]
    panel[:] = (panel * 0.35).astype(np.uint8)
    done = len(runner.results)
    text(view, f"{args.source.upper()} measurement   trial {done + (trial is not None)}/{len(runner.plan)}",
         (12, 26), GREY, 0.55)
    if runner.state == "done":
        text(view, "Finished - thank you!", (12, 75), GREEN, 1.0, 2)
        text(view, f"Saved to {args.out.name}. Press Q.", (12, 115), WHITE)
        return
    assert trial is not None
    number, size = runner.block_progress()
    if runner.state == "waiting":
        where = f"Go to {trial.condition}" if args.source == "gesture" else f"Set up: {trial.condition}"
        text(view, where, (12, 75), YELLOW, 1.1, 2)
        text(view, "then press SPACE", (12, 115), WHITE, 0.8)
    else:
        colour = WHITE if runner.state == "prepare" else (GREEN if runner.state == "active" else GREY)
        text(view, ("Next: " if runner.state == "prepare" else "") + trial.instruction, (12, 75), colour, 1.0, 2)
        # Start every trial from rest, as in real use: a hand already held in shape skews accuracy and latency
        rest = "hands DOWN, wait for NOW" if args.source == "gesture" else "wait for NOW"
        label = {"prepare": rest, "active": "NOW", "result": "done"}[runner.state]
        text(view, f"{label}  {runner.seconds_left(now):.0f} s     {trial.condition}   ({number}/{size} in block)",
             (12, 115), colour, 0.65)
    result = runner.last_result
    if runner.state == "result" and result is not None:
        good = result.outcome in (CORRECT, NO_TRIGGER)
        verdict = result.outcome.replace("_", " ") + (f": got {result.got}" if result.got else "")
        text(view, verdict, (12, 142), GREEN if good else (GREY if result.outcome == SKIPPED else RED), 0.6)
    text(view, "SPACE start block   K skip   Q stop", (12, height - 12), GREY, 0.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["gesture", "voice"], required=True)
    parser.add_argument("--conditions", nargs="+", help="Block names (default: 1m 2m 3m 5m / quiet noisy).")
    parser.add_argument("--repeats", type=int, default=5, help="Trials per command per condition.")
    parser.add_argument("--controls", type=int, help="'No command' trials per condition (default: repeats/2).")
    parser.add_argument("--window", type=float, default=6.0, help="Seconds to give each command.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", type=Path, help="Use a video file instead of the webcam (testing).")
    parser.add_argument("--mic", type=int, help="Microphone device index (default: system default).")
    parser.add_argument("--hfov", type=float, default=70.0, help="Camera horizontal field of view (deg).")
    parser.add_argument("--out", type=Path, help="CSV file (default: runs/measurements/<source>_<time>.csv).")
    parser.add_argument("--record", action="store_true",
                        help="Also save the camera video + a frame log next to the CSV (stays on this laptop), so "
                             "the session can be re-analysed later without repeating it.")
    parser.add_argument("--headless", action="store_true", help="No window, blocks start by themselves (checks).")
    parser.add_argument("--frames", type=int, help="Stop after this many frames.")
    args = parser.parse_args()
    if args.out is None:
        args.out = OUT_DIR / f"{args.source}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    conditions = args.conditions or DEFAULT_CONDITIONS[args.source]
    plan = make_plan(args.source, conditions, args.repeats, controls=args.controls, seed=args.seed)
    runner = TrialRunner(plan, window=args.window, auto_start=args.headless)
    print(f"{len(plan)} trials in {len(conditions)} blocks ({', '.join(conditions)}) -> {args.out}")

    capture = cv2.VideoCapture(str(args.video)) if args.video else cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    ok, frame = capture.read()
    if not ok and args.source == "gesture":
        raise SystemExit("Could not read from the camera / video. Try --camera 1, or close other apps using it.")
    if not ok:
        frame = np.zeros((480, 640, 3), np.uint8)  # voice works without a camera
        capture = None
    height, width = frame.shape[:2]

    tracker = hand_reader = gestures = controller = voice = None
    if args.source == "gesture":
        from src.control.follow_controller import FollowController
        from src.perception.gestures import GestureTimer, HandReader, hands_of, main_hand, raised_hands
        from src.perception.person_tracker import PersonTracker

        print("Loading YOLO11s-pose and the MediaPipe gesture model ...")
        tracker, hand_reader, gestures = PersonTracker(), HandReader(), GestureTimer()
        controller = FollowController(width, height, args.hfov)
    else:
        from src.voice.commands import VoiceListener

        voice = VoiceListener(device=args.mic)
        voice.start()
        print("Voice: loading Vosk (first run downloads the English model ~40 MB) ...")

    video = frame_log = None
    if args.record:
        video_path = args.out.with_name(args.out.stem + "_video.mp4")
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
        frame_log = video_path.with_suffix(".tsv").open("w", encoding="utf-8")
        frame_log.write("frame\tt\tstate\ttrial\n")  # t = time.monotonic(), same clock as the CSV
        print(f"Recording video to {video_path}")

    go_count = 0
    frame_count = 0
    last_heard = ""
    while args.frames is None or frame_count < args.frames:
        frame_count += 1
        if capture is not None:
            ok, frame = capture.read()
            if not ok:
                if args.video:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
        now = time.monotonic()
        events: list[CommandEvent] = []
        hand_visible = shape_seen = hand_up = hand_found = hand_own = False
        distance = None
        view = frame.copy()

        if args.source == "gesture":
            if runner.go_count != go_count:  # a new active window: forget earlier hand shapes
                go_count = runner.go_count
                gestures = GestureTimer()
            people = tracker(frame)
            hands = hand_reader(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), now)
            tester = max(people, key=lambda p: p.height) if people else None
            hand_found = bool(hands)
            if tester is not None:
                hand_own = bool(hands_of(tester.box, hands))
                hand = main_hand(raised_hands(tester, hands))
                hand_up = hand is not None
                hand_visible = hand is not None
                trial = runner.trial
                shape_seen = hand is not None and trial is not None and hand.label == expected_shape(trial.expected)
                command = gestures.update(hand, now)
                if command is not None:
                    events.append(CommandEvent(command, "gesture", now))
                distance = controller.measure(tester).distance
                x1, y1, x2, y2 = tester.box.astype(int)
                cv2.rectangle(view, (x1, y1), (x2, y2), GREEN, 2)
                if distance is not None:
                    text(view, f"~{distance:.1f} m", (x1 + 4, y2 - 8), GREEN, 0.6)
            for h in hands:
                for px, py in h.landmarks.astype(int):
                    cv2.circle(view, (px, py), 2, CYAN, -1)
                text(view, h.label.replace("_", " "), (int(h.wrist[0]), int(h.wrist[1]) + 18), CYAN, 0.45)
        else:
            while not voice.events.empty():
                events.append(voice.events.get())
            if voice.error:
                text(view, voice.error[:80], (12, 180), RED, 0.5)
            elif voice.last_text:
                text(view, f"heard: {voice.last_text}", (12, 180), WHITE, 0.55)

        key = -1 if args.headless else cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            runner.start_block(now)
        if hand_up:
            runner.hold(now)  # countdown starts again once the hand is down
        result = runner.skip(now) if key == ord("k") else None
        if result is None:
            heard = ""
            if voice is not None and voice.last_text != last_heard:  # only new text belongs to this trial
                heard = last_heard = voice.last_text
            result = runner.update(now, events, hand_visible, shape_seen, distance, heard, hand_found, hand_own)
        if result is not None:
            append_csv(args.out, result)
            print(f"  {result.condition:>6}  {result.expected or '(nothing)':>10}  ->  {result.outcome}"
                  f"{'  ' + result.got if result.got else ''}")

        if video is not None:
            video.write(frame)
            frame_log.write(f"{frame_count}\t{now:.4f}\t{runner.state}\t{runner.index}\n")
        draw(view, runner, now, args)
        if hand_up and runner.state == "prepare":
            text(view, "LOWER YOUR HAND", (12, 175), RED, 0.9, 2)
        if not args.headless:
            cv2.imshow("Command measurement (Q to stop)", view)
        if runner.state == "done" and args.headless:
            break

    if capture is not None:
        capture.release()
    if video is not None:
        video.release()
        frame_log.close()
    cv2.destroyAllWindows()
    if voice is not None:
        voice.stop()
    print(f"{len(runner.results)} trials saved to {args.out}")


if __name__ == "__main__":
    main()
