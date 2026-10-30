"""Follow-me prototype on the laptop webcam: tracking, re-ID, hand gestures, voice, arbiter.

No drone needed: the window shows what the drone *would* fly (velocity commands).

How to use:
    python scripts/demo/webcam_followme.py
    python scripts/demo/webcam_followme.py --no-voice          # without microphone
    python scripts/demo/webcam_followme.py --video clip.mp4    # test on a recorded video

1. Stand in front of the camera and hold up an OPEN PALM for 1 s -> you are locked as the operator
   (green box "ME"). Other people get grey boxes and their gestures are ignored.
2. Say "take off" (or press T), then "follow me" (or F).
3. Gestures (hold ~1 s): open palm = stop, fist = land (1.5 s), wave = come back.
4. Voice: take off, follow me, stop, land, come back, go higher, go lower, closer, further, take a photo.

Keys: T take off  F follow  S stop  L land  C come back  U/J higher/lower  [ ] closer/further
      P photo  E lock the biggest person  R release lock  X toggle simulated obstacle  Q/Esc quit

Webcam distances are short, so the defaults here are a 2 m follow distance and 1 m minimum; the real
drone uses >= 3 m (CLAUDE.md safety rules).
"""

from __future__ import annotations

import argparse
import math
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
from src.control.arbiter import Arbiter, Mode  # noqa: E402
from src.control.follow_controller import FollowController, Velocity  # noqa: E402
from src.perception.gestures import OPEN_PALM, GestureTimer, HandReader, hands_of, main_hand  # noqa: E402
from src.perception.person_tracker import PersonTracker  # noqa: E402
from src.perception.reid import OperatorLock  # noqa: E402

KEYS = {
    ord("t"): Command.TAKE_OFF, ord("f"): Command.FOLLOW, ord("s"): Command.STOP, ord("l"): Command.LAND,
    ord("c"): Command.COME_BACK, ord("u"): Command.HIGHER, ord("j"): Command.LOWER, ord("["): Command.CLOSER,
    ord("]"): Command.FURTHER, ord("p"): Command.PHOTO,
}
GREEN, GREY, WHITE, YELLOW, RED, CYAN = (60, 200, 60), (150, 150, 150), (255, 255, 255), (0, 220, 255), (40, 40, 230), (230, 200, 40)
PHOTO_DIR = ROOT / "runs" / "photos"
LOCK_HOLD = 1.0  # seconds of open palm to become the operator


def text(image: np.ndarray, line: str, origin: tuple[int, int], colour=WHITE, scale: float = 0.55) -> None:
    cv2.putText(image, line, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, line, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def draw_velocity(image: np.ndarray, velocity: Velocity, hover: bool, origin: tuple[int, int]) -> None:
    """Small top-down drone icon: arrow = forward/back, arc = turn, bar = climb."""
    x, y = origin
    cv2.rectangle(image, (x - 70, y - 70), (x + 90, y + 70), (30, 30, 30), -1)
    cv2.circle(image, (x, y), 10, WHITE, 2)
    if hover:
        text(image, "HOVER", (x - 28, y + 55), YELLOW)
        return
    length = int(25 * velocity.vx)
    cv2.arrowedLine(image, (x, y), (x, y - length), GREEN if velocity.vx >= 0 else RED, 3, tipLength=0.3)
    turn = math.degrees(velocity.yaw_rate)
    if abs(turn) > 1:
        cv2.ellipse(image, (x, y), (35, 35), -90, 0, -turn * 2, CYAN, 2)
    bar = int(50 * velocity.vz)
    cv2.rectangle(image, (x + 60, y), (x + 75, y - bar), CYAN, -1)
    text(image, f"vx {velocity.vx:+.1f} m/s", (x - 65, y + 50), WHITE, 0.45)
    text(image, f"yaw {turn:+.0f} deg/s  vz {velocity.vz:+.1f}", (x - 65, y + 65), WHITE, 0.4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", type=Path, help="Use a video file instead of the webcam.")
    parser.add_argument("--no-voice", action="store_true")
    parser.add_argument("--mic", type=int, help="Microphone device index (default: system default).")
    parser.add_argument("--hfov", type=float, default=70.0, help="Camera horizontal field of view (deg).")
    parser.add_argument("--distance", type=float, default=2.0, help="Follow distance (m).")
    parser.add_argument("--min-distance", type=float, default=1.0, help="Never closer than this (m).")
    parser.add_argument("--headless", action="store_true", help="No window (for automated checks).")
    parser.add_argument("--frames", type=int, help="Stop after this many frames.")
    parser.add_argument("--save", type=Path, help="Save the last drawn frame as an image.")
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.video)) if args.video else cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    ok, frame = capture.read()
    if not ok:
        raise SystemExit("Could not read from the camera / video. Try --camera 1, or close other apps using it.")
    height, width = frame.shape[:2]

    print("Loading models (first run downloads YOLO11s-pose ~20 MB and the gesture model ~8 MB) ...")
    tracker = PersonTracker()
    hand_reader = HandReader()
    lock = OperatorLock()
    gestures = GestureTimer()
    arbiter = Arbiter(follow_distance=args.distance, min_distance=args.min_distance, max_distance=6.0)
    controller = FollowController(width, height, args.hfov)
    voice = None
    if not args.no_voice:
        from src.voice.commands import VoiceListener

        voice = VoiceListener(device=args.mic)
        voice.start()
        print("Voice: listening (first run downloads the Vosk English model ~40 MB).")

    lock_candidate: tuple[int | None, float] | None = None  # (track id, palm since)
    simulated_altitude = 0.0
    obstacle = False
    last_event = ""
    fps, last_tick = 0.0, time.monotonic()
    frame_count = 0
    view = frame

    while args.frames is None or frame_count < args.frames:
        frame_count += 1
        ok, frame = capture.read()
        if not ok:
            if args.video:  # loop the video
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        now = time.monotonic()
        people = tracker(frame)
        hands = hand_reader(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), now)
        operator = lock.update(frame, people)
        events: list[CommandEvent] = []

        # Become the operator: hold an open palm for LOCK_HOLD seconds (when nobody is locked)
        if not lock.locked:
            palm_person = next((p for p in people if any(h.label == OPEN_PALM for h in hands_of(p.box, hands))), None)
            if palm_person is None:
                lock_candidate = None
            elif lock_candidate is None or lock_candidate[0] != palm_person.track_id:
                lock_candidate = (palm_person.track_id, now)
            elif now - lock_candidate[1] >= LOCK_HOLD:
                lock.lock(frame, palm_person)
                operator = palm_person
                gestures.shape, gestures.since, gestures.fired = OPEN_PALM, now, True  # the lock palm is not a STOP
                last_event = "operator locked"
                lock_candidate = None

        # Gestures count only from the operator
        if operator is not None:
            command = gestures.update(main_hand(hands_of(operator.box, hands)), now)
            if command is not None:
                events.append(CommandEvent(command, "gesture", now))
        if voice is not None:
            while not voice.events.empty():
                events.append(voice.events.get())

        key = -1 if args.headless else cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in KEYS:
            events.append(CommandEvent(KEYS[key], "keyboard", now))
        elif key == ord("e") and people:
            biggest = max(people, key=lambda p: p.height)
            lock.lock(frame, biggest)
            operator = biggest
            last_event = "operator locked (key)"
        elif key == ord("r"):
            lock.release()
            last_event = "operator released"
        elif key == ord("x"):
            obstacle = not obstacle

        # Decide and compute the velocity command
        measurement = controller.measure(operator) if operator is not None else None
        arrived = measurement is not None and measurement.distance is not None and (
            measurement.distance <= args.min_distance + 0.5
        )
        decision = arbiter.update(events, now, operator is not None, safety_stop=obstacle, arrived=arrived)
        velocity = Velocity.hover()
        if decision.mode in (Mode.FOLLOW, Mode.COME_BACK) and not decision.hover and operator is not None:
            target = args.min_distance + 0.5 if decision.mode is Mode.COME_BACK else decision.follow_distance
            velocity, measurement = controller.command(operator, target, args.min_distance)
        # Altitude: climb/descend towards the commanded height (simulated here; MTF-01 on the real drone)
        if decision.mode not in (Mode.LANDED, Mode.LANDING):
            velocity.vz = max(-0.5, min(0.5, 0.8 * (decision.altitude - simulated_altitude)))
        elif decision.mode is Mode.LANDING:
            velocity = Velocity(vz=-0.5)
        simulated_altitude = max(0.0, simulated_altitude + velocity.vz * (now - last_tick))
        if decision.mode is Mode.LANDED:
            simulated_altitude = 0.0
        for event in decision.accepted:
            last_event = f"{event.command.value}  ({event.source})"
        if decision.photo:
            PHOTO_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(PHOTO_DIR / f"photo_{datetime.now():%Y%m%d_%H%M%S}.jpg"), frame)

        # ---- drawing
        view = frame.copy()
        for person in people:
            is_operator = operator is not None and person is operator
            colour = GREEN if is_operator else GREY
            x1, y1, x2, y2 = person.box.astype(int)
            cv2.rectangle(view, (x1, y1), (x2, y2), colour, 3 if is_operator else 1)
            label = "ME" if is_operator else f"id {person.track_id}"
            text(view, label, (x1 + 4, y1 + 20), colour, 0.6)
        for hand in hands:
            for px, py in hand.landmarks.astype(int):
                cv2.circle(view, (px, py), 2, CYAN, -1)
            text(view, hand.label.replace("_", " "), (int(hand.wrist[0]), int(hand.wrist[1]) + 18), CYAN, 0.45)

        if lock_candidate is not None:
            progress = min(1.0, (now - lock_candidate[1]) / LOCK_HOLD)
            text(view, "Hold your palm to become the operator", (10, height - 40), YELLOW)
            cv2.rectangle(view, (10, height - 30), (10 + int(200 * progress), height - 20), YELLOW, -1)
        holding, progress = gestures.progress(now)
        if holding:
            text(view, f"holding: {holding}", (10, height - 40), YELLOW)
            cv2.rectangle(view, (10, height - 30), (10 + int(200 * progress), height - 20), YELLOW, -1)

        fps = 0.9 * fps + 0.1 / max(now - last_tick, 1e-3)
        last_tick = now
        lines = [
            (f"MODE: {decision.mode.value.upper()}", YELLOW if decision.hover else GREEN),
            (decision.reason, RED if "SAFETY" in decision.reason or "lost" in decision.reason else WHITE),
        ]
        if lock.locked:
            seen = "visible" if operator is not None else "searching"
            lines.append((f"operator: locked, {seen}  (look match {1 - lock.last_distance:.2f})", WHITE))
        else:
            lines.append(("operator: none - hold up an open palm", YELLOW))
        if measurement is not None and measurement.distance is not None:
            lines.append((f"distance {measurement.distance:.1f} m  (target {decision.follow_distance:.0f} m)", WHITE))
        lines.append((f"altitude {simulated_altitude:.1f} m  (target {decision.altitude:.1f} m)", WHITE))
        lines.append((f"last command: {last_event}", CYAN))
        if voice is not None:
            heard = voice.error or voice.last_text or "-"
            lines.append((f"heard: {heard}", RED if voice.error else WHITE))
        if obstacle:
            lines.append(("simulated obstacle ON (X)", RED))
        lines.append((f"{fps:.0f} FPS", GREY))
        panel_width = 20 + max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] for line, _ in lines)
        panel = view[0 : 12 + 22 * len(lines), 0:panel_width]
        panel[:] = (panel * 0.35).astype(np.uint8)  # darken behind the text
        for i, (line, colour) in enumerate(lines):
            cv2.putText(view, line, (10, 25 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
        draw_velocity(view, velocity, decision.hover or decision.mode is Mode.HOVER, (width - 100, 90))
        if not args.headless:
            cv2.imshow("Follow-me prototype (Q to quit)", view)

    if args.save:
        cv2.imwrite(str(args.save), view)
    capture.release()
    cv2.destroyAllWindows()
    if voice is not None:
        voice.stop()


if __name__ == "__main__":
    main()
