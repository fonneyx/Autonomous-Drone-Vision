"""20-second gesture check: where are the hands relative to the body? Saves annotated snapshots.

    python scripts/tools/gesture_check.py

Follow the screen: hands DOWN (5 s), raise an OPEN PALM (5 s), raise a FIST (5 s), WAVE (5 s).
Writes to runs/diagnostics/<time>/:
    frames.csv    per frame and hand: label, wrist height, shoulder line, the "raised" limit, raised yes/no
    phase_*.jpg   one annotated picture per phase (shoulder line yellow, raised limit red, wrists cyan)
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.perception.gestures import HandReader, hands_of, raised_hands  # noqa: E402
from src.perception.person_tracker import L_SHOULDER, R_SHOULDER, PersonTracker  # noqa: E402

PHASES = ["hands DOWN", "raise an OPEN PALM", "raise a FIST", "WAVE your palm"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--phase-seconds", type=float, default=5.0)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    out = ROOT / "runs" / "diagnostics" / f"{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(args.video)) if args.video else cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    tracker, reader = PersonTracker(), HandReader()
    rows = []
    start = time.monotonic() + 3.0  # 3 s to get in place
    saved: set[int] = set()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        now = time.monotonic()
        t = now - start
        phase = int(t // args.phase_seconds) if t >= 0 else -1
        if phase >= len(PHASES):
            break
        people = tracker(frame)
        hands = reader(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), now)
        view = frame.copy()
        tester = max(people, key=lambda p: p.height) if people else None
        if tester is not None:
            own = hands_of(tester.box, hands)
            raised = raised_hands(tester, hands)
            shoulders_ok = tester.visible(L_SHOULDER, R_SHOULDER)
            shoulder_y = float(tester.keypoints[[L_SHOULDER, R_SHOULDER], 1].mean())
            limit = None
            if shoulders_ok:
                # same limit as raised_hands: a wrist must be above (smaller y than) this line
                torso = tester.torso_length()
                if torso is None:
                    torso = max(1.3 * abs(float(tester.keypoints[L_SHOULDER, 0] - tester.keypoints[R_SHOULDER, 0])),
                                0.25 * tester.height)
                limit = shoulder_y + 0.5 * torso
                cv2.line(view, (0, int(shoulder_y)), (view.shape[1], int(shoulder_y)), (0, 220, 255), 1)
                cv2.line(view, (0, int(limit)), (view.shape[1], int(limit)), (40, 40, 230), 2)
            x1, y1, x2, y2 = tester.box.astype(int)
            cv2.rectangle(view, (x1, y1), (x2, y2), (60, 200, 60), 2)
            for i in (L_SHOULDER, R_SHOULDER):
                cv2.circle(view, tuple(tester.keypoints[i, :2].astype(int)), 6, (0, 220, 255), -1)
            for h in hands:
                cv2.circle(view, (int(h.wrist[0]), int(h.wrist[1])), 7, (230, 200, 40), -1)
                cv2.putText(view, f"{h.label} {'RAISED' if h in raised else ('own' if h in own else 'other')}",
                            (int(h.wrist[0]) + 8, int(h.wrist[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 200, 40), 1)
                rows.append({"t": f"{t:.2f}", "phase": PHASES[phase] if phase >= 0 else "get ready",
                             "label": h.label, "score": f"{h.score:.2f}", "wrist_x": f"{h.wrist[0]:.0f}",
                             "wrist_y": f"{h.wrist[1]:.0f}", "own": h in own, "raised": h in raised,
                             "shoulders_visible": shoulders_ok, "shoulder_y": f"{shoulder_y:.0f}",
                             "limit_y": f"{limit:.0f}" if limit is not None else "",
                             "box": " ".join(f"{v:.0f}" for v in tester.box), "frame_h": view.shape[0]})
            if not hands:
                rows.append({"t": f"{t:.2f}", "phase": PHASES[phase] if phase >= 0 else "get ready", "label": "(no hand)",
                             "shoulders_visible": shoulders_ok, "shoulder_y": f"{shoulder_y:.0f}",
                             "box": " ".join(f"{v:.0f}" for v in tester.box), "frame_h": view.shape[0]})
        title = "get in place" if phase < 0 else f"{PHASES[phase]}  ({args.phase_seconds - t % args.phase_seconds:.0f} s)"
        cv2.putText(view, title, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(view, title, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        if phase >= 0 and phase not in saved and t % args.phase_seconds > args.phase_seconds * 0.6:
            cv2.imwrite(str(out / f"phase_{phase + 1}.jpg"), view)
            saved.add(phase)
        if not args.headless:
            cv2.imshow("Gesture check", view)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    capture.release()
    cv2.destroyAllWindows()
    fields = ["t", "phase", "label", "score", "wrist_x", "wrist_y", "own", "raised", "shoulders_visible",
              "shoulder_y", "limit_y", "box", "frame_h"]
    with (out / "frames.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
