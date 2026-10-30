"""Hand gestures with MediaPipe: open palm = stop, fist = land, wave = come back.

MediaPipe's ready-made GestureRecognizer finds hands and labels their shape
("Open_Palm", "Closed_Fist", ...). Only the operator's hands count: a hand belongs
to the operator if its wrist lies inside the operator's body box.

Shapes become commands only after timing rules, so random movements do not
trigger anything:
    STOP       open palm held still for 1 s
    LAND       fist held for 1.5 s (longer: landing is the most serious command)
    COME_BACK  open palm moving side to side (a wave) within 1.5 s

A command fires once; the hand must change before it can fire again. The timing
logic (GestureTimer) is independent of MediaPipe, so it is unit-tested with
made-up observations.
"""

from __future__ import annotations

import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.commands import Command

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/"
    "gesture_recognizer.task"
)
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "gesture_recognizer.task"

OPEN_PALM, FIST = "Open_Palm", "Closed_Fist"


@dataclass
class Hand:
    label: str  # MediaPipe gesture name, e.g. "Open_Palm", "Closed_Fist", "None"
    score: float
    wrist: tuple[float, float]  # pixels
    palm_width: float  # pixels (index-finger base to little-finger base): the hand's own scale
    landmarks: np.ndarray  # (21, 2) pixels, for drawing


class HandReader:
    """MediaPipe GestureRecognizer on video frames."""

    def __init__(self, model_path: Path = MODEL_PATH, max_hands: int = 4, min_score: float = 0.5):
        from mediapipe.tasks.python import BaseOptions, vision

        if not model_path.is_file():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading MediaPipe gesture model to {model_path} (~8 MB) ...")
            urllib.request.urlretrieve(MODEL_URL, model_path)
        options = vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_score,
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)
        self._last_ms = -1

    def __call__(self, frame_rgb: np.ndarray, time_s: float) -> list[Hand]:
        import mediapipe as mp

        ms = max(int(time_s * 1000), self._last_ms + 1)  # timestamps must increase
        self._last_ms = ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        result = self.recognizer.recognize_for_video(image, ms)
        height, width = frame_rgb.shape[:2]
        hands = []
        for gestures, landmarks in zip(result.gestures, result.hand_landmarks):
            points = np.array([[p.x * width, p.y * height] for p in landmarks])
            top = gestures[0] if gestures else None
            hands.append(
                Hand(
                    label=top.category_name if top else "None",
                    score=float(top.score) if top else 0.0,
                    wrist=(float(points[0, 0]), float(points[0, 1])),
                    palm_width=float(np.linalg.norm(points[5] - points[17])),
                    landmarks=points,
                )
            )
        return hands


def hands_of(box: np.ndarray, hands: list[Hand], margin: float = 0.25) -> list[Hand]:
    """Hands whose wrist is inside the person's box (widened by `margin` for raised/outstretched arms)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return [
        hand for hand in hands
        if x1 - margin * w <= hand.wrist[0] <= x2 + margin * w and y1 - margin * h <= hand.wrist[1] <= y2 + margin * h
    ]


@dataclass
class HandObservation:
    label: str
    x: float  # wrist x in pixels
    palm_width: float  # pixels


def main_hand(hands: list[Hand]) -> HandObservation | None:
    """The hand that matters most this frame: an open palm or fist first, else the most confident."""
    if not hands:
        return None
    meaningful = [h for h in hands if h.label in (OPEN_PALM, FIST)]
    hand = max(meaningful or hands, key=lambda h: h.score)
    return HandObservation(hand.label, hand.wrist[0], max(hand.palm_width, 1.0))


class GestureTimer:
    """Turns per-frame hand shapes into commands (hold times, wave detection, fire once)."""

    def __init__(
        self,
        stop_hold: float = 1.0,
        land_hold: float = 1.5,
        wave_window: float = 1.5,
        wave_reversals: int = 2,
        wave_amplitude: float = 1.0,
        still_amplitude: float = 0.6,
    ):
        """wave_amplitude / still_amplitude: side-to-side hand movement in palm widths that counts as a
        wave / as holding still. wave_reversals: direction changes needed for a wave."""
        self.stop_hold = stop_hold
        self.land_hold = land_hold
        self.wave_window = wave_window
        self.wave_reversals = wave_reversals
        self.wave_amplitude = wave_amplitude
        self.still_amplitude = still_amplitude
        self.history: deque[tuple[float, str, float, float]] = deque()  # (time, label, x, palm width)
        self.shape = "None"
        self.since = 0.0
        self.fired = False

    def _window(self, now: float, seconds: float) -> list[tuple[float, str, float, float]]:
        return [entry for entry in self.history if now - entry[0] <= seconds]

    def _is_wave(self, now: float) -> bool:
        recent = self._window(now, self.wave_window)
        palms = [(x, w) for _, label, x, w in recent if label == OPEN_PALM]
        if len(palms) < 5 or len(palms) < 0.6 * len(recent):
            return False
        xs = np.array([x for x, _ in palms])
        scale = float(np.median([w for _, w in palms]))
        if xs.max() - xs.min() < self.wave_amplitude * scale:
            return False
        steps = np.diff(xs)
        steps = steps[np.abs(steps) > 0.05 * scale]  # ignore jitter
        reversals = int(np.sum(np.sign(steps[1:]) != np.sign(steps[:-1]))) if len(steps) > 1 else 0
        return reversals >= self.wave_reversals

    def _is_still(self, now: float) -> bool:
        recent = [(x, w) for t, label, x, w in self._window(now, self.stop_hold) if label == OPEN_PALM]
        if not recent:
            return False
        xs = np.array([x for x, _ in recent])
        return xs.max() - xs.min() < self.still_amplitude * float(np.median([w for _, w in recent]))

    def progress(self, now: float) -> tuple[str, float]:
        """(what is being held, 0..1) for the on-screen progress bar."""
        if self.fired or self.shape not in (OPEN_PALM, FIST):
            return "", 0.0
        hold = self.stop_hold if self.shape == OPEN_PALM else self.land_hold
        return ("stop" if self.shape == OPEN_PALM else "land"), min(1.0, (now - self.since) / hold)

    def update(self, hand: HandObservation | None, now: float) -> Command | None:
        """Feed one frame (the operator's main hand, or None); returns a command when one is confirmed."""
        label = hand.label if hand is not None else "None"
        if hand is not None:
            self.history.append((now, label, hand.x, hand.palm_width))
        while self.history and now - self.history[0][0] > max(self.wave_window, self.land_hold) + 0.5:
            self.history.popleft()

        if label != self.shape:
            # Waving can make the recognizer flicker for a frame: keep the palm "held" through short gaps
            if not (self.shape == OPEN_PALM and label == "None" and self._window(now, 0.3)):
                self.shape, self.since, self.fired = label, now, False

        if self.fired:
            return None
        if self.shape == OPEN_PALM and self._is_wave(now):
            self.fired = True
            return Command.COME_BACK
        if self.shape == OPEN_PALM and now - self.since >= self.stop_hold and self._is_still(now):
            self.fired = True
            return Command.STOP
        if self.shape == FIST and now - self.since >= self.land_hold:
            self.fired = True
            return Command.LAND
        return None
