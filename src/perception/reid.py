"""Re-identification: keep following *the operator*, not a stranger walking by.

The operator is locked once (they raise an open palm towards the drone, or press a
key). From then on the tracker's id is followed. Because tracker ids can switch when
people cross or the operator is hidden for a moment, every locked person is also
compared with an appearance template: colour histograms of the torso and the legs
(clothes). The template is learnt at lock time and slowly updated while tracking is
certain.

- Lost operator (no track with that id): another person is re-acquired only if they
  match the template for several frames in a row.
- Suspected id switch (tracked person stops matching): the lock is dropped.

Colour histograms are simple, fast and need no training; their limit (two people in
the same clothes) is a known limitation — a deep re-ID network (e.g. OSNet) is the
upgrade path.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.perception.person_tracker import L_HIP, L_KNEE, L_SHOULDER, R_HIP, R_KNEE, R_SHOULDER, Person

H_BINS, S_BINS = 16, 8


def _region_hist(frame_bgr: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray | None:
    h, w = frame_bgr.shape[:2]
    x0, x1 = int(max(0, min(x0, x1))), int(min(w, max(x0, x1)))
    y0, y1 = int(max(0, min(y0, y1))), int(min(h, max(y0, y1)))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    hsv = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    # Ignore very dark / washed-out pixels: their hue is noise
    mask = cv2.inRange(hsv, (0, 30, 30), (180, 255, 255))
    hist = cv2.calcHist([hsv], [0, 1], mask, [H_BINS, S_BINS], [0, 180, 0, 256])
    total = hist.sum()
    if total < 10:
        return None
    return (hist / total).astype(np.float32)


def appearance(frame_bgr: np.ndarray, person: Person) -> list[np.ndarray | None]:
    """[torso histogram, legs histogram] of a person; parts that are not visible are None."""
    x1, y1, x2, y2 = person.box
    if person.visible(L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
        k = person.keypoints
        left = min(k[L_SHOULDER, 0], k[R_SHOULDER, 0], k[L_HIP, 0], k[R_HIP, 0])
        right = max(k[L_SHOULDER, 0], k[R_SHOULDER, 0], k[L_HIP, 0], k[R_HIP, 0])
        top = min(k[L_SHOULDER, 1], k[R_SHOULDER, 1])
        hip = max(k[L_HIP, 1], k[R_HIP, 1])
        torso = _region_hist(frame_bgr, left, top, right, hip)
        if person.visible(L_KNEE, R_KNEE):
            legs = _region_hist(frame_bgr, left, hip, right, max(k[L_KNEE, 1], k[R_KNEE, 1]))
        else:
            legs = _region_hist(frame_bgr, left, hip, right, y2)
    else:  # fall back to fixed fractions of the box
        width, height = x2 - x1, y2 - y1
        torso = _region_hist(frame_bgr, x1 + 0.2 * width, y1 + 0.2 * height, x2 - 0.2 * width, y1 + 0.5 * height)
        legs = _region_hist(frame_bgr, x1 + 0.2 * width, y1 + 0.55 * height, x2 - 0.2 * width, y1 + 0.85 * height)
    return [torso, legs]


def distance(a: list[np.ndarray | None], b: list[np.ndarray | None]) -> float:
    """0 = identical clothes, 1 = completely different (Bhattacharyya, averaged over visible parts)."""
    values = [
        cv2.compareHist(x, y, cv2.HISTCMP_BHATTACHARYYA) for x, y in zip(a, b) if x is not None and y is not None
    ]
    return float(np.mean(values)) if values else 1.0


class OperatorLock:
    """Which tracked person is the operator, kept robust to id switches and short losses."""

    def __init__(
        self,
        match_threshold: float = 0.45,
        reacquire_frames: int = 3,
        mismatch_frames: int = 8,
        learning_rate: float = 0.05,
    ):
        self.match_threshold = match_threshold
        self.reacquire_frames = reacquire_frames
        self.mismatch_frames = mismatch_frames
        self.learning_rate = learning_rate
        self.track_id: int | None = None
        self.template: list[np.ndarray | None] | None = None
        self._candidate: int | None = None
        self._candidate_count = 0
        self._mismatch_count = 0
        self.last_distance = 1.0

    @property
    def locked(self) -> bool:
        return self.template is not None

    def lock(self, frame_bgr: np.ndarray, person: Person) -> None:
        self.track_id = person.track_id
        self.template = appearance(frame_bgr, person)
        self._candidate, self._candidate_count, self._mismatch_count = None, 0, 0

    def release(self) -> None:
        self.track_id, self.template = None, None

    def _learn(self, current: list[np.ndarray | None]) -> None:
        assert self.template is not None
        for i, (old, new) in enumerate(zip(self.template, current)):
            if new is None:
                continue
            self.template[i] = new if old is None else (1 - self.learning_rate) * old + self.learning_rate * new

    def update(self, frame_bgr: np.ndarray, people: list[Person]) -> Person | None:
        """The operator in this frame, or None if not visible / not sure."""
        if self.template is None:
            return None
        tracked = next((p for p in people if p.track_id is not None and p.track_id == self.track_id), None)
        if tracked is not None:
            features = appearance(frame_bgr, tracked)
            self.last_distance = distance(self.template, features)
            if self.last_distance < self.match_threshold:
                self._mismatch_count = 0
                self._learn(features)
                return tracked
            self._mismatch_count += 1
            if self._mismatch_count >= self.mismatch_frames:
                self.track_id = None  # probably an id switch: stop trusting this track, search by looks
            return None

        # Operator not tracked: look for someone who matches the template
        best, best_distance = None, self.match_threshold
        for person in people:
            d = distance(self.template, appearance(frame_bgr, person))
            if d < best_distance:
                best, best_distance = person, d
        self.last_distance = best_distance
        if best is None:
            self._candidate, self._candidate_count = None, 0
            return None
        key = best.track_id
        self._candidate_count = self._candidate_count + 1 if key == self._candidate else 1
        self._candidate = key
        if self._candidate_count >= self.reacquire_frames:
            self.track_id = best.track_id
            self._mismatch_count = 0
            return best
        return None
