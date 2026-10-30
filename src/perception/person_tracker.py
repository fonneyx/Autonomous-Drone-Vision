"""Find and track people in a camera frame, with 17 body keypoints each.

One YOLO11-pose model does detection, keypoints and (with ByteTrack) tracking, so
the same person keeps the same track id from frame to frame. Keypoints follow the
COCO order (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles); the
re-identification uses them to find the torso and legs (clothes colours) and the
follow controller to measure distance.

Works the same on a laptop webcam and on the drone's video stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Downloaded by Ultralytics on first use (~20 MB)
POSE_WEIGHTS = str(Path(__file__).resolve().parents[2] / "models" / "yolo11s-pose.pt")

# COCO keypoint indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
MIN_VISIBILITY = 0.5


@dataclass
class Person:
    track_id: int | None  # stable id from the tracker; None until the tracker confirms it
    box: np.ndarray  # x1, y1, x2, y2 in pixels
    confidence: float
    keypoints: np.ndarray  # (17, 3): x, y, visibility

    @property
    def height(self) -> float:
        return float(self.box[3] - self.box[1])

    @property
    def width(self) -> float:
        return float(self.box[2] - self.box[0])

    @property
    def centre(self) -> tuple[float, float]:
        return float(self.box[0] + self.box[2]) / 2, float(self.box[1] + self.box[3]) / 2

    def visible(self, *indices: int) -> bool:
        return bool(all(self.keypoints[i, 2] >= MIN_VISIBILITY for i in indices))

    def torso_length(self) -> float | None:
        """Shoulder-to-hip distance in pixels, or None if not visible."""
        if not self.visible(L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
            return None
        shoulders = (self.keypoints[L_SHOULDER, :2] + self.keypoints[R_SHOULDER, :2]) / 2
        hips = (self.keypoints[L_HIP, :2] + self.keypoints[R_HIP, :2]) / 2
        return float(np.linalg.norm(shoulders - hips))


class PersonTracker:
    """YOLO11-pose + ByteTrack on each frame; returns the people in it."""

    def __init__(self, weights: str = POSE_WEIGHTS, confidence: float = 0.4, device: str | None = None):
        import torch
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.confidence = confidence
        self.device = device if device is not None else ("0" if torch.cuda.is_available() else "cpu")

    def __call__(self, frame_bgr: np.ndarray) -> list[Person]:
        result = self.model.track(
            frame_bgr, persist=True, classes=[0], conf=self.confidence, tracker="bytetrack.yaml",
            device=self.device, verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else [None] * len(boxes)
        if result.keypoints is not None:
            keypoints = result.keypoints.data.cpu().numpy()
        else:
            keypoints = np.zeros((len(boxes), 17, 3))
        return [Person(ids[i], boxes[i], float(confidences[i]), keypoints[i]) for i in range(len(boxes))]
