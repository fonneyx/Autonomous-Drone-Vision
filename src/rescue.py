"""Finding people: detect them in the camera, put them on the map, score the result.

Two detectors with the same output, so the rest of the pipeline is identical:
- SimulatedPersonDetector: fast stand-in for benchmarks over many caves. It
  "detects" a person only if they are inside the camera's field of view, within
  range, not hidden behind rock (ray test), and then only with a probability that
  drops with distance. Bearings get noise, like a real detector's box centre.
- YoloPersonDetector: the real thing; renders the drone's colour camera and runs
  the YOLO model trained on simulator + real images.

A detection is a bearing (angle from the nose) plus the image rows the person
covers. locate() turns it into a map position with the depth camera: the nearest
depth readings in that direction are the person. PersonMarkers fuses repeated
sightings and only confirms a marker after several hits, so a single false alarm
never becomes a marker.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from src.cave_people import PERSON_RADIUS
from src.sim_env import DEPTH_HEIGHT, DEPTH_RANGE, DEPTH_WIDTH, HORIZONTAL_FOV

TAN_H = math.tan(HORIZONTAL_FOV / 2.0)
TAN_V = TAN_H * DEPTH_HEIGHT / DEPTH_WIDTH
YOLO_WEIGHTS = Path(__file__).resolve().parent.parent / "runs" / "train" / "sim-vision" / "weights" / "best.pt"


@dataclass
class Detection:
    bearing: float  # rad from the nose, + = left
    rows: tuple[int, int]  # depth-image rows the person covers (top, bottom)
    confidence: float


@dataclass
class Marker:
    xy: np.ndarray
    hits: int = 1
    sightings: list[np.ndarray] = field(default_factory=list)


def _column(bearing: float) -> int:
    """Depth-image column looking in direction `bearing` (column 0 = left edge)."""
    u = -math.tan(bearing) / TAN_H  # -1 (left) .. 1 (right)
    return int(np.clip((u + 1.0) / 2.0 * DEPTH_WIDTH, 0, DEPTH_WIDTH - 1))


def _row(elevation_tan: float) -> int:
    v = -elevation_tan / TAN_V  # -1 (top) .. 1 (bottom)
    return int(np.clip((v + 1.0) / 2.0 * DEPTH_HEIGHT, 0, DEPTH_HEIGHT - 1))


def locate(env, detection: Detection) -> np.ndarray | None:
    """World (x, y) of a detected person from the depth image, or None if no depth there."""
    depth = env.depth_image_metres()
    column = _column(detection.bearing)
    top, bottom = detection.rows
    patch = depth[max(0, top) : bottom + 1, max(0, column - 1) : column + 2]
    readings = patch[patch < DEPTH_RANGE - 0.01]
    if readings.size == 0:
        return None
    # The person is the nearest thing in their own box; the 30th percentile ignores
    # stray pixels of floor in front and background beside them
    distance = float(np.percentile(readings, 30)) + PERSON_RADIUS
    camera, rotation = env.camera_pose()
    heading = env.yaw + detection.bearing
    return camera[:2] + distance * np.array([math.cos(heading), math.sin(heading)])


class SimulatedPersonDetector:
    """Ground-truth-based detector with realistic limits (field of view, range, occlusion, misses)."""

    def __init__(
        self,
        max_range: float = 8.0,
        sure_range: float = 4.0,
        far_probability: float = 0.4,
        bearing_noise_deg: float = 1.5,
        false_alarm_rate: float = 0.01,
        seed: int | None = None,
    ):
        self.max_range = max_range
        self.sure_range = sure_range
        self.far_probability = far_probability
        self.bearing_noise = math.radians(bearing_noise_deg)
        self.false_alarm_rate = false_alarm_rate
        self.rng = np.random.default_rng(seed)
        self._hit = np.zeros(1, dtype=np.int32)

    def detect(self, env) -> list[Detection]:
        camera, _ = env.camera_pose()
        detections = []
        for slot, xy in zip(env._person_slots, env.people_xy):
            dx, dy = xy - camera[:2]
            distance = math.hypot(dx, dy)
            bearing = math.atan2(math.sin(math.atan2(dy, dx) - env.yaw), math.cos(math.atan2(dy, dx) - env.yaw))
            if distance > self.max_range or abs(bearing) > HORIZONTAL_FOV / 2 - 0.05:
                continue
            # Occlusion: does a ray to the person's chest hit the person first?
            chest = np.array([xy[0], xy[1], 1.2])
            direction = chest - camera
            length = float(np.linalg.norm(direction))
            mujoco.mj_ray(env.model, env.data, camera, direction / length, None, 1, env._drone_body, self._hit)
            if self._hit[0] != slot.proxy_geom:
                continue
            chance = 1.0 if distance <= self.sure_range else 1.0 - (1.0 - self.far_probability) * (
                (distance - self.sure_range) / (self.max_range - self.sure_range)
            )
            if self.rng.random() > chance:
                continue
            head = _row((slot.height - camera[2]) / distance)
            feet = _row((0.0 - camera[2]) / distance)
            detections.append(
                Detection(bearing + self.rng.normal(0.0, self.bearing_noise), (head, feet), float(chance))
            )
        if self.rng.random() < self.false_alarm_rate:
            # A false alarm somewhere in view (rock that looks like a person)
            detections.append(
                Detection(self.rng.uniform(-0.7, 0.7), (DEPTH_HEIGHT // 3, DEPTH_HEIGHT - 4), 0.3)
            )
        return detections


class YoloPersonDetector:
    """Run YOLO on the drone's rendered colour camera; keep 'person' boxes."""

    def __init__(self, weights: Path = YOLO_WEIGHTS, confidence: float = 0.4, image_size: tuple[int, int] = (480, 640)):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.person_class = next(k for k, v in self.model.names.items() if v == "person")
        self.confidence = confidence
        self.height, self.width = image_size
        self._renderer: mujoco.Renderer | None = None
        self._model: mujoco.MjModel | None = None  # the model the renderer was built for
        self.last_frame: np.ndarray | None = None
        self.last_boxes: list[tuple[float, float, float, float, float]] = []

    def detect(self, env) -> list[Detection]:
        # Compare the model object itself, not id(): a freed model's id can be reused by
        # the next cave's model, which would leave the renderer drawing a dead scene
        if self._renderer is None or self._model is not env.model:
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(env.model, self.height, self.width)
            self._model = env.model
        self._renderer.update_scene(env.data, camera="fpv")
        frame = self._renderer.render()
        self.last_frame = frame
        result = self.model.predict(frame[:, :, ::-1], conf=self.confidence, verbose=False)[0]
        detections, self.last_boxes = [], []
        for box in result.boxes:
            if int(box.cls) != self.person_class:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            self.last_boxes.append((x1, y1, x2, y2, float(box.conf)))
            u = (x1 + x2) / self.width - 1.0  # -1 .. 1, box centre
            bearing = math.atan(-u * TAN_H)
            rows = (int(y1 / self.height * DEPTH_HEIGHT), int(y2 / self.height * DEPTH_HEIGHT))
            detections.append(Detection(bearing, rows, float(box.conf)))
        return detections


class PersonMarkers:
    """Fuse located detections into map markers; confirm after `confirm_hits` sightings."""

    def __init__(self, merge_distance: float = 1.5, confirm_hits: int = 3):
        self.merge_distance = merge_distance
        self.confirm_hits = confirm_hits
        self.markers: list[Marker] = []

    def add(self, xy: np.ndarray) -> None:
        for marker in self.markers:
            if np.linalg.norm(marker.xy - xy) < self.merge_distance:
                marker.sightings.append(xy)
                marker.hits += 1
                marker.xy = np.median(np.array(marker.sightings), axis=0)
                return
        self.markers.append(Marker(xy=xy.copy(), sightings=[xy.copy()]))

    def confirmed(self) -> list[np.ndarray]:
        return [marker.xy for marker in self.markers if marker.hits >= self.confirm_hits]


def score_markers(markers: list[np.ndarray], people_xy: np.ndarray, match_distance: float = 2.0) -> dict:
    """people found, position error of found people, false markers."""
    remaining = list(range(len(markers)))
    errors = []
    for person in people_xy:
        if not remaining:
            break
        distances = [float(np.linalg.norm(markers[i] - person)) for i in remaining]
        best = int(np.argmin(distances))
        if distances[best] <= match_distance:
            errors.append(distances[best])
            remaining.pop(best)
    return {
        "people": len(people_xy),
        "found": len(errors),
        "marker_error_m": float(np.mean(errors)) if errors else float("nan"),
        "false_markers": len(remaining),
    }
