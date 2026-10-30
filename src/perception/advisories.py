"""Convert image detections into conservative visual flight advisories.

These advisories are not flight-control commands. A monocular image cannot
reliably estimate obstacle distance or guarantee a safe direction of travel.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class Guidance(str, Enum):
    CLEAR = "CLEAR"
    CAUTION = "CAUTION"
    HOLD = "HOLD"


@dataclass(frozen=True)
class Detection:
    """A detection box in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str


def assess_detections(
    detections: Iterable[Detection],
    frame_width: int,
    frame_height: int,
    confidence_threshold: float = 0.35,
    center_zone_ratio: float = 0.40,
) -> Guidance:
    """Return HOLD for a central detection, CAUTION for other detections."""
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame dimensions must be positive.")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1.")
    if not 0.0 < center_zone_ratio <= 1.0:
        raise ValueError("center_zone_ratio must be greater than 0 and at most 1.")

    center_left = frame_width * (1.0 - center_zone_ratio) / 2.0
    center_right = frame_width - center_left
    has_detection = False

    for detection in detections:
        if detection.confidence < confidence_threshold:
            continue
        has_detection = True
        box_center_x = (detection.x1 + detection.x2) / 2.0
        if center_left <= box_center_x <= center_right:
            return Guidance.HOLD

    return Guidance.CAUTION if has_detection else Guidance.CLEAR
