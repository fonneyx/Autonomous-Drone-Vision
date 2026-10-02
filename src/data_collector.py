"""Persist frames that contain detections below the reporting threshold."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from typing import Iterable

import cv2
import numpy as np


class LowConfidenceFrameCollector:
    """Write a frame when any detection falls in the configured confidence band."""

    def __init__(self, output_dir: str | Path, minimum_confidence: float = 0.10):
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1.")
        self.output_dir = Path(output_dir)
        self.minimum_confidence = minimum_confidence

    def save_if_low_confidence(
        self,
        frame: np.ndarray,
        confidences: Iterable[float],
        reporting_threshold: float,
    ) -> Path | None:
        """Save and return the frame path if a confidence is below the report threshold."""
        if not 0.0 <= reporting_threshold <= 1.0:
            raise ValueError("reporting_threshold must be between 0 and 1.")
        if not any(
            self.minimum_confidence <= confidence < reporting_threshold
            for confidence in confidences
        ):
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_path = self.output_dir / f"{timestamp}_{uuid4().hex[:8]}.jpg"
        if not cv2.imwrite(str(output_path), frame):
            raise OSError(f"OpenCV could not write collected frame: {output_path}")
        return output_path
