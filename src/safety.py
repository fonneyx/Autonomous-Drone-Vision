"""Rule-based safety shield between the pilot and the flight controller.

The RL pilot is fast but has no guarantees. The shield watches the same range scan
and, when anything is closer than the safety distances, replaces the pilot's command
with the classical follow-the-gap command (which slows down and steers away). It
also caps speed. This is the "safety shield can always override RL" rule from the
project architecture, and it runs unchanged on the real drone.
"""

from __future__ import annotations

import numpy as np

from src.baselines import follow_the_gap


class SafetyShield:
    """Override unsafe pilot actions; counts how often it had to step in."""

    def __init__(
        self,
        max_speed: float,
        max_yaw_rate: float,
        front_distance: float = 1.2,
        side_distance: float = 0.6,
        front_cone: float = 0.35,
        speed_limit: float | None = None,
    ):
        """front_distance applies to rays within +/- front_cone rad of the nose; side_distance
        to all other rays. speed_limit (m/s) caps any command; None = no cap."""
        self.max_speed = max_speed
        self.max_yaw_rate = max_yaw_rate
        self.front_distance = front_distance
        self.side_distance = side_distance
        self.front_cone = front_cone
        self.speed_limit = speed_limit
        self.steps = 0
        self.overrides = 0

    @property
    def override_rate(self) -> float:
        return self.overrides / max(1, self.steps)

    def unsafe(self, angles: np.ndarray, ranges: np.ndarray) -> bool:
        front = np.abs(angles) <= self.front_cone
        return bool((ranges[front] < self.front_distance).any() or (ranges[~front] < self.side_distance).any())

    def filter(
        self, action: np.ndarray, angles: np.ndarray, ranges: np.ndarray, target_angle: float = 0.0
    ) -> np.ndarray:
        """Return the action to send: the pilot's own, or the classical one if unsafe."""
        self.steps += 1
        action = np.asarray(action, dtype=np.float32).copy()
        if self.unsafe(angles, ranges):
            self.overrides += 1
            action = follow_the_gap(angles, ranges, self.max_speed, self.max_yaw_rate, target_angle=target_angle)
        if self.speed_limit is not None:
            limit = self.speed_limit / self.max_speed * 2.0 - 1.0
            action[0] = min(action[0], limit)
        return action
