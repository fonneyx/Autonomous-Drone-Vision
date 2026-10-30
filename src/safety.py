"""Rule-based safety shield between the pilot and the flight controller.

The RL pilot is fast but has no guarantees. The shield watches the same range scan
and, when anything is closer than the safety distances, replaces the pilot's command
with the classical follow-the-gap command (which slows down and steers away). Near
a person it always takes over and flies slowly. It can also cap speed. This is the "safety shield can always override RL" rule from the
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
        person_speed: float = 0.5,
    ):
        """front_distance applies to rays within +/- front_cone rad of the nose; side_distance
        to all other rays. speed_limit (m/s) caps any command; None = no cap. person_speed
        (m/s) is the cap while a person is close."""
        self.max_speed = max_speed
        self.max_yaw_rate = max_yaw_rate
        self.front_distance = front_distance
        self.side_distance = side_distance
        self.front_cone = front_cone
        self.speed_limit = speed_limit
        self.person_speed = person_speed
        self.steps = 0
        self.overrides = 0

    @property
    def override_rate(self) -> float:
        return self.overrides / max(1, self.steps)

    def unsafe(self, angles: np.ndarray, ranges: np.ndarray) -> bool:
        front = np.abs(angles) <= self.front_cone
        return bool((ranges[front] < self.front_distance).any() or (ranges[~front] < self.side_distance).any())

    def filter(
        self,
        action: np.ndarray,
        angles: np.ndarray,
        ranges: np.ndarray,
        target_angle: float = 0.0,
        near_person: bool = False,
        target_distance: float | None = None,
    ) -> np.ndarray:
        """Return the action to send: the pilot's own, or the classical one if unsafe.

        near_person: a person was seen close by. People are thin and can stand beside
        the drone where the camera cannot see, so the shield then always takes over and
        caps the speed at person_speed.
        """
        self.steps += 1
        action = np.asarray(action, dtype=np.float32).copy()
        if near_person or self.unsafe(angles, ranges):
            self.overrides += 1
            action = follow_the_gap(
                angles, ranges, self.max_speed, self.max_yaw_rate,
                target_angle=target_angle, target_distance=target_distance,
            )
        limit = self.speed_limit
        if near_person:
            limit = self.person_speed if limit is None else min(limit, self.person_speed)
        if limit is not None:
            action[0] = min(action[0], limit / self.max_speed * 2.0 - 1.0)
        return action
