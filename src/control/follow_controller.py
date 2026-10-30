"""Follow controller: where the operator is in the image -> velocity command.

Distance from one camera: a person's shoulder-to-hip length is about 0.5 m, so
    distance = focal length (px) * 0.5 m / torso length (px)
(pinhole camera). If the torso is not visible, the full body height (~1.7 m) is used,
but only when the whole body is in the picture. Bearing comes from how far the
person is left/right of the image centre.

Commands (drone body frame, as sent to ArduPilot in GUIDED mode):
    vx        forward (+) / back (-) to hold the follow distance
    vy        0 here (the drone turns to face the operator instead of sliding)
    vz        towards the commanded altitude (the drone's altitude comes from the MTF-01)
    yaw_rate  turn to keep the operator in the image centre
All are proportional controllers with limits (max 2 m/s, 45 deg/s). Closer than the
minimum distance the drone always backs off.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.perception.person_tracker import Person

TORSO_M = 0.5  # typical shoulder-to-hip length of an adult
HEIGHT_M = 1.7  # typical body height


@dataclass
class Velocity:
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_rate: float = 0.0  # rad/s, + = turn left

    @staticmethod
    def hover() -> Velocity:
        return Velocity()


@dataclass
class Measurement:
    distance: float | None  # metres, None if it cannot be estimated
    bearing: float  # rad, + = operator is to the left


class FollowController:
    def __init__(
        self,
        image_width: int,
        image_height: int,
        horizontal_fov_deg: float = 70.0,
        max_speed: float = 2.0,
        max_yaw_rate: float = math.radians(45.0),
        max_climb: float = 0.5,
        distance_gain: float = 0.6,
        yaw_gain: float = 1.2,
        altitude_gain: float = 0.8,
        dead_zone: float = 0.2,
    ):
        self.width, self.height = image_width, image_height
        self.focal = (image_width / 2) / math.tan(math.radians(horizontal_fov_deg) / 2)
        self.max_speed = max_speed
        self.max_yaw_rate = max_yaw_rate
        self.max_climb = max_climb
        self.distance_gain = distance_gain
        self.yaw_gain = yaw_gain
        self.altitude_gain = altitude_gain
        self.dead_zone = dead_zone

    def measure(self, person: Person) -> Measurement:
        cx, _ = person.centre
        bearing = math.atan((self.width / 2 - cx) / self.focal)
        torso = person.torso_length()
        if torso is not None and torso > 5:
            return Measurement(self.focal * TORSO_M / torso, bearing)
        x1, y1, x2, y2 = person.box
        whole_body = y1 > 3 and y2 < self.height - 3
        if whole_body and person.height > 10:
            return Measurement(self.focal * HEIGHT_M / person.height, bearing)
        return Measurement(None, bearing)

    def command(
        self,
        person: Person,
        target_distance: float,
        min_distance: float,
        target_altitude: float | None = None,
        altitude: float | None = None,
    ) -> tuple[Velocity, Measurement]:
        m = self.measure(person)
        velocity = Velocity()
        velocity.yaw_rate = max(-self.max_yaw_rate, min(self.max_yaw_rate, self.yaw_gain * m.bearing))
        if m.distance is not None:
            error = m.distance - target_distance
            if abs(error) > self.dead_zone:
                velocity.vx = max(-self.max_speed, min(self.max_speed, self.distance_gain * error))
            if m.distance < min_distance:
                velocity.vx = min(velocity.vx, -0.3)  # too close: always back off
            # Turn first, then drive: slow down while the operator is far off-centre
            velocity.vx *= max(0.0, math.cos(m.bearing)) ** 2
        if target_altitude is not None and altitude is not None:
            velocity.vz = max(-self.max_climb, min(self.max_climb, self.altitude_gain * (target_altitude - altitude)))
        return velocity, m
