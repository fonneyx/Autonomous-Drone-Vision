"""Classical (non-learned) pilots, the baseline the RL pilot is compared against.

follow_the_gap: steer towards the most open direction in the depth scan and slow
down when the way ahead is blocked. Simple, explainable, needs no training, and
runs unchanged on a real drone with any forward range sensor.
"""

from __future__ import annotations

import math

import numpy as np


def follow_the_gap(
    angles: np.ndarray,
    ranges: np.ndarray,
    max_speed: float,
    max_yaw_rate: float,
    cruise_speed: float = 1.5,
    drone_radius: float = 0.35,
    stop_distance: float = 0.9,
    look_ahead: float = 6.0,
    straight_bias: float = 0.6,
    side_guard: float = 1.0,
    target_angle: float = 0.0,
    target_distance: float | None = None,
) -> np.ndarray:
    """Return the env action [speed, yaw rate] in [-1, 1] from a horizontal scan.

    angles: ray angle relative to the nose (rad, positive = left); ranges: horizontal
    distance per ray (m). Same order as CaveExploreEnv.scan(). Each ray's clearance is the shortest range among
    neighbours within the drone's width at that distance (obstacle inflation).
    target_angle: direction to a waypoint relative to the nose (+ = left); the pilot
    prefers gaps close to it. 0 = keep flying where the nose points.
    target_distance: distance to that waypoint (m). If the direct line to it is clear,
    the pilot heads straight for it instead of for the widest gap (otherwise a wide
    opening elsewhere can lure it away from a narrower but free path, back and forth).
    """
    ranges = np.minimum(np.asarray(ranges, dtype=np.float64), look_ahead)
    angles = np.asarray(angles, dtype=np.float64)
    spacing = abs(float(angles[1] - angles[0])) if len(angles) > 1 else 1.0
    clearance = ranges.copy()
    for index, distance in enumerate(ranges):
        # Half-width of the drone seen from here, in rays
        half = int(math.ceil(math.atan2(drone_radius, max(distance, 0.3)) / spacing))
        clearance[index] = ranges[max(0, index - half) : index + half + 1].min()

    target_angle = math.atan2(math.sin(target_angle), math.cos(target_angle))
    score = clearance - straight_bias * np.abs(angles - target_angle)
    best = int(np.argmax(score))
    direct_clear = False
    if target_distance is not None and abs(target_angle) <= abs(float(angles[0])):
        direct = int(np.argmin(np.abs(angles - target_angle)))
        if clearance[direct] >= min(target_distance, look_ahead) or clearance[direct] >= 2.0:
            best, direct_clear = direct, True
    centre = int(np.argmin(np.abs(angles)))
    ahead = clearance[max(0, centre - 2) : centre + 3].min()

    if abs(target_angle) > abs(float(angles[0])):
        # Waypoint outside the field of view: turn on the spot to face it first
        speed, yaw_rate = 0.0, math.copysign(max_yaw_rate * 0.6, target_angle)
    elif clearance[best] < stop_distance * 1.5 and not direct_clear:
        # Dead end or boxed in (and no clear line to a nearby waypoint): stop and turn on
        # the spot, towards the waypoint (else left)
        speed, yaw_rate = 0.0, math.copysign(max_yaw_rate, target_angle) if target_angle else max_yaw_rate
    else:
        yaw_rate = 1.5 * angles[best]
        # Side guard: walls closer than side_guard push the nose away from them. The
        # camera cannot see the drone's flanks, so never turn into something close.
        closeness = np.clip((side_guard - ranges) / side_guard, 0.0, 1.0)
        yaw_rate -= 2.0 * max_yaw_rate * float(np.sum(closeness * np.sign(angles)) / max(1, np.count_nonzero(closeness)))
        yaw_rate = float(np.clip(yaw_rate, -max_yaw_rate, max_yaw_rate))
        # Slow for obstacles ahead, for anything close by and for sharp turns
        speed = cruise_speed * float(np.clip((ahead - stop_distance) / 2.0, 0.0, 1.0))
        if direct_clear:
            speed = max(speed, 0.3)  # creep along a clear line to a close waypoint
        speed *= 1.0 - 0.7 * float(closeness.max())
        speed *= max(0.2, 1.0 - abs(yaw_rate) / max_yaw_rate)
    return np.array([speed / max_speed * 2.0 - 1.0, yaw_rate / max_yaw_rate], dtype=np.float32)
