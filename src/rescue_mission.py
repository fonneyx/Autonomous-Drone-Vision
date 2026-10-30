"""The full search-and-rescue mission: explore, mark people, come home before the battery runs out.

One RescueMission.step() is one 10 Hz control cycle:
1. explorer gives a waypoint (frontier, or home once returning),
2. every few steps the detector looks for people; sightings are placed on the map
   with the depth camera and fused into markers, and a keep-out zone around each
   sighting stops paths from passing close to a person,
3. pilot (classical or RL) turns the waypoint into a velocity command,
4. optional safety shield overrides it near walls,
5. battery check: return home when the time left is close to the time needed to
   fly home (planned path length / assumed return speed, with a safety factor and
   reserve), or as soon as the cave is fully explored.

The mission ends at home (success), on a crash, or when the battery runs out.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.baselines import follow_the_gap
from src.cave_env import CaveExploreEnv
from src.explorer import FrontierExplorer
from src.rescue import PersonMarkers, locate, score_markers
from src.safety import SafetyShield
from src.sim_env import CONTROL_DT, MAX_SPEED, MAX_YAW_RATE


class RescueMission:
    def __init__(
        self,
        env: CaveExploreEnv,
        detector,
        policy=None,
        shield: SafetyShield | None = None,
        battery_seconds: float = 240.0,
        detect_every: int = 3,
        return_speed: float = 1.0,
        return_factor: float = 1.3,
        return_margin: float = 15.0,
        person_keep_out: float = 1.0,
        person_caution: float = 2.0,
    ):
        """policy: a Stable-Baselines3 model, or None for the classical follow-the-gap pilot.
        Go home when time left < return_factor * (path home / return_speed) + return_margin (s).
        return_speed is an assumed speed along the known path home (m/s); the average speed
        while exploring is no guide, because exploring includes a lot of turning on the spot.
        person_keep_out: radius (m) of the soft keep-out zone around each person sighting.
        person_caution: within this distance (m) of a sighting, the safety shield (if any)
        takes over and flies slowly.
        """
        self.env = env
        self.detector = detector
        self.policy = policy
        self.shield = shield
        self.battery_seconds = battery_seconds
        self.detect_every = detect_every
        self.return_speed = return_speed
        self.return_factor = return_factor
        self.return_margin = return_margin
        self.person_keep_out = person_keep_out
        self.person_caution = person_caution
        env.max_steps = round(battery_seconds / CONTROL_DT)
        env.done_coverage = None

    def reset(self, seed: int | None = None) -> None:
        self.obs, self.info = self.env.reset(seed=seed)
        self.explorer = FrontierExplorer(self.env.map, home=self.env.layout.home_xy)
        self.markers = PersonMarkers()
        self.phase = "explore"  # explore -> return -> home | crashed | battery empty
        self.return_reason = ""
        self.return_time = math.nan
        self.flown = 0.0
        self._last_xy = self.env.position[:2].copy()
        if self.shield is not None:
            self.shield.steps = self.shield.overrides = 0

    @property
    def elapsed(self) -> float:
        return self.env._step_count * CONTROL_DT

    @property
    def finished(self) -> bool:
        return self.phase in ("home", "crashed", "battery empty")

    def _check_battery(self) -> None:
        if self.explorer.returning:
            return
        reason = ""
        if self.explorer.done:
            reason = "cave explored"
        elif math.isfinite(self.explorer.distance_home):  # inf = not planned yet
            time_home = self.explorer.distance_home / self.return_speed
            if self.battery_seconds - self.elapsed < self.return_factor * time_home + self.return_margin:
                reason = "battery"
        if reason:
            self.explorer.go_home()
            self.phase, self.return_reason, self.return_time = "return", reason, self.elapsed

    def _pilot(self, target_angle: float, target_distance: float | None) -> np.ndarray:
        angles, ranges = self.env.scan()
        if self.policy is None:
            action = follow_the_gap(
                angles, ranges, MAX_SPEED, MAX_YAW_RATE, target_angle=target_angle, target_distance=target_distance
            )
        else:
            action, _ = self.policy.predict(self.env._observation(), deterministic=True)
        if self.shield is not None:
            xy = self.env.position[:2]
            near_person = any(
                np.linalg.norm(marker.xy - xy) < self.person_caution for marker in self.markers.markers
            )
            action = self.shield.filter(
                action, angles, ranges, target_angle=target_angle,
                near_person=near_person, target_distance=target_distance,
            )
        return action

    def step(self) -> bool:
        """Run one control cycle; returns True when the mission is over."""
        env = self.env
        self._check_battery()
        env.goal = self.explorer.update(env.position[:2])
        if self.explorer.at_home:
            self.phase = "home"
            return True

        if env._step_count % self.detect_every == 0:
            for detection in self.detector.detect(env):
                xy = locate(env, detection)
                if xy is not None:
                    self.markers.add(xy)
                    # Never fly close to a (possibly) injured person: plan around them
                    self.explorer.add_keep_out(xy, self.person_keep_out)

        target_angle, target_distance = 0.0, None
        if env.goal is not None:
            dx, dy = env.goal - env.position[:2]
            target_angle, target_distance = math.atan2(dy, dx) - env.yaw, math.hypot(dx, dy)
        _, _, terminated, truncated, self.info = env.step(self._pilot(target_angle, target_distance))
        self.flown += float(np.linalg.norm(env.position[:2] - self._last_xy))
        self._last_xy = env.position[:2].copy()

        if self.info["collision"]:
            self.phase = "crashed"
        elif truncated:
            self.phase = "battery empty"
        return self.finished

    def result(self) -> dict[str, Any]:
        found = score_markers(self.markers.confirmed(), self.env.people_xy)
        return {
            **found,
            "outcome": self.phase,
            "home": self.phase == "home",
            "coverage": self.env.coverage(),
            "time": self.elapsed,
            "return_reason": self.return_reason,
            "return_time": self.return_time,
            "shield_rate": self.shield.override_rate if self.shield is not None else math.nan,
        }
