"""Command arbiter: one decision from gestures, voice, keyboard and safety.

Priority (highest first):
    1. SAFETY      obstacle too close / emergency -> hover this instant, whatever else is asked
    2. LAND, STOP  always accepted while flying
    3. other operator commands (follow, come back, higher/lower, closer/further, photo)
    4. follow      the default behaviour once FOLLOW was given

Conflicts: a STOP (or LAND) cancels any FOLLOW / COME_BACK that arrives within
`conflict_window` seconds of it, in either order — when unsure, hover. Losing the
operator while following means hover; following resumes by itself if they are found
again within `lost_timeout`, otherwise the drone stays hovering until a new FOLLOW.

The arbiter only decides the *mode* and targets; the follow controller turns them
into velocities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from src.commands import Command, CommandEvent


class Mode(Enum):
    LANDED = "landed"
    HOVER = "hover"
    FOLLOW = "follow"
    COME_BACK = "come back"
    LANDING = "landing"


@dataclass
class Decision:
    mode: Mode
    hover: bool  # True: send zero velocity this cycle (safety, lost operator, mode HOVER)
    reason: str
    follow_distance: float
    altitude: float
    photo: bool = False
    accepted: list[CommandEvent] = field(default_factory=list)


PRIORITY = {Command.LAND: 0, Command.STOP: 1}  # everything else: 2


class Arbiter:
    def __init__(
        self,
        follow_distance: float = 3.0,
        min_distance: float = 3.0,
        max_distance: float = 8.0,
        altitude: float = 2.0,
        min_altitude: float = 1.0,
        max_altitude: float = 5.0,
        conflict_window: float = 0.5,
        lost_timeout: float = 10.0,
        landing_time: float = 3.0,
    ):
        self.follow_distance = follow_distance
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.altitude = altitude
        self.min_altitude = min_altitude
        self.max_altitude = max_altitude
        self.conflict_window = conflict_window
        self.lost_timeout = lost_timeout
        self.landing_time = landing_time
        self.mode = Mode.LANDED
        self._last_stop = -1e9
        self._pending_motion: CommandEvent | None = None  # FOLLOW/COME_BACK waiting out the conflict window
        self._lost_since: float | None = None
        self._landing_since = 0.0

    def _apply(self, event: CommandEvent, now: float) -> bool:
        """Apply one command; returns whether it was accepted."""
        command = event.command
        if self.mode in (Mode.LANDED, Mode.LANDING):
            if command is Command.TAKE_OFF and self.mode is Mode.LANDED:
                self.mode = Mode.HOVER
                return True
            return False
        if command is Command.LAND:
            self.mode, self._landing_since, self._pending_motion = Mode.LANDING, now, None
            return True
        if command is Command.STOP:
            self.mode, self._last_stop, self._pending_motion = Mode.HOVER, event.time, None
            return True
        if command in (Command.FOLLOW, Command.COME_BACK):
            if abs(event.time - self._last_stop) <= self.conflict_window:
                return False  # conflicting with a STOP: hover wins
            self._pending_motion = event
            return True
        if command is Command.HIGHER:
            self.altitude = min(self.max_altitude, self.altitude + 0.5)
        elif command is Command.LOWER:
            self.altitude = max(self.min_altitude, self.altitude - 0.5)
        elif command is Command.CLOSER:
            self.follow_distance = max(self.min_distance, self.follow_distance - 1.0)
        elif command is Command.FURTHER:
            self.follow_distance = min(self.max_distance, self.follow_distance + 1.0)
        return command is not Command.TAKE_OFF

    def update(
        self,
        events: list[CommandEvent],
        now: float,
        operator_visible: bool,
        safety_stop: bool = False,
        arrived: bool = False,
    ) -> Decision:
        """One control cycle. arrived: COME_BACK reached the minimum distance."""
        accepted = []
        photo = False
        for event in sorted(events, key=lambda e: (PRIORITY.get(e.command, 2), e.time)):
            if self._apply(event, now):
                accepted.append(event)
                photo |= event.command is Command.PHOTO

        # A FOLLOW / COME_BACK only takes effect once no STOP came right after it
        if self._pending_motion is not None and now - self._pending_motion.time > self.conflict_window:
            self.mode = Mode.FOLLOW if self._pending_motion.command is Command.FOLLOW else Mode.COME_BACK
            self._pending_motion = None
            self._lost_since = None

        if self.mode is Mode.LANDING and now - self._landing_since >= self.landing_time:
            self.mode = Mode.LANDED
        if self.mode is Mode.COME_BACK and arrived:
            self.mode = Mode.HOVER

        hover, reason = self.mode is Mode.HOVER, self.mode.value
        if self.mode in (Mode.FOLLOW, Mode.COME_BACK):
            if operator_visible:
                self._lost_since = None
            else:
                self._lost_since = now if self._lost_since is None else self._lost_since
                hover, reason = True, "operator lost - hovering"
                if now - self._lost_since >= self.lost_timeout:
                    self.mode, reason = Mode.HOVER, "operator lost too long - stopped following"
        if safety_stop and self.mode not in (Mode.LANDED, Mode.LANDING):
            hover, reason = True, "SAFETY: obstacle - hovering"
        return Decision(self.mode, hover, reason, self.follow_distance, self.altitude, photo, accepted)
