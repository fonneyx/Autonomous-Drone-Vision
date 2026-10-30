"""The operator's commands, shared by gestures, voice and keyboard."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Command(Enum):
    TAKE_OFF = "take off"
    FOLLOW = "follow me"
    STOP = "stop"  # hover in place
    LAND = "land"
    COME_BACK = "come back"  # fly back to the operator (minimum distance) and hover
    HIGHER = "go higher"
    LOWER = "go lower"
    CLOSER = "closer"  # follow distance -1 m
    FURTHER = "further"  # follow distance +1 m
    PHOTO = "take a photo"


@dataclass(frozen=True)
class CommandEvent:
    command: Command
    source: str  # "gesture", "voice" or "keyboard"
    time: float  # seconds (time.monotonic())
