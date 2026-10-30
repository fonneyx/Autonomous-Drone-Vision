"""Tests for the follow-me prototype: gestures, voice parsing, arbiter, follow controller, re-ID, tracker.

Run: python -m pytest tests -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.commands import Command, CommandEvent  # noqa: E402
from src.control.arbiter import Arbiter, Mode  # noqa: E402
from src.control.follow_controller import FollowController  # noqa: E402
from src.perception.gestures import FIST, OPEN_PALM, GestureTimer, HandObservation  # noqa: E402
from src.perception.person_tracker import L_HIP, L_SHOULDER, R_HIP, R_SHOULDER, Person  # noqa: E402
from src.perception.reid import OperatorLock  # noqa: E402
from src.voice.commands import PHRASES, parse  # noqa: E402

FPS = 20


def feed(timer: GestureTimer, frames: list[HandObservation | None], start: float = 0.0) -> list[Command]:
    fired = []
    for i, hand in enumerate(frames):
        command = timer.update(hand, start + i / FPS)
        if command is not None:
            fired.append(command)
    return fired


def palm(x: float = 300.0) -> HandObservation:
    return HandObservation(OPEN_PALM, x, 40.0)


# ------------------------------------------------------------------ gestures
def test_open_palm_held_still_is_stop() -> None:
    assert feed(GestureTimer(), [palm() for _ in range(25)]) == [Command.STOP]


def test_short_palm_does_nothing() -> None:
    assert feed(GestureTimer(), [palm() for _ in range(10)] + [None] * 20) == []


def test_fist_held_is_land_and_fires_once() -> None:
    fist = HandObservation(FIST, 300.0, 40.0)
    assert feed(GestureTimer(), [fist for _ in range(60)]) == [Command.LAND]


def test_fist_needs_longer_than_stop() -> None:
    fist = HandObservation(FIST, 300.0, 40.0)
    assert feed(GestureTimer(), [fist for _ in range(25)]) == []  # 1.25 s < 1.5 s


def test_wave_is_come_back_not_stop() -> None:
    xs = [300 + 60 * math.sin(2 * math.pi * 2.0 * i / FPS) for i in range(30)]  # 2 Hz, +-1.5 palm widths
    assert feed(GestureTimer(), [palm(x) for x in xs]) == [Command.COME_BACK]


def test_slow_drift_is_not_a_wave() -> None:
    xs = [300 + 3 * i for i in range(30)]  # hand drifting one way
    assert Command.COME_BACK not in feed(GestureTimer(), [palm(x) for x in xs])


# --------------------------------------------------------------------- voice
@pytest.mark.parametrize(
    ("text", "command"),
    [
        ("follow me", Command.FOLLOW),
        ("[unk] go higher", Command.HIGHER),
        ("take a photo", Command.PHOTO),
        ("come back", Command.COME_BACK),
        ("land", Command.LAND),
        ("hello there", None),
        ("", None),
    ],
)
def test_voice_parse(text: str, command: Command | None) -> None:
    assert parse(text) is command


def test_voice_covers_all_ten_commands() -> None:
    assert set(PHRASES.values()) == set(Command)
    assert len(Command) == 10


# ------------------------------------------------------------------- arbiter
def ev(command: Command, t: float, source: str = "voice") -> CommandEvent:
    return CommandEvent(command, source, t)


def test_take_off_then_follow() -> None:
    arbiter = Arbiter()
    assert arbiter.update([ev(Command.FOLLOW, 0.0)], 0.0, True).mode is Mode.LANDED, "must take off first"
    assert arbiter.update([ev(Command.TAKE_OFF, 1.0)], 1.0, True).mode is Mode.HOVER
    arbiter.update([ev(Command.FOLLOW, 2.0)], 2.0, True)
    decision = arbiter.update([], 2.6, True)
    assert decision.mode is Mode.FOLLOW and not decision.hover


def test_stop_beats_follow_in_conflict() -> None:
    arbiter = Arbiter()
    arbiter.update([ev(Command.TAKE_OFF, 0.0)], 0.0, True)
    arbiter.update([ev(Command.FOLLOW, 1.0, "gesture"), ev(Command.STOP, 1.2, "voice")], 1.2, True)
    assert arbiter.update([], 2.0, True).mode is Mode.HOVER
    arbiter.update([ev(Command.STOP, 3.0)], 3.0, True)
    arbiter.update([ev(Command.FOLLOW, 3.3)], 3.3, True)  # right after a STOP: rejected
    assert arbiter.update([], 4.0, True).mode is Mode.HOVER


def test_land_overrides_everything_and_finishes() -> None:
    arbiter = Arbiter(landing_time=2.0)
    arbiter.update([ev(Command.TAKE_OFF, 0.0)], 0.0, True)
    decision = arbiter.update([ev(Command.FOLLOW, 1.0), ev(Command.LAND, 1.0)], 1.0, True)
    assert decision.mode is Mode.LANDING
    assert arbiter.update([], 3.5, True).mode is Mode.LANDED


def test_safety_forces_hover() -> None:
    arbiter = Arbiter()
    arbiter.update([ev(Command.TAKE_OFF, 0.0)], 0.0, True)
    arbiter.update([ev(Command.FOLLOW, 1.0)], 1.0, True)
    decision = arbiter.update([], 2.0, True, safety_stop=True)
    assert decision.hover and "SAFETY" in decision.reason


def test_lost_operator_hover_then_give_up() -> None:
    arbiter = Arbiter(lost_timeout=5.0)
    arbiter.update([ev(Command.TAKE_OFF, 0.0)], 0.0, True)
    arbiter.update([ev(Command.FOLLOW, 1.0)], 1.0, True)
    arbiter.update([], 2.0, True)
    lost = arbiter.update([], 3.0, False)
    assert lost.hover and lost.mode is Mode.FOLLOW, "hover while lost, still wanting to follow"
    assert arbiter.update([], 4.0, True).hover is False, "resumes when found again"
    arbiter.update([], 5.0, False)
    assert arbiter.update([], 10.5, False).mode is Mode.HOVER


def test_distance_and_altitude_limits() -> None:
    arbiter = Arbiter(follow_distance=3.0, min_distance=3.0, max_distance=5.0, altitude=2.0, max_altitude=3.0)
    arbiter.update([ev(Command.TAKE_OFF, 0.0)], 0.0, True)
    for t in range(1, 6):
        decision = arbiter.update([ev(Command.FURTHER, t), ev(Command.HIGHER, t), ev(Command.CLOSER, t + 0.1)],
                                  t + 0.1, True)
    assert 3.0 <= decision.follow_distance <= 5.0
    assert decision.altitude == pytest.approx(3.0)


# ------------------------------------------------------------ follow controller
def person_at(cx: float, torso_px: float, width: float = 80.0) -> Person:
    keypoints = np.zeros((17, 3))
    top = 200.0
    for index, x in ((L_SHOULDER, cx - 20), (R_SHOULDER, cx + 20)):
        keypoints[index] = [x, top, 0.9]
    for index, x in ((L_HIP, cx - 15), (R_HIP, cx + 15)):
        keypoints[index] = [x, top + torso_px, 0.9]
    box = np.array([cx - width / 2, top - 40, cx + width / 2, top + 3 * torso_px])
    return Person(1, box, 0.9, keypoints)


def test_controller_distance_and_bearing() -> None:
    controller = FollowController(640, 480, horizontal_fov_deg=70.0)
    focal = 320 / math.tan(math.radians(35))
    m = controller.measure(person_at(320, focal * 0.5 / 4.0))  # torso size of someone 4 m away, centred
    assert m.distance == pytest.approx(4.0, rel=0.02)
    assert abs(m.bearing) < 0.01
    assert controller.measure(person_at(100, 50)).bearing > 0.3, "person on the left -> positive bearing"


def test_controller_moves_forward_when_far_and_backs_off_when_close() -> None:
    controller = FollowController(640, 480)
    focal = controller.focal
    far, _ = controller.command(person_at(320, focal * 0.5 / 6.0), target_distance=3.0, min_distance=2.0)
    close, _ = controller.command(person_at(320, focal * 0.5 / 1.5), target_distance=3.0, min_distance=2.0)
    assert far.vx > 0.5 and close.vx < 0
    assert far.vx <= 2.0


def test_controller_turns_towards_the_operator() -> None:
    controller = FollowController(640, 480)
    velocity, _ = controller.command(person_at(80, 60), target_distance=3.0, min_distance=2.0)
    assert velocity.yaw_rate > 0


# ------------------------------------------------------------------- re-ID
def painted_person(frame: np.ndarray, cx: int, shirt: tuple[int, int, int], pants: tuple[int, int, int],
                   track_id: int | None) -> Person:
    person = person_at(cx, 120)
    x1, y1, x2, y2 = person.box.astype(int)
    cv2.rectangle(frame, (x1, 200), (x2, 320), shirt, -1)
    cv2.rectangle(frame, (x1, 320), (x2, y2), pants, -1)
    person.track_id = track_id
    return person


def test_reid_keeps_operator_and_rejects_stranger() -> None:
    frame = np.zeros((600, 800, 3), np.uint8)
    me = painted_person(frame, 200, (30, 30, 200), (200, 80, 20), track_id=1)  # red shirt, blue pants (BGR)
    stranger = painted_person(frame, 600, (40, 200, 40), (20, 200, 220), track_id=2)
    lock = OperatorLock()
    lock.lock(frame, me)
    assert lock.update(frame, [me, stranger]) is me


def test_reid_reacquires_operator_after_id_change() -> None:
    frame = np.zeros((600, 800, 3), np.uint8)
    me = painted_person(frame, 200, (30, 30, 200), (200, 80, 20), track_id=1)
    lock = OperatorLock(reacquire_frames=3)
    lock.lock(frame, me)
    stranger = painted_person(frame, 600, (40, 200, 40), (20, 200, 220), track_id=2)
    me.track_id = 7  # tracker lost me and gave me a new id
    found = [lock.update(frame, [stranger, me]) for _ in range(3)]
    assert found[-1] is me and found[0] is None, "needs a few matching frames before switching"
    assert lock.update(frame, [stranger]) is None, "the stranger is never taken for me"


# ------------------------------------------------------------------- tracker
def test_tracker_finds_people_on_a_photo() -> None:
    from ultralytics.utils import ASSETS

    from src.perception.person_tracker import PersonTracker

    people = PersonTracker()(cv2.imread(str(ASSETS / "zidane.jpg")))
    assert len(people) >= 2
    assert all(p.keypoints.shape == (17, 3) for p in people)
