"""Tests for Stage 3: people in caves, detection, map markers, keep-out zones, full mission.

Run: python -m pytest tests -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.cave_env import CaveExploreEnv  # noqa: E402
from src.explorer import wavefront  # noqa: E402
from src.rescue import PersonMarkers, SimulatedPersonDetector, locate, score_markers  # noqa: E402
from src.rescue_mission import RescueMission  # noqa: E402


@pytest.fixture(scope="module")
def env() -> CaveExploreEnv:
    return CaveExploreEnv(world_seed=1, people=3)


def face_person(env: CaveExploreEnv, index: int, distance: float) -> None:
    """Put the drone `distance` m from person `index`, looking at them, with a clear view."""
    person = env.people_xy[index]
    for angle in np.linspace(0, 2 * math.pi, 64, endpoint=False):
        x, y = person[0] - distance * math.cos(angle), person[1] - distance * math.sin(angle)
        line = np.linspace(0.0, 0.85, 20)
        if env.layout.is_free_world(x + (person[0] - x) * line, y + (person[1] - y) * line).all():
            env.data.qpos[:4] = [x, y, 1.5, angle]
            mujoco.mj_forward(env.model, env.data)
            env._cast_depth()
            return
    pytest.skip("no clear view of this person in this cave")


# ------------------------------------------------------------------ people
def test_people_are_placed_in_open_space_far_from_home(env: CaveExploreEnv) -> None:
    env.reset(seed=1)
    assert env.people_xy.shape == (3, 2)
    home = np.array(env.layout.home_xy)
    for xy in env.people_xy:
        assert env.layout.is_free_world(np.array([xy[0]]), np.array([xy[1]]))[0]
        assert np.linalg.norm(xy - home) > 4.0
    gaps = [np.linalg.norm(a - b) for i, a in enumerate(env.people_xy) for b in env.people_xy[i + 1 :]]
    assert min(gaps) >= 4.0


def test_people_move_between_resets(env: CaveExploreEnv) -> None:
    env.reset(seed=1)
    first = env.people_xy.copy()
    env.reset(seed=2)
    assert not np.allclose(first, env.people_xy)


# ---------------------------------------------------------------- detection
def test_detected_person_is_located_accurately(env: CaveExploreEnv) -> None:
    env.reset(seed=1)
    face_person(env, 0, 3.0)
    detections = SimulatedPersonDetector(seed=0, false_alarm_rate=0.0).detect(env)
    assert len(detections) == 1
    xy = locate(env, detections[0])
    assert xy is not None and np.linalg.norm(xy - env.people_xy[0]) < 0.3


def test_no_detection_when_looking_away(env: CaveExploreEnv) -> None:
    env.reset(seed=1)
    face_person(env, 0, 3.0)
    env.data.qpos[3] += math.pi  # turn around
    mujoco.mj_forward(env.model, env.data)
    assert SimulatedPersonDetector(seed=0, false_alarm_rate=0.0).detect(env) == []


def test_no_detection_through_rock(env: CaveExploreEnv) -> None:
    env.reset(seed=1)
    face_person(env, 0, 3.0)
    # Move the person into solid rock just beyond the boundary wall: the wall hides them
    rock = np.argwhere(~env.layout.free & ~env.layout.boundary_rock())
    target = env.people_xy[0]
    centres = (rock + 0.5) * env.layout.cell_size
    xy = centres[np.argmin(np.linalg.norm(centres - target, axis=1))]
    slot = env._person_slots[0]
    env.data.mocap_pos[slot.mocap] = [xy[0], xy[1], 0.0]
    env.people_xy[0] = xy
    mujoco.mj_forward(env.model, env.data)
    detections = SimulatedPersonDetector(seed=0, false_alarm_rate=0.0, max_range=30.0).detect(env)
    assert all(abs(d.bearing - math.atan2(xy[1] - env.position[1], xy[0] - env.position[0]) + env.yaw) > 0.05
               for d in detections)


# ------------------------------------------------------------------ markers
def test_markers_confirm_after_repeated_sightings() -> None:
    markers = PersonMarkers(confirm_hits=3)
    markers.add(np.array([5.0, 5.0]))
    markers.add(np.array([5.2, 4.9]))
    assert markers.confirmed() == []
    markers.add(np.array([4.9, 5.1]))
    markers.add(np.array([20.0, 20.0]))  # one-off false alarm elsewhere
    confirmed = markers.confirmed()
    assert len(confirmed) == 1 and np.linalg.norm(confirmed[0] - [5.0, 5.0]) < 0.2


def test_score_markers() -> None:
    people = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    markers = [np.array([0.3, 0.0]), np.array([10.0, 0.5]), np.array([30.0, 0.0])]
    score = score_markers(markers, people)
    assert score["found"] == 2 and score["false_markers"] == 1
    assert score["marker_error_m"] == pytest.approx(0.4)


# ---------------------------------------------------------------- keep-out
def test_soft_keep_out_detours_when_possible_and_passes_when_not() -> None:
    passable = np.ones((20, 20), dtype=bool)
    cost = np.ones((20, 20))
    cost[8:12, 0:12] = 6.0  # keep-out zone, open on the right side
    plain = wavefront(passable, (2, 5))
    soft = wavefront(passable, (2, 5), cost)
    assert soft[18, 5] > plain[18, 5], "path should detour around the zone"
    assert soft[18, 5] < plain[18, 5] + 20, "but not by a huge amount"
    cost[8:12, :] = 6.0  # zone now spans the whole width: only way is through it
    assert np.isfinite(wavefront(passable, (2, 5), cost)[18, 5])


# ----------------------------------------------------------------- mission
def test_short_mission_returns_home_before_battery_runs_out() -> None:
    env = CaveExploreEnv(world_seed=1, people=3)
    mission = RescueMission(env, SimulatedPersonDetector(seed=1), battery_seconds=90.0)
    mission.reset(seed=1)
    while not mission.step():
        pass
    result = mission.result()
    assert result["outcome"] == "home", result
    assert result["return_reason"] == "battery"
    assert result["time"] < 90.0


def test_full_mission_finds_people_and_marks_them_accurately() -> None:
    env = CaveExploreEnv(world_seed=1, people=3)
    mission = RescueMission(env, SimulatedPersonDetector(seed=1))
    mission.reset(seed=1)
    while not mission.step():
        pass
    result = mission.result()
    assert result["home"] and result["outcome"] != "crashed"
    assert result["found"] >= 2
    assert result["marker_error_m"] < 0.3
    assert result["false_markers"] == 0
