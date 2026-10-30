"""Tests for the classical pilot (follow-the-gap) and the safety shield.

Run: python -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.control.classical_pilot import follow_the_gap  # noqa: E402
from src.control.safety_shield import SafetyShield  # noqa: E402
from src.sim.avoidance_env import MAX_SPEED, MAX_YAW_RATE  # noqa: E402

ANGLES = np.linspace(0.78, -0.78, 64)  # left to right, like the depth camera


# ------------------------------------------------------------- classical pilot
def test_turns_towards_the_opening() -> None:
    ranges = np.full(64, 1.5)
    ranges[:16] = 10.0  # open on the left
    assert follow_the_gap(ANGLES, ranges, MAX_SPEED, MAX_YAW_RATE)[1] > 0.2


def test_stops_and_turns_in_a_dead_end() -> None:
    action = follow_the_gap(ANGLES, np.full(64, 0.8), MAX_SPEED, MAX_YAW_RATE)
    assert action[0] == pytest.approx(-1.0) and abs(action[1]) == pytest.approx(1.0)


def test_turns_on_the_spot_to_a_waypoint_behind() -> None:
    action = follow_the_gap(ANGLES, np.full(64, 8.0), MAX_SPEED, MAX_YAW_RATE, target_angle=-2.5)
    assert action[0] == pytest.approx(-1.0) and action[1] < 0


def test_prefers_the_waypoint_direction() -> None:
    open_space = np.full(64, 8.0)
    assert follow_the_gap(ANGLES, open_space, MAX_SPEED, MAX_YAW_RATE, target_angle=0.5)[1] > 0.2
    assert follow_the_gap(ANGLES, open_space, MAX_SPEED, MAX_YAW_RATE, target_angle=-0.5)[1] < -0.2


def test_flies_straight_at_a_close_waypoint_with_a_clear_line() -> None:
    ranges = np.full(64, 1.9)  # narrow but clear on the left, wide open on the right
    ranges[50:] = 5.0
    action = follow_the_gap(ANGLES, ranges, MAX_SPEED, MAX_YAW_RATE, target_angle=0.6, target_distance=1.0)
    assert action[1] > 0, "must not be lured to the wide gap on the right"
    assert action[0] > -1.0, "must keep moving towards the waypoint"


# --------------------------------------------------------------- safety shield
def test_shield_passes_safe_actions_through() -> None:
    shield = SafetyShield(MAX_SPEED, MAX_YAW_RATE)
    action = np.array([0.8, -0.3], dtype=np.float32)
    assert np.allclose(shield.filter(action, ANGLES, np.full(64, 5.0)), action)
    assert shield.override_rate == 0.0


def test_shield_overrides_when_an_obstacle_is_close() -> None:
    shield = SafetyShield(MAX_SPEED, MAX_YAW_RATE)
    ranges = np.full(64, 5.0)
    ranges[30:34] = 0.7  # obstacle dead ahead
    assert shield.filter(np.array([1.0, 0.0]), ANGLES, ranges)[0] < 0.0
    assert shield.override_rate == 1.0


def test_shield_slows_down_near_a_person() -> None:
    shield = SafetyShield(MAX_SPEED, MAX_YAW_RATE, person_speed=0.5)
    action = shield.filter(np.array([1.0, 0.0]), ANGLES, np.full(64, 9.0), near_person=True)
    assert (action[0] + 1.0) / 2.0 * MAX_SPEED <= 0.5 + 1e-6


def test_shield_speed_limit() -> None:
    shield = SafetyShield(MAX_SPEED, MAX_YAW_RATE, speed_limit=1.0)
    action = shield.filter(np.array([1.0, 0.0]), ANGLES, np.full(64, 9.0))
    assert (action[0] + 1.0) / 2.0 * MAX_SPEED == pytest.approx(1.0)
