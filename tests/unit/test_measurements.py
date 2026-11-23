"""Tests for the command measurement tools: trial plan, trial runner, CSV and report.

Run: python -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from dronevision.commands import Command, CommandEvent  # noqa: E402
from dronevision.evaluation.command_report import accuracy, false_triggers, latency, markdown, wilson  # noqa: E402
from dronevision.evaluation.command_trials import (  # noqa: E402
    CORRECT,
    FALSE_TRIGGER,
    MISSED,
    NO_TRIGGER,
    SKIPPED,
    WRONG,
    Trial,
    TrialRunner,
    append_csv,
    make_plan,
    read_csv,
)

DT = 0.05


def run_until_active(runner: TrialRunner, t: float) -> float:
    """Step frames until the active window opens; returns the time."""
    while runner.state != "active":
        runner.update(t, [])
        t += DT
    return t


# ---------------------------------------------------------------------- plan
def test_plan_has_blocks_in_order_with_controls() -> None:
    plan = make_plan("gesture", ["1m", "3m"], repeats=4, seed=1)
    assert [t.condition for t in plan] == ["1m"] * 14 + ["3m"] * 14  # 3 gestures x 4 + 2 controls
    block = plan[:14]
    assert sum(t.expected is None for t in block) == 2
    assert {t.expected for t in block} == {Command.STOP, Command.LAND, Command.COME_BACK, None}


def test_plan_is_shuffled_but_repeatable() -> None:
    a = make_plan("voice", ["quiet"], repeats=3, seed=5)
    assert a == make_plan("voice", ["quiet"], repeats=3, seed=5)
    assert a != make_plan("voice", ["quiet"], repeats=3, seed=6)
    assert len(a) == 10 * 3 + 1


def test_instructions() -> None:
    assert Trial("gesture", "1m", Command.LAND).instruction == "Raise a FIST, hold"
    assert Trial("voice", "quiet", Command.PHOTO).instruction == 'Say "take a photo"'
    assert "NO" in Trial("voice", "quiet", None).instruction


# -------------------------------------------------------------------- runner
def test_waits_for_space_then_prepares() -> None:
    runner = TrialRunner([Trial("gesture", "2m", Command.STOP)], prepare=1.0)
    runner.update(0.0, [])
    runner.update(5.0, [])
    assert runner.state == "waiting"
    runner.start_block(5.0)
    assert runner.state == "prepare"
    runner.update(6.1, [])
    assert runner.state == "active" and runner.go_count == 1


def test_correct_gesture_and_latency_from_first_seen_shape() -> None:
    runner = TrialRunner([Trial("gesture", "2m", Command.STOP)], prepare=1.0, auto_start=True)
    t = run_until_active(runner, 0.0)
    go = runner.state_since
    runner.update(t, [], hand_visible=True, shape_seen=False, distance=2.1)
    runner.update(t + 0.2, [], hand_visible=True, shape_seen=True, distance=1.9)  # palm appears
    result = runner.update(t + 1.3, [CommandEvent(Command.STOP, "gesture", t + 1.3)], hand_visible=True)
    assert result is not None and result.outcome == CORRECT
    assert result.latency_s == pytest.approx(1.1)
    assert result.response_s == pytest.approx(t + 1.3 - go)
    assert result.hand_seen == pytest.approx(1.0)
    assert result.est_distance_m == pytest.approx(2.0)


def test_wrong_command_and_other_sources_ignored() -> None:
    runner = TrialRunner([Trial("gesture", "2m", Command.COME_BACK)], prepare=0.5, auto_start=True)
    t = run_until_active(runner, 0.0)
    assert runner.update(t, [CommandEvent(Command.LAND, "keyboard", t)]) is None  # not the tested source
    result = runner.update(t + DT, [CommandEvent(Command.STOP, "gesture", t + DT)])
    assert result.outcome == WRONG and result.got == "STOP" and result.latency_s is None


def test_missed_after_window_and_events_before_window_ignored() -> None:
    runner = TrialRunner([Trial("voice", "noisy", Command.HIGHER)], prepare=0.5, window=2.0, auto_start=True)
    runner.update(0.0, [])
    runner.update(0.3, [CommandEvent(Command.HIGHER, "voice", 0.3)])  # said too early: during prepare
    t = run_until_active(runner, 0.35)
    late = runner.update(t, [CommandEvent(Command.HIGHER, "voice", 0.3)])  # recognised before the window opened
    assert late is None
    result = None
    while result is None:
        t += DT
        result = runner.update(t, [])
    assert result.outcome == MISSED and t - runner.results[0].t_go == pytest.approx(2.0, abs=DT)


def test_voice_latency_from_end_of_speech() -> None:
    runner = TrialRunner([Trial("voice", "quiet", Command.PHOTO)], prepare=0.5, auto_start=True)
    t = run_until_active(runner, 0.0)
    result = runner.update(t + 2.0, [CommandEvent(Command.PHOTO, "voice", t + 2.0, spoken_at=t + 1.6)],
                           heard="take a photo (0.95)")
    assert result.outcome == CORRECT and result.latency_s == pytest.approx(0.4)
    assert result.heard == "take a photo (0.95)"


def test_control_trials() -> None:
    plan = [Trial("gesture", "1m", None), Trial("gesture", "1m", None)]
    runner = TrialRunner(plan, prepare=0.5, window=1.0, show_result=0.5, auto_start=True)
    t = run_until_active(runner, 0.0)
    first = runner.update(t + 0.1, [CommandEvent(Command.STOP, "gesture", t + 0.1)])
    assert first.outcome == FALSE_TRIGGER
    t = run_until_active(runner, t + 0.2)
    second = None
    while second is None:
        t += DT
        second = runner.update(t, [])
    assert second.outcome == NO_TRIGGER
    t += 1.0
    runner.update(t, [])
    assert runner.state == "done"


def test_new_condition_waits_for_space_again() -> None:
    plan = [Trial("gesture", "1m", None), Trial("gesture", "3m", None)]
    runner = TrialRunner(plan, prepare=0.1, window=0.2, show_result=0.1)
    runner.start_block(0.0)
    t = 0.0
    while not runner.results:
        t += DT
        runner.update(t, [])
    for _ in range(5):
        t += DT
        runner.update(t, [])
    assert runner.state == "waiting" and runner.trial.condition == "3m"


def test_hold_restarts_countdown_while_hand_is_up() -> None:
    runner = TrialRunner([Trial("gesture", "1m", Command.STOP)], prepare=1.0, auto_start=True)
    runner.update(0.0, [])
    for t in (0.5, 1.0, 1.5):  # hand still up
        runner.hold(t)
        runner.update(t, [])
    assert runner.state == "prepare"
    runner.update(2.6, [])
    assert runner.state == "active"


def test_skip() -> None:
    runner = TrialRunner([Trial("gesture", "1m", Command.LAND)], prepare=0.1, auto_start=True)
    runner.update(0.0, [])
    assert runner.skip(0.05).outcome == SKIPPED


# --------------------------------------------------------------- CSV + report
def make_rows(tmp_path: Path) -> list[dict[str, str]]:
    plan = [Trial("gesture", "1m", Command.STOP)] * 4 + [Trial("gesture", "3m", Command.STOP)] * 2 + [
        Trial("gesture", "3m", None)]
    answers = [Command.STOP, Command.STOP, Command.STOP, Command.LAND, Command.STOP, None, Command.STOP]
    runner = TrialRunner(plan, prepare=0.1, window=0.5, show_result=0.1, auto_start=True)
    path = tmp_path / "gesture.csv"
    t = 0.0
    for answer in answers:
        t = run_until_active(runner, t)
        runner.update(t, [], shape_seen=True)
        events = [CommandEvent(answer, "gesture", t + 1.0)] if answer else []
        result = runner.update(t + (1.0 if answer else 0.6), events)
        append_csv(path, result)
        t += 1.5
        runner.update(t, [])
        if runner.state == "waiting":
            runner.start_block(t)
    return read_csv(path)


def test_report_numbers(tmp_path: Path) -> None:
    rows = make_rows(tmp_path)
    assert [r["outcome"] for r in rows] == [CORRECT] * 3 + [WRONG, CORRECT, MISSED, FALSE_TRIGGER]
    acc = {(a.condition, a.command): a for a in accuracy(rows)}
    assert acc[("1m", "STOP")].rate == pytest.approx(0.75)
    assert acc[("3m", "STOP")].missed == 1 and acc[("3m", "ALL")].n == 2
    assert false_triggers(rows)[0].triggers == 1
    lat = latency(rows)[0]
    assert lat.n == 4 and lat.median == pytest.approx(1.0)
    text = markdown(rows)
    assert "| 1m | STOP | 4 | 3 | 1 | 0 | 75 %" in text and "(nothing) | STOP | 1" in text


def test_wilson_interval() -> None:
    low, high = wilson(10, 10)
    assert high == pytest.approx(1.0) and 0.65 < low < 0.75  # 10/10 is not proof of 100 %
    assert wilson(0, 0) == (0.0, 0.0)
