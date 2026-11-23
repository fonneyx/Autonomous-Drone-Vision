"""Command trials: how often gestures and voice commands are recognised, and how fast.

A session is a list of trials grouped in blocks, one block per condition (a distance such
as "3m", or a noise level such as "noisy"). Each trial asks the tester for one command:

    prepare  (3 s)  the screen shows what to do, e.g. "Hold a FIST"
    active   (6 s)  the first recognised command from the tested source is the answer
    result          correct / wrong / missed, then the next trial

Control trials ask for *no* command ("move naturally"): anything recognised then is a
false trigger. Blocks start when the tester presses SPACE, so there is time to walk to
the next distance.

Latency per trial:
    gesture  command fired - first frame the expected hand shape was seen
             (includes the designed hold time: 1 s stop, 1.5 s land)
    voice    command fired - end of the last spoken word (Vosk word times)

The runner only does timing and bookkeeping; the script feeds it events and frames, so
it is unit-tested without a camera or microphone.
"""

from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from src.commands import Command, CommandEvent

GESTURE_COMMANDS = (Command.STOP, Command.LAND, Command.COME_BACK)
VOICE_COMMANDS = tuple(Command)

GESTURE_INSTRUCTIONS = {
    Command.STOP: "Raise an OPEN PALM, hold still",
    Command.LAND: "Raise a FIST, hold",
    Command.COME_BACK: "Raise and WAVE your palm",
    None: "Move naturally - NO gesture",
}
VOICE_NOTHING = "Talk normally - NO command"

# Outcomes
CORRECT, WRONG, MISSED = "correct", "wrong", "missed"  # trials that ask for a command
NO_TRIGGER, FALSE_TRIGGER = "no_trigger", "false_trigger"  # control trials
SKIPPED = "skipped"  # tester pressed K (e.g. stood in the wrong place); left out of reports


@dataclass(frozen=True)
class Trial:
    source: str  # "gesture" or "voice"
    condition: str  # "3m", "quiet", ...
    expected: Command | None  # None = control trial, nothing should fire

    @property
    def instruction(self) -> str:
        if self.source == "gesture":
            return GESTURE_INSTRUCTIONS[self.expected]
        return f'Say "{self.expected.value}"' if self.expected is not None else VOICE_NOTHING


@dataclass
class TrialResult:
    index: int
    source: str
    condition: str
    expected: str  # command name, or "" for a control trial
    got: str  # first recognised command, or ""
    outcome: str
    t_go: float  # monotonic seconds: start of the active window
    t_seen: float | None  # gesture: expected hand shape first seen
    t_fired: float | None  # command recognised
    t_spoken: float | None  # voice: end of the last word
    response_s: float | None  # t_fired - t_go (includes the tester's reaction)
    latency_s: float | None  # see module docstring
    hand_seen: float  # gesture: share of frames with a usable (raised) hand on the tester (0..1)
    hand_found: float  # gesture: share of frames where MediaPipe found any hand at all
    hand_own: float  # gesture: share of frames with a hand inside the tester's box (raised or not)
    est_distance_m: float | None  # gesture: mean camera distance estimate during the trial
    heard: str  # voice: last text Vosk heard


def make_plan(source: str, conditions: list[str], repeats: int, commands: tuple[Command, ...] | None = None,
              controls: int | None = None, seed: int = 0) -> list[Trial]:
    """Blocks of trials, one per condition; order inside a block is shuffled (same seed = same order).

    controls: control trials per block (default: one per two repeats, at least 1)."""
    if commands is None:
        commands = GESTURE_COMMANDS if source == "gesture" else VOICE_COMMANDS
    if controls is None:
        controls = max(1, repeats // 2)
    rng = random.Random(seed)
    plan: list[Trial] = []
    for condition in conditions:
        block = [Trial(source, condition, c) for c in commands for _ in range(repeats)]
        block += [Trial(source, condition, None) for _ in range(controls)]
        rng.shuffle(block)
        plan += block
    return plan


class TrialRunner:
    """Steps through a plan: waiting (for SPACE) -> prepare -> active -> result -> ... -> done."""

    def __init__(self, plan: list[Trial], prepare: float = 3.0, window: float = 6.0, show_result: float = 1.5,
                 auto_start: bool = False):
        self.plan = plan
        self.prepare = prepare
        self.window = window
        self.show_result = show_result
        self.auto_start = auto_start  # no SPACE needed between blocks (automated runs)
        self.index = 0
        self.state = "waiting" if plan else "done"
        self.state_since = 0.0
        self.go_count = 0  # increases at each active window: the script resets its gesture timer on change
        self.results: list[TrialResult] = []
        self.last_result: TrialResult | None = None
        self._reset_trial()

    # ---------------------------------------------------------------- info for the screen
    @property
    def trial(self) -> Trial | None:
        return self.plan[self.index] if self.index < len(self.plan) else None

    def seconds_left(self, now: float) -> float:
        length = {"prepare": self.prepare, "active": self.window, "result": self.show_result}.get(self.state, 0.0)
        return max(0.0, length - (now - self.state_since))

    def block_progress(self) -> tuple[int, int]:
        """(trial number inside the current block, block size)."""
        trial = self.trial
        if trial is None:
            return 0, 0
        same = [i for i, t in enumerate(self.plan) if t.condition == trial.condition]
        return same.index(self.index) + 1, len(same)

    # ---------------------------------------------------------------- control
    def start_block(self, now: float) -> None:
        """SPACE pressed: the tester is in place for this condition."""
        if self.state == "waiting":
            self._enter("prepare", now)

    def hold(self, now: float) -> None:
        """Restart the get-ready countdown (the tester's hand is still up): every trial starts from rest."""
        if self.state == "prepare":
            self.state_since = now

    def skip(self, now: float) -> TrialResult | None:
        """K pressed: record the current trial as skipped and move on."""
        if self.state in ("prepare", "active"):
            return self._finish(now, SKIPPED, None)
        return None

    def update(self, now: float, events: list[CommandEvent], hand_visible: bool = False,
               shape_seen: bool = False, distance: float | None = None, heard: str = "",
               hand_found: bool = False, hand_own: bool = False) -> TrialResult | None:
        """Feed one frame. Returns the result when a trial ends."""
        if self.state == "waiting" and self.auto_start:
            self._enter("prepare", now)
        if self.state == "prepare" and now - self.state_since >= self.prepare:
            self._enter("active", now)
            self.go_count += 1
            return None  # events from before the window opened do not count
        if self.state == "result" and now - self.state_since >= self.show_result:
            self._next(now)
            return None
        if self.state != "active":
            return None

        trial = self.trial
        assert trial is not None
        self._frames += 1
        self._hand_frames += int(hand_visible)
        self._found_frames += int(hand_found or hand_own or hand_visible)
        self._own_frames += int(hand_own or hand_visible)
        if shape_seen and self._t_seen is None:
            self._t_seen = now
        if distance is not None:
            self._distances.append(distance)
        if heard:
            self._heard = heard

        answer = next((e for e in events if e.source == trial.source and e.time >= self.state_since), None)
        if answer is not None:
            if trial.expected is None:
                return self._finish(now, FALSE_TRIGGER, answer)
            return self._finish(now, CORRECT if answer.command is trial.expected else WRONG, answer)
        if now - self.state_since >= self.window:
            return self._finish(now, NO_TRIGGER if trial.expected is None else MISSED, None)
        return None

    # ---------------------------------------------------------------- internals
    def _enter(self, state: str, now: float) -> None:
        self.state, self.state_since = state, now

    def _reset_trial(self) -> None:
        self._frames = 0
        self._hand_frames = 0
        self._found_frames = 0
        self._own_frames = 0
        self._t_seen: float | None = None
        self._distances: list[float] = []
        self._heard = ""

    def _finish(self, now: float, outcome: str, answer: CommandEvent | None) -> TrialResult:
        trial = self.trial
        assert trial is not None
        t_go = self.state_since if self.state == "active" else now
        t_fired = answer.time if answer is not None else None
        t_spoken = answer.spoken_at if answer is not None else None
        latency = None
        if outcome == CORRECT and t_fired is not None:
            start = self._t_seen if trial.source == "gesture" else t_spoken
            latency = t_fired - start if start is not None else None
        result = TrialResult(
            index=self.index,
            source=trial.source,
            condition=trial.condition,
            expected=trial.expected.name if trial.expected is not None else "",
            got=answer.command.name if answer is not None else "",
            outcome=outcome,
            t_go=t_go,
            t_seen=self._t_seen,
            t_fired=t_fired,
            t_spoken=t_spoken,
            response_s=t_fired - t_go if t_fired is not None else None,
            latency_s=latency,
            hand_seen=self._hand_frames / self._frames if self._frames else 0.0,
            hand_found=self._found_frames / self._frames if self._frames else 0.0,
            hand_own=self._own_frames / self._frames if self._frames else 0.0,
            est_distance_m=sum(self._distances) / len(self._distances) if self._distances else None,
            heard=self._heard,
        )
        self.results.append(result)
        self.last_result = result
        self._enter("result", now)
        return result

    def _next(self, now: float) -> None:
        previous = self.trial
        self.index += 1
        self._reset_trial()
        if self.trial is None:
            self._enter("done", now)
        elif previous is not None and self.trial.condition != previous.condition:
            self._enter("waiting", now)  # new condition: wait for the tester to move
        else:
            self._enter("prepare", now)


# ---------------------------------------------------------------- CSV
COLUMNS = [f.name for f in fields(TrialResult)]


def append_csv(path: Path, result: TrialResult) -> None:
    """Append one result (writes the header on a new file), so nothing is lost if the session stops early."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        row = asdict(result)
        writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else ("" if v is None else v)) for k, v in row.items()})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))
