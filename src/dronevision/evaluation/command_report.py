"""Turn command-trial CSV rows into thesis tables: accuracy, false triggers, latency, confusions.

Accuracy = correct / (correct + wrong + missed) for each command and condition, with a
95 % Wilson interval (honest for small samples such as 5-20 trials). Control trials give
the false-trigger rate. Skipped trials are left out.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass

from dronevision.evaluation.command_trials import CORRECT, FALSE_TRIGGER, MISSED, NO_TRIGGER, SKIPPED, WRONG


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class Accuracy:
    source: str
    condition: str
    command: str  # "ALL" for the condition total
    n: int
    correct: int
    wrong: int
    missed: int

    @property
    def rate(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def interval(self) -> tuple[float, float]:
        return wilson(self.correct, self.n)


@dataclass
class FalseTriggers:
    source: str
    condition: str
    n: int
    triggers: int


@dataclass
class Latency:
    source: str
    command: str
    n: int
    median: float
    mean: float
    p90: float


def _ordered(values: list[str]) -> list[str]:
    """Unique values in first-seen order (keeps the session's condition order: 1m, 2m, 3m ...)."""
    return list(dict.fromkeys(values))


def accuracy(rows: list[dict[str, str]]) -> list[Accuracy]:
    counts: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    trials = [r for r in rows if r["expected"] and r["outcome"] in (CORRECT, WRONG, MISSED)]
    for r in trials:
        counts[(r["source"], r["condition"], r["expected"])][r["outcome"]] += 1
        counts[(r["source"], r["condition"], "ALL")][r["outcome"]] += 1
    out = []
    for source in _ordered([r["source"] for r in trials]):
        for condition in _ordered([r["condition"] for r in trials if r["source"] == source]):
            commands = _ordered([r["expected"] for r in trials if r["source"] == source]) + ["ALL"]
            for command in commands:
                c = counts.get((source, condition, command))
                if c:
                    out.append(Accuracy(source, condition, command, sum(c.values()), c[CORRECT], c[WRONG], c[MISSED]))
    return out


def false_triggers(rows: list[dict[str, str]]) -> list[FalseTriggers]:
    controls = [r for r in rows if not r["expected"] and r["outcome"] in (NO_TRIGGER, FALSE_TRIGGER)]
    out = []
    for source in _ordered([r["source"] for r in controls]):
        for condition in _ordered([r["condition"] for r in controls if r["source"] == source]):
            group = [r for r in controls if r["source"] == source and r["condition"] == condition]
            out.append(FalseTriggers(source, condition, len(group), sum(r["outcome"] == FALSE_TRIGGER for r in group)))
    return out


def latency(rows: list[dict[str, str]]) -> list[Latency]:
    """Latency of correct trials per command (all conditions together)."""
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        if r["outcome"] == CORRECT and r.get("latency_s"):
            values[(r["source"], r["expected"])].append(float(r["latency_s"]))
    out = []
    for (source, command), xs in values.items():
        xs.sort()
        p90 = xs[min(len(xs) - 1, math.ceil(0.9 * len(xs)) - 1)]
        out.append(Latency(source, command, len(xs), statistics.median(xs), statistics.fmean(xs), p90))
    return out


def confusions(rows: list[dict[str, str]]) -> Counter:
    """(expected, got) counts for wrong answers and false triggers ("-" = control trial)."""
    return Counter((r["expected"] or "-", r["got"]) for r in rows if r["outcome"] in (WRONG, FALSE_TRIGGER))


def _pct(x: float) -> str:
    return f"{100 * x:.0f} %"


def markdown(rows: list[dict[str, str]]) -> str:
    """All tables as Markdown, ready to paste into the thesis notes."""
    rows = [r for r in rows if r["outcome"] != SKIPPED]
    lines = ["# Command recognition measurements", ""]
    lines.append(f"Trials: {len(rows)} (skipped trials left out). Accuracy = correct / attempts, "
                 "95 % Wilson interval in brackets.")
    for source in _ordered([r["source"] for r in rows]):
        acc = [a for a in accuracy(rows) if a.source == source]
        if acc:
            lines += ["", f"## {source.capitalize()}: accuracy", "",
                      "| Condition | Command | Attempts | Correct | Wrong | Missed | Accuracy |",
                      "|---|---|---|---|---|---|---|"]
            for a in acc:
                low, high = a.interval
                name = "**all**" if a.command == "ALL" else a.command
                lines.append(f"| {a.condition} | {name} | {a.n} | {a.correct} | {a.wrong} | {a.missed} | "
                             f"{_pct(a.rate)} [{_pct(low)}–{_pct(high)}] |")
        ft = [f for f in false_triggers(rows) if f.source == source]
        if ft:
            lines += ["", f"## {source.capitalize()}: false triggers (control trials, no command given)", "",
                      "| Condition | Trials | False triggers | Rate |", "|---|---|---|---|"]
            for f in ft:
                lines.append(f"| {f.condition} | {f.n} | {f.triggers} | {_pct(f.triggers / f.n if f.n else 0)} |")
        lat = [x for x in latency(rows) if x.source == source]
        if lat:
            what = ("first frame the hand shape was seen → command (includes the 1 s / 1.5 s hold)"
                    if source == "gesture" else "end of the last spoken word → command")
            lines += ["", f"## {source.capitalize()}: latency, {what}", "",
                      "| Command | Correct trials | Median (s) | Mean (s) | 90th pct (s) |", "|---|---|---|---|---|"]
            for x in lat:
                lines.append(f"| {x.command} | {x.n} | {x.median:.2f} | {x.mean:.2f} | {x.p90:.2f} |")
        conf = {k: v for k, v in confusions([r for r in rows if r["source"] == source]).items()}
        if conf:
            lines += ["", f"## {source.capitalize()}: confusions", "", "| Asked for | Recognised | Times |",
                      "|---|---|---|"]
            for (expected, got), n in sorted(conf.items(), key=lambda kv: -kv[1]):
                lines.append(f"| {expected if expected != '-' else '(nothing)'} | {got} | {n} |")
    return "\n".join(lines) + "\n"
