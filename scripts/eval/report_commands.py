"""Tables and charts from the command measurements (scripts/eval/measure_commands.py).

    python scripts/eval/report_commands.py                      # all CSVs in runs/measurements
    python scripts/eval/report_commands.py runs/measurements/gesture_20261105_181500.csv

Writes to runs/measurements/report/:
    report.md              accuracy (95 % Wilson interval), false triggers, latency, confusions
    gesture_accuracy.png   accuracy vs distance, one line per gesture
    voice_accuracy.png     accuracy per phrase, one bar per noise condition
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.commands import Command  # noqa: E402
from src.evaluation.command_report import accuracy, markdown  # noqa: E402
from src.evaluation.command_trials import SKIPPED, read_csv  # noqa: E402

DATA_DIR = ROOT / "runs" / "measurements"
# Validated categorical palette (fixed order); text stays in ink colours, not series colours
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.yaxis.label.set_color(INK_2)
    ax.xaxis.label.set_color(INK_2)


def gesture_chart(rows: list[dict[str, str]], path: Path) -> None:
    import matplotlib.pyplot as plt

    acc = [a for a in accuracy(rows) if a.source == "gesture" and a.command != "ALL"]
    conditions = list(dict.fromkeys(a.condition for a in acc))
    commands = list(dict.fromkeys(a.command for a in acc))
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=200, facecolor=SURFACE)
    _style(ax)
    for i, command in enumerate(commands):
        dodge = (i - (len(commands) - 1) / 2) * 0.06  # equal values (e.g. all 100 %) stay visible side by side
        points = [(conditions.index(a.condition) + dodge, a) for a in acc if a.command == command]
        xs = [x for x, _ in points]
        ys = [100 * a.rate for _, a in points]
        low = [100 * (a.rate - a.interval[0]) for _, a in points]
        high = [100 * (a.interval[1] - a.rate) for _, a in points]
        colour = SERIES[i]
        ax.errorbar(xs, ys, yerr=[low, high], color=colour, linewidth=2, marker="o", markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=2, capsize=0, elinewidth=1, alpha=0.95,
                    label=Command[command].value)
    ax.set_xticks(range(len(conditions)), conditions)
    ax.set_xlim(-0.3, len(conditions) - 0.3)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Recognised correctly (%)")
    ax.set_xlabel("Distance to camera")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, -0.2),
              ncol=len(commands))
    ax.set_title("Gesture accuracy vs distance (bars: 95 % interval)", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def voice_chart(rows: list[dict[str, str]], path: Path) -> None:
    import matplotlib.pyplot as plt

    acc = [a for a in accuracy(rows) if a.source == "voice" and a.command != "ALL"]
    conditions = list(dict.fromkeys(a.condition for a in acc))
    commands = list(dict.fromkeys(a.command for a in acc))
    bar = 0.8 / max(1, len(conditions))
    fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(commands) + 1.4), dpi=200, facecolor=SURFACE)
    _style(ax)
    for i, condition in enumerate(conditions):
        for a in (a for a in acc if a.condition == condition):
            y = commands.index(a.command) + (i - (len(conditions) - 1) / 2) * bar
            ax.barh(y, 100 * a.rate, height=bar - 0.04, color=SERIES[i], label=condition)
            ax.text(100 * a.rate + 1.5, y, f"{100 * a.rate:.0f}%", va="center", fontsize=8, color=INK_2)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=9, labelcolor=INK,
              loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(unique))
    ax.set_yticks(range(len(commands)), [Command[c].value for c in commands])
    ax.invert_yaxis()
    ax.set_xlim(0, 110)
    ax.set_xlabel("Recognised correctly (%)")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Voice command accuracy", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="*", type=Path, help="CSV files (default: all in runs/measurements).")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "report")
    args = parser.parse_args()
    files = args.csv or sorted(DATA_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files in {DATA_DIR}. Run scripts/eval/measure_commands.py first.")
    rows = [row for f in files for row in read_csv(f)]
    rows = [r for r in rows if r["outcome"] != SKIPPED]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text(markdown(rows), encoding="utf-8")
    written = ["report.md"]
    sources = {r["source"] for r in rows if r["expected"]}
    if "gesture" in sources:
        gesture_chart(rows, args.out / "gesture_accuracy.png")
        written.append("gesture_accuracy.png")
    if "voice" in sources:
        voice_chart(rows, args.out / "voice_accuracy.png")
        written.append("voice_accuracy.png")
    print(f"{len(rows)} trials from {len(files)} file(s) -> {args.out}: {', '.join(written)}")


if __name__ == "__main__":
    main()
