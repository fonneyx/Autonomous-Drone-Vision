"""Run docs/database/schema.sql in a real PostgreSQL and save the query results as docs/database/query-results.png.

Needs no installed PostgreSQL: `pgserver` ships the PostgreSQL binaries as a Python package.

    uv run --no-project --with pgserver --with matplotlib python scripts/docs/run_schema.py

The script stops with an error if any statement fails (psql ON_ERROR_STOP).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pgserver  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SQL = ROOT / "docs" / "database" / "schema.sql"
PNG = ROOT / "docs" / "database" / "query-results.png"
TXT = ROOT / "docs" / "database" / "query-results.txt"


def main() -> None:
    data_dir = Path(tempfile.mkdtemp(prefix="dronevision-pg-"))
    server = pgserver.get_server(data_dir, cleanup_mode="stop")
    try:
        psql = Path(pgserver.__file__).parent / "pginstall" / "bin" / ("psql.exe" if sys.platform == "win32" else "psql")
        version = server.psql("SELECT version();").splitlines()[2].strip()
        result = subprocess.run(
            [str(psql), "-X", "-v", "ON_ERROR_STOP=1", "-P", "pager=off", "-d", server.get_uri(),
             "-f", str(SQL)],
            capture_output=True, text=True, encoding="utf-8",
        )
        if result.returncode != 0:
            print(result.stdout, result.stderr, sep="\n")
            raise SystemExit("schema.sql failed")
    finally:
        server.cleanup()
        shutil.rmtree(data_dir, ignore_errors=True)

    # status lines (DROP/CREATE/INSERT) become one summary line; the three result tables are kept as they are
    status = re.findall(r"^(SET|DROP TABLE|CREATE TABLE|CREATE INDEX|INSERT 0 (\d+))$", result.stdout, re.M)
    rows = sum(int(n) for _, n in status if n)
    count = {k: sum(1 for s, _ in status if s.startswith(k)) for k in ("CREATE TABLE", "CREATE INDEX", "INSERT")}
    out = "\n".join(line for line in result.stdout.splitlines()
                    if not re.match(r"^(SET|DROP TABLE|CREATE TABLE|CREATE INDEX|INSERT 0 \d+)$", line))
    blocks = re.split(r"\n\s*\n", out.strip("\n"))
    tables = [b.strip("\n") for b in blocks if re.search(r"^\s*-+\+", b, re.M)]
    titles = ["Q1  JOIN (command_event, flight, operator, drone, ml_model): command log",
              "Q2  GROUP BY source: accepted share and latency",
              "Q3  GROUP BY flight over the M:N link flight_model: models used"]
    lines = [f"$ psql -v ON_ERROR_STOP=1 -f docs/database/schema.sql      -- {version.split(',')[0]}",
             f"DROP TABLE, CREATE TABLE x{count['CREATE TABLE']}, CREATE INDEX x{count['CREATE INDEX']}, "
             f"INSERT x{count['INSERT']} ({rows} rows) - no errors", ""]
    for title, table in zip(titles, tables):
        lines += [f"-- {title}", *table.splitlines(), ""]
    TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    width = max(len(line) for line in lines)
    fig = plt.figure(figsize=(width * 0.075 + 0.6, len(lines) * 0.168 + 0.3), dpi=150)
    fig.patch.set_facecolor("#1e1e1e")
    fig.text(0.012, 0.985, "\n".join(lines), family="DejaVu Sans Mono", fontsize=9, color="#e6e6e6", va="top",
             linespacing=1.32)
    fig.savefig(PNG, facecolor=fig.get_facecolor())
    print(f"ok: {len(tables)} query results -> {PNG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
