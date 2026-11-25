# Database

PostgreSQL schema for flights, telemetry, commands with the arbiter's decisions, and model versions.

| File | What it is |
|---|---|
| [erd.png](erd.png) · [erd.drawio](erd.drawio) | ER diagram |
| [schema.sql](schema.sql) | `CREATE TABLE` (PK, FK, NOT NULL, CHECK), test data, three queries |
| [query-results.png](query-results.png) · [.txt](query-results.txt) | Output of the queries in PostgreSQL 16 |

## Tables

| Table | Holds | Relationships |
|---|---|---|
| `operator` | people who control the drone, consent time, appearance template | 1:N `flight` |
| `drone` | registered drones (simulated or real) | 1:N `flight` |
| `flight` | one row per flight: who, which drone, sim/real, start, end, why it ended | 1:N `telemetry_sample`, 1:N `command_event` |
| `telemetry_sample` | flight mode, altitude, battery, distance to the operator over time | - |
| `command_event` | every gesture/voice/dashboard command, confidence, latency, **decision and reason** | - |
| `ml_model` | versions of the models (pose, gesture, voice, avoidance) with file hash | 1:N `command_event` |
| `flight_model` | junction table: which model versions ran in which flight | `flight` **M:N** `ml_model` |

The tables are in third normal form: one value per cell, every non-key column depends only on the key, and data
that repeats (operators, drones, models) lives in its own table and is referenced by a foreign key.

## Queries

1. **Command log** - `JOIN` of `command_event`, `flight`, `operator`, `drone` and `ml_model`.
2. **Accepted share and latency per command source** - `GROUP BY source`.
3. **Flight summary with the models used** - `GROUP BY flight` over the M:N table `flight_model`.

## Run it

Any PostgreSQL 14+: `psql -v ON_ERROR_STOP=1 -f docs/database/schema.sql` (the script drops and recreates its
tables, so it can be run again). Without an installed PostgreSQL:

```bash
uv run --no-project --python 3.12 --with pgserver --with matplotlib python scripts/docs/run_schema.py
```

This starts a temporary PostgreSQL, runs the script and regenerates `query-results.png`. CI runs the same script
against a PostgreSQL 16 service on every push.
