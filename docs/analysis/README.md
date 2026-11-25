# Analysis: how data moves through the system

Diagrams are edited in draw.io (`.drawio`) and exported to `.png`. Both are generated from one definition by
`python scripts/docs/build_diagrams.py`, so they never drift apart.

| Diagram | File |
|---|---|
| Sequence: land the drone with a fist gesture | [sequence-land-by-gesture.png](sequence-land-by-gesture.png) · [.drawio](sequence-land-by-gesture.drawio) |
| DFD context diagram | [dfd-context.png](dfd-context.png) · [.drawio](dfd-context.drawio) |
| DFD level 0 | [dfd-level0.png](dfd-level0.png) · [.drawio](dfd-level0.drawio) |
| DFD level 1: 2.0 Recognize operator commands | [dfd-level1.png](dfd-level1.png) · [.drawio](dfd-level1.drawio) |

## External entities

They are the users and external systems listed in the [README](../../README.md#users): **Operator**, **Analyst** and
**Drone** (flight controller + camera).

## Level 0 processes

| No. | Process | Reads | Writes |
|---|---|---|---|
| 1.0 | Register operators, drones and models | - | D1, D2, D6 |
| 2.0 | Recognize operator commands (the sequence scenario) | D1, D6 | D5 |
| 3.0 | Control drone flight | D2 | D3, D4 |
| 4.0 | Monitor flight | D3, D4, D5 | - |
| 5.0 | Generate flight reports | D3, D5 | - |

## Data stores = database tables

| Store | Table in [schema.sql](../database/schema.sql) |
|---|---|
| D1 | `operator` |
| D2 | `drone` |
| D3 | `flight` (with the junction table `flight_model`: models used in the flight) |
| D4 | `telemetry_sample` |
| D5 | `command_event` |
| D6 | `ml_model` |

## Balancing

**Context → level 0.** Every flow of the context diagram appears at level 0 with the same name:

| Flow | Context | Level 0 |
|---|---|---|
| registration details (operator, drone, consent) | Operator → 0 | Operator → 1.0 |
| registration confirmation | 0 → Operator | 1.0 → Operator |
| voice commands | Operator → 0 | Operator → 2.0 |
| live view and flight status | 0 → Operator | 4.0 → Operator |
| camera video with gestures | Drone → 0 | Drone → 2.0 |
| telemetry | Drone → 0 | Drone → 3.0 |
| flight commands (MAVLink) | 0 → Drone | 3.0 → Drone |
| model version details | Analyst → 0 | Analyst → 1.0 |
| report request | Analyst → 0 | Analyst → 5.0 |
| flight reports | 0 → Analyst | 5.0 → Analyst |

**Level 0 → level 1 (process 2.0).** Inputs: camera video with gestures (Drone), voice commands (Operator),
appearance template (D1), model versions (D6). Outputs: recognized command (→ 3.0), command record (→ D5).
The same flows enter and leave sub-processes 2.1-2.4.
