# Progress log

## 2026-10-30 — Stage 1: underground world + live map

**Goal change.** The project now explores unknown underground areas (tunnels, mines, caves) instead of following a
person. CLAUDE.md was rewritten: new title, architecture (map → explorer → pilot → safety shield → ArduPilot),
stages 1–6, metrics and hardware notes (no GPS underground → MTF-01 optical flow + lidar; LEDs; short Wi-Fi range).

**Simulator choice.** Train in MuJoCo (fast, free, already used for the avoidance pilot); validate later in ArduPilot
SITL (free, same MAVLink interface as the real drone). One codebase, two simulators. Cost: $0.

### What was built
| File | What it does |
|---|---|
| `src/cave_world.py` | Procedural caves: random "worm" walkers carve tunnels (3–4 m wide), branches and chambers in a 48×48 m grid. Every open cell is reachable from home by construction. Walls merged into ~80–100 boxes for fast ray casting. |
| `src/occupancy_map.py` | 2D log-odds occupancy grid (0.25 m cells) from range scans. Needs 2 observations to call a cell free, stops rays at confirmed walls (sensor dropouts cannot "see through" walls), finds frontiers. Independent of MuJoCo → reusable on the real drone. |
| `src/cave_env.py` | Gymnasium env `CaveExplore-v0`: same drone, flight model, 64×48 depth camera and action space as the avoidance env, in a dark cave with a headlamp. Middle depth rows → horizontal scan → map. Reward = new area mapped. Metrics: coverage, wall precision. `goal` hook ready for Stage 2. |
| `src/baselines.py` | Classical follow-the-gap pilot with obstacle inflation and a side guard (the camera cannot see the drone's flanks). |
| `scripts/explore_cave.py` | Demo: 3D viewer + live map window, pilots `gap` / `rl` / `keys` (fly yourself), `--headless` scoring, `--record` video. |
| `tests/test_cave.py` | 17 pytest tests: generator connectivity/enclosure/determinism, wall boxes tile walls exactly, map free/wall/unknown logic, dropout robustness, pilot behaviour, Gymnasium API, crash detection, exploration without crashing. |
| `src/sim_env.py` | Refactor only: drone XML moved into `drone_xml()` so both worlds share it. Compiled avoidance model verified identical. |

### Results (10 caves, seeds 0–9, 150 s limit, depth noise 2 % + 1 % dropouts)
| Pilot | Crashes | Mean coverage | Wall precision (±0.25 m) |
|---|---|---|---|
| Follow-the-gap (classical) | 0 / 10 | 54.3 % | 99.9–100 % |
| RL avoidance pilot (trained only in the forest, never saw a cave) | 0 / 10 | 50.1 % | 100 % |

Notes for the thesis:
- The RL pilot transfers zero-shot from the forest course to caves without a crash → evidence that the depth-based
  policy learned general avoidance, not the forest layout.
- Both pilots only "wander" (no memory of where they have been), so coverage plateaus at ~50 % and varies a lot per
  cave (26–90 %). This is the motivation for Stage 2: frontier-based exploration.
- Simulation speed ≈ 300 steps/s per process (≈ 30× real time).

Bugs found and fixed while building (useful for "Implementation challenges"):
- Projecting a "no hit" depth reading (10 m) onto the horizontal plane made it 9.999 m → mapped as a phantom wall.
- Noise was applied to "no hit" readings too → same phantom walls. Noise now applies to real hits only.
- Single-frame dropouts cleared cells behind walls (spikes in the map) → rays now stop at confirmed walls.
- The gap pilot turned into walls at the edge of its 90° field of view → side guard added.

Images: `docs/images/stage1_cave7_map.png`, `docs/images/stage1_cave7_flight.png`, video `docs/images/stage1_cave7.mp4`.

### How to run
```
python scripts/explore_cave.py                  # watch it explore (3D viewer + live map)
python scripts/explore_cave.py --pilot keys     # fly yourself (W/S speed, A/D turn) in the map window
python scripts/explore_cave.py --headless --episodes 10 --pilot rl
python -m pytest tests -q
```

### Next steps (Stage 2 — frontier exploration)
1. Frontier clustering + choose the next frontier (nearest / information gain) → waypoint.
2. A* path on the occupancy map from drone to frontier; feed waypoints through `env.goal`.
3. Retrain the RL pilot in caves with waypoint goals (short run, announced first); compare with follow-the-gap.
4. Target metric: coverage ≥ 90 % in most caves within the time limit, 0 crashes.
