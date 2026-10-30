# Progress log

## 2026-10-30 — Stage 2: frontier exploration + safety shield

### What was built
| File | What it does |
|---|---|
| `src/explorer.py` | `FrontierExplorer`: safe grid (free cells ≥ 0.5 m from walls) → path distances from the drone (vectorised grid Dijkstra, 8-connected) → frontier cells grouped into clusters, clusters narrower than 0.75 m dropped → target = safe cell next to a frontier with the lowest *path distance − frontier width* → path traced back → waypoint = farthest path point ≤ 2.5 m ahead in line of sight. Replans every 0.5 s (≈ 15–30 ms). Reached or unreachable (20 s) targets are blacklisted. If nothing is left, a 360° look-around comes first; only then is the cave "explored". |
| `src/safety.py` | `SafetyShield`: if anything is closer than 1.2 m ahead (±20°) or 0.6 m to the side, the pilot's command is replaced by the classical follow-the-gap command. Optional speed cap. Counts how often it intervenes. |
| `src/baselines.py` | follow-the-gap gets `target_angle`: prefers gaps towards the waypoint, turns on the spot when the waypoint is outside the camera view, and turns towards the waypoint when boxed in. Behaviour with no waypoint is unchanged (wander results identical to Stage 1). |
| `src/cave_env.py` | Heading-error sign fixed to match what the RL pilot learned in the forest (`sin(nose − goal bearing)`). |
| `scripts/explore_cave.py` | `--explore`, `--shield`, `--time`; draws the planned path, waypoint and target; RL pilot defaults to `runs/avoidance/avoidance-2` (choosing the newest file by date picked an older model after `git pull` changed file dates). |
| `tests/test_explorer.py` | 13 tests: wavefront detours/unreachable cells, clusters, explorer heads for the open end, ignores 1-cell scan gaps, look-around before giving up, blacklisting, pilot waypoint behaviour, shield, and explorer beats wandering in real caves. 30 tests total. |

### Results (same 10 caves, seeds 0–9; episode ends at 90 % coverage or the time limit)
| Setup | Crashes | Mean coverage (150 s) | Caves reaching 90 % | Mean time |
|---|---|---|---|---|
| Wander, follow-the-gap (Stage 1) | 0 | 54.3 % | 1 / 10 | 145 s |
| Wander, RL pilot (Stage 1) | 0 | 50.1 % | 0 / 10 | 150 s |
| **Explorer + follow-the-gap** | **0** | **85.8 %** | 4 / 10 | 136 s |
| Explorer + follow-the-gap, 300 s limit | 0 | 90.1 % | **10 / 10** | 155 s |
| Explorer + RL pilot (no shield) | **2** | 79.2 % | 7 / 10 | 62 s |
| **Explorer + RL pilot + safety shield** | **0** | 82.4 % | 7 / 10 | **89 s** |

Shield intervention rate: 1–12 % of control steps (median ≈ 6 %).

Notes for the thesis:
- Frontier exploration raises coverage from ~52 % to ~86 % in the same time; with 300 s the classical setup maps ≥ 90 % of every cave without a crash.
- Early in a flight a lucky wanderer can be ahead (cave 4 at 60 s: wander 69 % vs explorer 46 %); the explorer wins on completeness (cave 4 at 150 s: 87 % vs 69 %). Report the time curve, not one snapshot.
- The RL pilot is ~2× faster when it finishes (it flies up to 4 m/s vs 1.5 m/s cruise) but, never trained with waypoints in caves, it crashed in 2 / 10 caves. The safety shield removes all crashes while intervening only ~6 % of the time — a direct argument for the "RL + rule-based shield" architecture.
- Remaining weakness of RL + shield: it gets stuck in 2 caves (6, 9) where the classical pilot does not. Next step: train the RL pilot in caves with waypoint goals.

Bugs found (for "Implementation challenges"):
- With a forward-only camera, unknown space right behind the drone was always the "best" frontier → the drone never moved. Fix: minimum target distance 1.5 m.
- Unseen wall cells created fake 1-cell frontiers, and unknown rock behind walls looked like "space to explore" → drone hovered next to walls. Fix: cluster frontiers, ignore clusters narrower than the drone.
- The map is empty for the first few steps, so the explorer declared the cave finished at 0.2 s. Fix: look-around before giving up.
- Heading-error sign was opposite to the forest training convention → RL pilot steered away from waypoints.

Images: `docs/images/stage2_cave1_map.png`, `docs/images/stage2_cave1_flight.png`, video `docs/images/stage2_cave1_explore.mp4`.

### How to run
```
python scripts/explore_cave.py --explore                         # watch it explore
python scripts/explore_cave.py --explore --pilot rl --shield     # RL pilot with safety shield
python scripts/explore_cave.py --headless --explore --episodes 10
```

### Next steps
1. (Optional, ~1 h laptop run, announce first) Train the RL pilot in caves with waypoint goals + shield; compare again.
2. Stage 3: person models in caves, YOLO detection → map markers, return home with the same path planner.

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
