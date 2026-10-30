# Autonomous-Drone-Vision — Project Brief for Claude

You are my engineering partner and supervisor-level advisor for my university diploma (final-year IT thesis).
Help me plan, code, debug, test and write the thesis. Explain things in simple English, step by step.

## 1. The project in one sentence
An autonomous quadcopter that **explores an unknown underground area** (tunnel, mine, cave, basement) with
**no prior map**, **builds the map while flying**, **avoids obstacles**, **marks people who need rescue** on the
map, and **returns home** — trained and tested **in simulation first**, then demonstrated on a **real DIY drone**,
using **reinforcement learning (RL)** for the low-level avoidance pilot.

- Repo: `Autonomous-Drone-Vision` (GitHub, public later)
- Thesis title (EN, working): "Autonomous exploration and mapping of unknown underground environments by a
  quadcopter with learned obstacle avoidance"
- Timeline: speedrun ~3 months (Nov 2026 – Jan/Feb 2027), defense ~May–June 2027
- Budget: very low. Prefer free tools. Uni provides a 3D printer; drone will be DIY.
- Purpose boundary: search-and-rescue / inspection only. No target homing, terminal guidance or attack
  behaviour toward people or vehicles.

## 2. Hardware (planned, may change)
- Frame/motors/ESC: F450 + 4× 2212 920KV + 4× 30A ESC, 1045 props (consider a smaller frame for tunnels)
- Flight controller: SpeedyBee F405 V4 running **ArduPilot**
- **No GPS underground** → position from MicoAir MTF-01 (optical flow + 8 m ToF lidar, downward) with
  ArduPilot EKF3 non-GPS mode. GPS module only useful for outdoor tests.
- Obstacles / mapping: VL53L5CX 8×8 ToF (front, ~4 m range). Cheap upgrade worth considering: a 360° 2D lidar
  (e.g. LDROBOT LD06/LD19) — matches the simulator's horizontal scan and makes mapping far more reliable.
- LEDs on the drone: tunnels are dark, camera and optical flow need light.
- Companion: Raspberry Pi Zero 2 W + camera (streams video + bridges MAVLink over Wi-Fi; Wi-Fi range
  underground is short → onboard stop-and-hover / return-home on link loss)
- Heavy compute: **my laptop** (vision, mapping, RL policy inference)
- Safety: RadioMaster Pocket ELRS + RP1 receiver (manual override ALWAYS available)
- Battery: 3S LiPo (bought locally)

## 3. Architecture
```
 depth / ToF / 2D lidar ──► occupancy map (live 2D grid: unknown / free / wall) ──┐
 drone camera ──► person detector (YOLO, trained on sim + real images) ──► people marked on map
                                                                                  ▼
              Explorer (classical): pick next frontier (edge of known space) → waypoint
              Return-home planner (A* on the map) when done / low battery / link lost
                                                                                  ▼
              Pilot: RL policy (or classical follow-the-gap baseline) → velocity setpoints
                                                                                  ▼
              Safety shield (max speed, min wall distance, stop-and-hover on doubt)
                                                                                  ▼
              MAVLink (pymavlink) → ArduPilot GUIDED mode → motors
```
Key rule: **RL never controls motors directly.** ArduPilot keeps low-level stabilisation; RL only outputs
bounded velocity setpoints. A rule-based safety shield can always override RL.

## 4. Stages
### Stage 1 — Underground world + live map (simulation)
Procedural cave/tunnel generator in MuJoCo, same drone + depth camera as the avoidance env, 2D occupancy
map updated live, frontier detection, coverage metric, demo script with map window.

### Stage 2 — Frontier exploration
Explorer picks frontiers, pilot flies to waypoints (retrain the avoidance pilot with waypoint goals in caves).
Compare RL pilot vs classical follow-the-gap pilot on the same caves and seeds.

### Stage 3 — People marking + return home + scoring
Person models in the caves, YOLO detection → map markers, A* return home, mission score.

### Stage 4 — Realistic validation
ArduPilot SITL (free) with the same MAVLink interface as the real drone. Gazebo only if time is left
(needs WSL2 on Windows).

### Stage 5 — Real drone
Build, register (Kazakhstan rules), ArduPilot setup, non-GPS position hold with MTF-01, manual flights, then
the SAME code: classical pilot first, then RL, indoors in a corridor/basement standing in for a tunnel,
low speed, prop guards. Measure the sim-to-real gap.

### Stage 6 — Evaluation & thesis
Metrics, videos, thesis chapters, slides, rehearsal.

## 5. Metrics to collect (sim and real)
- Exploration: coverage (% of reachable area mapped) vs time, map accuracy (wall cells correct), path length
- Avoidance: collisions per run, min distance to walls, success rate
- Rescue: people found / people present, marker position error (m)
- Return home: success rate, time
- RL vs baseline: same caves, same seeds; also smoothness (jerk) and energy proxy
- Sim-to-real gap: same metric in sim vs real

## 6. Safety rules (non-negotiable)
- Manual RC override and failsafe (land) configured before any autonomous flight
- Speed limits (e.g. ≤ 1–2 m/s indoors), min altitude, battery failsafe, link-loss failsafe
- Unclear situation → stop and hover
- First real tests: indoors, low height, nobody under the drone, prop guards
- Never fly unregistered, in restricted areas, or in real mines/caves without permission and supervision

## 7. Fallback plan
Core = exploration + mapping in simulation. If a feature fails on the real drone: autonomous exploration →
show in sim + short autonomous corridor flight on the real drone; people marking → sim only; RL pilot →
classical pilot on hardware. Document failures as "Limitations & future work" with measurements.

## 8. Tech stack
Python 3.11/3.12, MuJoCo, Gymnasium, Stable-Baselines3 (PPO), NumPy, OpenCV, Ultralytics YOLO, Blender
(models), ArduPilot SITL, pymavlink, Mission Planner/QGroundControl, Git/GitHub, pytest.

## 9. How to work with me
- Before big changes, give a short plan; then implement in small, testable steps.
- Write clean, commented code with type hints; add a test or a demo script for each module.
- Tell me honestly when something is risky, too hard for the timeline, or won't transfer to the real drone.
- Keep a `docs/progress.md` log: date, what was done, results, next steps (this feeds the thesis).
- When I ask for thesis text, write it academically but clearly, with citations I can check.
- Budget-aware: always suggest the free/cheapest option first.
- Long runs (training, big evaluations): keep them short and tell me before starting.
