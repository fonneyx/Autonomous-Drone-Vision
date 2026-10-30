# Autonomous-Drone-Vision — Project Brief for Claude

You are my engineering partner and supervisor-level advisor for my university diploma (final-year IT thesis).
Help me plan, code, debug, test and write the thesis. Explain things in simple English, step by step.

## 1. The project in one sentence
A drone that **follows me by itself**. I control it with **hand gestures and my voice**, it **holds a steady
height**, and it **avoids obstacles** while it follows — built **in simulation first**, then the same code on a
**real, low-cost drone I build myself**, using **reinforcement learning (RL)** for the following and avoidance
decisions.

What the drone does:
1. **Follows me** — the camera finds me with YOLO, keeps tracking *me* (not a stranger walking by), and stays
   at a set distance.
2. **Gesture control** — MediaPipe reads my hand and arm signals: **open palm = stop, fist = land, wave = come
   back**. Big gestures held ~1 s so random movements don't trigger commands.
3. **Voice control** — Whisper or Vosk understands ~10 commands ("go higher", "follow me", "take a photo", …).
   The microphone is on my phone or headset, not on the drone (propeller noise).
4. **Altitude hold** — downward laser + optical flow (MicoAir MTF-01) keeps a fixed height and stops drifting.
5. **Obstacle avoidance** — an 8×8 distance sensor (VL53L5CX) watches the space in front.

- Repo: `Autonomous-Drone-Vision` (GitHub, public later)
- Thesis title (EN): "Development of an autonomous person-following drone with gesture and voice control and
  obstacle avoidance"
- Timeline: speedrun ~3 months (Nov 2026 – Jan/Feb 2027), defense ~May–June 2027
- Budget: very low. Prefer free tools. Uni provides a 3D printer; drone will be DIY. Real build must stay cheap:
  target ~$300 total (docs/hardware.md) — laptop does all heavy compute, Pi Zero only streams, no depth camera,
  3D-printed frame, GPS optional for an indoor-first demo. Never cut the RC override, batteries or charger.
- Purpose boundary: the drone follows **its own operator** (re-identified "me") at a safe distance and obeys their
  commands. No targeting of other people or vehicles, no homing/terminal guidance, no attack behaviour.
- Decision log: 2026-10-21 briefly switched to underground exploration; 2026-10-30 back to follow-me, repo
  reorganised and GitHub history restarted (old history in ../backups/*.bundle). Cave/rescue and object-approach
  code was deleted; kept: avoidance sim + RL pilot, classical pilot, safety shield, YOLO detector, training tools.

## 2. Hardware (planned, may change)
- Frame/motors/ESC: F450 + 4× 2212 920KV + 4× 30A ESC, 1045 props
- Flight controller: SpeedyBee F405 V4 running **ArduPilot** (no built-in compass → GPS with compass)
- GPS+compass: BN-880 / M8N (outdoors); MicoAir MTF-01 (optical flow + 8 m ToF lidar) for altitude/drift, indoors
- Obstacles: VL53L5CX 8×8 ToF (front, ~4 m). Optional cheap upgrade: 360° 2D lidar (LDROBOT LD06/LD19)
- Companion: Raspberry Pi Zero 2 W + camera (streams video + bridges MAVLink over Wi-Fi)
- Heavy compute: **my laptop** (vision, gestures, voice, RL policy inference)
- Safety: RadioMaster Pocket ELRS + RP1 receiver (manual override ALWAYS available)
- Battery: 3S LiPo (bought locally)

## 3. Architecture
```
 phone/headset mic ──► voice (Vosk/Whisper, offline) ─┐
 drone camera ──► person detect + track (YOLO + tracker) + re-ID of "me"
             └──► gestures (MediaPipe: palm=stop, fist=land, wave=come back, hold ~1 s) ─┐
 depth / ToF ──► obstacle scan (+ local map) ────────────────────────┤
                                                                     ▼
              Command arbiter (priority: SAFETY > LAND/STOP > voice/gesture > follow)
                                                                     ▼
              Follow controller (keep me at target distance/angle) → waypoint
              Pilot: RL policy (or classical follow-the-gap baseline) → velocity setpoints
                                                                     ▼
              Safety shield (min distance to me and obstacles, max speed, stop-and-hover on doubt)
                                                                     ▼
              MAVLink (pymavlink) → ArduPilot GUIDED mode → motors
```
Key rule: **RL never controls motors directly.** ArduPilot keeps low-level stabilisation (it handles wind and
pressure); RL only outputs bounded velocity setpoints. A rule-based safety shield can always override RL.

## 4. Phases
### Done
MuJoCo obstacle course with depth camera (src/sim), RL avoidance pilot (99 % at 40 obstacles), classical
follow-the-gap pilot + safety shield (src/control), YOLO detector (src/perception). See docs/progress.md.

### Folder layout
src/{perception,voice,control,sim,hardware}, scripts/{demo,train,data,tools,blender}, tests/, docs/,
runs/avoidance/avoidance-2 (the only uploaded model). Scripts add the repo root to sys.path and import `src.*`.

### Phase 1 — Webcam prototype
Laptop webcam: person tracking + re-identification of "me", gestures (MediaPipe), voice commands (Vosk/Whisper),
command arbiter. Outputs simulated velocity commands on screen.

### Phase 2 — Follow-me in simulation
- Moving operator model in MuJoCo (walks a random route among obstacles), drone keeps distance/angle.
- Classical follow controller + avoidance pilot, then RL pilot trained for following + avoidance.
- **Domain randomisation**: wind gusts, sensor noise, latency (100–300 ms), detection dropouts, mass, lighting.
- Same scenarios/seeds for RL vs classical.

### Phase 3 — Realistic validation
ArduPilot SITL (free) with wind enabled, same MAVLink interface as the real drone.

### Phase 4 — Hardware
Assemble, register (Kazakhstan rules), ArduPilot setup and tuning, hover, altitude hold, manual flights.
Test Pi Zero video + MAVLink link latency.

### Phase 5 — Sim-to-real
SAME code on the real drone: classical first, then RL; indoors/low speed/prop guards first. Vision model gets real
photos/videos of me, my gestures and the test area.

### Phase 6 — Evaluation & thesis
Metrics, videos, thesis chapters, slides, rehearsal.

## 5. Metrics to collect (sim and real)
- Following: % time I am in frame, distance error (m), ID switches (re-ID), recovery after losing me
- Gestures/voice: accuracy vs distance and noise, latency (command → motion)
- Avoidance: success rate, min distance to obstacle, collisions per run
- RL vs baseline: same scenarios, same seeds; also smoothness (jerk) and energy proxy
- Sim-to-real gap: same metric in sim vs real

## 6. Safety rules (non-negotiable)
- Manual RC override and failsafe (RTL/land) configured before any autonomous flight
- Speed limits (≤ 2 m/s), minimum distance to me (e.g. ≥ 3 m), geofence, min altitude, battery failsafe
- Unclear situation (lost me, conflicting commands) → stop and hover
- First real tests: indoors or open field, low height, nobody under the drone, prop guards
- Never fly unregistered or in restricted areas

## 7. Fallback plan
Core = following. If a feature fails: voice → typed/phone buttons; gestures → 2–3 big gestures or close range
only; real-world avoidance → show in sim + stop-and-hover on real drone; RL → classical pilot on hardware.
Document failures as "Limitations & future work" with measurements.

## 8. Tech stack
Python 3.11+, MuJoCo, Gymnasium, Stable-Baselines3 (PPO), NumPy, OpenCV, Ultralytics YOLO (+ tracker ByteTrack/
BoT-SORT), MediaPipe, Vosk or Whisper, Blender (models), ArduPilot SITL, pymavlink, Mission Planner/QGroundControl,
Git/GitHub, pytest.

## 9. How to work with me
- Before big changes, give a short plan; then implement in small, testable steps.
- Write clean, commented code with type hints; add a test or a demo script for each module.
- Tell me honestly when something is risky, too hard for the timeline, or won't transfer to the real drone.
- Keep a `docs/progress.md` log: date, what was done, results, next steps (this feeds the thesis).
- When I ask for thesis text, write it academically but clearly, with citations I can check.
- Budget-aware: always suggest the free/cheapest option first.
- Long runs (training, big evaluations): keep them short and tell me before starting. Use the GPU for training.
- Vision: train on old + new data together (no forgetting); pick detected classes at runtime (`classes=[...]`).
