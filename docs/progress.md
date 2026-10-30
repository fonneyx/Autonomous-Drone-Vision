# Progress log

## 2026-10-30 — Fresh start: repository reorganised around follow-me

**Goal (final):** a drone that follows its operator, obeys hand gestures (MediaPipe: open palm = stop, fist =
land, wave = come back) and ~10 voice commands (Vosk/Whisper, phone/headset mic), holds altitude (MTF-01) and
avoids obstacles (8×8 distance sensor), with an RL pilot for following/avoidance. Built in simulation, then on
a low-cost DIY drone (~$300, see `docs/hardware.md`).

**What was kept**
- MuJoCo obstacle course with a 64×48 depth camera and velocity-command interface (`src/sim/avoidance_env.py`).
- Trained RL avoidance pilot (PPO), `runs/avoidance/avoidance-2/best_model.zip`: 99 % success at 40
  obstacles; re-checked after the move: 19/20 courses, 0 crashes (1 timeout).
- Classical follow-the-gap pilot (`src/control/classical_pilot.py`) — the baseline RL is compared against.
- Safety shield (`src/control/safety_shield.py`): replaces the pilot's command near obstacles, slows down
  near people.
- YOLO camera detector + low-confidence frame collector (`src/perception/`), YOLO / RL training scripts,
  dataset tools, Blender person/object models.

**What was removed** (earlier experiments, not part of the follow-me goal): an underground-exploration
prototype (procedural caves, occupancy map, frontier explorer, rescue mission) and an object-approach mission.
Full history is kept in a local git bundle (`../backups/Autonomous-Drone-Vision-history-2026-10-30.bundle`).

**Lessons carried over** (useful for "Implementation challenges")
- A forward-only camera cannot see the drone's flanks: turning on the spot near thin obstacles (people,
  poles) caused collisions → the pilot needs a side guard and the shield must slow down near people.
- Pick the trained model explicitly; "newest file by date" picked an older model after `git pull` changed
  file dates.
- When the pilot follows a waypoint, prefer the direct line if it is clear; otherwise a wider gap elsewhere
  lures it away and it oscillates.

**Tests:** 13 pytest tests (obstacle course API, depth sensing, crash detection, altitude hold, simulation
speed, classical pilot, safety shield) — all pass.

### Next steps
1. Webcam prototype: YOLO person tracking + re-identification of "me", MediaPipe gestures, Vosk voice
   commands, command arbiter, on-screen velocity commands.
2. Follow-me world in MuJoCo (walking operator model) with wind, sensor noise and latency randomisation.
3. RL pilot for following + avoidance with the real sensor (8×8, ~4 m) as input; compare with classical.
