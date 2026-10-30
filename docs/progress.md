# Progress log

## 2026-10-30 — Webcam follow-me prototype (Phase 1)

Everything the drone's "brain" needs, running on the laptop webcam and microphone; the window shows the velocity
commands the drone *would* fly.

| File | What it does |
|---|---|
| `src/perception/person_tracker.py` | YOLO11s-pose + ByteTrack: people with stable track ids and 17 body keypoints. 24 ms/frame on the RTX 3050 laptop GPU. |
| `src/perception/reid.py` | `OperatorLock`: follows *the operator*. Appearance template = HSV colour histograms of torso and legs (from keypoints). Re-acquires the operator after a lost/new track id only after 3 matching frames; drops a track that stops matching (id switch). Strangers are never taken for the operator. |
| `src/perception/gestures.py` | MediaPipe GestureRecognizer (ready-made, no training) + `GestureTimer`: **open palm held still 1 s = STOP, fist 1.5 s = LAND, open palm waved side to side = COME BACK**. Only hands whose wrist is inside the operator's box count. Fires once; the hand must change before firing again. Open palm held 1 s by anyone = "lock me as the operator" (when nobody is locked). |
| `src/voice/commands.py` | Vosk offline speech with a fixed grammar (only command phrases + "[unk]") → 10 commands: take off, follow me, stop, land, come back, go higher, go lower, closer, further, take a photo. Confidence threshold 0.6. |
| `src/commands.py` | One `Command` list shared by gestures, voice and keyboard. |
| `src/control/arbiter.py` | Priority SAFETY > LAND/STOP > other commands > follow. A STOP within 0.5 s of a FOLLOW/COME BACK (either order) wins → hover. Lost operator: hover, resume if found within 10 s, else stop following. Must TAKE OFF before anything else. Distance 3–8 m and altitude 1–5 m limits. |
| `src/control/follow_controller.py` | Distance from torso length (0.5 m) or body height (1.7 m) with a pinhole camera model; bearing from the image centre. Proportional control, max 2 m/s and 45 °/s; turns first, then drives; always backs off inside the minimum distance. |
| `scripts/demo/webcam_followme.py` | Live demo window: boxes ("ME" in green), hand landmarks, hold progress bars, mode, distance, simulated altitude, last command, what the microphone heard, velocity widget. Keyboard fallback for every command; `X` toggles a simulated obstacle to show the safety priority; photos saved to `runs/photos`. |
| `tests/test_followme.py` | 26 tests: gesture timing (stop, land, wave vs drift, short gestures ignored), voice parsing, arbiter (take-off first, STOP beats FOLLOW, LAND overrides, safety, lost operator, limits), controller (distance, bearing, forward/back-off, turning), re-ID (keeps operator, rejects stranger, re-acquires after id change), tracker on a photo. **39 tests total, all pass.** |

Checked automatically: the demo runs at ~30 FPS on a test video (both people found, correct "landed / show an open
palm" state).

**Live test (2026-10-30, laptop webcam + microphone, by the operator):** everything worked — lock with open palm,
take off, follow, the three gestures (stop / land / come back) and the voice commands. Not yet measured: accuracy
vs distance and noise, and command latency (next step 2).

Free downloads on first run: YOLO11s-pose (~20 MB), MediaPipe gesture model (~8 MB), Vosk English model (~40 MB).
Known limits: colour re-ID confuses two people in very similar clothes (upgrade: a deep re-ID network); hand
shapes are reliable within a few metres — from the drone at 3–6 m this needs testing (fallback in CLAUDE.md: fewer
gestures / closer range).

### Next steps
1. ~~Live test with the user~~ — done, all features worked.
2. Measure: gesture accuracy and voice accuracy vs distance and noise, command latency (thesis metrics).
3. Follow-me world in MuJoCo with a walking operator, then the RL follow + avoidance pilot.

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
