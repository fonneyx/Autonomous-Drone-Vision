# Progress log

## 2026-11-23 — Approved title, system plan, repository reorganised as an information system

- **Title approved by the university:** "Development of an Information System for Intelligent Drone Control,
  Monitoring, and Data Processing" (RU: "Разработка информационной системы для интеллектуального управления
  дроном, мониторинга и обработки данных"). The follow-me drone is the "intelligent control" part; the thesis
  adds monitoring (web dashboard) and data processing (database, ML analytics, audit log).
- **Research:** `../reports/Drone information system software design.md` (architecture, database, testing,
  laptop-as-brain hardware). **Plan:** `docs/system_plan.md` (functions, architecture, tables, UML list, phases).
- **Reorganisation:** code moved to one package `src/dronevision/` (imports `dronevision.*`), `pyproject.toml`
  replaces `requirements.txt`, ruff + pre-commit config, tests in `tests/unit/`, `dataset/` → ignored `data/`,
  new `backend/`, `frontend/`, `db/`, `docs/thesis/`, `docs/uml/`, stubs `dronevision.link` and
  `dronevision.storage`. Removed unused `weights/yolo26n.pt`, the Blender auto-backup and stale local branches
  (all kept in `../backups/*.bundle`). 56 tests pass; every script starts.
- **Next:** Phase 1 of the plan: MAVLink link + ArduPilot SITL (in WSL), webcam follow-me steering a SITL copter.

## 2026-11-05 — Measurement tools for gesture and voice commands

Tools to collect the thesis metrics "gesture/voice accuracy vs distance and noise" and "latency".

| File | What it does |
|---|---|
| `scripts/eval/measure_commands.py` | Guided session on the webcam/microphone. The screen says what to do ("Hold a FIST", `Say "go higher"`, or "Move naturally - NO gesture" for control trials); 3 s to get ready, 6 s to do it. Blocks per condition (default gesture: 1, 2, 3, 5 m; voice: quiet, noisy), shuffled order with a fixed seed, SPACE between blocks to walk to the next distance, K to skip a botched trial. Every trial is appended to `runs/measurements/<source>_<time>.csv` immediately. |
| `scripts/eval/report_commands.py` | `report.md` (accuracy with 95 % Wilson interval, false-trigger rate, latency median/mean/90th percentile, confusions) and charts `gesture_accuracy.png` (accuracy vs distance) and `voice_accuracy.png` (per phrase, quiet vs noisy). |
| `src/evaluation/command_trials.py` | Trial plan + `TrialRunner` (prepare → active → result; first command from the tested source counts; events before the window are ignored) + CSV. |
| `src/evaluation/command_report.py` | The statistics and Markdown tables. |
| `tests/test_measurements.py` | 13 tests (plan, runner timing, outcomes, latency, CSV, report numbers, Wilson interval). **52 tests total, all pass.** |

Definitions (for the methods chapter):
- **Accuracy** = correct / attempts; wrong command and no command (6 s) are both failures, counted separately.
- **False-trigger rate** = control trials (tester moves/talks naturally) in which any command fired.
- **Gesture latency** = command fired − first frame the expected hand shape was seen. Includes the designed hold
  (1 s stop, 1.5 s land), so the recognition overhead is latency − hold.
- **Voice latency** = command fired − end of the last word (Vosk word timestamps vs audio fed to the recognizer).
- Each trial also logs the share of frames with a hand on the tester and the camera's own distance estimate
  (pinhole, torso length), to check the tester stood at the right distance.
- Not included yet: the drone link (Pi Zero video + MAVLink) and the arbiter's 0.5 s conflict window for FOLLOW /
  COME BACK; STOP and LAND act at once. Link latency is measured on hardware (Phase 4).

Checked: a headless dry run on a test clip (8 trials, CSV and report written; the photo has no gestures, so all
"missed", no false triggers) and the voice recorder starts. Real measurements: needs the user.

**First real session, 1 m block only (17 trials, `runs/measurements/gesture_20261105_200458.csv`):**

| Gesture | Correct | Notes |
|---|---|---|
| LAND (fist) | 5/5 | latency 1.52 s = 1.5 s hold + ~20 ms |
| STOP (palm) | 4/5 | latency 1.02 s = 1 s hold + ~20 ms; 1 × read as COME BACK after 0.25 s |
| COME BACK (wave) | 3/5 | latency 0.2–0.47 s; 2 × read as STOP (palm held still ≥ 1 s before waving) |
| control (no gesture) | 0/2 | both fired STOP at ~1.05 s |

Findings and fixes:
- In almost every trial the expected hand shape was already visible 50 ms after the window opened: the tester
  raised the hand during "get ready". Both false triggers came right after a palm trial (palm still up). → The
  prepare screen now says **"hands DOWN, wait for NOW"**, so each trial starts from rest as in real use. 1 m data
  from before this change is kept but should be re-recorded for the thesis.
- STOP → COME BACK in 0.25 s: the wave detector counted landmark jitter while the palm moved into place as
  direction changes. → Fixed with **hysteresis**: a direction change counts only after the hand has swung back
  ≥ 0.5 palm widths from its last turning point (`GestureTimer._swings`). New regression test fails on the old
  code and passes now.
- COME BACK → STOP when the palm is held still for 1 s before waving is the designed behaviour (still palm =
  stop); the operator must start waving within 1 s. Accepted; documented as a usage rule.
- Recognition overhead beyond the designed hold is ~1 frame (~20 ms) on the laptop.
- The camera distance estimate is empty at 1 m (hips/feet out of frame) — expected, it needs the torso or full body.

**54 tests, all pass.**

**Second session, all distances (68 trials, `runs/measurements/gesture_20261105_201051.csv`), after the wave fix:**

| Distance | Correct (of 15) | False triggers (of 2) |
|---|---|---|
| 1 m | 9 (60 %) | 2 |
| 2 m | 8 (53 %) | 2 |
| 3 m | 10 (67 %) | 2 |
| 5 m | 9 (60 %) | 2 |

By gesture: STOP 16/20, COME BACK 12/20, LAND 8/20. Errors: 19 of 27 were a STOP that nobody asked for (7 × during
a wave, 5 × during a fist, 7 × in "no gesture" control trials); the expected shape was already visible within
0.15 s of NOW in 47 of 60 trials. Accuracy did not fall with distance.

Diagnosis: an open palm was in view and still *before* the tester did anything. Most likely a relaxed hand hanging
at the side, which MediaPipe labels Open_Palm; held still for 1 s, that is a STOP by the rules. (To confirm with the
tester.) Fixes:
- **Raised-hand rule** (`gestures.raised_hands`): a hand counts only with the wrist above mid-chest (shoulder line +
  0.5 × torso; torso from shoulder width when the hips are out of frame). Matches the spec "big gestures"; used in
  the demo (also for the lock palm) and in the measurements.
- Measurement: the get-ready countdown restarts while a hand is raised ("LOWER YOUR HAND"), so every trial
  starts from rest. Instructions now say "Raise ...".
- Distance check: the camera estimate was empty at 1–2 m (hips out of frame) and ~2.0 m at "3m", ~1.55 m at "5m" —
  it should grow with distance, so either the tester stood closer than the tape marks or the estimate is off
  (webcam field of view assumed 70°). To check with the tester before reporting distances.

These 68 trials are kept as the "before" baseline for the thesis.

**Third session with the raised-hand rule (68 trials, `gesture_20261105_205141.csv`): 4/60 correct**, but also
0/8 false triggers. A usable hand was seen in ~0 % of frames (before: 100 %). The rule or the detection broke it;
the CSV could not tell which. Lesson: check a rule on the live camera for a minute before a 30-minute session.

**20 s gesture check** (`scripts/tools/gesture_check.py`, annotated snapshots in `runs/diagnostics/`, not uploaded),
tester at ~1 m:
- Hands down: hands are below the picture, **no hand detected** — so the "hanging hand looks like a palm"
  explanation was wrong, at least at close range. The earlier false STOPs came from hands still raised.
- Palm, fist and wave: detected and labelled correctly (fist 168/210 frames, palm 180/212), and the rule kept
  every hand (at 1 m its limit is below the picture).
- So at close range the rule is harmless; why hands vanished in the session (rule cutting lower gestures at 2–5 m,
  or MediaPipe not finding small hands far away) is still open.

Measurement now logs three shares per trial — `hand_found` (MediaPipe found any hand), `hand_own` (on the
tester), `hand_seen` (raised, usable) — and `--record` saves the camera video + frame log next to the CSV (kept
off GitHub), so sessions can be re-analysed and new rules tested offline instead of repeating sessions.
**56 tests, all pass.**

### Next steps
1. User re-runs the gesture session with the hands-down rule (1/2/3/5 m, 5 repeats ≈ 15 min) and the voice
   session (quiet + noisy ≈ 15 min).
2. Report → thesis tables; decide whether the 5 m gesture accuracy is good enough or the fallback (closer range /
   fewer gestures) is needed.
3. Phase 2: follow-me world in MuJoCo.

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
