# Autonomous Drone Vision

**A drone that follows you by itself.** You control it with hand gestures and your voice, it holds a steady
height, and it avoids obstacles while it follows. Built in simulation first, then moved — same code — to a
real, low-cost drone you build yourself, using reinforcement learning (RL) for the following and avoidance
decisions.

> Final-year IT diploma project.
> Thesis: *"Development of an autonomous person-following drone with gesture and voice control and obstacle
> avoidance"*

## What the drone does

| Feature | How |
|---|---|
| **Follows you** | The camera finds you with YOLO, keeps tracking **you** (not a stranger walking by) and stays at a set distance. |
| **Gesture control** | MediaPipe reads your hand and arm signals: **open palm = stop**, **fist = land**, **wave = come back**. Big gestures held for about a second, so random movements don't trigger commands. |
| **Voice control** | Vosk or Whisper understands about 10 commands ("follow me", "go higher", "take a photo", …). The microphone is on your phone or headset, not on the drone — propeller noise would drown out your voice. |
| **Altitude hold** | A downward laser + optical-flow sensor (MicoAir MTF-01) keeps a fixed height above the ground and stops drifting. |
| **Obstacle avoidance** | An 8×8 distance sensor watches the space in front; an RL pilot (with a classical pilot as baseline) steers around obstacles. |

## How it fits together

```
 phone/headset mic ──► voice commands ───────────────┐
 drone camera ──► YOLO person tracking + "is it me?" │
             └──► MediaPipe gestures ────────────────┤
 8×8 distance sensor ──► obstacles ──────────────────┤
                                                     ▼
        Command arbiter (priority: SAFETY > LAND/STOP > voice/gesture > follow)
                                                     ▼
        Follow controller + RL pilot → velocity commands (vx, vy, vz, yaw rate)
                                                     ▼
        Safety shield (min distance, max speed, stop and hover when unsure)
                                                     ▼
        MAVLink → ArduPilot (GUIDED mode) → motors
```

**RL never drives the motors directly.** ArduPilot keeps the drone stable (and handles wind and pressure);
the RL pilot only sends bounded velocity commands, and a rule-based safety shield can always override it.

## Project status

| Part | Status |
|---|---|
| MuJoCo simulator with depth camera + RL obstacle-avoidance pilot (PPO) | ✅ done — 99 % success at 40 obstacles |
| Classical pilot (follow-the-gap) + safety shield | ✅ done |
| Webcam prototype: tracking + re-ID, gestures, voice, arbiter | ✅ done — live-tested on webcam + mic |
| Follow-me in simulation (moving operator, wind, noise, latency) | ⏳ next |
| ArduPilot SITL validation | ⏳ |
| Real drone build and tests | ⏳ — see [docs/hardware.md](docs/hardware.md) (~$300 budget) |

Progress and results are logged in [docs/progress.md](docs/progress.md).

## Folder structure

```
config/           settings (YOLO model, thresholds, training)
src/
  perception/     person tracking (YOLO11-pose), re-ID of the operator, MediaPipe gestures, YOLO detector
  voice/          offline voice commands (Vosk)
  control/        command arbiter, follow controller, classical pilot, safety shield
  sim/            MuJoCo worlds — obstacle course for the RL pilot
  hardware/       (later) MAVLink link to ArduPilot
scripts/
  demo/           watch the trained pilot fly
  train/          train the RL pilot (PPO) and YOLO
  data/           dataset tools
  tools/          GPU check
  blender/        builds the 3D person / object models used in simulation
assets/sim_models 3D models
runs/             trained RL pilot used by the project
tests/            pytest tests
docs/             progress log, hardware guide
```

## Setup

Python 3.11+ on Windows, Linux or macOS. From this folder:

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

For an NVIDIA GPU, install the CUDA build of PyTorch first (see pytorch.org), then check it:

```bash
python scripts/tools/check_gpu.py
```

## Quick start

```bash
python scripts/demo/webcam_followme.py                      # follow-me prototype on your webcam + microphone
python scripts/demo/fly_avoidance.py                        # watch the RL pilot fly the obstacle course
python scripts/demo/fly_avoidance.py --headless --episodes 100   # success-rate test
python scripts/train/train_pilot.py --timesteps 200000      # short RL training run
python -m pytest tests -q                                   # run the tests
```

## Safety

Manual RC override and failsafes are configured before any autonomous flight; speed is limited, the drone
keeps a minimum distance to people, and anything unclear (lost the operator, conflicting commands) means
**stop and hover**. First real flights: indoors or an open field, low and slow, prop guards, nobody underneath.

## Licence

See [LICENCE](LICENCE).
