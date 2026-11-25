# Autonomous Drone Vision

[![CI](https://github.com/fonneyx/autonomous-drone-vision/actions/workflows/ci.yml/badge.svg)](https://github.com/fonneyx/autonomous-drone-vision/actions/workflows/ci.yml)

A ground station for ArduPilot drones. You pick a person or vehicle on the camera view, and the drone follows it at a
safe distance, avoids obstacles and records the flight. The laptop does the vision and the decisions; the drone only
needs a flight controller, a camera and a radio link.

**Status: early development.** Nothing flies yet. The parts below are what exists today.

## What works now

- **Webcam prototype.** Tracks one person with YOLO11-pose, keeps the lock on that person with re-identification,
  and turns voice commands (Vosk, offline) and hand gestures into simulated drone commands through a command arbiter.
- **Obstacle avoidance in simulation.** A PPO pilot trained in a MuJoCo obstacle course (99 % of runs finished with
  40 obstacles), a classical follow-the-gap pilot to compare against, and a safety shield that can override both.
- **Flight database.** PostgreSQL schema for flights, telemetry and command decisions, with test data and queries
  checked in CI.
- **Command measurements.** Scripts that measure gesture and voice accuracy by distance and noise, and command
  latency.

## Not built yet

Selecting any object to follow, ArduPilot SITL and MAVLink link, depth from the camera, the web dashboard, and flight
on a real drone. The plan is in [docs/architecture.md](docs/architecture.md).

## How it works

ArduPilot flies the drone and keeps it stable. This project runs on a laptop and sends it velocity commands over
MAVLink. Each camera frame passes through four parts:

- **Perception** finds people and vehicles and keeps the lock on the chosen target.
- **Follow controller** works out where the drone should be to keep its distance.
- **Pilot** steers around obstacles: a reinforcement-learning pilot or a classical one.
- **Safety shield** enforces the minimum distance and speed limit, and hovers when the target is lost.

Everything is written to PostgreSQL for replay and reports; the database is never in the control loop. Details are
in [docs/architecture.md](docs/architecture.md).

## Getting started

Python 3.11 or newer.

```bash
git clone https://github.com/fonneyx/autonomous-drone-vision.git
cd autonomous-drone-vision
python -m venv .venv
.venv\Scripts\activate                    # Linux/macOS: source .venv/bin/activate
pip install -e . --group dev
python scripts/tools/download_models.py  # trained avoidance pilot
```

```bash
python scripts/demo/webcam_followme.py   # webcam prototype
python scripts/demo/fly_avoidance.py     # RL pilot flying the obstacle course
python -m pytest -q                      # tests
```

## Layout

```
src/dronevision/   perception, control, voice, sim, evaluation
configs/           settings
scripts/           demos, training, evaluation, tools, diagram builders
results/           measurement results
docs/              architecture, requirements, diagrams, database, interface design
```

## License

All rights reserved, see [LICENCE](LICENCE). Ultralytics YOLO is AGPL-3.0.
