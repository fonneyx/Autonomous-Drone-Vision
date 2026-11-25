# Architecture

Autonomous Drone Vision is a ground-station system for an ArduPilot drone. The laptop is the brain: it runs
perception, decides what the drone should do, keeps the drone inside a safety envelope, and records every flight.
The drone carries only a flight controller, a camera link and a radio.

## Overview

```
camera ──► detect + track (YOLO11) ──► target lock (re-ID, lock confidence) ◄── operator selects a target
                                              │                                  (dashboard click or voice)
                                              ▼
              follow controller ──► avoidance pilot (depth from one camera) ──► safety shield
                                                                                    │
                                              MAVLink adapter (GUIDED velocity) ◄───┘
                                                     │
                                ArduPilot SITL + Webots  ·  real quad  ·  later: FPV quad, plane

every stage ──► flight recorder ──► PostgreSQL ──► backend API ──► HUD dashboard, replay, reports
```

## Design rules

1. **Decisions happen in code, in real time.** The arbiter, the pilots and the safety shield run in one Python
   process. The database is a flight recorder: it only stores what happened and never sits in the control loop.
2. **ArduPilot flies the aircraft.** The laptop sends bounded velocity setpoints in GUIDED mode
   (`SET_POSITION_TARGET_LOCAL_NED`, 5-10 Hz). ArduPilot keeps the drone stable and stops on its own if setpoints
   stop arriving.
3. **Authority order:** RC transmitter > ArduPilot failsafes (RC, GCS heartbeat, battery, geofence) > safety shield
   > arbiter > operator requests.
4. **One MAVLink adapter.** The simulator and every real vehicle are integrations behind the same interface, so the
   rest of the code does not change between them.
5. **Plain Python, not ROS 2,** because ROS 2 support on Windows is weak. The adapter layer leaves room for a ROS 2
   bridge later.

## Components

| Component | Package | What it does | Status |
|---|---|---|---|
| Detector + tracker | `perception/` | YOLO11 detection, track IDs per object | Built for people; general objects next |
| Target lock | `perception/reid.py` | Keeps the selected target through occlusions, reports lock confidence | Built for the operator; any allowed object next |
| Voice commands | `voice/` | Offline Vosk: follow, stop, come back, land | Built |
| Gestures | `perception/gestures.py` | MediaPipe palm / fist / wave, short range | Built, optional |
| Arbiter | `control/arbiter.py` | Priorities, mode changes, accepts or rejects requests with a reason | Built |
| Follow controller | `control/follow_controller.py` | Keeps distance and bearing to the target | Built |
| Avoidance pilots | `control/`, `sim/` | RL pilot (PPO), classical follow-the-gap, ArduPilot built-in avoidance | RL and classical built; depth model next |
| Safety shield | `control/safety_shield.py` | Limits speed and distance, hovers when unsure | Built |
| Evaluation | `evaluation/` | Command trials, accuracy and latency reports | Built |
| MAVLink adapter | `link/` | pymavlink connection, GUIDED setpoints, telemetry | Next |
| Simulation | `sim/` | ArduPilot SITL + Webots world with a camera, cars and pedestrians | Next |
| Flight recorder | `recorder/` | Writes flights, telemetry, track samples and decisions to PostgreSQL | Planned |
| Backend | `apps/backend/` | FastAPI: REST for flights and reports, WebSocket for live telemetry | Planned |
| Dashboard | `apps/dashboard/` | HUD-style live view, target selection, replay, reports | Planned ([wireframes](ui/README.md)) |

## Tap to track

The operator selects an object on the live video, and the drone follows it.

- **Allowed targets:** the operator (the default), registered people who agreed to be followed, and vehicles or
  boats the operator owns. Aircraft, drones and animals are not allowed targets.
- **Lock states:** `idle → selected → locked → following ⇄ reacquiring → lost (hover) → return home`.
- **Safety envelope:**
  - at least 5 m from the target
  - never closing in faster than about 1 m/s
  - a maximum speed and a geofence
  - lost lock: hover for 3 s, try to re-acquire, then return home

## Obstacle avoidance

Distance to obstacles comes from one camera through a monocular depth model, so no lidar is needed. Three pilots
are compared on the same courses: the RL pilot, a classical follow-the-gap pilot, and ArduPilot's built-in
avoidance. The metrics are success rate, minimum obstacle distance and smoothness.

## Data

PostgreSQL, with the schema in [`database/schema.sql`](database/schema.sql) and the ER diagram in
[`database/`](database/README.md). Media files stay on disk, and the database keeps their path and SHA-256. Next
tables to add: `target_lock` (what was selected, by whom, start and end, end reason) and `track_sample` (box,
confidence, estimated distance).

## Vehicles

| Vehicle | Changes needed |
|---|---|
| ArduPilot SITL (copter) | Main test bench |
| Real quad (SpeedyBee F405, ArduPilot) | Bench test with props off first; see [hardware](hardware.md) |
| FPV quad | Same code |
| Plane (ArduPilot Plane) | New follow rule (orbit or trail behind the target); lost lock means loiter instead of hover |

## Measurements

- **Lock:** percentage of time locked, time to re-acquire after occlusion, wrong-target switches.
- **Follow:** distance error against 5 m, minimum distance ever reached, speed limit respected.
- **Avoidance:** success rate and minimum obstacle distance for each pilot.
- **System:** delay from camera frame to command, dashboard latency, depth-model FPS.
