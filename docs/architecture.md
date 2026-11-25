# Architecture and roadmap

**Project:** an information system for intelligent drone control, monitoring and data processing.


## 1. What the information system does

The existing follow-me code (tracking, re-identification, gestures, voice, arbiter, follow controller, RL pilot) is the **intelligent control** module. The information system is everything around it that a person can see, use and check afterwards.

| Function | What the user gets |
|---|---|
| Control | Gestures and voice give commands; the arbiter decides; the laptop sends safe velocity commands to ArduPilot. The dashboard gives high-level buttons only (start follow, hold, land, return), never raw setpoints. |
| Monitoring | A live web dashboard: video with the operator box, mode, battery, altitude, distance to the operator, map, command log, warnings. |
| Data processing | Every flight is recorded in a database: telemetry, perception results, commands, arbiter decisions **with a reason**, avoidance overrides, photos and video, model versions. Reports are built from it: accuracy by distance, latency, operator-in-frame %, ID switches, rejected commands. |
| Records and audit | A tamper-evident (hash-chained) log of every decision, so any flight can be explained afterwards: "why did it land at 12:03:41?" |
| Users and consent | Operator profiles, consent records, roles (operator, analyst, admin), retention and blurring of bystanders. |

What this system adds compared with existing platforms (PX4 Flight Review, DJI FlightHub, WebODM, Aerobridge): none of them store operator consent, perception outputs, model versions, arbiter decisions with reasons, or evaluation trials.

## 2. Architecture

```
 [Drone or ArduPilot SITL] --MAVLink UDP 14550--> [Link service (pymavlink)]
 [Camera / video link] ----frames----> [Perception: YOLO + re-ID + gestures]  [Mic -> Vosk]
                                                  \            |            /
                                                   [Command arbiter + follow controller + safety]
                                                              |
                              [Recorder] --> PostgreSQL (+PostGIS, optional TimescaleDB), media files on disk
                                                              |
                     [FastAPI backend: REST + WebSocket]  [MediaMTX: WebRTC video]
                                                              |
                                          [Web dashboard: React + Vite + Leaflet]
 QGroundControl / Mission Planner stays connected in parallel (mavlink-router) as the safety GCS.
```

- **Laptop is the brain.** One Python process (asyncio) owns the MAVLink link. Only the command service writes MAVLink, with validation, clamps and rate limits.
- **GUIDED velocity commands:** `SET_POSITION_TARGET_LOCAL_NED`, velocity-only type_mask 3527, sent at 5-10 Hz only while the operator track is fresh. Lost track → zero velocity, then BRAKE/LOITER. ArduPilot stops by itself 3 s after the last command.
- **Authority order:** RC transmitter > ArduPilot failsafes (GCS heartbeat, RC, battery, geofence) > laptop > dashboard.
- **Same code for sim and real drone:** both are MAVLink on UDP 14550, so SITL is the main test bench.
- **Plain Python, not ROS 2** (ROS 2 on Windows 11 is weak). A clean interface layer leaves room for a ROS 2 bridge later.

## 3. Database (first version)

PostgreSQL + PostGIS; TimescaleDB only if telemetry becomes large. Media files stay on disk, and the database keeps only their path and SHA-256.

Main tables: `operator`, `consent`, `appearance_template`, `drone`, `flight`, `telemetry` (10-50 Hz), `video`, `photo`, `perception_frame`, `track_event`, `command_event`, `arbiter_decision` (reason code), `avoidance_action`, `model_version`, `dataset_version`, `evaluation_session`, `evaluation_trial`, `alert`, `audit_event` (hash chain), `app_user`/`role`.

The current data model is in `docs/database/` (ER diagram + `schema.sql`); migrations will use Alembic.

## 4. Engineering documents

- Requirements with IDs: FR-xx, NFR-xx, SR-xx (safety), plus a traceability matrix (ISO/IEC/IEEE 29148, ISO/IEC 25010:2023, ГОСТ 34.602-2020).
- UML in PlantUML: use case (+ descriptions), class, sequence (gesture → arbiter → MAVLink → DB → dashboard), activity, **state machine** (arbiter modes), component, deployment (laptop + drone + browser), ER.
- Test plan and results: unit tests, replay tests from recorded flights, SITL tests (`pytest -m sitl`), load test of telemetry ingestion, dashboard end-to-end tests (Playwright).

## 5. Optional real drone (laptop as the brain)

The drone carries only cheap parts; all ML runs on the laptop.

| Part | Choice |
|---|---|
| Flight controller | SpeedyBee F405 V4 stack with ArduPilot |
| Altitude/position indoors | MicoAir MTF-01 optical flow + rangefinder |
| RC + telemetry | RadioMaster Pocket ELRS in MAVLink mode (one radio; laptop connects over the transmitter's Wi-Fi on UDP 14550) |
| Video | RunCam WiFiLink 2 / OpenIPC (best image for YOLO) or analog 5.8 GHz + USB receiver (cheapest) |
| Never cut | RC transmitter, good battery and charger, prop guards, GPS + compass |

- About $330-450 at Western prices; AliExpress and Kazakh prices are still to check.
- Kazakh rules: autonomous flight is outside the open category, so the real drone is a **props-off bench demo** plus, at most, a tethered test outside settlements. It is not part of the required result.
- Licence-free radio limits (2020 list, re-check): 2.4 GHz 10 mW, 5.8 GHz 25 mW.

## 6. Phases

| # | Phase | Done when |
|---|---|---|
| 0 | Repository reorganisation | New layout, tests pass, pushed to GitHub |
| 1 | MAVLink link + ArduPilot SITL (WSL) | Webcam follow-me steers a SITL copter in GUIDED; failsafe tests pass |
| 2 | Recorder + database | Each run creates a flight with telemetry, commands, decisions, perception rows; audit chain verifies |
| 3 | Backend API + live video | REST + WebSocket telemetry; WebRTC video through MediaMTX |
| 4 | Web dashboard | Live page, flights list, flight replay, reports, operator/consent pages |
| 5 | Analytics + measurements | Accuracy-by-distance, latency and tracking reports from the database |
| 6 | Engineering documents | Requirements, UML, ER, test plan with results |
| 7 | Optional hardware | Bench test with props off: same code, real link |
