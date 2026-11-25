# Requirements

## Purpose

Let one person control a drone hands-free, while the system checks that every command really comes from that
person and keeps a complete, queryable record of each flight.

## Users

| User | What they do with the system |
|---|---|
| **Operator** | Registers and gives consent, controls the drone by voice, gestures and the dashboard, and watches the live view and flight status. |
| **Analyst** | Registers model versions and requests flight reports: command accuracy, latency, flight history. |
| **Drone** *(external system)* | Flight controller (ArduPilot) and camera. It sends video and telemetry and receives flight commands over MAVLink. |

These are the external entities of the [data flow diagrams](analysis/README.md).

## Functional requirements

| ID | Requirement |
|---|---|
| R1 | The operator registers with a full name, an email and recorded consent before the first flight. |
| R2 | The analyst registers drones and ML model versions (name, task, version, file hash). |
| R3 | The operator starts a flight by choosing a drone and the environment (simulator or real). |
| R4 | The system accepts gesture and voice commands only from the identified operator; commands from anyone else are rejected and logged with the reason. |
| R5 | The operator lands or stops the drone with a held gesture (fist = land, open palm = stop), a voice command or a dashboard button. |
| R6 | The system sends accepted commands to ArduPilot over MAVLink and records telemetry (flight mode, altitude, battery, distance to the operator). |
| R7 | The operator sees the live view, the flight status and the command log on the dashboard. |
| R8 | The analyst views flight reports: command accuracy and latency by source, and the models used in each flight. |

The screens that cover each requirement are listed in the [interface design](ui/README.md#traceability).
