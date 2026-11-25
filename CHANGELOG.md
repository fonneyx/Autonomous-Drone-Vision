# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). There is no software release yet;
trained model files are published separately on the
[Model files](https://github.com/fonneyx/autonomous-drone-vision/releases/tag/models) pre-release.

## [Unreleased]

### Added
- Tap-to-track architecture: target lock states, a safety envelope and allowed-target rules.
- Interface design: navigation diagram, wireframes, colour rules and a usability evaluation plan.
- Activity and sequence diagrams for landing on command.
- Data model: data flow diagrams, ER diagram, PostgreSQL schema with test data and queries, CI check of the schema.
- Command measurement tools: gesture and voice accuracy by distance and noise, latency reports.
- Webcam prototype: operator tracking with re-identification, gestures, offline voice, command arbiter.
- RL obstacle-avoidance pilot (PPO in MuJoCo), classical pilot and safety shield.
- Model download script with SHA-256 checks.
