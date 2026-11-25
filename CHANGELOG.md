# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Tap-to-track architecture: target lock states, a safety envelope and allowed-target rules.
- Interface design: navigation diagram, wireframes, colour rules and a usability evaluation plan.
- Activity and sequence diagrams for landing on command.
- Model download script with SHA-256 checks; CONTRIBUTING, SECURITY, issue and PR templates.

### Changed
- The trained avoidance pilot is downloaded from the release instead of being stored in git.
- Settings moved to `configs/`, and published measurements moved to `results/`.
- Documentation reorganised under `docs/` with an index.

### Removed
- Unused 3D assets, dataset tools, YOLO training script and empty placeholder folders.

## [0.2.0]

### Added
- Data model: data flow diagrams, ER diagram, PostgreSQL schema with test data and queries, CI check of the schema.
- Command measurement tools: gesture and voice accuracy by distance and noise, latency reports.
- Webcam prototype: operator tracking with re-identification, gestures, offline voice, command arbiter.
- RL obstacle-avoidance pilot (PPO in MuJoCo), classical pilot and safety shield.

[Unreleased]: https://github.com/fonneyx/autonomous-drone-vision/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/fonneyx/autonomous-drone-vision/releases/tag/v0.2.0
