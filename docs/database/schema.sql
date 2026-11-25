-- Autonomous Drone Vision - database schema, test data and example queries (PostgreSQL 14+)
--
-- Tables match docs/database/erd.png and the data stores of docs/analysis/dfd-level0.png:
--   D1 operator, D2 drone, D3 flight, D4 telemetry_sample, D5 command_event, D6 ml_model,
--   plus flight_model, the junction table of the many-to-many link flight <-> ml_model.
-- The script can be run again: it drops and recreates everything.

SET TIME ZONE INTERVAL '+05:00' HOUR TO MINUTE;  -- Kazakhstan time (UTC+5)

DROP TABLE IF EXISTS command_event, telemetry_sample, flight_model, flight, ml_model, drone, operator CASCADE;

-- ---------------------------------------------------------------- tables

-- D1: people who control the drone; consent is required before any video of them is processed
CREATE TABLE operator (
    operator_id       SERIAL       PRIMARY KEY,
    full_name         VARCHAR(100) NOT NULL,
    email             VARCHAR(120) NOT NULL UNIQUE,
    consent_given_at  TIMESTAMPTZ  NOT NULL,
    template_path     VARCHAR(255),                       -- appearance template file used for re-identification
    created_at        TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- D2: registered drones (simulated or real)
CREATE TABLE drone (
    drone_id           SERIAL      PRIMARY KEY,
    name               VARCHAR(60) NOT NULL,
    model              VARCHAR(60) NOT NULL,
    flight_controller  VARCHAR(60) NOT NULL,
    registration_no    VARCHAR(30) NOT NULL UNIQUE
);

-- D6: versions of the machine-learning models the system runs
CREATE TABLE ml_model (
    model_id     SERIAL      PRIMARY KEY,
    name         VARCHAR(60) NOT NULL,
    task         VARCHAR(20) NOT NULL CHECK (task IN ('person_pose', 'gesture', 'voice', 'avoidance')),
    version      VARCHAR(20) NOT NULL,
    file_sha256  CHAR(64)    NOT NULL UNIQUE,
    UNIQUE (name, version)
);

-- D3: one row per flight (operator 1:N flight, drone 1:N flight)
CREATE TABLE flight (
    flight_id    SERIAL      PRIMARY KEY,
    operator_id  INTEGER     NOT NULL REFERENCES operator (operator_id),
    drone_id     INTEGER     NOT NULL REFERENCES drone (drone_id),
    environment  VARCHAR(10) NOT NULL CHECK (environment IN ('sim', 'real')),
    started_at   TIMESTAMPTZ NOT NULL,
    ended_at     TIMESTAMPTZ,
    end_reason   VARCHAR(40),
    CHECK (ended_at IS NULL OR ended_at > started_at)
);

-- Junction table: which model versions were used in which flight (flight M:N ml_model)
CREATE TABLE flight_model (
    flight_id  INTEGER NOT NULL REFERENCES flight (flight_id) ON DELETE CASCADE,
    model_id   INTEGER NOT NULL REFERENCES ml_model (model_id),
    PRIMARY KEY (flight_id, model_id)
);

-- D4: telemetry from the flight controller (flight 1:N telemetry_sample)
CREATE TABLE telemetry_sample (
    sample_id               BIGSERIAL    PRIMARY KEY,
    flight_id               INTEGER      NOT NULL REFERENCES flight (flight_id) ON DELETE CASCADE,
    recorded_at             TIMESTAMPTZ  NOT NULL,
    flight_mode             VARCHAR(20)  NOT NULL,
    altitude_m              NUMERIC(5,2) NOT NULL,
    battery_pct             SMALLINT     NOT NULL CHECK (battery_pct BETWEEN 0 AND 100),
    distance_to_operator_m  NUMERIC(5,2)
);

-- D5: every recognized command and the arbiter's decision with its reason
-- (flight 1:N command_event, ml_model 1:N command_event)
CREATE TABLE command_event (
    command_id  SERIAL       PRIMARY KEY,
    flight_id   INTEGER      NOT NULL REFERENCES flight (flight_id) ON DELETE CASCADE,
    model_id    INTEGER      REFERENCES ml_model (model_id),  -- NULL for dashboard buttons
    source      VARCHAR(10)  NOT NULL CHECK (source IN ('gesture', 'voice', 'dashboard')),
    command     VARCHAR(20)  NOT NULL,
    confidence  NUMERIC(4,3) CHECK (confidence BETWEEN 0 AND 1),
    latency_ms  INTEGER      CHECK (latency_ms >= 0),
    decision    VARCHAR(10)  NOT NULL CHECK (decision IN ('accepted', 'rejected')),
    reason      VARCHAR(80)  NOT NULL,
    created_at  TIMESTAMPTZ  NOT NULL
);

CREATE INDEX idx_telemetry_flight_time ON telemetry_sample (flight_id, recorded_at);
CREATE INDEX idx_command_flight_time ON command_event (flight_id, created_at);

-- ---------------------------------------------------------------- test data

INSERT INTO operator (full_name, email, consent_given_at, template_path) VALUES
    ('Aidar Serikov',      'operator1@dronevision.local', '2026-11-20 10:00+05', 'templates/operator_1.npz'),
    ('Aruzhan Sadykova',   'operator2@dronevision.local', '2026-11-21 09:30+05', 'templates/operator_2.npz'),
    ('Timur Akhmetov',     'operator3@dronevision.local', '2026-11-22 14:15+05', 'templates/operator_3.npz'),
    ('Madi Zhaksylykov',   'operator4@dronevision.local', '2026-11-22 15:00+05', NULL);

INSERT INTO drone (name, model, flight_controller, registration_no) VALUES
    ('SITL copter',  'ArduPilot SITL quad', 'ArduCopter 4.6 (SITL)',  'SIM-001'),
    ('MuJoCo quad',  'MuJoCo simulated quad', 'built-in controller', 'SIM-002'),
    ('Falcon F450',  'F450 quadcopter',     'SpeedyBee F405 V4',     'KZ-UAV-0450');

INSERT INTO ml_model (name, task, version, file_sha256) VALUES
    ('yolo11s-pose',       'person_pose', '11.0',   repeat('a', 64)),
    ('gesture_recognizer', 'gesture',     '0.10.30', repeat('b', 64)),
    ('vosk-small-en',      'voice',       '0.15',   repeat('c', 64)),
    ('ppo-avoidance',      'avoidance',   '2',      repeat('d', 64));

INSERT INTO flight (operator_id, drone_id, environment, started_at, ended_at, end_reason) VALUES
    (1, 1, 'sim',  '2026-11-23 11:00+05', '2026-11-23 11:12+05', 'landed by gesture'),
    (2, 1, 'sim',  '2026-11-23 15:30+05', '2026-11-23 15:41+05', 'landed by voice'),
    (1, 2, 'sim',  '2026-11-24 10:05+05', '2026-11-24 10:20+05', 'operator lost - hover and land'),
    (3, 3, 'real', '2026-11-24 17:00+05', '2026-11-24 17:04+05', 'bench test, props off');

INSERT INTO flight_model (flight_id, model_id) VALUES
    (1, 1), (1, 2), (1, 3),
    (2, 1), (2, 3),
    (3, 1), (3, 2), (3, 4),
    (4, 1), (4, 2);

INSERT INTO telemetry_sample (flight_id, recorded_at, flight_mode, altitude_m, battery_pct, distance_to_operator_m) VALUES
    (1, '2026-11-23 11:00:05+05', 'GUIDED', 2.00, 98, 3.10),
    (1, '2026-11-23 11:06:00+05', 'GUIDED', 2.40, 81, 4.05),
    (1, '2026-11-23 11:11:40+05', 'LAND',   0.30, 67, 3.60),
    (2, '2026-11-23 15:35:00+05', 'GUIDED', 2.10, 85, 3.80),
    (2, '2026-11-23 15:40:30+05', 'LAND',   0.20, 72, 3.20),
    (3, '2026-11-24 10:19:00+05', 'BRAKE',  1.80, 60, NULL),
    (4, '2026-11-24 17:02:00+05', 'STABILIZE', 0.00, 99, 2.50);

INSERT INTO command_event (flight_id, model_id, source, command, confidence, latency_ms, decision, reason, created_at) VALUES
    (1, 2,    'gesture',   'stop',       0.910, 1080, 'accepted', 'operator gesture, held 1.0 s',        '2026-11-23 11:03:10+05'),
    (1, 3,    'voice',     'go_higher',  0.870,  640, 'accepted', 'voice command from operator',         '2026-11-23 11:05:00+05'),
    (1, 2,    'gesture',   'land',       0.930, 1540, 'accepted', 'operator gesture, held 1.5 s',        '2026-11-23 11:11:30+05'),
    (2, 3,    'voice',     'follow_me',  0.820,  710, 'accepted', 'voice command from operator',         '2026-11-23 15:31:00+05'),
    (2, 3,    'voice',     'take_photo', 0.540,  690, 'rejected', 'confidence below 0.60',               '2026-11-23 15:36:20+05'),
    (2, 3,    'voice',     'land',       0.900,  600, 'accepted', 'voice command from operator',         '2026-11-23 15:40:10+05'),
    (3, 2,    'gesture',   'come_back',  0.880, 1250, 'accepted', 'wave detected (3 swings)',            '2026-11-24 10:10:00+05'),
    (3, 2,    'gesture',   'stop',       0.770, 1100, 'rejected', 'hand belongs to another person',      '2026-11-24 10:14:00+05'),
    (4, NULL, 'dashboard', 'land',       NULL,   120, 'accepted', 'dashboard button',                    '2026-11-24 17:03:30+05');

-- ---------------------------------------------------------------- queries

-- Q1 (JOIN of 5 tables): command log with operator, drone and model names
SELECT c.created_at::timestamp(0)       AS time,
       o.full_name                      AS operator,
       d.name                           AS drone,
       c.source,
       c.command,
       COALESCE(m.name, '-')            AS model,
       c.decision,
       c.reason
FROM command_event c
JOIN flight   f ON f.flight_id   = c.flight_id
JOIN operator o ON o.operator_id = f.operator_id
JOIN drone    d ON d.drone_id    = f.drone_id
LEFT JOIN ml_model m ON m.model_id = c.model_id
ORDER BY c.created_at;

-- Q2 (GROUP BY): accuracy and latency of each command source
SELECT c.source,
       COUNT(*)                                                    AS commands,
       COUNT(*) FILTER (WHERE c.decision = 'accepted')             AS accepted,
       ROUND(100.0 * COUNT(*) FILTER (WHERE c.decision = 'accepted') / COUNT(*), 1) AS accepted_pct,
       ROUND(AVG(c.latency_ms))                                    AS avg_latency_ms
FROM command_event c
GROUP BY c.source
ORDER BY commands DESC;

-- Q3 (GROUP BY over the M:N link): flight summary with the models used and the lowest battery
SELECT f.flight_id,
       o.full_name                                              AS operator,
       f.environment,
       ROUND(EXTRACT(EPOCH FROM (f.ended_at - f.started_at)) / 60, 1) AS minutes,
       STRING_AGG(DISTINCT m.name, ', ' ORDER BY m.name)        AS models_used,
       (SELECT MIN(t.battery_pct) FROM telemetry_sample t WHERE t.flight_id = f.flight_id) AS min_battery_pct
FROM flight f
JOIN operator     o  ON o.operator_id = f.operator_id
JOIN flight_model fm ON fm.flight_id  = f.flight_id
JOIN ml_model     m  ON m.model_id    = fm.model_id
GROUP BY f.flight_id, o.full_name, f.environment, f.started_at, f.ended_at
ORDER BY f.flight_id;
