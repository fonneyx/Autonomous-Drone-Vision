"""Build the analysis and database diagrams (draw.io + PNG) from one definition.

    python scripts/docs/build_diagrams.py

Writes docs/analysis/{sequence-land-by-gesture,dfd-context,dfd-level0,dfd-level1}.{drawio,png} and
docs/database/erd.{drawio,png}. The names here match README "Users", the DFD stores and docs/database/schema.sql.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagram_kit import Diagram  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "docs" / "analysis"
DATABASE = ROOT / "docs" / "database"

SYSTEM = "Drone control, monitoring\nand data processing system"


def sequence() -> Diagram:
    """Scenario: the operator lands the drone with a fist gesture (process 2.0, then 3.0)."""
    d = Diagram("sequence-land-by-gesture", 1540, 800, "Sequence: land the drone with a fist gesture")
    names = ["Operator", "Drone\n(camera, ArduPilot)", "Person tracker\n(YOLO11-pose)", "Operator identifier\n(re-ID)",
             "Command recognizer\n(MediaPipe / Vosk)", "Command arbiter", "Flight controller\nlink (MAVLink)",
             "Database"]
    xs = []
    for i, name in enumerate(names):
        x = 30 + i * 185
        d.node(f"l{i}", "lifeline", x, 50, 165, 730, name)
        xs.append(x + 82.5)
    op, drone, tracker, ident, recog, arb, link, db = xs
    steps = [
        (op, drone, "1: raises a fist, holds 1.5 s"),
        (drone, tracker, "2: camera video frame"),
        (tracker, ident, "3: person tracks + keypoints"),
        (ident, db, "4: read appearance template (operator)"),
        (db, ident, "5: appearance template", True),
        (ident, recog, "6: operator track + hand"),
        (recog, arb, "7: candidate LAND, conf. 0.93"),
        (arb, arb, "8: check priority and flight mode"),
        (arb, db, "9: save command_event (accepted)"),
        (arb, link, "10: recognized command LAND"),
        (link, drone, "11: SET_MODE LAND (MAVLink)"),
        (drone, link, "12: telemetry: mode LAND, altitude", True),
        (link, db, "13: save telemetry_sample"),
        (link, op, '14: status "Landing" on the dashboard', True),
    ]
    y = 135
    for x1, x2, label, *reply in steps:
        d.message(x1, x2, y, label, reply=bool(reply))
        y += 62 if x1 == x2 else 44
    return d


def context() -> Diagram:
    d = Diagram("dfd-context", 1300, 540, "DFD - context diagram")
    d.node("op", "entity", 40, 140, 170, 90, "Operator")
    d.node("drone", "entity", 1090, 140, 170, 90, "Drone\n(flight controller\n+ camera)")
    d.node("an", "entity", 565, 420, 170, 80, "Analyst")
    d.node("sys", "process", 455, 80, 390, 210, SYSTEM, number="0")
    d.edge("op", (1, 0.15), "sys", (0, 0.2), "registration details\n(operator, drone, consent)")
    d.edge("sys", (0, 0.4), "op", (1, 0.4), "registration confirmation")
    d.edge("op", (1, 0.65), "sys", (0, 0.6), "voice commands")
    d.edge("sys", (0, 0.85), "op", (1, 0.9), "live view and\nflight status")
    d.edge("drone", (0, 0.2), "sys", (1, 0.25), "camera video\nwith gestures")
    d.edge("drone", (0, 0.5), "sys", (1, 0.5), "telemetry")
    d.edge("sys", (1, 0.8), "drone", (0, 0.85), "flight commands\n(MAVLink)")
    d.edge("an", (0.2, 0), "sys", (0.35, 1), "model version details,\nreport request", label_at=(560, 355))
    d.edge("sys", (0.65, 1), "an", (0.8, 0), "flight reports", label_at=(740, 355))
    return d


def level0() -> Diagram:
    d = Diagram("dfd-level0", 1500, 940, "DFD - level 0")
    d.node("op", "entity", 40, 200, 150, 90, "Operator")
    d.node("an", "entity", 40, 760, 150, 100, "Analyst")
    d.node("drone", "entity", 1320, 380, 150, 100, "Drone\n(flight controller\n+ camera)")
    d.node("p1", "process", 330, 90, 210, 110, "Register operators,\ndrones and models", number="1.0")
    d.node("p4", "process", 330, 500, 210, 110, "Monitor flight", number="4.0")
    d.node("p5", "process", 330, 790, 210, 110, "Generate flight\nreports", number="5.0")
    d.node("p2", "process", 980, 130, 210, 110, "Recognize operator\ncommands", number="2.0")
    d.node("p3", "process", 980, 560, 210, 110, "Control drone\nflight", number="3.0")
    for sid, no, name, y in [("d1", "D1", "operator", 70), ("d6", "D6", "ml_model", 170), ("d2", "D2", "drone", 290),
                             ("d5", "D5", "command_event", 430), ("d4", "D4", "telemetry_sample", 560),
                             ("d3", "D3", "flight", 690)]:
        d.node(sid, "store", 650, y, 230, 46, name, number=no)

    def fy(node: str, y: float) -> float:
        """Anchor fraction on a node's side for an absolute y."""
        n = d.nodes[node]
        return round((y - n.y) / n.h, 4)

    # 1.0 Register operators, drones and models
    d.edge("op", (0.3, 0), "p1", (0, fy("p1", 120)), "registration details\n(operator, drone, consent)",
           points=[(85, 120)], label_at=(212, 120))
    d.edge("p1", (0, fy("p1", 180)), "op", (0.7, 0), "registration confirmation", points=[(145, 180)],
           label_at=(250, 180))
    d.edge("an", (1, 0.1), "p1", (0.15, 1), "model version\ndetails", points=[(250, 770), (250, 235), (361.5, 235)],
           label_at=(250, 690))
    d.edge("p1", (1, fy("p1", 105)), "d1", (0, 0.5), "operator record")
    d.edge("p1", (1, fy("p1", 150)), "d6", (0, 0.5), "model record")
    d.edge("p1", (1, fy("p1", 190)), "d2", (0, 0.5), "drone record")
    # 2.0 Recognize operator commands
    d.edge("op", (0, 0.5), "p2", (0.5, 0), "voice commands", points=[(20, 245), (20, 48), (1085, 48)],
           label_at=(560, 48))
    d.edge("drone", (0.3, 0), "p2", (1, fy("p2", 185)), "camera video\nwith gestures", points=[(1365, 185)],
           label_at=(1365, 290))
    d.edge("d1", (1, 0.5), "p2", (0, fy("p2", 150)), "appearance template")
    d.edge("d6", (1, 0.5), "p2", (0, fy("p2", 200)), "model versions")
    d.edge("p2", (0, fy("p2", 230)), "d5", (1, 0.5), "command record", label_at=(930, 360))
    d.edge("p2", (0.5, 1), "p3", (0.5, 0), "recognized command", label_at=(1085, 400))
    # 3.0 Control drone flight
    d.edge("d2", (1, 0.5), "p3", (0, fy("p3", 580)), "drone profile", points=[(940, 313), (940, 580)],
           label_at=(940, 470))
    d.edge("drone", (0.3, 1), "p3", (1, fy("p3", 600)), "telemetry", points=[(1365, 600)], label_at=(1365, 540))
    d.edge("p3", (1, fy("p3", 645)), "drone", (0.65, 1), "flight commands\n(MAVLink)", points=[(1417.5, 645)],
           label_at=(1300, 645))
    d.edge("p3", (0, fy("p3", 610)), "d4", (1, 0.5), "telemetry\nsamples", label_at=(930, 600))
    d.edge("p3", (0, fy("p3", 650)), "d3", (1, 0.5), "flight record", label_at=(930, 685))
    # 4.0 Monitor flight
    d.edge("d5", (0, 0.5), "p4", (1, fy("p4", 520)), "recent commands", label_at=(595, 480))
    d.edge("d4", (0, 0.5), "p4", (1, fy("p4", 560)), "latest telemetry", label_at=(595, 570))
    d.edge("d3", (0, 0.5), "p4", (1, fy("p4", 595)), "flight status", label_at=(595, 660))
    d.edge("p4", (0, 0.5), "op", (0.5, 1), "live view and\nflight status", points=[(115, 555)], label_at=(115, 420))
    # 5.0 Generate flight reports
    d.edge("d5", (0, 0.75), "p5", (1, fy("p5", 820)), "command history", points=[(610, 464.5), (610, 820)],
           label_at=(610, 760))
    d.edge("d3", (0.3, 1), "p5", (1, fy("p5", 875)), "flight history", points=[(719, 875)], label_at=(640, 875))
    d.edge("an", (1, 0.4), "p5", (0, fy("p5", 815)), "report request", label_at=(262, 800))
    d.edge("p5", (0, fy("p5", 870)), "an", (1, 0.8), "flight reports", label_at=(262, 860))
    return d


def level1() -> Diagram:
    d = Diagram("dfd-level1", 1520, 720, "DFD - level 1: 2.0 Recognize operator commands")
    d.node("drone", "entity", 30, 330, 150, 90, "Drone\n(flight controller\n+ camera)")
    d.node("op", "entity", 30, 600, 150, 80, "Operator")
    labels = [("p21", "Detect and track\npersons"), ("p22", "Identify the\noperator"),
              ("p23", "Classify gesture\nor voice command"), ("p24", "Arbitrate and log\ncommand")]
    for i, (pid, label) in enumerate(labels):
        d.node(pid, "process", 280 + i * 330, 320, 200, 110, label, number=f"2.{i + 1}")
    d.node("d6", "store", 595, 100, 230, 46, "ml_model", number="D6")
    d.node("d1", "store", 595, 210, 230, 46, "operator", number="D1")
    d.node("d5", "store", 1255, 600, 230, 46, "command_event", number="D5")
    d.node("p3", "ref", 1290, 110, 160, 80, "Control drone flight", number="3.0")
    d.edge("drone", (1, 0.6), "p21", (0, 0.6), "camera video\nwith gestures", label_at=(230, 355))
    d.edge("p21", (1, 0.6), "p22", (0, 0.6), "person tracks\n+ keypoints", label_at=(545, 355))
    d.edge("p22", (1, 0.6), "p23", (0, 0.6), "operator track\n+ hand region", label_at=(875, 355))
    d.edge("p23", (1, 0.6), "p24", (0, 0.6), "candidate command\n+ confidence", label_at=(1205, 355))
    d.edge("d6", (0, 0.5), "p21", (0.5, 0), "model versions", points=[(380, 123)], label_at=(480, 123))
    d.edge("d6", (1, 0.5), "p23", (0.5, 0), "model versions", points=[(1040, 123)], label_at=(940, 123))
    d.edge("d1", (0.5, 1), "p22", (0.5, 0), "appearance template", label_at=(710, 288))
    d.edge("op", (1, 0.5), "p23", (0.5, 1), "voice commands", points=[(1040, 640)], label_at=(600, 640))
    d.edge("p24", (0.5, 1), "d5", (0.5, 0), "command record", label_at=(1370, 515))
    d.edge("p24", (0.5, 0), "p3", (0.5, 1), "recognized command", label_at=(1370, 260))
    return d


def erd() -> Diagram:
    d = Diagram("erd", 1500, 900, "ER diagram (PostgreSQL)")
    d.node("operator", "table", 40, 60, 330, 0, "operator", rows=[
        ("PK", "operator_id", "SERIAL"), ("", "full_name", "VARCHAR(100)"), ("", "email", "VARCHAR(120) UQ"),
        ("", "consent_given_at", "TIMESTAMPTZ"), ("", "template_path", "VARCHAR(255)"),
        ("", "created_at", "TIMESTAMPTZ")])
    d.node("drone", "table", 40, 420, 330, 0, "drone", rows=[
        ("PK", "drone_id", "SERIAL"), ("", "name", "VARCHAR(60)"), ("", "model", "VARCHAR(60)"),
        ("", "flight_controller", "VARCHAR(60)"), ("", "registration_no", "VARCHAR(30) UQ")])
    d.node("flight", "table", 560, 240, 330, 0, "flight", rows=[
        ("PK", "flight_id", "SERIAL"), ("FK", "operator_id", "INTEGER"), ("FK", "drone_id", "INTEGER"),
        ("", "environment", "VARCHAR(10)"), ("", "started_at", "TIMESTAMPTZ"), ("", "ended_at", "TIMESTAMPTZ"),
        ("", "end_reason", "VARCHAR(40)")])
    d.node("telemetry_sample", "table", 1110, 40, 350, 0, "telemetry_sample", rows=[
        ("PK", "sample_id", "BIGSERIAL"), ("FK", "flight_id", "INTEGER"), ("", "recorded_at", "TIMESTAMPTZ"),
        ("", "flight_mode", "VARCHAR(20)"), ("", "altitude_m", "NUMERIC(5,2)"), ("", "battery_pct", "SMALLINT"),
        ("", "distance_to_operator_m", "NUMERIC(5,2)")])
    d.node("command_event", "table", 1110, 360, 350, 0, "command_event", rows=[
        ("PK", "command_id", "SERIAL"), ("FK", "flight_id", "INTEGER"), ("FK", "model_id", "INTEGER"),
        ("", "source", "VARCHAR(10)"), ("", "command", "VARCHAR(20)"), ("", "confidence", "NUMERIC(4,3)"),
        ("", "latency_ms", "INTEGER"), ("", "decision", "VARCHAR(10)"), ("", "reason", "VARCHAR(80)"),
        ("", "created_at", "TIMESTAMPTZ")])
    d.node("flight_model", "table", 560, 600, 330, 0, "flight_model", rows=[
        ("PK,FK", "flight_id", "INTEGER"), ("PK,FK", "model_id", "INTEGER")])
    d.node("ml_model", "table", 1110, 680, 350, 0, "ml_model", rows=[
        ("PK", "model_id", "SERIAL"), ("", "name", "VARCHAR(60)"), ("", "task", "VARCHAR(20)"),
        ("", "version", "VARCHAR(20)"), ("", "file_sha256", "CHAR(64) UQ")])

    def h(name: str) -> float:
        return 30 + len(d.nodes[name].rows) * 22 + 6

    for n in d.nodes.values():  # real heights for anchor maths
        n.h = h(n.id)
    # operator 1 - N flight, drone 1 - N flight
    d.edge("operator", (1, 0.5), "flight", (0, 0.3), "conducts", kind="relation",
           points=[(465, 60 + h("operator") / 2), (465, 240 + h("flight") * 0.3)], label_at=(465, 200))
    d.edge("drone", (1, 0.4), "flight", (0, 0.75), "flies", kind="relation",
           points=[(465, 420 + h("drone") * 0.4), (465, 240 + h("flight") * 0.75)], label_at=(465, 450))
    # flight 1 - N telemetry_sample, flight 1 - N command_event
    d.edge("flight", (1, 0.2), "telemetry_sample", (0, 0.6), "records", kind="relation",
           points=[(1000, 240 + h("flight") * 0.2), (1000, 40 + h("telemetry_sample") * 0.6)], label_at=(1000, 220))
    d.edge("flight", (1, 0.7), "command_event", (0, 0.3), "receives", kind="relation",
           points=[(1000, 240 + h("flight") * 0.7), (1000, 360 + h("command_event") * 0.3)], label_at=(1000, 410))
    # ml_model 1 - N command_event
    d.edge("ml_model", (0.5, 0), "command_event", (0.5, 1), "recognizes", kind="relation", label_at=(1285, 665))
    # M:N flight <-> ml_model through the junction table flight_model
    d.edge("flight", (0.5, 1), "flight_model", (0.5, 0), "uses", kind="relation", label_at=(725, 570))
    d.edge("ml_model", (0, 0.5), "flight_model", (1, 0.5), "used in", kind="relation",
           points=[(1000, 680 + h("ml_model") / 2), (1000, 600 + h("flight_model") / 2)], label_at=(1000, 720))
    return d


def main() -> None:
    for diagram, folder in [(sequence(), ANALYSIS), (context(), ANALYSIS), (level0(), ANALYSIS),
                            (level1(), ANALYSIS), (erd(), DATABASE)]:
        diagram.save(folder, diagram.name)
        print("wrote", folder.relative_to(ROOT) / diagram.name, "(.drawio + .png)")


if __name__ == "__main__":
    main()
