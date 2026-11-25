"""Build the user-interface design: navigation diagram and screen wireframes (draw.io + PNG).

    python scripts/docs/build_ui.py

Writes docs/ui/navigation.{drawio,png} and docs/ui/screens/NN-name.{drawio,png}. Form fields follow the columns of
docs/database/schema.sql, and lists show its INSERT test data. Requirement IDs (R1-R8) are those in README.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagram_kit import Diagram  # noqa: E402
from wireframe_kit import BLUE, GREY, INK, LINE, PAPER, RED, Screen  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
UI = ROOT / "docs" / "ui"
SCREENS = UI / "screens"
USER = "Aidar Serikov"


# ---------------------------------------------------------------- screens
def sign_in() -> Screen:
    s = Screen("01-sign-in", "Sign in")
    s.rect(290, 150, 380, 300, fill="#ffffff", stroke=LINE, rounded=True)
    s.text(320, 170, 320, 30, "Operator sign-in", 15, bold=True)
    s.field(320, 215, 320, "Email", "operator1@dronevision.local", required=True,
            hint="The email you registered with")
    s.button(320, 310, 320, "Sign in", primary=True)
    s.text(320, 366, 320, 24, "New operator?", 12, color=GREY, align="center")
    s.button(400, 392, 160, "Register")
    return s


def register(errors: bool = False) -> Screen:
    name = "02-register-errors" if errors else "02-register"
    s = Screen(name, "Register operator", back="Sign in")
    s.text(24, 116, 900, 24, "Fields are stored in the operator table. * = required.", 12, color=GREY)
    if errors:
        s.field(24, 150, 440, "Full name", "", required=True,
                error="Enter your full name, for example: Aidar Serikov.")
        s.field(24, 240, 440, "Email", "aidar.serikov@mail", required=True,
                error="Email format is wrong. Example: name@mail.com")
    else:
        s.field(24, 150, 440, "Full name", "Aidar Serikov", required=True, hint="Up to 100 characters")
        s.field(24, 240, 440, "Email", "operator1@dronevision.local", required=True,
                hint="Used to sign in; must be unique")
    s.text(24, 330, 440, 20, "Consent *", 12, bold=True)
    s.checkbox(24, 352, "I agree that my appearance is stored so the drone recognises only me.",
               checked=not errors,
               error="Tick the box to continue: without consent the drone cannot recognise you." if errors else "")
    s.field(24, 410, 440, "Consent given at", "2026-11-20 10:00" if not errors else "set when the box is ticked",
            kind="datetime", readonly=True, hint="Filled in automatically")
    # appearance template (optional)
    s.text(520, 150, 400, 20, "Appearance template", 12, bold=True)
    s.image(520, 172, 400, 210, "Camera preview")
    s.button(520, 392, 190, "Capture from camera")
    s.text(720, 392, 220, 36, "templates/operator_1.npz" if not errors else "optional", 11, color=GREY)
    s.line(0, 560, 960, 560)
    s.button(24, 580, 140, "Save", primary=True)
    s.button(176, 580, 120, "Cancel")
    return s


def home(ignored: bool = False) -> Screen:
    name = "03-home-command-ignored" if ignored else "03-home"
    s = Screen(name, "Live flight", user=USER)
    if ignored:  # command_event 8: flight 3, gesture stop, rejected, "hand belongs to another person"
        flight = ("Flight 3", "MuJoCo quad", "sim", "BRAKE", "1.80 m", "60 %", "not in view", "2026-11-24 10:14")
    else:  # telemetry_sample 2 of flight 1
        flight = ("Flight 1", "SITL copter", "sim", "GUIDED", "2.40 m", "81 %", "4.05 m", "2026-11-23 11:06")
    s.image(24, 128, 560, 330, f"Live view - {flight[1]} camera")
    s.text(24, 462, 560, 24, f"Operator locked: {USER}" + ("" if not ignored else "  -  another person in view"),
           11.5, color=GREY)
    # status card
    s.rect(604, 128, 332, 330, fill="#ffffff", stroke=LINE, rounded=True)
    s.text(620, 138, 200, 30, flight[0], 15, bold=True)
    s.chip(830, 142, "in flight", w=90)
    rows = [("Drone", flight[1]), ("Environment", flight[2]), ("Flight mode", flight[3]), ("Altitude", flight[4]),
            ("Battery", flight[5]), ("Distance to operator", flight[6]), ("Updated", flight[7])]
    for i, (k, v) in enumerate(rows):
        y = 178 + i * 38
        s.text(620, y, 160, 30, k, 12, color=GREY)
        s.text(780, y, 150, 30, v, 12.5, bold=True)
        s.line(620, y + 34, 920, y + 34, color=PAPER)
    if ignored:
        s.banner(24, 494, 912, 80, "Command ignored: the LAND fist was shown by another person.",
                 f"Only {USER} can command this drone. Step into the camera view alone, face the camera\n"
                 "and hold the fist for 1.5 s - or press Send command below and choose land.")
    else:
        s.text(24, 492, 300, 24, "Last commands", 12, bold=True)
        s.text(24, 516, 600, 22, "11:03:10  gesture  stop  - accepted", 11.5, color=INK)
        s.text(24, 538, 600, 22, "11:05:00  voice  go_higher  - accepted", 11.5, color=INK)
    s.line(0, 584, 960, 584)
    s.button(24, 594, 170, "Send command", primary=True)
    s.button(206, 594, 150, "Start flight")
    s.button(368, 594, 150, "Command log")
    s.button(530, 594, 150, "Flight reports")
    if ignored:
        s.button(816, 594, 120, "Dismiss", color=GREY)
    return s


def start_flight() -> Screen:
    s = Screen("04-start-flight", "Start flight", user=USER, back="Home")
    s.text(24, 116, 900, 24, "Creates a row in the flight table and links the selected models (flight_model).", 12,
           color=GREY)
    s.field(24, 150, 420, "Operator", USER, readonly=True, hint="The signed-in operator")
    s.field(24, 240, 420, "Drone", "SITL copter  (SIM-001)", kind="select", required=True)
    s.field(24, 330, 420, "Environment", "sim", kind="select", required=True, hint="sim or real")
    s.field(24, 420, 420, "Start time", "2026-11-23 11:00", kind="datetime", required=True,
            hint="Now by default")
    # open drone list to show the choices from the drone table
    s.rect(500, 150, 436, 150, fill="#ffffff", stroke=LINE, rounded=True)
    s.text(516, 156, 400, 26, "Drone list (from the drone table)", 12, bold=True)
    for i, d in enumerate(["SITL copter  -  ArduPilot SITL quad  -  SIM-001",
                           "MuJoCo quad  -  MuJoCo simulated quad  -  SIM-002",
                           "Falcon F450  -  F450 quadcopter  -  KZ-UAV-0450"]):
        s.text(516, 186 + i * 34, 410, 30, d, 11.5)
    s.text(500, 320, 436, 20, "Models used in this flight", 12, bold=True)
    for i, (m, on) in enumerate([("yolo11s-pose  (person_pose 11.0)", True),
                                 ("gesture_recognizer  (gesture 0.10.30)", True),
                                 ("vosk-small-en  (voice 0.15)", True), ("ppo-avoidance  (avoidance 2)", False)]):
        s.checkbox(500, 346 + i * 30, m, checked=on)
    s.line(0, 560, 960, 560)
    s.button(24, 580, 160, "Start flight", primary=True)
    s.button(196, 580, 120, "Cancel")
    return s


def send_command() -> Screen:
    s = Screen("05-send-command", "Send command", user="Timur Akhmetov", back="Home")  # command_event 9
    s.text(24, 116, 900, 24, "The dashboard path of the landing scenario: saved in command_event with "
           "source = dashboard, then checked by the command arbiter.", 12, color=GREY)
    s.field(24, 150, 420, "Flight", "Flight 4  -  Falcon F450  -  real", kind="select", required=True,
            hint="Flights of the signed-in operator")
    s.field(24, 240, 420, "Command", "land", kind="select", required=True,
            hint="land, stop, come_back, follow_me, go_higher, take_photo")
    s.field(24, 330, 420, "Reason", "dashboard button", required=True, hint="Up to 80 characters")
    s.field(24, 420, 200, "Source", "dashboard", readonly=True)
    s.field(244, 420, 200, "Time", "2026-11-24 17:03:30", kind="datetime", readonly=True)
    s.rect(500, 150, 436, 170, fill=PAPER, stroke=LINE, rounded=True)
    s.text(516, 160, 400, 26, "What happens next", 12.5, bold=True)
    s.text(516, 192, 410, 110,
           "1. The arbiter checks priority and flight mode.\n2. The command is saved as accepted or rejected\n"
           "    with its reason.\n3. LAND is sent to ArduPilot over MAVLink.\n4. The status on Home changes "
           "to Landing.", 12)
    s.line(0, 560, 960, 560)
    s.button(24, 580, 160, "Send LAND", primary=True)
    s.button(196, 580, 120, "Cancel")
    return s


COMMANDS = [  # command_event INSERT rows joined as in query Q1
    ("2026-11-23 11:03:10", "Aidar Serikov", "SITL copter", "gesture", "stop", "accepted", "operator gesture, held 1.0 s"),
    ("2026-11-23 11:05:00", "Aidar Serikov", "SITL copter", "voice", "go_higher", "accepted", "voice command from operator"),
    ("2026-11-23 11:11:30", "Aidar Serikov", "SITL copter", "gesture", "land", "accepted", "operator gesture, held 1.5 s"),
    ("2026-11-23 15:31:00", "Aruzhan Sadykova", "SITL copter", "voice", "follow_me", "accepted", "voice command from operator"),
    ("2026-11-23 15:36:20", "Aruzhan Sadykova", "SITL copter", "voice", "take_photo", "rejected", "confidence below 0.60"),
    ("2026-11-23 15:40:10", "Aruzhan Sadykova", "SITL copter", "voice", "land", "accepted", "voice command from operator"),
    ("2026-11-24 10:10:00", "Aidar Serikov", "MuJoCo quad", "gesture", "come_back", "accepted", "wave detected (3 swings)"),
    ("2026-11-24 10:14:00", "Aidar Serikov", "MuJoCo quad", "gesture", "stop", "rejected", "hand belongs to another person"),
    ("2026-11-24 17:03:30", "Timur Akhmetov", "Falcon F450", "dashboard", "land", "accepted", "dashboard button"),
]


def command_log() -> Screen:
    s = Screen("06-command-log", "Command log", user=USER, back="Home")
    s.field(24, 116, 180, "Decision", "All", kind="select")
    s.field(220, 116, 180, "Source", "All", kind="select")
    s.text(420, 138, 500, 34, "9 commands  -  7 accepted, 2 rejected", 12, color=GREY)
    s.table(24, 196, [("Time", 150), ("Operator", 138), ("Drone", 104), ("Source", 84), ("Command", 92),
                      ("Decision", 100), ("Reason", 244)], [list(r) for r in COMMANDS], row_h=34, chip_col=5)
    return s


def reports() -> Screen:
    s = Screen("07-flight-reports", "Flight reports", user=USER, back="Home")
    s.text(24, 116, 900, 24, "Flights (query Q3)", 13, bold=True)
    flights = [
        ["1", "Aidar Serikov", "sim", "12.0", "gesture_recognizer, vosk-small-en, yolo11s-pose", "67 %", "ended"],
        ["2", "Aruzhan Sadykova", "sim", "11.0", "vosk-small-en, yolo11s-pose", "72 %", "ended"],
        ["3", "Aidar Serikov", "sim", "15.0", "gesture_recognizer, ppo-avoidance, yolo11s-pose", "60 %", "ended"],
        ["4", "Timur Akhmetov", "real", "4.0", "gesture_recognizer, yolo11s-pose", "99 %", "ended"],
    ]
    s.table(24, 144, [("Flight", 64), ("Operator", 140), ("Env.", 60), ("Minutes", 76), ("Models used", 362),
                      ("Min battery", 100), ("Status", 110)], flights, row_h=34, chip_col=6)
    s.text(24, 330, 900, 24, "Commands by source (query Q2)", 13, bold=True)
    s.table(24, 358, [("Source", 140), ("Commands", 110), ("Accepted", 110), ("Accepted %", 120),
                      ("Avg latency, ms", 150)],
            [["voice", "4", "3", "75.0", "660"], ["gesture", "4", "3", "75.0", "1243"],
             ["dashboard", "1", "1", "100.0", "120"]], row_h=34)
    s.line(0, 560, 960, 560)
    s.button(24, 580, 160, "Export CSV", primary=True)
    s.button(196, 580, 120, "Back")
    return s


# ---------------------------------------------------------------- navigation
def navigation() -> Diagram:
    d = Diagram("navigation", 1300, 800, "Navigation between screens (R = requirement in README)")
    box = dict(w=240, h=90)
    d.node("s01", "process", 40, 200, label="01 Sign in", number="R1", **box)
    d.node("s02", "process", 40, 500, label="02 Register operator", number="R1", **box)
    d.node("s03", "process", 520, 60, 240, 630, "03 Home\n(live flight)", number="R4, R5, R6, R7")
    right = [("s04", "04 Start flight", "R3"), ("s05", "05 Send command", "R4, R5"),
             ("s06", "06 Command log", "R4, R7"), ("s07", "07 Flight reports", "R8")]
    for i, (sid, label, req) in enumerate(right):
        d.node(sid, "process", 1000, 60 + i * 180, label=label, number=req, **box)

    def fy(node: str, y: float) -> float:
        n = d.nodes[node]
        return round((y - n.y) / n.h, 4)

    # sign in <-> home, sign in <-> register
    d.edge("s01", (1, 0.3), "s03", (0, fy("s03", 227)), "Sign in button\n(email found)", label_at=(390, 205))
    d.edge("s03", (0, fy("s03", 263)), "s01", (1, 0.7), "Sign out link", label_at=(390, 280))
    d.edge("s01", (0.3, 1), "s02", (0.3, 0), "Register\nbutton", label_at=(112, 395))
    d.edge("s02", (0.7, 0), "s01", (0.7, 1), "Save (valid)\nor Cancel", label_at=(208, 395))
    d.edge("s02", (1, 0.3), "s02", (1, 0.75), "Save with an empty or\nwrong field: show errors",
           points=[(330, 527), (330, 567.5)], label_at=(420, 547))
    # home <-> screens on the right
    forward = ["Start flight button", "Send command button", "Command log button", "Flight reports button"]
    back = ["Start flight or Cancel", "Send LAND or Cancel", "← Home link", "Back button"]
    for i, (sid, _, _) in enumerate(right):
        cy = 105 + i * 180
        d.edge("s03", (1, fy("s03", cy - 18)), sid, (0, 0.3), forward[i], label_at=(880, cy - 36))
        d.edge(sid, (0, 0.7), "s03", (1, fy("s03", cy + 18)), back[i], label_at=(880, cy + 36))
    # home: rejected gesture shows the error banner
    d.edge("s03", (0.35, 1), "s03", (0.65, 1), "Gesture from another person:\n\"Command ignored\" banner",
           points=[(604, 735), (676, 735)], label_at=(640, 768))
    return d


def main() -> None:
    for screen in [sign_in(), register(), register(errors=True), home(), home(ignored=True), start_flight(),
                   send_command(), command_log(), reports()]:
        screen.save(SCREENS)
        print("wrote", (SCREENS / screen.name).relative_to(ROOT), "(.drawio + .png)")
    nav = navigation()
    nav.save(UI, nav.name)
    print("wrote", (UI / nav.name).relative_to(ROOT), "(.drawio + .png)")


if __name__ == "__main__":
    main()
