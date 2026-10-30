"""Draw the live map with the drone, its flown path, the explorer's plan and people markers.

Colours: grey = unknown, white = free, black = wall, orange = frontier, blue = flown
path, red arrow = drone, green ring = home, magenta = planned path / waypoint /
target, red cross = confirmed person marker, small grey ring = true person position
(simulation only, for checking).
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from src.sim_env import CONTROL_DT

MAP_SCALE = 3  # pixels per map cell
PLAN_BGR = (200, 40, 200)
MARKER_BGR = (40, 40, 230)
TRUTH_BGR = (150, 150, 150)


def draw_map(
    env,
    info: dict,
    explorer=None,
    markers: Sequence[np.ndarray] = (),
    show_truth: bool = False,
    status: str = "",
) -> np.ndarray:
    grid = env.map
    image = grid.to_bgr(MAP_SCALE)

    def pixel(x: float, y: float) -> tuple[int, int]:
        return grid.world_to_pixel(x, y, MAP_SCALE)

    points = [pixel(x, y) for x, y in env.trail]
    if len(points) > 1:
        cv2.polylines(image, [np.array(points, dtype=np.int32)], False, (200, 120, 30), 1, cv2.LINE_AA)
    if explorer is not None and explorer.path:
        plan = [pixel(x, y) for x, y in explorer.path]
        cv2.polylines(image, [np.array(plan, dtype=np.int32)], False, PLAN_BGR, 1, cv2.LINE_AA)
        cv2.circle(image, plan[-1], 7, PLAN_BGR, 2, cv2.LINE_AA)
        if explorer.waypoint is not None:
            cv2.circle(image, pixel(*explorer.waypoint), 4, PLAN_BGR, -1, cv2.LINE_AA)
    cv2.circle(image, pixel(*env.layout.home_xy), 6, (60, 180, 60), 2)
    if show_truth:
        for x, y in env.people_xy:
            cv2.circle(image, pixel(x, y), 5, TRUTH_BGR, 1, cv2.LINE_AA)
    for x, y in markers:
        cv2.drawMarker(image, pixel(x, y), MARKER_BGR, cv2.MARKER_CROSS, 14, 3, cv2.LINE_AA)

    x, y = env.position[:2]
    nose = (x + 0.8 * np.cos(env.yaw), y + 0.8 * np.sin(env.yaw))
    cv2.arrowedLine(image, pixel(x, y), pixel(*nose), (30, 30, 230), 2, cv2.LINE_AA, tipLength=0.5)

    text = f"coverage {info.get('coverage', 0.0) * 100:4.1f}%   t {env._step_count * CONTROL_DT:5.1f}s"
    if status:
        text += f"   {status}"
    width = 8 + cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] + 8
    cv2.rectangle(image, (0, 0), (width, 28), (30, 30, 30), -1)
    cv2.putText(image, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return image
