"""People to rescue in the cave: MuJoCo models built from the Blender person meshes.

Each person slot is a mocap body with the detailed meshes (rendered only, group 2)
and a capsule proxy (group 3) that the depth camera sees and the drone can hit, so
a person is also an obstacle in the map, as in reality. Poses: standing, walking,
waving. Colours (skin, clothes, hair) are re-randomised every reset.

Placement keeps people in open space (not inside walls), far enough from home that
finding them takes exploring, and apart from each other.
"""

from __future__ import annotations

import colorsys
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from src.cave_world import CaveLayout

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets" / "sim_models"
POSES = ("person_stand", "person_walk", "person_wave")
PERSON_RADIUS = 0.25  # proxy capsule radius (m)
HIDDEN_Z = -50.0


@dataclass
class PersonSlot:
    body: int
    mocap: int
    proxy_geom: int
    role_geoms: dict[str, int]
    height: float


def people_xml(count: int, rng: np.random.Generator) -> tuple[str, str]:
    """(assets, bodies) XML for `count` person slots with random pose and size."""
    manifest = json.loads((ASSET_DIR / "manifest.json").read_text(encoding="utf-8"))
    assets, bodies = [], []
    for index in range(count):
        pose = POSES[index % len(POSES)]
        info = manifest[pose]
        scale = rng.uniform(0.9, 1.1)
        lo, hi = np.array(info["bounds"][0]) * scale, np.array(info["bounds"][1]) * scale
        geoms = []
        for part in info["parts"]:
            mesh = f"person{index}_{part['role']}"
            path = (ASSET_DIR / part["file"]).as_posix()
            assets.append(f'<mesh name="{mesh}" file="{path}" scale="{scale:.3f} {scale:.3f} {scale:.3f}"/>')
            geoms.append(
                f'<geom name="{mesh}" type="mesh" mesh="{mesh}" group="2" contype="0" conaffinity="0" '
                f'density="0" rgba="{" ".join(f"{c:.3f}" for c in part["rgba"])}"/>'
            )
        cx, cy = (lo[:2] + hi[:2]) / 2
        proxy = (
            f'<geom name="person{index}_proxy" type="capsule" group="3" rgba="0 1 0 .3" '
            f'fromto="{cx:.3f} {cy:.3f} {lo[2] + PERSON_RADIUS:.3f} {cx:.3f} {cy:.3f} {hi[2] - PERSON_RADIUS:.3f}" '
            f'size="{PERSON_RADIUS}"/>'
        )
        bodies.append(
            f'<body name="person{index}" mocap="true" pos="{-100 - 3 * index} 0 {HIDDEN_Z}">'
            f'<inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>{proxy}{"".join(geoms)}</body>'
        )
    return "\n    ".join(assets), "\n    ".join(bodies)


def index_people(model: mujoco.MjModel, count: int) -> list[PersonSlot]:
    slots = []
    for index in range(count):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"person{index}")
        role_geoms = {
            role: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"person{index}_{role}")
            for role in ("hair", "pants", "shirt", "shoes", "skin")
        }
        proxy = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"person{index}_proxy")
        top = model.geom_pos[proxy][2] + model.geom_size[proxy][1] + PERSON_RADIUS
        slots.append(PersonSlot(body, int(model.body_mocapid[body]), proxy, role_geoms, float(top)))
    return slots


def _colour(role: str, rng: np.random.Generator) -> list[float]:
    if role == "skin":
        t = rng.random()
        return [*(np.array([0.95, 0.78, 0.66]) * (1 - t) + np.array([0.3, 0.19, 0.12]) * t), 1.0]
    if role == "hair":
        options = [(0.03, 0.02, 0.02), (0.15, 0.09, 0.05), (0.35, 0.22, 0.1), (0.8, 0.65, 0.35), (0.6, 0.6, 0.6)]
        return [*options[rng.integers(len(options))], 1.0]
    if role == "pants":
        return [*colorsys.hsv_to_rgb(rng.random(), rng.uniform(0.0, 0.5), rng.uniform(0.1, 0.5)), 1.0]
    # shirt and shoes: anything, including high-visibility work clothes
    return [*colorsys.hsv_to_rgb(rng.random(), rng.uniform(0.1, 0.95), rng.uniform(0.2, 0.95)), 1.0]


def choose_spots(
    layout: CaveLayout,
    count: int,
    rng: np.random.Generator,
    min_from_home: float = 8.0,
    min_apart: float = 4.0,
) -> np.ndarray:
    """World (x, y) for each person: open cells (3x3 all free), path distance from home and spacing."""
    free = layout.free
    open_space = free.copy()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            open_space &= np.roll(np.roll(free, dx, 0), dy, 1)
    # Path distance from home over the layout (BFS, cells)
    distance = np.full(free.shape, np.inf)
    distance[layout.home] = 0
    queue = deque([layout.home])
    while queue:
        x, y = queue.popleft()
        for nx_, ny_ in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if free[nx_, ny_] and np.isinf(distance[nx_, ny_]):
                distance[nx_, ny_] = distance[x, y] + 1
                queue.append((nx_, ny_))
    cells = np.argwhere(open_space & (distance * layout.cell_size >= min_from_home))
    if len(cells) == 0:  # tiny cave: relax the distance rule
        cells = np.argwhere(open_space)
    spots: list[np.ndarray] = []
    for cell in rng.permutation(cells):
        xy = (cell + 0.5 + rng.uniform(-0.3, 0.3, 2)) * layout.cell_size
        if all(np.linalg.norm(xy - other) >= min_apart for other in spots):
            spots.append(xy)
        if len(spots) == count:
            break
    return np.array(spots).reshape(-1, 2)


def place_people(
    data: mujoco.MjData, model: mujoco.MjModel, slots: list[PersonSlot], spots: np.ndarray, rng: np.random.Generator
) -> None:
    """Move the first len(spots) people to their spots (random facing and colours); hide the rest."""
    for index, slot in enumerate(slots):
        if index < len(spots):
            yaw = rng.uniform(-math.pi, math.pi)
            data.mocap_pos[slot.mocap] = [spots[index][0], spots[index][1], 0.0]
            data.mocap_quat[slot.mocap] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
            for role, geom in slot.role_geoms.items():
                model.geom_rgba[geom] = _colour(role, rng)
        else:
            data.mocap_pos[slot.mocap] = [-100.0 - 3 * index, 0.0, HIDDEN_Z]
