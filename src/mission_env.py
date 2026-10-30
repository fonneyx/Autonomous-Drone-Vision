"""Target mission: find a person, vehicle or drone and fly up to it without crashing.

Builds on the obstacle course in sim_env.py and adds the people, vehicles and
drones modelled in Blender (assets/sim_models, made by
scripts/blender/build_sim_models.py). Each episode one object of the requested
class is the target; objects of the other classes are decoys and obstacles.

The pilot gets two kinds of vision:
* the depth camera (as in the avoidance task), for not hitting things;
* a detection of the target in the colour camera image, i.e. the box YOLO
  would output: [found, centre x, centre y, width, height].

During RL training the detection comes from a fast simulated detector (the
target's true box, with misses and jitter like a real detector). When testing,
a real YOLO model looks at the rendered camera image instead
(detector="external", see scripts/fly_mission.py).
"""

from __future__ import annotations

import colorsys
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .sim_env import (
    COLLISION_PENALTY,
    COURSE_LENGTH,
    DANGER_DISTANCE,
    DANGER_WEIGHT,
    DEPTH_HEIGHT,
    DEPTH_WIDTH,
    HALF_WIDTH,
    HIDDEN_Z,
    HORIZONTAL_FOV,
    SENSOR_GROUPS,
    SUCCESS_REWARD,
    TIME_PENALTY,
    TURN_PENALTY,
    DroneAvoidanceEnv,
)

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets" / "sim_models"
CLASSES = ("person", "vehicle", "drone")
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480  # colour camera resolution
TAN_H = math.tan(HORIZONTAL_FOV / 2.0)
TAN_V = TAN_H * DEPTH_HEIGHT / DEPTH_WIDTH

# Size variants built into the pool: one instance per (variant, scale)
SCALES = {
    "person": (0.9, 1.0, 1.1),
    "vehicle": (0.95, 1.05),
    "drone_quad": (1.0, 1.5, 2.0),
    "drone_hexa": (0.8, 1.0, 1.2),
    "drone_fpv": (1.0, 1.6, 2.2),
}

REACH_DISTANCE = 2.0  # success: this close to the target (metres) while seeing it
SEEN_BONUS = 0.02  # small reward per step with the target in view
MIN_DETECTABLE_PX = 10  # objects smaller than this in the image are not detected
TARGET_VECTOR_SIZE = 7 + len(CLASSES)


@dataclass
class Instance:
    name: str
    class_id: int
    body: int
    mocap: int
    proxy_geom: int
    mesh_geoms: np.ndarray
    role_geoms: dict[str, list[int]]
    centre: np.ndarray  # local bounding-box centre
    half: np.ndarray  # local bounding-box half extents
    radius: float  # footprint radius on the ground
    surface_points: list[tuple[int, np.ndarray]]  # (geom, mesh vertices) for tight image boxes


def _rgb(h: float, s: float, v: float) -> list[float]:
    return [*colorsys.hsv_to_rgb(h % 1.0, s, v), 1.0]


class DroneMissionEnv(DroneAvoidanceEnv):
    """Find the target object (person / vehicle / drone) and fly within 2 m of it.

    Observation (dict): depth and state as in DroneAvoidanceEnv, plus
        target: float32 (10,) [found, centre x, centre y, width, height,
                 recently seen (decays 1 -> 0), last seen centre x,
                 one-hot target class (person, vehicle, drone)]
    Action: as in DroneAvoidanceEnv (forward speed, yaw rate).
    """

    def __init__(
        self,
        min_obstacles: int = 8,
        max_obstacles: int = 30,
        max_steps: int = 600,
        target_class: str | None = None,
        detector: str = "simulated",
        **kwargs: Any,
    ):
        if target_class is not None and target_class not in CLASSES:
            raise ValueError(f"target_class must be one of {CLASSES} or None.")
        if detector not in ("simulated", "external"):
            raise ValueError("detector must be 'simulated' or 'external'.")
        self.target_class = target_class
        self.detector = detector
        manifest_path = ASSET_DIR / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"{manifest_path} not found. Build the models: open Blender and run "
                "scripts/blender/build_sim_models.py"
            )
        self._manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        super().__init__(min_obstacles=min_obstacles, max_obstacles=max_obstacles,
                         max_steps=max_steps, **kwargs)

        self._instances = self._index_instances()
        self._geom_instance = np.full(self.model.ngeom, -1, dtype=np.int32)
        for index, instance in enumerate(self._instances):
            self._geom_instance[instance.mesh_geoms] = index
        self._ground_material = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "ground")
        self._default_ground_rgba = self.model.mat_rgba[self._ground_material].copy()

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces,
             "target": spaces.Box(-1.0, 1.0, (TARGET_VECTOR_SIZE,), dtype=np.float32)}
        )
        self._target: Instance | None = None
        self._placed: list[Instance] = []
        self._ray_hit = np.zeros(1, dtype=np.int32)
        self._reset_target_memory()

    # ---------------------------------------------------------------- building
    def _scales(self, variant: str, object_class: str) -> tuple[float, ...]:
        return SCALES.get(variant, SCALES.get(object_class, (1.0,)))

    def _extra_world_xml(self, rng: np.random.Generator) -> tuple[str, str]:
        assets, bodies = [], []
        slot = 0
        for variant, info in self._manifest.items():
            lo, hi = np.array(info["bounds"][0]), np.array(info["bounds"][1])
            for k, scale in enumerate(self._scales(variant, info["class"])):
                name = f"{variant}_{k}"
                geoms = []
                for part in info["parts"]:
                    mesh = f"{name}_{part['role']}"
                    path = (ASSET_DIR / part["file"]).as_posix()
                    assets.append(
                        f'<mesh name="{mesh}" file="{path}" scale="{scale} {scale} {scale}" inertia="shell"/>'
                    )
                    rgba = " ".join(f"{c:.3f}" for c in part["rgba"])
                    geoms.append(
                        f'<geom name="{mesh}" type="mesh" mesh="{mesh}" group="2" contype="0" '
                        f'conaffinity="0" density="0" rgba="{rgba}"/>'
                    )
                centre = (lo + hi) / 2 * scale
                half = (hi - lo) / 2 * scale
                if info["class"] == "person":
                    radius = min(0.25, max(half[0], half[1]))
                    proxy = (
                        f'<geom name="{name}_proxy" type="capsule" group="3" rgba="0 1 0 .3" '
                        f'fromto="{centre[0]:.3f} {centre[1]:.3f} {lo[2] * scale + radius:.3f} '
                        f'{centre[0]:.3f} {centre[1]:.3f} {hi[2] * scale - radius:.3f}" size="{radius:.3f}"/>'
                    )
                else:
                    proxy = (
                        f'<geom name="{name}_proxy" type="box" group="3" rgba="0 1 0 .3" '
                        f'pos="{centre[0]:.3f} {centre[1]:.3f} {centre[2]:.3f}" '
                        f'size="{half[0]:.3f} {half[1]:.3f} {half[2]:.3f}"/>'
                    )
                bodies.append(
                    f'<body name="{name}" mocap="true" pos="-300 {slot * 5} {HIDDEN_Z}">'
                    f'<inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>{proxy}{"".join(geoms)}</body>'
                )
                slot += 1
        return "\n".join(assets), "\n".join(bodies)

    def _index_instances(self) -> list[Instance]:
        model = self.model
        instances = []
        for variant, info in self._manifest.items():
            lo, hi = np.array(info["bounds"][0]), np.array(info["bounds"][1])
            for k, scale in enumerate(self._scales(variant, info["class"])):
                name = f"{variant}_{k}"
                body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                role_geoms = {
                    part["role"]: [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_{part['role']}")]
                    for part in info["parts"]
                }
                half = (hi - lo) / 2 * scale
                surface_points = []
                for geom in (g for ids in role_geoms.values() for g in ids):
                    mesh = model.geom_dataid[geom]
                    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                    stride = max(1, count // 80)
                    surface_points.append((geom, model.mesh_vert[start:start + count:stride].copy()))
                instances.append(
                    Instance(
                        name=name,
                        class_id=CLASSES.index(info["class"]),
                        body=body,
                        mocap=int(model.body_mocapid[body]),
                        proxy_geom=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_proxy"),
                        mesh_geoms=np.array([g for ids in role_geoms.values() for g in ids]),
                        role_geoms=role_geoms,
                        centre=(lo + hi) / 2 * scale,
                        half=half,
                        radius=float(math.hypot(half[0], half[1])),
                        surface_points=surface_points,
                    )
                )
        return instances

    # --------------------------------------------------------------- placement
    def _hide_instances(self) -> None:
        for index, instance in enumerate(self._instances):
            self.data.mocap_pos[instance.mocap] = [-300.0, index * 5.0, HIDDEN_Z]
        self._placed = []

    def _place_instance(self, instance: Instance, x: float, y: float, yaw: float) -> None:
        z = 0.0
        if CLASSES[instance.class_id] == "drone":
            z = self.np_random.uniform(1.0, 2.6)  # hovering
        self.data.mocap_pos[instance.mocap] = [x, y, z]
        self.data.mocap_quat[instance.mocap] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        self._placed.append(instance)

    def _free_spot(self, placed, radius, x_range, gap=1.0):
        # Whole object stays inside the walls
        y_limit = max(0.0, HALF_WIDTH - radius - 0.2)
        for _ in range(40):
            x = self.np_random.uniform(*x_range)
            y = self.np_random.uniform(-y_limit, y_limit)
            if all(math.hypot(x - ox, y - oy) > radius + r + gap for ox, oy, r in placed):
                return x, y
        return None

    def _random_colour(self, role: str) -> list[float]:
        rng = self.np_random
        if role == "paint":
            palette = [(0.92, 0.92, 0.92), (0.04, 0.04, 0.05), (0.62, 0.63, 0.66), (0.33, 0.34, 0.36),
                       (0.62, 0.05, 0.05), (0.08, 0.18, 0.52), (0.05, 0.08, 0.2), (0.1, 0.3, 0.15),
                       (0.85, 0.7, 0.1), (0.85, 0.35, 0.05), (0.55, 0.47, 0.35)]
            base = np.array(palette[rng.integers(len(palette))])
            return [*np.clip(base + rng.normal(0, 0.03, 3), 0, 1), 1.0]
        if role == "skin":
            t = rng.random()
            light, dark = np.array([0.95, 0.78, 0.66]), np.array([0.3, 0.19, 0.12])
            return [*(light * (1 - t) + dark * t), 1.0]
        if role == "shirt":
            return _rgb(rng.random(), rng.uniform(0.1, 0.9), rng.uniform(0.15, 0.95))
        if role == "pants":
            options = [_rgb(0.6, 0.6, rng.uniform(0.25, 0.55)), _rgb(0, 0, rng.uniform(0.05, 0.4)),
                       _rgb(0.1, 0.35, rng.uniform(0.4, 0.7)), _rgb(rng.random(), 0.4, rng.uniform(0.15, 0.5))]
            return options[rng.integers(len(options))]
        if role == "hair":
            options = [(0.03, 0.02, 0.02), (0.15, 0.09, 0.05), (0.35, 0.22, 0.1), (0.8, 0.65, 0.35),
                       (0.6, 0.6, 0.6), (0.45, 0.15, 0.05)]
            return [*options[rng.integers(len(options))], 1.0]
        if role == "shoes":
            return _rgb(rng.random(), rng.uniform(0, 0.3), rng.choice([0.08, 0.3, 0.9]))
        if role == "body":
            options = [(0.9, 0.9, 0.92), (0.75, 0.76, 0.78), (0.25, 0.25, 0.27), (0.06, 0.06, 0.07),
                       (0.7, 0.1, 0.08), (0.9, 0.65, 0.1), (0.9, 0.4, 0.05)]
            return [*options[rng.integers(len(options))], 1.0]
        if role == "props":
            options = [(0.08, 0.08, 0.09), (0.5, 0.5, 0.52), (0.9, 0.9, 0.9), (0.9, 0.4, 0.05)]
            return [*options[rng.integers(len(options))], 1.0]
        if role == "frame":
            return _rgb(0, 0, rng.uniform(0.05, 0.3))
        return None  # fixed colour (glass, tyres, lights, ...)

    def _recolour(self, instances) -> None:
        for instance in instances:
            for role, geoms in instance.role_geoms.items():
                colour = self._random_colour(role)
                if colour is not None:
                    self.model.geom_rgba[geoms] = colour

    def _place_extra_objects(self) -> list[tuple[float, float, float]]:
        self._hide_instances()
        rng = self.np_random
        target_class = self._options.get("target", self.target_class)
        if target_class is None:
            target_class = CLASSES[rng.integers(len(CLASSES))]
        class_id = CLASSES.index(target_class)
        candidates = [inst for inst in self._instances if inst.class_id == class_id]
        self._target = candidates[rng.integers(len(candidates))]

        x, y = self._free_spot([], self._target.radius + 1.0, (18.0, COURSE_LENGTH - 3.0))
        self._place_instance(self._target, x, y, rng.uniform(-math.pi, math.pi))
        # Keep the target's surroundings clear so it can be reached
        reserved = [(x, y, self._target.radius + 2.5)]

        decoys = [inst for inst in self._instances if inst.class_id != class_id]
        for instance in rng.permutation(len(decoys))[: rng.integers(2, 6)]:
            decoy = decoys[instance]
            spot = self._free_spot(reserved, decoy.radius, (6.0, COURSE_LENGTH - 1.0))
            if spot is None:
                continue
            self._place_instance(decoy, *spot, rng.uniform(-math.pi, math.pi))
            reserved.append((*spot, decoy.radius))

        self._recolour(self._placed)
        self._ignored_ray_geoms = np.array([self._floor_geom, self._target.proxy_geom])
        return reserved

    # --------------------------------------------------------------- detection
    def target_box(self, instance: Instance | None = None) -> np.ndarray | None:
        """True box of an object in the colour camera: [cx, cy, w, h] or None if unseen.

        cx, cy in [-1, 1] (image left/top = -1); w, h as a fraction of image size.
        """
        instance = instance or self._target
        origin, rotation = self.camera_pose()
        position = self.data.xpos[instance.body]
        orientation = self.data.xmat[instance.body].reshape(3, 3)
        # Project the model's surface points, giving a tight box like a hand-drawn label
        surface = np.concatenate([
            self.data.geom_xpos[geom] + vertices @ self.data.geom_xmat[geom].reshape(3, 3).T
            for geom, vertices in instance.surface_points
        ])
        relative = (surface - origin) @ rotation  # columns: forward, left, up
        forward = relative[:, 0]
        if forward.max() < 0.3:
            return None
        forward = np.maximum(forward, 0.05)
        u = -relative[:, 1] / forward / TAN_H
        v = -relative[:, 2] / forward / TAN_V
        if u.max() <= -1 or u.min() >= 1 or v.max() <= -1 or v.min() >= 1:
            return None
        u0, u1 = max(u.min(), -1.0), min(u.max(), 1.0)
        v0, v1 = max(v.min(), -1.0), min(v.max(), 1.0)
        width, height = (u1 - u0) / 2, (v1 - v0) / 2
        if max(width * IMAGE_WIDTH, height * IMAGE_HEIGHT) < MIN_DETECTABLE_PX:
            return None

        # Occlusion: is any part of the object in line of sight?
        offsets = np.array([[0, 0, 0], [0, 0.4, 0.4], [0, -0.4, 0.4], [0, 0.4, -0.4], [0, -0.4, -0.4],
                            [0.4, 0, 0], [-0.4, 0, 0]])
        points = position + (instance.centre + offsets * instance.half) @ orientation.T
        for point in points:
            mujoco.mj_ray(self.model, self.data, origin, point - origin, SENSOR_GROUPS, True,
                          self._drone_body, self._ray_hit)
            if self._ray_hit[0] == instance.proxy_geom:
                return np.array([(u0 + u1) / 2, (v0 + v1) / 2, width, height])
        return None

    def _simulated_detection(self, true_box: np.ndarray | None) -> np.ndarray | None:
        """A realistic detector: misses small objects more often, jitters boxes."""
        if true_box is None:
            return None
        rng = self.np_random
        height_px = true_box[3] * IMAGE_HEIGHT
        if rng.random() < 0.05 + 0.3 * max(0.0, 1.0 - height_px / 40.0):
            return None
        box = true_box.copy()
        box[:2] += rng.normal(0.0, 0.015, 2) + rng.normal(0.0, 0.1, 2) * box[2:]
        box[2:] *= np.exp(rng.normal(0.0, 0.08, 2))
        return np.clip(box, [-1, -1, 0, 0], [1, 1, 1, 1])

    def _reset_target_memory(self) -> None:
        self._detection: np.ndarray | None = None
        self._steps_since_seen = 10_000
        self._last_seen_x = 0.0

    def set_detection(self, box: np.ndarray | None) -> None:
        """Feed a detection [cx, cy, w, h] (or None) from an external detector such as YOLO."""
        self._detection = None if box is None else np.asarray(box, dtype=np.float64)
        if self._detection is not None:
            self._steps_since_seen = 0
            self._last_seen_x = float(self._detection[0])
        else:
            self._steps_since_seen += 1

    def _target_vector(self) -> np.ndarray:
        vector = np.zeros(TARGET_VECTOR_SIZE, dtype=np.float32)
        if self._detection is not None:
            vector[0] = 1.0
            vector[1:5] = self._detection
        if self._steps_since_seen < 10_000:
            vector[5] = math.exp(-self._steps_since_seen / 10.0)
            vector[6] = self._last_seen_x
        vector[7 + self._target.class_id] = 1.0
        return vector

    def observe(self) -> dict[str, np.ndarray]:
        """Current observation (use after set_detection in external-detector mode)."""
        return self._observation()

    def _observation(self) -> dict[str, np.ndarray]:
        observation = super()._observation()
        observation["target"] = self._target_vector()
        return observation

    # -------------------------------------------------------------- task logic
    def target_distance(self) -> float:
        """Distance from the drone to the target's bounding box (metres)."""
        position = self.data.xpos[self._target.body]
        orientation = self.data.xmat[self._target.body].reshape(3, 3)
        local = orientation.T @ (self.position - position) - self._target.centre
        return float(np.linalg.norm(np.maximum(np.abs(local) - self._target.half, 0.0)))

    def _after_reset(self) -> None:
        self._reset_target_memory()
        self._previous_distance = self.target_distance()
        if self.detector == "simulated":
            self.set_detection(self._simulated_detection(self.target_box()))

    def _evaluate(self, action: np.ndarray, collided: bool) -> tuple[float, bool, dict[str, Any]]:
        true_box = self.target_box()
        if self.detector == "simulated":
            self.set_detection(self._simulated_detection(true_box))

        distance = self.target_distance()
        reward = self._previous_distance - distance
        self._previous_distance = distance
        nearest = self.nearest_obstacle_distance()
        reward -= DANGER_WEIGHT * max(0.0, (DANGER_DISTANCE - nearest) / DANGER_DISTANCE)
        reward -= TURN_PENALTY * abs(action[1]) + TIME_PENALTY
        if true_box is not None:
            reward += SEEN_BONUS

        reached = distance < REACH_DISTANCE and true_box is not None
        if collided:
            reward += COLLISION_PENALTY
        elif reached:
            reward += SUCCESS_REWARD
        info = {
            "is_success": bool(reached and not collided),
            "collision": collided,
            "target_class": CLASSES[self._target.class_id],
            "target_distance": distance,
            "target_visible": true_box is not None,
            "nearest_obstacle": nearest,
            "progress": float(self.position[0] / COURSE_LENGTH),
        }
        return reward, collided or reached, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        observation, info = super().reset(seed=seed, options=options)
        info["target_class"] = CLASSES[self._target.class_id]
        return observation, info

    # ------------------------------------------------------- synthetic dataset
    def randomize_for_dataset(self) -> list[Instance]:
        """Random scene and camera pose for rendering labelled training images."""
        rng = self.np_random
        mujoco.mj_resetData(self.model, self.data)
        self._hide_instances()
        x0, y0 = rng.uniform(-1.0, 22.0), rng.uniform(-6.0, 6.0)
        z0, yaw0 = rng.uniform(1.0, 3.0), rng.uniform(-0.6, 0.6)
        reserved = [(x0, y0, 1.0)]
        unused = list(rng.permutation(len(self._instances)))
        for _ in range(rng.integers(1, 7)):
            instance = self._instances[unused.pop()]
            distance = 3.0 + 25.0 * rng.random() ** 1.5
            bearing = yaw0 + rng.uniform(-0.75, 0.75)
            x, y = x0 + distance * math.cos(bearing), y0 + distance * math.sin(bearing)
            if abs(y) + instance.radius > HALF_WIDTH - 0.2 or x > COURSE_LENGTH + 4:
                continue
            if any(math.hypot(x - ox, y - oy) < instance.radius + r + 0.5 for ox, oy, r in reserved):
                continue
            self._place_instance(instance, x, y, rng.uniform(-math.pi, math.pi))
            reserved.append((x, y, instance.radius))
        self._place_obstacles(int(rng.integers(0, 35)), reserved)
        self._recolour(self._placed)

        # Lighting and ground variety
        elevation, azimuth = rng.uniform(0.4, 1.4), rng.uniform(0, 2 * math.pi)
        self.model.light_dir[0] = [-math.cos(elevation) * math.cos(azimuth),
                                   -math.cos(elevation) * math.sin(azimuth), -math.sin(elevation)]
        self.model.light_diffuse[0] = [rng.uniform(0.35, 0.9)] * 3
        self.model.vis.headlight.ambient[:] = [rng.uniform(0.2, 0.45)] * 3
        tint = rng.uniform(0.6, 1.4, 3) * [1.0, 1.0, rng.uniform(0.6, 1.2)]
        if rng.random() < 0.3:
            tint = [rng.uniform(0.6, 1.3)] * 3  # grey, like asphalt or concrete
        self.model.mat_rgba[self._ground_material] = [*np.clip(self._default_ground_rgba[:3] * tint, 0, 1), 1]

        self.data.qpos[:4] = [x0, y0, z0, yaw0]
        mujoco.mj_forward(self.model, self.data)
        return list(self._placed)

    def boxes_from_segmentation(self, segmentation: np.ndarray, min_pixels: int = 30) -> list[tuple[int, float, float, float, float]]:
        """YOLO labels (class, cx, cy, w, h normalised 0..1) from a segmentation render."""
        geom_ids = np.where(segmentation[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, segmentation[..., 0], -1)
        instance_map = np.where(geom_ids >= 0, self._geom_instance[np.clip(geom_ids, 0, None)], -1)
        height, width = instance_map.shape
        labels = []
        for instance in self._placed:
            index = self._instances.index(instance)
            ys, xs = np.nonzero(instance_map == index)
            if len(xs) < min_pixels:
                continue
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            labels.append((instance.class_id, (x0 + x1) / 2 / width, (y0 + y1) / 2 / height,
                           (x1 - x0) / width, (y1 - y0) / height))
        return labels


gym.register(id="DroneMission-v0", entry_point=DroneMissionEnv)
