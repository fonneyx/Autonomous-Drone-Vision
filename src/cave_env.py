"""Explore an unknown cave in MuJoCo and build a 2D map while flying.

Same drone, flight model, depth camera and action interface as DroneAvoidanceEnv
(forward speed + yaw rate setpoints), so the trained avoidance pilot and the
classical baseline both plug in unchanged. The world is a procedural tunnel
network (src/cave_world.py) with a hidden ceiling, dark rock and a headlamp.

Every step, the two middle rows of the depth image are used as a horizontal
range scan (like a 2D lidar) and fused into an OccupancyMap. The reward is new
area mapped, so this env can train an exploring policy directly; Stage 2 adds
frontier waypoints through the `goal` attribute.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from src.cave_people import choose_spots, index_people, people_xml, place_people
from src.cave_world import TUNNEL_HEIGHT, CaveLayout, generate_cave
from src.occupancy_map import OccupancyMap
from src.sim_env import (
    CONTROL_DT,
    DEPTH_HEIGHT,
    DEPTH_RANGE,
    DEPTH_WIDTH,
    FLIGHT_ALTITUDE,
    MAX_SPEED,
    MAX_YAW_RATE,
    PHYSICS_DT,
    DroneAvoidanceEnv,
    _camera_ray_directions,
    drone_xml,
)

SCAN_ROWS = (DEPTH_HEIGHT // 2 - 1, DEPTH_HEIGHT // 2)  # middle rows: almost horizontal rays
GOAL_SCALE = 40.0  # metres; goal distance in the observation is divided by this

AREA_REWARD = 0.1  # per square metre newly mapped as free
COLLISION_PENALTY = -20.0
TIME_PENALTY = 0.005
DONE_COVERAGE = 0.9  # episode ends once this share of the reachable cave is mapped


def cave_xml(layout: CaveLayout, extra_assets: str = "", extra_bodies: str = "") -> str:
    """MuJoCo scene: floor, rock wall boxes, hidden ceiling and the drone with a headlamp.

    extra_assets / extra_bodies are inserted into <asset> / <worldbody> (people).
    """
    size_x, size_y = layout.size_m
    half_z = TUNNEL_HEIGHT / 2.0
    walls = "".join(
        f'<geom type="box" pos="{cx:.3f} {cy:.3f} {half_z:.3f}" size="{hx:.3f} {hy:.3f} {half_z:.3f}" '
        f'material="rock"/>'
        for cx, cy, hx, hy in layout.wall_boxes()
    )
    home_x, home_y = layout.home_xy
    headlamp = (
        '<light name="headlamp" pos="0.25 0 0" dir="1 0 -0.15" cutoff="55" exponent="4" '
        'diffuse="1 .95 .85" specular=".2 .2 .2" attenuation="1 0.05 0.01" castshadow="false"/>'
    )
    drone_body, actuators = drone_xml(headlamp)
    return f"""
<mujoco model="drone_cave">
  <option timestep="{PHYSICS_DT}" gravity="0 0 0" integrator="implicitfast"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight ambient=".32 .32 .34" diffuse=".3 .3 .3" specular="0 0 0"/>
    <map znear="0.003"/>
  </visual>
  <asset>
    <texture name="rock_tex" type="2d" builtin="flat" mark="random" random="0.25"
             rgb1=".34 .30 .26" markrgb=".48 .43 .37" width="256" height="256"/>
    <texture name="gravel_tex" type="2d" builtin="flat" mark="random" random="0.4"
             rgb1=".20 .19 .17" markrgb=".32 .30 .27" width="512" height="512"/>
    <material name="rock" texture="rock_tex" texuniform="true" texrepeat="1 1"/>
    <material name="gravel" texture="gravel_tex" texrepeat="{size_x / 2:.0f} {size_y / 2:.0f}"/>
    {extra_assets}
  </asset>
  <default>
    <geom contype="2" conaffinity="1" condim="1"/>
    <default class="visual">
      <geom contype="0" conaffinity="0" group="1" density="0"/>
    </default>
  </default>
  <worldbody>
    <geom name="floor" type="plane" pos="{size_x / 2} {size_y / 2} 0" size="{size_x / 2} {size_y / 2} 0.1"
          material="gravel"/>
    <!-- Ceiling: seen by the depth camera and collides, but hidden so the viewer can look down -->
    <geom name="ceiling" type="box" pos="{size_x / 2} {size_y / 2} {TUNNEL_HEIGHT + 0.05}"
          size="{size_x / 2} {size_y / 2} 0.05" group="3" rgba=".3 .3 .3 .3"/>
    <geom name="home_pad" class="visual" type="cylinder" pos="{home_x} {home_y} 0.01" size="0.6 0.005"
          rgba=".1 .8 .3 1"/>
    {walls}
    {extra_bodies}
{drone_body}
  </worldbody>{actuators}</mujoco>
"""


class CaveExploreEnv(DroneAvoidanceEnv):
    """Map as much of an unknown cave as possible without crashing.

    Observation (dict), same layout as DroneAvoidanceEnv so its pilot is reusable:
        depth: uint8 (1, 48, 64) depth image, 0 = touching, 255 = 10 m or more
        state: float32 (6,) [sin(heading error), cos(heading error), 0, goal distance,
               forward speed, yaw rate]; heading error = nose angle minus goal bearing.
               With no goal set it is 0 ("fly where you are pointing") and goal
               distance reads 0.5
    Action: float32 (2,) in [-1, 1] -> [forward speed 0..4 m/s, yaw rate +/-90 deg/s]
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": int(1 / CONTROL_DT)}

    def __init__(
        self,
        cave_size: tuple[int, int] = (48, 48),
        new_cave_each_reset: bool = False,
        max_steps: int = 1500,
        depth_noise: float = 0.02,
        map_resolution: float = 0.25,
        render_mode: str | None = None,
        world_seed: int | None = None,
        people: int = 0,
        done_coverage: float | None = DONE_COVERAGE,
    ):
        """people: how many people to rescue are placed in the cave each reset (Stage 3).
        done_coverage: end the episode at this coverage; None = never (a mission ends at home)."""
        gym.Env.__init__(self)
        self.cave_size = cave_size
        self.people = people
        self.done_coverage = done_coverage
        self.people_xy = np.zeros((0, 2))
        self.new_cave_each_reset = new_cave_each_reset
        self.max_steps = max_steps
        self.depth_noise = depth_noise
        self.map_resolution = map_resolution
        self.render_mode = render_mode
        self._world_rng = np.random.default_rng(world_seed)

        self._ray_directions_body = _camera_ray_directions()
        self._ray_count = len(self._ray_directions_body)
        self._ray_geom_ids = np.zeros(self._ray_count, dtype=np.int32)
        self._ray_distances = np.zeros(self._ray_count, dtype=np.float64)
        self._substeps = round(CONTROL_DT / PHYSICS_DT)
        # Horizontal scan rays (rows x columns): body-frame angle and horizontal share of each ray
        scan_dirs = self._ray_directions_body.reshape(DEPTH_HEIGHT, DEPTH_WIDTH, 3)[list(SCAN_ROWS)]
        self._scan_angles_body = np.arctan2(scan_dirs[0, :, 1], scan_dirs[0, :, 0])  # left (+) to right
        self._scan_horizontal = np.hypot(scan_dirs[..., 0], scan_dirs[..., 1])

        self.observation_space = spaces.Dict(
            {
                "depth": spaces.Box(0, 255, (1, DEPTH_HEIGHT, DEPTH_WIDTH), dtype=np.uint8),
                "state": spaces.Box(-1.0, 1.0, (6,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

        self.goal: np.ndarray | None = None  # world (x, y) waypoint, set by an explorer (Stage 2)
        self.trail: list[tuple[float, float]] = []
        self._renderer: mujoco.Renderer | None = None
        self._step_count = 0
        self._start_x = 0.0
        self._build_world()

    # ------------------------------------------------------------------- world
    def _build_world(self) -> None:
        self.layout = generate_cave(self._world_rng, self.cave_size)
        extra = people_xml(self.people, self._world_rng) if self.people else ("", "")
        self.model = mujoco.MjModel.from_xml_string(cave_xml(self.layout, *extra))
        self.data = mujoco.MjData(self.model)
        self._person_slots = index_people(self.model, self.people)
        self._drone_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "drone")
        self._hull_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "hull")
        self._floor_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        ceiling = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ceiling")
        self._ignored_ray_geoms = np.array([self._floor_geom, ceiling])
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

        # Ground truth for metrics: which map cells are open and reachable from home
        self.map = OccupancyMap(self.layout.size_m, self.map_resolution)
        reachable = self.layout.reachable()
        cx, cy = self.map.cell_centres()
        lx = np.floor(cx / self.layout.cell_size).astype(int)
        ly = np.floor(cy / self.layout.cell_size).astype(int)
        self._true_free = reachable[lx, ly]
        self._true_rock = ~self.layout.free[lx, ly]
        self._cell_centres = (cx, cy)

    # ----------------------------------------------------------------- sensing
    def scan(self) -> tuple[np.ndarray, np.ndarray]:
        """Horizontal range scan, one ray per image column, left to right.

        Returns (angles relative to the nose, rad, + = left; ranges, m). The scan rows
        are merged by taking the nearer reading. "Nothing hit" stays exactly
        DEPTH_RANGE so the map never mistakes it for a wall.
        """
        raw = self._ray_distances.reshape(DEPTH_HEIGHT, DEPTH_WIDTH)[list(SCAN_ROWS)]
        ranges = np.where(raw >= DEPTH_RANGE, DEPTH_RANGE, raw * self._scan_horizontal).min(axis=0)
        return self._scan_angles_body, ranges

    def _cast_depth(self) -> None:
        super()._cast_depth()
        if not hasattr(self, "map"):
            return
        angles, ranges = self.scan()
        if self.depth_noise > 0:
            # Noise on real hits only (capped below max range); misses stay misses
            noisy = ranges * (1.0 + self.np_random.normal(0.0, self.depth_noise, ranges.shape))
            ranges = np.where(ranges < DEPTH_RANGE, np.minimum(noisy, DEPTH_RANGE - 0.01), DEPTH_RANGE)
            ranges[self.np_random.random(ranges.shape) < 0.01] = DEPTH_RANGE  # stereo dropouts
        sensor_xy, _ = self.camera_pose()
        self._new_cells = self.map.integrate_scan(sensor_xy[:2], angles + self.yaw, ranges, DEPTH_RANGE)
        self.trail.append((float(self.position[0]), float(self.position[1])))

    # ----------------------------------------------------------------- metrics
    def coverage(self) -> float:
        """Share of the reachable cave the map knows is open (0..1)."""
        return float((self.map.free_mask() & self._true_free).sum() / max(1, self._true_free.sum()))

    def wall_precision(self, tolerance_cells: int = 1) -> float:
        """Share of mapped wall cells within tolerance_cells of real rock (map accuracy, 0..1).

        A one-cell (0.25 m) tolerance is the usual allowance for range noise. People are
        real obstacles too, so cells where a person stands count as correct.
        """
        rock = self._true_rock.copy()
        cx, cy = self._cell_centres
        for px, py in self.people_xy:
            rock |= np.hypot(cx - px, cy - py) < 0.5
        padded = np.pad(rock, tolerance_cells, constant_values=True)
        near_rock = np.zeros_like(rock)
        nx, ny = rock.shape
        for dx in range(2 * tolerance_cells + 1):
            for dy in range(2 * tolerance_cells + 1):
                near_rock |= padded[dx : dx + nx, dy : dy + ny]
        occupied = self.map.occupied_mask()
        return float((occupied & near_rock).sum() / max(1, occupied.sum()))

    # ------------------------------------------------------------- observation
    def _observation(self) -> dict[str, np.ndarray]:
        observation = super()._observation()
        heading_error, goal_distance = 0.0, 0.5 * GOAL_SCALE
        if self.goal is not None:
            # Nose angle relative to the goal bearing (+ = turned left of the goal). Same
            # sign as sin(yaw) in the forest course, where the goal bearing is always 0.
            dx, dy = self.goal - self.position[:2]
            heading_error = self.yaw - math.atan2(dy, dx)
            goal_distance = math.hypot(dx, dy)
        state = observation["state"]
        state[0], state[1] = math.sin(heading_error), math.cos(heading_error)
        state[2] = 0.0
        state[3] = min(1.0, goal_distance / GOAL_SCALE)
        return observation

    def _evaluate(self, action: np.ndarray, collided: bool) -> tuple[float, bool, dict[str, Any]]:
        coverage = self.coverage()
        new_area = self._new_cells * self.map.resolution**2
        reward = AREA_REWARD * new_area - TIME_PENALTY
        if collided:
            reward += COLLISION_PENALTY
        finished = self.done_coverage is not None and coverage >= self.done_coverage
        info = {
            "is_success": bool(finished and not collided),
            "collision": collided,
            "coverage": coverage,
            "new_area_m2": new_area,
            "nearest_obstacle": self.nearest_obstacle_distance(),
        }
        return reward, collided or finished, info

    # ------------------------------------------------------------------ gym API
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        gym.Env.reset(self, seed=seed)
        options = options or {}
        if self.new_cave_each_reset or options.get("new_cave"):
            self._build_world()
        else:
            self.map = OccupancyMap(self.layout.size_m, self.map_resolution)
        mujoco.mj_resetData(self.model, self.data)
        if self.people:
            self.people_xy = choose_spots(self.layout, self.people, self.np_random)
            place_people(self.data, self.model, self._person_slots, self.people_xy, self.np_random)
        home_x, home_y = self.layout.home_xy
        self.data.qpos[:4] = [home_x, home_y, FLIGHT_ALTITUDE, self.np_random.uniform(-0.3, 0.3)]
        mujoco.mj_forward(self.model, self.data)
        self.goal = None
        self.trail = []
        self._step_count = 0
        self._cast_depth()
        return self._observation(), {"coverage": self.coverage()}

    def render(self) -> np.ndarray | None:
        """Top-down view following the drone (ceiling is hidden)."""
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = self._drone_body
        camera.distance, camera.elevation, camera.azimuth = 14.0, -70.0, 90.0
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()


gym.register(id="CaveExplore-v0", entry_point=CaveExploreEnv)
