"""MuJoCo obstacle course for training drone obstacle avoidance with RL.

The drone flies at a fixed altitude through a random forest of poles, trees and
boxes towards a finish line. It sees the world through a forward-facing depth
camera (ray-cast, behaving like a stereo depth camera on a real drone) and is
commanded with velocity setpoints, forward speed and yaw rate. That is the same
interface PX4/ArduPilot flight controllers expose in offboard/guided mode, so a
policy trained here talks to a real autopilot the same way.

Simplification: the airframe's roll/pitch dynamics are not modelled; the drone
tracks velocity commands with realistic acceleration limits instead.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

COURSE_LENGTH = 40.0  # metres from start to finish line
HALF_WIDTH = 8.0  # corridor half width (side walls at +/- this)
FLIGHT_ALTITUDE = 1.5
PHYSICS_DT = 0.005
CONTROL_DT = 0.1  # 10 Hz policy, typical for companion-computer control loops
MAX_SPEED = 4.0  # m/s
MAX_YAW_RATE = math.radians(90.0)

DEPTH_WIDTH = 64
DEPTH_HEIGHT = 48
DEPTH_RANGE = 10.0  # metres; farther or no hit reads as max range
HORIZONTAL_FOV = math.radians(90.0)
CAMERA_OFFSET = 0.22  # camera sits on the drone's nose

POOL_SIZE = 60  # obstacle bodies built into the model, placed at each reset
HIDDEN_Z = -50.0

COLLISION_PENALTY = -20.0
SUCCESS_REWARD = 20.0
DANGER_DISTANCE = 1.5  # start penalising when an obstacle is closer than this
DANGER_WEIGHT = 0.5
TURN_PENALTY = 0.02
TIME_PENALTY = 0.005


def _camera_ray_directions() -> np.ndarray:
    """Unit ray directions in the drone body frame (x forward, y left, z up), row-major."""
    tan_h = math.tan(HORIZONTAL_FOV / 2.0)
    tan_v = tan_h * DEPTH_HEIGHT / DEPTH_WIDTH
    u = (np.arange(DEPTH_WIDTH) + 0.5) / DEPTH_WIDTH * 2.0 - 1.0
    v = (np.arange(DEPTH_HEIGHT) + 0.5) / DEPTH_HEIGHT * 2.0 - 1.0
    uu, vv = np.meshgrid(u, v)
    directions = np.stack([np.ones_like(uu), -uu * tan_h, -vv * tan_v], axis=-1).reshape(-1, 3)
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def _build_model_xml(rng: np.random.Generator) -> tuple[str, np.ndarray, np.ndarray]:
    """Return the scene XML plus each pooled obstacle's half height and clearance radius."""
    half_heights = np.empty(POOL_SIZE)
    radii = np.empty(POOL_SIZE)
    obstacle_bodies = []
    for index in range(POOL_SIZE):
        half_height = rng.uniform(1.4, 4.0)
        half_heights[index] = half_height
        if rng.random() < 0.7:
            # Tree trunk or pole
            radius = rng.uniform(0.1, 0.6)
            radii[index] = radius
            geom = f'<geom type="cylinder" size="{radius:.3f} {half_height:.3f}" material="bark"/>'
        else:
            # Building corner, container, rock
            sx, sy = rng.uniform(0.3, 1.5, size=2)
            radii[index] = math.hypot(sx, sy)
            geom = (
                f'<geom type="box" size="{sx:.3f} {sy:.3f} {half_height:.3f}" '
                f'material="concrete"/>'
            )
        obstacle_bodies.append(
            f'<body name="obstacle{index}" mocap="true" pos="-200 {index} {HIDDEN_Z}">{geom}</body>'
        )

    centre_x = COURSE_LENGTH / 2.0
    wall_half_length = COURSE_LENGTH / 2.0 + 5.0
    xml = f"""
<mujoco model="drone_obstacle_course">
  <option timestep="{PHYSICS_DT}" gravity="0 0 0" integrator="implicitfast"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="4096"/>
    <headlight ambient=".35 .35 .35" diffuse=".5 .5 .5"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1=".55 .72 .92" rgb2=".95 .97 1"
             width="256" height="256"/>
    <texture name="grass" type="2d" builtin="checker" rgb1=".27 .36 .22" rgb2=".22 .30 .18"
             width="512" height="512"/>
    <material name="ground" texture="grass" texrepeat="30 15"/>
    <material name="bark" rgba=".45 .30 .18 1"/>
    <material name="concrete" rgba=".62 .62 .65 1"/>
    <material name="wall" rgba=".75 .75 .8 .25"/>
  </asset>
  <default>
    <geom contype="2" conaffinity="1" condim="1"/>
    <default class="visual">
      <geom contype="0" conaffinity="0" group="1" density="0"/>
    </default>
  </default>
  <worldbody>
    <light pos="{centre_x} 0 40" dir="0.3 0.2 -1" directional="true" castshadow="true"/>
    <geom name="floor" type="plane" pos="{centre_x} 0 0"
          size="{wall_half_length + 5} {HALF_WIDTH + 5} 0.1" material="ground"/>
    <geom name="wall_left" type="box" pos="{centre_x} {HALF_WIDTH + 0.1} 3"
          size="{wall_half_length} 0.1 3" material="wall"/>
    <geom name="wall_right" type="box" pos="{centre_x} {-HALF_WIDTH - 0.1} 3"
          size="{wall_half_length} 0.1 3" material="wall"/>
    <geom name="wall_back" type="box" pos="-3 0 3" size="0.1 {HALF_WIDTH} 3" material="wall"/>
    <geom name="finish_line" class="visual" type="box" pos="{COURSE_LENGTH} 0 0.01"
          size="0.15 {HALF_WIDTH} 0.01" rgba=".1 .9 .3 1"/>

    <body name="drone">
      <joint name="slide_x" type="slide" axis="1 0 0" armature="0.01"/>
      <joint name="slide_y" type="slide" axis="0 1 0" armature="0.01"/>
      <joint name="slide_z" type="slide" axis="0 0 1" armature="0.01"/>
      <joint name="yaw" type="hinge" axis="0 0 1" armature="0.001"/>
      <!-- Collision hull covering the arms and propellers -->
      <geom name="hull" type="box" size="0.2 0.2 0.06" mass="1" contype="1" conaffinity="2"
            group="3" rgba="1 0 0 .2"/>
      <geom class="visual" type="box" size="0.07 0.05 0.03" rgba=".12 .12 .14 1"/>
      <geom class="visual" type="capsule" fromto="-.17 -.17 0 .17 .17 0" size=".012"
            rgba=".2 .2 .22 1"/>
      <geom class="visual" type="capsule" fromto="-.17 .17 0 .17 -.17 0" size=".012"
            rgba=".2 .2 .22 1"/>
      <geom class="visual" type="cylinder" pos=".17 .17 .02" size=".08 .004" rgba=".8 .8 .85 .5"/>
      <geom class="visual" type="cylinder" pos=".17 -.17 .02" size=".08 .004" rgba=".8 .8 .85 .5"/>
      <geom class="visual" type="cylinder" pos="-.17 .17 .02" size=".08 .004" rgba=".8 .8 .85 .5"/>
      <geom class="visual" type="cylinder" pos="-.17 -.17 .02" size=".08 .004" rgba=".8 .8 .85 .5"/>
      <geom class="visual" type="sphere" pos="{CAMERA_OFFSET} 0 0" size=".025" rgba=".9 .2 .1 1"/>
      <camera name="fpv" pos="{CAMERA_OFFSET} 0 0" xyaxes="0 -1 0 0 0 1" fovy="74"/>
      <camera name="chase" pos="-3 0 1.2" xyaxes="0 -1 0 .34 0 .94" fovy="60"/>
    </body>

    {"".join(obstacle_bodies)}
  </worldbody>
  <actuator>
    <velocity name="vel_x" joint="slide_x" kv="5" forcelimited="true" forcerange="-6 6"/>
    <velocity name="vel_y" joint="slide_y" kv="5" forcelimited="true" forcerange="-6 6"/>
    <velocity name="vel_z" joint="slide_z" kv="5" forcelimited="true" forcerange="-6 6"/>
    <velocity name="yaw_rate" joint="yaw" kv="0.2" forcelimited="true" forcerange="-0.5 0.5"/>
  </actuator>
</mujoco>
"""
    return xml, half_heights, radii


class DroneAvoidanceEnv(gym.Env):
    """Fly through a random obstacle course to the finish line without crashing.

    Observation (dict):
        depth: uint8 (1, 48, 64) depth image, 0 = touching, 255 = 10 m or more
        state: float32 (6,) [sin(yaw), cos(yaw), lateral position, distance left,
               forward speed, yaw rate], all scaled to [-1, 1]
    Action: float32 (2,) in [-1, 1] -> [forward speed 0..4 m/s, yaw rate +/-90 deg/s]
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": int(1 / CONTROL_DT)}

    def __init__(
        self,
        min_obstacles: int = 10,
        max_obstacles: int = 40,
        max_steps: int = 600,
        depth_noise: float = 0.02,
        render_mode: str | None = None,
        world_seed: int | None = None,
    ):
        super().__init__()
        if not 0 <= min_obstacles <= max_obstacles <= POOL_SIZE:
            raise ValueError(f"Need 0 <= min_obstacles <= max_obstacles <= {POOL_SIZE}.")
        self.min_obstacles = min_obstacles
        self.max_obstacles = max_obstacles
        self.max_steps = max_steps
        self.depth_noise = depth_noise
        self.render_mode = render_mode

        xml, self._half_heights, self._radii = _build_model_xml(np.random.default_rng(world_seed))
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._substeps = round(CONTROL_DT / PHYSICS_DT)

        self._drone_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "drone")
        self._hull_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "hull")
        self._floor_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._obstacle_mocap = np.array(
            [
                self.model.body_mocapid[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle{index}")
                ]
                for index in range(POOL_SIZE)
            ]
        )

        self._ray_directions_body = _camera_ray_directions()
        self._ray_count = len(self._ray_directions_body)
        self._ray_geom_ids = np.zeros(self._ray_count, dtype=np.int32)
        self._ray_distances = np.zeros(self._ray_count, dtype=np.float64)
        # Rays only see collision geoms (group 0), never decorations or the drone itself
        self._ray_groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)

        self.observation_space = spaces.Dict(
            {
                "depth": spaces.Box(0, 255, (1, DEPTH_HEIGHT, DEPTH_WIDTH), dtype=np.uint8),
                "state": spaces.Box(-1.0, 1.0, (6,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

        self._step_count = 0
        self._renderer: mujoco.Renderer | None = None

    # ----------------------------------------------------------------- helpers
    @property
    def position(self) -> np.ndarray:
        return self.data.qpos[:3]

    @property
    def yaw(self) -> float:
        return float(self.data.qpos[3])

    def _place_obstacles(self, count: int) -> None:
        self.data.mocap_pos[:] = [-200.0, 0.0, HIDDEN_Z]
        placed: list[tuple[float, float, float]] = []
        for slot in self.np_random.permutation(POOL_SIZE)[:count]:
            radius = self._radii[slot]
            for _ in range(30):
                x = self.np_random.uniform(5.0, COURSE_LENGTH - 1.0)
                y = self.np_random.uniform(-HALF_WIDTH + 0.5, HALF_WIDTH - 0.5)
                # Keep a flyable gap between neighbours
                if all(math.hypot(x - ox, y - oy) > radius + other + 1.0 for ox, oy, other in placed):
                    break
            else:
                continue
            placed.append((x, y, radius))
            mocap = self._obstacle_mocap[slot]
            self.data.mocap_pos[mocap] = [x, y, self._half_heights[slot]]
            angle = self.np_random.uniform(0, math.pi)
            self.data.mocap_quat[mocap] = [math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]

    def _cast_depth(self) -> None:
        """Fill the ray buffers with distances (metres) and hit geom ids."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        origin = self.position + rotation @ np.array([CAMERA_OFFSET, 0.0, 0.0])
        directions = (self._ray_directions_body @ rotation.T).ravel()
        mujoco.mj_multiRay(
            self.model,
            self.data,
            origin,
            directions,
            self._ray_groups,
            True,
            self._drone_body,
            self._ray_geom_ids,
            self._ray_distances,
            None,
            self._ray_count,
            DEPTH_RANGE,
        )
        misses = self._ray_distances < 0
        self._ray_distances[misses] = DEPTH_RANGE

    def nearest_obstacle_distance(self) -> float:
        """Closest non-ground surface seen by the depth camera, in metres."""
        not_floor = self._ray_geom_ids != self._floor_geom
        return float(self._ray_distances[not_floor].min()) if not_floor.any() else DEPTH_RANGE

    def depth_image_metres(self) -> np.ndarray:
        return self._ray_distances.reshape(DEPTH_HEIGHT, DEPTH_WIDTH).copy()

    def _observation(self) -> dict[str, np.ndarray]:
        depth = self._ray_distances.copy()
        if self.depth_noise > 0:
            depth += self.np_random.normal(0.0, self.depth_noise, depth.shape) * depth
            # Stereo cameras drop pixels on textureless or shiny surfaces
            depth[self.np_random.random(depth.shape) < 0.01] = DEPTH_RANGE
        depth_image = np.clip(depth / DEPTH_RANGE * 255.0, 0, 255).astype(np.uint8)

        velocity = self.data.qvel
        forward_speed = velocity[0] * math.cos(self.yaw) + velocity[1] * math.sin(self.yaw)
        remaining = (COURSE_LENGTH - self.position[0]) / COURSE_LENGTH
        state = np.array(
            [
                math.sin(self.yaw),
                math.cos(self.yaw),
                self.position[1] / HALF_WIDTH,
                remaining,
                forward_speed / MAX_SPEED,
                velocity[3] / MAX_YAW_RATE,
            ],
            dtype=np.float32,
        )
        return {
            "depth": depth_image.reshape(1, DEPTH_HEIGHT, DEPTH_WIDTH),
            "state": np.clip(state, -1.0, 1.0),
        }

    def _hull_in_contact(self) -> bool:
        contacts = self.data.contact.geom[: self.data.ncon]
        return bool(np.any(contacts == self._hull_geom))

    # --------------------------------------------------------------- gym API
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        count = self.np_random.integers(self.min_obstacles, self.max_obstacles + 1)
        if options and "obstacles" in options:
            count = int(options["obstacles"])
        self._place_obstacles(int(count))

        self.data.qpos[:4] = [
            0.0,
            self.np_random.uniform(-3.0, 3.0),
            FLIGHT_ALTITUDE,
            self.np_random.uniform(-0.4, 0.4),
        ]
        mujoco.mj_forward(self.model, self.data)
        self._cast_depth()
        self._step_count = 0
        return self._observation(), {"obstacles": int(count)}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        speed = (action[0] + 1.0) / 2.0 * MAX_SPEED
        yaw_rate = action[1] * MAX_YAW_RATE
        start_x = self.position[0]

        collided = False
        for _ in range(self._substeps):
            # Velocity setpoint in the body frame, converted to world frame each tick
            self.data.ctrl[:] = [
                speed * math.cos(self.yaw),
                speed * math.sin(self.yaw),
                2.0 * (FLIGHT_ALTITUDE - self.position[2]),
                yaw_rate,
            ]
            mujoco.mj_step(self.model, self.data)
            if self._hull_in_contact():
                collided = True
                break

        self._step_count += 1
        self._cast_depth()
        nearest = self.nearest_obstacle_distance()
        reached_goal = self.position[0] >= COURSE_LENGTH

        reward = float(self.position[0] - start_x)
        reward -= DANGER_WEIGHT * max(0.0, (DANGER_DISTANCE - nearest) / DANGER_DISTANCE)
        reward -= TURN_PENALTY * abs(action[1]) + TIME_PENALTY
        if collided:
            reward += COLLISION_PENALTY
        elif reached_goal:
            reward += SUCCESS_REWARD

        terminated = collided or reached_goal
        truncated = not terminated and self._step_count >= self.max_steps
        info = {
            "is_success": bool(reached_goal and not collided),
            "collision": collided,
            "progress": float(self.position[0] / COURSE_LENGTH),
            "nearest_obstacle": nearest,
        }
        return self._observation(), reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera="chase")
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


gym.register(id="DroneAvoidance-v0", entry_point=DroneAvoidanceEnv)
