#!/usr/bin/env python3
"""
RL Environment for Autonomous Drone Obstacle Avoidance
Gymnasium-compatible custom environment integrating YOLOv11 detections.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DetectionBbox:
    """Represents a detected bounding box from YOLO."""
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float
    class_id: int
    class_name: str


class DroneAvoidanceEnv(gym.Env):
    """
    Gymnasium-compatible environment for autonomous drone obstacle avoidance.
    
    Integrates YOLOv11 detections as state observations.
    
    State Space: Normalized bounding box features [x, y, w, h, confidence, class_id]
    Action Space: Continuous [yaw_rate, forward_velocity]
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        max_obstacles: int = 5,
        max_steps: int = 500,
        collision_area_threshold: float = 0.3,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the drone avoidance environment.
        
        Args:
            max_obstacles: Maximum number of obstacles to track per observation
            max_steps: Maximum steps per episode
            collision_area_threshold: Normalized bbox area above which collision occurs
            render_mode: Optional render mode
        """
        super().__init__()
        
        self.max_obstacles = max_obstacles
        self.max_steps = max_steps
        self.collision_area_threshold = collision_area_threshold
        self.render_mode = render_mode
        
        # Observation space: [x, y, w, h, conf, class_id] per obstacle
        # + [drone_velocity, drone_yaw] = 6 * max_obstacles + 2
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.max_obstacles * 6 + 2,),
            dtype=np.float32
        )
        
        # Action space: [yaw_rate, forward_velocity]
        # yaw_rate: [-45, 45] deg/s → normalized to [-1, 1]
        # forward_velocity: [0, 15] m/s → normalized to [0, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Internal state
        self.current_step = 0
        self.current_detections: list[DetectionBbox] = []
        self.drone_velocity = 0.0
        self.drone_yaw = 0.0
        self.episode_reward = 0.0
        
        # Reward constants
        self.STEP_REWARD = 1.0
        self.COLLISION_PENALTY = -100.0
        self.STEERING_PENALTY = -5.0
        
    def set_detections(self, detections: list[DetectionBbox]) -> None:
        """
        Update environment with new YOLO detections.
        This is called from external detection pipeline.
        
        Args:
            detections: List of DetectionBbox objects from YOLO
        """
        self.current_detections = detections[:self.max_obstacles]
    
    def _normalize_bbox(self, bbox: DetectionBbox, image_width: int = 640, image_height: int = 480) -> np.ndarray:
        """
        Normalize bounding box coordinates to [-1, 1] range.
        
        Args:
            bbox: DetectionBbox object
            image_width: Width of detection image
            image_height: Height of detection image
            
        Returns:
            Normalized feature array [x_norm, y_norm, w_norm, h_norm, conf, class_id_norm]
        """
        # Normalize coordinates to [-1, 1]
        x_norm = (bbox.x_center / image_width) * 2.0 - 1.0
        y_norm = (bbox.y_center / image_height) * 2.0 - 1.0
        w_norm = (bbox.width / image_width) * 2.0
        h_norm = (bbox.height / image_height) * 2.0
        
        # Confidence already in [0, 1], scale to [-1, 1]
        conf_norm = bbox.confidence * 2.0 - 1.0
        
        # Normalize class_id (assuming max 10 classes for this example)
        class_id_norm = (bbox.class_id / 10.0) * 2.0 - 1.0
        
        return np.array([x_norm, y_norm, w_norm, h_norm, conf_norm, class_id_norm], dtype=np.float32)
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct observation from current detections and drone state.
        
        Returns:
            Normalized observation array
        """
        obs = np.zeros(self.max_obstacles * 6 + 2, dtype=np.float32)
        
        # Add normalized bounding boxes
        for i, bbox in enumerate(self.current_detections):
            if i >= self.max_obstacles:
                break
            normalized_bbox = self._normalize_bbox(bbox)
            obs[i * 6:(i + 1) * 6] = normalized_bbox
        
        # Add drone state (velocity and yaw normalized to [-1, 1])
        obs[-2] = self.drone_velocity / 15.0  # Max velocity 15 m/s
        obs[-1] = self.drone_yaw / 180.0      # Max yaw 180 degrees
        
        return obs
    
    def _check_collision(self) -> bool:
        """
        Check if drone has collided with any obstacle.
        Collision occurs when normalized bbox area exceeds threshold.
        
        Returns:
            True if collision detected, False otherwise
        """
        for bbox in self.current_detections:
            # Calculate normalized area
            normalized_area = (bbox.width / 640.0) * (bbox.height / 480.0)
            if normalized_area > self.collision_area_threshold:
                return True
        
        return False
    
    def _calculate_reward(self, action: np.ndarray, collision: bool) -> float:
        """
        Calculate reward based on action and environment state.
        
        Reward structure:
        - +1.0 for each step without collision while moving forward
        - -100.0 for collision
        - -5.0 penalty for erratic steering
        
        Args:
            action: [yaw_rate, forward_velocity]
            collision: Whether collision occurred
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Collision penalty (highest priority)
        if collision:
            reward += self.COLLISION_PENALTY
        else:
            # Positive reward for forward movement without collision
            forward_velocity = action[1]
            if forward_velocity > 0.0:
                reward += self.STEP_REWARD
            
            # Penalize erratic steering
            yaw_rate = abs(action[0])
            if yaw_rate > 0.7:  # High steering rate
                reward += self.STEERING_PENALTY
        
        return reward
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: [yaw_rate, forward_velocity]
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1
        
        # Update drone state based on action
        self.drone_yaw += action[0] * 45.0  # Yaw rate: [-45, 45] deg/s
        self.drone_yaw = np.clip(self.drone_yaw, -180.0, 180.0)
        self.drone_velocity = action[1] * 15.0  # Velocity: [0, 15] m/s
        
        # Check collision
        collision = self._check_collision()
        
        # Calculate reward
        reward = self._calculate_reward(action, collision)
        self.episode_reward += reward
        
        # Get observation
        observation = self._get_observation()
        
        # Determine termination conditions
        terminated = collision  # Episode ends on collision
        truncated = self.current_step >= self.max_steps  # Episode ends at max steps
        
        # Info dictionary
        info = {
            'step': self.current_step,
            'collision': collision,
            'episode_reward': self.episode_reward,
            'drone_velocity': self.drone_velocity,
            'drone_yaw': self.drone_yaw,
            'num_obstacles': len(self.current_detections)
        }
        
        return observation, reward, terminated, truncated, info
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Additional reset options
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        self.drone_velocity = 0.0
        self.drone_yaw = 0.0
        self.episode_reward = 0.0
        self.current_detections = []
        
        observation = self._get_observation()
        
        info = {
            'step': 0,
            'collision': False,
            'episode_reward': 0.0,
            'drone_velocity': 0.0,
            'drone_yaw': 0.0,
            'num_obstacles': 0
        }
        
        return observation, info
    
    def render(self) -> None:
        """
        Render environment state (optional visualization).
        """
        if self.render_mode == "human":
            print(f"\n[Step {self.current_step}]")
            print(f"  Drone Velocity: {self.drone_velocity:.2f} m/s")
            print(f"  Drone Yaw: {self.drone_yaw:.1f}°")
            print(f"  Obstacles: {len(self.current_detections)}")
            print(f"  Episode Reward: {self.episode_reward:.2f}")
    
    def close(self) -> None:
        """Clean up environment resources."""
        pass


class DroneDetectionWrapper:
    """
    Wrapper to integrate YOLO detection pipeline with DroneAvoidanceEnv.
    Handles conversion from YOLO output to DroneAvoidanceEnv input.
    """
    
    def __init__(self, env: DroneAvoidanceEnv):
        """
        Initialize wrapper.
        
        Args:
            env: DroneAvoidanceEnv instance
        """
        self.env = env
    
    def update_from_yolo_detections(self, yolo_results) -> None:
        """
        Convert YOLO detection results to DroneAvoidanceEnv format.
        
        Args:
            yolo_results: Ultralytics YOLO detection results object
        """
        detections = []
        
        if yolo_results.boxes is not None:
            boxes = yolo_results.boxes
            
            for i in range(len(boxes)):
                # Extract bbox in x_center, y_center, width, height format
                bbox_xyxy = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = bbox_xyxy
                
                x_center = (x1 + x2) / 2.0
                y_center = (y1 + y2) / 2.0
                width = x2 - x1
                height = y2 - y1
                
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = yolo_results.names[class_id]
                
                detection = DetectionBbox(
                    x_center=x_center,
                    y_center=y_center,
                    width=width,
                    height=height,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name
                )
                detections.append(detection)
        
        self.env.set_detections(detections)


if __name__ == "__main__":
    # Example usage
    env = DroneAvoidanceEnv(max_obstacles=5, max_steps=100, render_mode="human")
    
    # Reset environment
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Simulate some steps
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated or truncated:
            break
    
    env.close()
