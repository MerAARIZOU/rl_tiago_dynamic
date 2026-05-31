#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from map_target import MapTargetValidator
from gazebo_tools import GazeboTools

class TiagoEnv(gym.Env, Node):
    def __init__(self):
        super().__init__('tiago_gym_env')
        
        # 1. Espaces (20 LiDAR + 1 Dist + 2 Angle = 23)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=10.0, shape=(23,), dtype=np.float32)
        
        # 2. Variables d'état
        self.robot_pos = np.array([0.0, 0.0])
        self.robot_yaw = 0.0
        self.target_pos = np.array([0.0, 0.0])
        self.prev_dist = 0.0
        self.last_scan = np.zeros(20, dtype=np.float32)
        self.last_action = np.array([0.0, 0.0])
        self.collided = False
        
        # 3. ROS 2 Interfaces (Retour à l'odométrie native stable)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/mobile_base_controller/odom', self._odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan_raw', self._scan_callback, 10)
        
        self.target_validator = MapTargetValidator(self)
        self.gazebo_tools = GazeboTools(self)

    def _scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.nan_to_num(ranges, nan=10.0, posinf=10.0, neginf=0.05)
        sectors = np.array_split(ranges, 20)
        self.last_scan = np.array([np.min(s) for s in sectors], dtype=np.float32)

    def _odom_callback(self, msg):
        self.robot_pos[0] = msg.pose.pose.position.x
        self.robot_pos[1] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.robot_yaw = float(np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)))

    def _get_obs(self):
        current_dist = float(np.linalg.norm(self.target_pos - self.robot_pos))
        angle_to_target = np.arctan2(self.target_pos[1] - self.robot_pos[1], self.target_pos[0] - self.robot_pos[0])
        angle_error = float(np.arctan2(np.sin(angle_to_target - self.robot_yaw), np.cos(angle_to_target - self.robot_yaw)))
        
        state = np.zeros(23, dtype=np.float32)
        state[0:20] = np.clip(self.last_scan / 10.0, 0.0, 1.0)
        state[20] = np.clip(current_dist / 4.0, 0.0, 1.0)
        state[21] = np.sin(angle_error)
        state[22] = np.cos(angle_error)
        return state

    def step(self, action):
        self.last_action = action
        msg = Twist()
        msg.linear.x = float((action[0] + 1.0) * 0.25)
        msg.angular.z = float(action[1] * 1.0)
        self.cmd_vel_pub.publish(msg)
        
        rclpy.spin_once(self, timeout_sec=0.1)
        
        obs = self._get_obs()
        reward, done, info = self._calculate_reward(obs[20] * 4.0, np.arctan2(obs[21], obs[22]))
        self.prev_dist = obs[20] * 4.0
        
        return obs, reward, done, False, info

    def _calculate_reward(self, current_dist, angle_error):
        reward_dist = (self.prev_dist - current_dist) * 30.0
        reward_align = -abs(angle_error) * 1.5
        reward_forward = (self.last_action[0] * 2.0) if (abs(angle_error) < 0.2 and self.last_action[0] > 0.0) else 0.0
        penalite_cercle = -2.0 if (abs(self.last_action[1]) > 0.3 and self.last_action[0] > 0.1) else 0.0
        
        reward = reward_dist + reward_align + reward_forward + penalite_cercle - 0.05
        done = False
        
        if np.min(self.last_scan) <= 0.25:
            reward -= 100.0
            done = True
            self.collided = True
        elif current_dist < 0.45:
            reward += 200.0
            done = True
            
        return float(reward), done, {}

    def _apply_movement(self, lin_x, ang_z, duration):
        msg = Twist()
        msg.linear.x = float(lin_x)
        msg.angular.z = float(ang_z)
        
        start_time = self.get_clock().now()
        while (self.get_clock().now() - start_time).nanoseconds / 1e9 < duration:
            self.cmd_vel_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.01)
        
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)
        rclpy.spin_once(self, timeout_sec=0.05)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.collided:
            self.get_logger().info("Séquence de dégagement : Recul -> Rotation -> Avance")
            self._apply_movement(-0.25, 0.0, duration=1.0)
            rotation_dir = random.choice([-1.0, 1.0])
            self._apply_movement(0.0, rotation_dir * 0.8, duration=1.0)
            self._apply_movement(0.20, 0.0, duration=0.5)
            self.collided = False

        while self.target_validator.map_grid is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            
        tx, ty = self.target_validator.generate_valid_target()
        self.target_pos = np.array([tx, ty], dtype=np.float32)
        
        # Rétablissement de l'outil Gazebo d'origine pour placer le modèle physique
        self.gazebo_tools.update_gazebo_target_marker(tx, ty)
        
        self.prev_dist = float(np.linalg.norm(self.target_pos - self.robot_pos))
        
        return self._get_obs(), {}