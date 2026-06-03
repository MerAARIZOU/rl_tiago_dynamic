#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import time

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, OccupancyGrid  # Ajout de OccupancyGrid
from sensor_msgs.msg import LaserScan
from gazebo_tools import GazeboTools
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy

class TiagoEnv(gym.Env, Node):
    def __init__(self):
        super().__init__('tiago_gym_env')

        self.max_episode_steps = 250  
        self.current_step = 0
        
        # Action space et Observation space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=10.0, shape=(23,), dtype=np.float32)
        
        # Variables d'état
        self.robot_pos = np.array([0.0, 0.0])
        self.robot_yaw = 0.0
        self.target_pos = np.array([0.0, 0.0])
        self.prev_dist = 0.0
        self.last_scan = np.ones(20, dtype=np.float32) * 10.0
        self.last_action = np.array([0.0, 0.0])
        self.collided = False

        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.latest_map = None
        self.map_sub = self.create_subscription(
            OccupancyGrid, 
            '/map', # Astuce : Utilisez '/global_costmap/costmap' si vous voulez éviter les zones trop proches des murs !
            self._map_callback, 
            map_qos 
        )
        
        # ROS 2 Interfaces
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/mobile_base_controller/odom', self._odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan_raw', self._scan_callback, 10)
        
        self.gazebo_tools = GazeboTools(self)

    def _map_callback(self, msg):
        """Callback qui récupère la carte d'occupation de l'environnement."""
        self.latest_map = msg

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
        
        state[20] = current_dist 
        state[21] = np.sin(angle_error)
        state[22] = np.cos(angle_error)
        return state

    def _generate_new_target(self):
        """Génère dynamiquement une cible uniquement dans les zones libres (cellule == 0)."""
        if self.latest_map is None:
            self.get_logger().warn("Carte non reçue, position par défaut (1.0, 1.0)")
            return 1.0, 1.0

        info = self.latest_map.info
        resolution = info.resolution
        origin_x = info.origin.position.x
        origin_y = info.origin.position.y

        # Calcul des limites physiques réelles de la carte
        min_x = origin_x
        max_x = origin_x + (info.width * resolution)
        min_y = origin_y
        max_y = origin_y + (info.height * resolution)

        while True:
            # On tire aléatoirement dans la boîte englobante de la carte
            tx = random.uniform(min_x, max_x)
            ty = random.uniform(min_y, max_y)

            # Conversion en coordonnées de grille (pixels)
            col = int((tx - origin_x) / resolution)
            row = int((ty - origin_y) / resolution)

            # Vérification des index pour éviter les débordements de tableau
            if 0 <= col < info.width and 0 <= row < info.height:
                index = row * info.width + col
                cell_value = self.latest_map.data[index]

                # 0 = Espace libre, 100 = Obstacle, -1 = Inconnu (extérieur de la maison)
                if cell_value == 0:
                    return float(tx), float(ty)

    def step(self, action):
        self.current_step += 1  
        self.last_action = action
        
        msg = Twist()
        msg.linear.x = float((action[0] + 1.0) * 0.2) 
        msg.angular.z = float(action[1] * 1.0)
        
        self.cmd_vel_pub.publish(msg)
        
        start_time = time.time()
        while time.time() - start_time < 0.1:
            rclpy.spin_once(self, timeout_sec=0.01)
        
        obs = self._get_obs()
        current_dist = obs[20]
        angle_error = np.arctan2(obs[21], obs[22])
        
        reward, done, info = self._calculate_reward(current_dist, angle_error)
        self.prev_dist = current_dist
        
        obs_normalized = obs.copy()
        obs_normalized[20] = np.clip(current_dist / 10.0, 0.0, 1.0)
        
        truncated = False
        if self.current_step >= self.max_episode_steps:
            truncated = True
        
        return obs_normalized, reward, done, truncated, info

    def _calculate_reward(self, current_dist, angle_error):
        reward_dist = (self.prev_dist - current_dist) * 50.0
        reward_align = -abs(angle_error) * 1.0
        reward_rotation_pure = -0.5 * abs(self.last_action[1])
        
        vitesse_lineaire = (self.last_action[0] + 1.0) * 0.2
        reward_forward = 0.0
        
        if abs(angle_error) < 0.3 and abs(self.last_action[1]) < 0.2 and vitesse_lineaire > 0.05:
            reward_forward = vitesse_lineaire * 5.0  
            
        reward = reward_dist + reward_align + reward_forward + reward_rotation_pure - 0.1
        done = False
        
        if np.min(self.last_scan) <= 0.32:
            reward -= 50.0
            done = True
            self.collided = True
            
        elif current_dist < 0.50:
            reward += 200.0
            done = True
            self.get_logger().info("CIBLE ATTEINTE !")
            
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
        self.current_step = 0        
        
        # Séquence de dégagement en cas de collision
        if self.collided:
            self.get_logger().info("Séquence de dégagement : Recul -> Rotation -> Avance")
            self._apply_movement(-0.25, 0.0, duration=1.2)
            rotation_dir = random.choice([-1.0, 1.0])
            self._apply_movement(0.0, rotation_dir * 1.0, duration=1.0)
            self._apply_movement(0.20, 0.0, duration=0.6)
            self.collided = False

        # Attente bloquante mais sûre du premier message de la carte
        while self.latest_map is None:
            self.get_logger().info("Attente de la carte /map...")
            rclpy.spin_once(self, timeout_sec=0.2)
            
        # Génération de la cible valide (Option 2 intégrée)
        tx, ty = self._generate_new_target()
        self.target_pos = np.array([tx, ty], dtype=np.float32)
        
        # Mise à jour graphique de la cible dans Gazebo via vos outils
        self.gazebo_tools.update_gazebo_target_marker(tx, ty)
        
        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.02)
            
        self.prev_dist = float(np.linalg.norm(self.target_pos - self.robot_pos))
        
        obs = self._get_obs()
        obs_normalized = obs.copy()
        obs_normalized[20] = np.clip(self.prev_dist / 10.0, 0.0, 1.0)
        
        return obs_normalized, {}