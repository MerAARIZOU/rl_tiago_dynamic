import rclpy
from rclpy.node import Node
import gymnasium
from gymnasium import spaces
import numpy as np
import random

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy


class MapTargetValidator:
    """
    Module autonome de supervision cartographique (Oracle).
    Gère l'abonnement asynchrone à Nav2 et la validation géométrique des coordonnées.
    """
    def __init__(self, node_instance):
        """
        Initialise le validateur de cible en utilisant le nœud ROS 2 principal.
        :param node_instance: L'instance de votre TiagoEnv (qui contient le nœud ROS2)
        """
        self.node = node_instance
        
        # Métadonnées réelles lues depuis votre topic /map de la scène House
        self.map_resolution = 0.05
        self.map_origin_x = -9.37
        self.map_origin_y = -5.6
        self.map_width = 374
        self.map_height = 223
        
        self.map_grid = None

        map_qos_profile = QoSProfile(depth=1)
        map_qos_profile.reliability = QoSReliabilityPolicy.RELIABLE
        map_qos_profile.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL

        # On utilise le nœud principal pour créer l'abonnement
        self.map_sub = self.node.create_subscription(
            OccupancyGrid,
            '/map',
            self._map_callback,
            map_qos_profile
        )

    def _map_callback(self, msg):
        """Callback pour stocker et formater la carte en matrice 2D"""
        if self.map_grid is None:
            self.map_grid = np.array(msg.data).reshape((self.map_height, self.map_width))
            self.node.get_logger().info("[Validator] Matrice d'occupation chargée et prête.")

    def is_target_free(self, x, y):
        """Vérifie si le point Gazebo (mètres) est une zone libre (0)"""
        if self.map_grid is None:
            return False

        # Conversion Mètres (Gazebo/Odom) -> Pixels (Matrice)
        pixel_x = int((x - self.map_origin_x) / self.map_resolution)
        pixel_y = int((y - self.map_origin_y) / self.map_resolution)

        # Sécurité : sortie de map
        if not (0 <= pixel_x < self.map_width and 0 <= pixel_y < self.map_height):
            return False

        # Retourne True uniquement si la case vaut 0 (Espace libre)
        return self.map_grid[pixel_y, pixel_x] == 0

    def generate_valid_target(self):
        """Boucle globale en mémoire (Alternative pour génération sur toute la maison)"""
        if self.map_grid is None:
            self.node.get_logger().warn("[Validator] Impossible de générer une cible : /map non reçue.")
            return None

        while True:
            target_x = random.uniform(-9.0, 9.0)
            target_y = random.uniform(-5.3, 5.3)

            if self.is_target_free(target_x, target_y):
                return target_x, target_y