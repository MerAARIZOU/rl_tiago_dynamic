import os
import subprocess
from visualization_msgs.msg import Marker

class GazeboTools:
    def __init__(self, node_instance):
        """
        Gestionnaire visuel double flux (RViz + Gazebo Sim Jazzy).
        :param node_instance: L'instance de votre TiagoEnv principal
        """
        self.node = node_instance
        self.has_spawned_marker = False
        
        # 1. Publication du marqueur standard (lu par RViz et par le Bridge Gazebo)
        self.marker_pub = self.node.create_publisher(Marker, '/simulation_target_marker', 10)
        
        # Chemin vers votre fichier URDF cible pour le premier affichage Gazebo
        self.urdf_path = os.path.expanduser('~/ros2_ws/src/rl_tiago_dynamic/models/target_marker.urdf')

    def update_gazebo_target_marker(self, x, y):
        """Met à jour la position du marqueur simultanément sur RViz et Gazebo"""
        
        # --- ÉTAPE A : Gestion du modèle physique dans Gazebo (Premier Spawn) ---
        if not self.has_spawned_marker:
            if os.path.exists(self.urdf_path):
                # On fait apparaître le modèle une première fois
                cmd = f"ros2 run ros_gz_sim create -file {self.urdf_path} -name target_marker -x {x} -y {y} -z 0.15"
                subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.has_spawned_marker = True
                self.node.get_logger().info("Marqueur initialisé dans Gazebo Sim.")
        
        # --- ÉTAPE B : Publication du message de mise à jour (RViz + Gazebo via Bridge) ---
        marker = Marker()
        marker.header.frame_id = "odom"  # Repère de votre odométrie
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.id = 999
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # Coordonnées générées par votre reset
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.15
        
        # Dimensions du marqueur visuel sur RViz
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        
        # Couleur Rouge éclatante
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        marker.lifetime.sec = 0  # Reste visible indéfiniment jusqu'au prochain update
        
        # Envoi du message
        self.marker_pub.publish(marker)