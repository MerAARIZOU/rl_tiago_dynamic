#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
from ros_gz_interfaces.srv import SetEntityPose 
from ros_gz_interfaces.msg import Entity  # Import pour utiliser les constantes de type

class DynamicObstacleMover(Node):
    def __init__(self):
        super().__init__('dynamic_obstacle_mover')
        
        self.srv_name = '/world/house_dynamic/set_pose'
        self.client = self.create_client(SetEntityPose, self.srv_name)
        self.obstacle_name = 'obstacle_cylindre'
        self.counter = 0.0
        
        # Variable pour suivre l'état de la requête asynchrone précédente
        self.current_future = None
        
        self.timer_init = self.create_timer(1.0, self._check_service_connection)

    def _check_service_connection(self):
        if self.client.wait_for_service(timeout_sec=0.1):
            self.get_logger().info(f"Connecté au service Gazebo : {self.srv_name}")
            self.timer_init.cancel()
            self.timer_move = self.create_timer(0.05, self._move_obstacle_callback)
        else:
            self.get_logger().warn(f"En attente du service {self.srv_name}...")

    def _move_obstacle_callback(self):
        # SÉCURITÉ : Si la requête précédente n'est pas encore terminée, 
        # on saute ce cycle pour ne pas surcharger Gazebo et ROS 2.
        if self.current_future is not None and not self.current_future.done():
            return
            
        self.counter += 0.03
        req = SetEntityPose.Request()
        
        req.entity.name = self.obstacle_name
        req.entity.type = Entity.MODEL  # Utilisation de la constante propre (vaut 2)
        
        req.pose.position.x = 1.8
        req.pose.position.y = 0.0 + math.sin(self.counter) * 1.5
        req.pose.position.z = 0.5
        req.pose.orientation.w = 1.0
        
        # On stocke le future pour le cycle suivant
        self.current_future = self.client.call_async(req)

def main(args=None):
    rclpy.init(args=args)
    node = DynamicObstacleMover()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()