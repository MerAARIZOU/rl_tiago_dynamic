#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
from ros_gz_interfaces.srv import SetEntityPose 

class DynamicObstacleMover(Node):
    def __init__(self):
        super().__init__('dynamic_obstacle_mover')
        
        # CORRECTION : On s'aligne sur le nom de l'instance de simulation lancée par le launch.py
        self.srv_name = '/world/house_dynamic/set_pose'
        self.client = self.create_client(SetEntityPose, self.srv_name)
        self.obstacle_name = 'obstacle_cylindre'
        self.counter = 0.0
        
        self.timer_init = self.create_timer(1.0, self._check_service_connection)

    def _check_service_connection(self):
        if self.client.wait_for_service(timeout_sec=0.1):
            self.get_logger().info(f"Connecté au service Gazebo : {self.srv_name}")
            self.timer_init.cancel()
            self.timer_move = self.create_timer(0.05, self._move_obstacle_callback)
        else:
            self.get_logger().warn(f"En attente du service {self.srv_name}...")

    def _move_obstacle_callback(self):
        self.counter += 0.03
        req = SetEntityPose.Request()
        
        # STRUCTURE DE REQUÊTE STANDARD ET DIRECTE
        req.entity.name = self.obstacle_name
        req.entity.type = 2  # Type Model
        
        req.pose.position.x = 1.8
        req.pose.position.y = 0.0 + math.sin(self.counter) * 1.5
        req.pose.position.z = 0.5
        req.pose.orientation.w = 1.0
        
        self.client.call_async(req)

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