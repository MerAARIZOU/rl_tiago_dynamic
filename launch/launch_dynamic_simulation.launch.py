from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    
    tiago_launch_file = os.path.join(
        get_package_share_directory('tiago_gazebo'),
        'launch',
        'tiago_gazebo.launch.py'
    )

    map_yaml_file = os.path.join(get_package_share_directory('pal_maps'), 'maps', 'small_house', 'map.yaml')

    return LaunchDescription([
        # 1. Lancement de la simulation TIAGo
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(tiago_launch_file),
            launch_arguments={
                'is_public_sim': 'True',
                'world_name': 'house_dynamic',
                'extra_gz_args': '--ros-args --log-level WARN:=ERROR'
            }.items()
        ),

        # 2. Le pont complet d'origine avec Gazebo (Horloge + Set Pose synchronisé sur house_dynamic)
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                '/world/house_dynamic/set_pose@ros_gz_interfaces/srv/SetEntityPose',
                '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'
            ],
            parameters=[{'use_sim_time': True}],
            output='screen'
        ),

        # 3. Le nœud qui déplace l'obstacle dynamique dans la pièce
        Node(
            package='rl_tiago_dynamic', 
            executable='dynamic_obstacle_mover', 
            output='screen'
        ),
        
        # 4. Le serveur de carte Nav2 pour valider les positions de spawn dans Gym
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[
                {'yaml_filename': map_yaml_file},
                {'frame_id': 'map'}
            ]
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_map',
            output='screen',
            parameters=[
                {'autostart': True},
                {'node_names': ['map_server']}
            ]
        ),
    ])