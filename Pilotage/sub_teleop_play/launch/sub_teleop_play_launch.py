import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():

    ld = LaunchDescription()

    config = os.path.join(
        get_package_share_directory('sub_teleop_play'),
        'config',
        'params.yaml'
        )
    
    joy = Node(
            package='joy',
            executable='joy_node',
            name='joy', 
            )
    
    interface_joy = Node(
            package='sub_teleop_play',
            namespace='sub_teleop_play',
            executable='teleop_twist_playstation',
            parameters = [config],
            name='teleop_playstation',
            remappings=[
                ('/sub_teleop_play/SousMarin/command/pos_vel', '/SousMarin/command/pos_vel'),
                ('/sub_teleop_play/SousMarin/command/vel', '/SousMarin/command/vel'),
                ('/sub_teleop_play/SousMarin/command/mode', '/SousMarin/command/mode'),
                ('/sub_teleop_play/joy', '/joy')
                ]
            )
    
    ld.add_action(joy)
    ld.add_action(interface_joy)

    return ld