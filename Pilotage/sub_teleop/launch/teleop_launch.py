from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='sub_teleop',
            namespace='teleop_cmd',
            executable='teleop_twist_keyboard_vf',
            name='sim'
        )
    ])