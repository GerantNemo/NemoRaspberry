from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        #Node(
        #    package='sub_teleop',
        #    namespace='teleop_cmd',
        #    executable='teleop_twist_keyboard_vf',
        #    name='sim'
        #),
        Node(
            package='sub_teleop',
            namespace='teleop_cmd',
            executable='teleop_twist_keyboard_vf_gazebo',
            name='sim2',
            remappings=[
                ('/teleop_cmd/SousMarin/command/mode', '/SousMarin/command/mode'),
                ('/teleop_cmd/SousMarin/command/pos_vel', '/SousMarin/command/pos_vel'),
                ('/teleop_cmd/SousMarin/command/vel', '/SousMarin/command/vel'),
                ('/teleop_cmd/keyboard/keypress', '/keyboard/keypress')
            ]
        )
    ])