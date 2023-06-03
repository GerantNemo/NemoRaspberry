from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='sub_control',
            namespace='sub_control',
            executable='position_controller_node',
            name='sim',
            remappings=[
                ('/sub/odometry', '/odometry/sub/odometry')
            ]
        ),
        Node(
            package='sub_control',
            namespace='sub_control',
            executable='velocity_controller_node',
            name='sim',
            remappings=[
                ('/cmd/velocity', '/cmd_vel'),
                ('/sub/odometry', '/odometry/sub/odometry')
            ]
        )
    ])