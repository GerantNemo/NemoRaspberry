from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='sub_messages_sim',
            namespace='imu',
            executable='imu_publisher',
            name='sim'
        ),
        Node(
            package='sub_messages_sim',
            namespace='odometry',
            executable='odometry_publisher',
            name='sim'
        )
    ])