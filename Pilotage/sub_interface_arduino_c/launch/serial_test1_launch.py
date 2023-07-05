import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    ld = LaunchDescription();

    config = os.path.join(
        get_package_share_directory('serial_test1'),
        'config',
        'params.yaml'
        )

    serial = Node(
        package='serial_test1',
        namespace='serial_test1',
        executable='serial_node',
        parameters = [config],
        name='serial',
        remappings=[
            ('/sub_interface_arduino_c/SousMarin/command/pwm', '/SousMarin/command/pwm'),
            ('/sub_interface_arduino_c/SousMarin/measured/Imu', '/SousMarin/measured/Imu'),
            ('/sub_interface_arduino_c/SousMarin/measured/MagneticField', '/SousMarin/measured/MagneticField'),
            ('/sub_interface_arduino_c/SousMarin/measured/Angle', '/SousMarin/measured/Angle'),
            ('/sub_interface_arduino_c/SousMarin/measured/Pressure', '/SousMarin/measured/Pressure'),
            ('/sub_interface_arduino_c/SousMarin/measured/Depth', '/SousMarin/measured/Depth'),
            ('/sub_interface_arduino_c/SousMarin/measured/Temperature', '/SousMarin/measured/Temperature'),
            ('/sub_interface_arduino_c/SousMarin/measured/Humidity', '/SousMarin/measured/Humidity'),
            ('/sub_interface_arduino_c/SousMarin/measured/Tension', '/SousMarin/measured/Tension')
        ]
    )

    ld.add_action(serial)

    return ld