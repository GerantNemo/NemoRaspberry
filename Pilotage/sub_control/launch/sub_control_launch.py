import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    ld = LaunchDescription();

    config = os.path.join(
        get_package_share_directory('sub_control'),
        'config',
        'params.yaml'
        )

    control_pos = Node(
        package='sub_control',
        namespace='sub_control',
        executable='position_controller_node',
        parameters = [config],
        name='sim_pose',
        remappings=[
            ('/sub_control/SousMarin/command/mode', '/SousMarin/command/mode'),
            ('/sub_control/model/SousMarin/odometry', '/model/SousMarin/odometry'),
            ('/sub_control/SousMarin/command/pos_vel_pos', '/SousMarin/command/pos_vel_pos'),
            ('/sub_control/SousMarin/command/pos', '/SousMarin/command/pos')
        ]
    )
    
    control_vel = Node(
        package='sub_control',
        namespace='sub_control',
        executable='velocity_controller_node',
        parameters = [config],
        name='sim_vel',
        remappings=[
            ('/sub_control/SousMarin/command/mode', '/SousMarin/command/mode'),
            ('/sub_control/SousMarin/command/pos_vel', '/SousMarin/command/pos_vel'),
            ('/sub_control/model/SousMarin/odometry', '/model/SousMarin/odometry'),
            ('/sub_control/SousMarin/command/pos_vel_pos', '/SousMarin/command/pos_vel_pos'),
            ('/sub_control/model/SousMarin/joint/CaissonPrinc_Moteur1/cmd_thrust', '/model/SousMarin/joint/CaissonPrinc_Moteur1/cmd_thrust'),
            ('/sub_control/model/SousMarin/joint/CaissonPrinc_Moteur2/cmd_thrust', '/model/SousMarin/joint/CaissonPrinc_Moteur2/cmd_thrust'),
            ('/sub_control/model/SousMarin/joint/CaissonPrinc_Moteur3/cmd_thrust', '/model/SousMarin/joint/CaissonPrinc_Moteur3/cmd_thrust'),
            ('/sub_control/model/SousMarin/joint/CaissonPrinc_Moteur4/cmd_thrust', '/model/SousMarin/joint/CaissonPrinc_Moteur4/cmd_thrust'),
            ('/sub_control/model/SousMarin/joint/CaissonPrinc_MoteurProf1/cmd_thrust', '/model/SousMarin/joint/CaissonPrinc_MoteurProf1/cmd_thrust'),
            ('/sub_control/model/SousMarin/joint/CaissonPrinc_MoteurProf2/cmd_thrust', '/model/SousMarin/joint/CaissonPrinc_MoteurProf2/cmd_thrust'),
            ('/sub_control/model/SousMarin/joint/CaissonPrinc_MoteurProf3/cmd_thrust', '/model/SousMarin/joint/CaissonPrinc_MoteurProf3/cmd_thrust'),
            ('/sub_control/model/SousMarin/joint/CaissonPrinc_MoteurProf4/cmd_thrust', '/model/SousMarin/joint/CaissonPrinc_MoteurProf4/cmd_thrust')
        ]
    )

    ld.add_action(control_pos)
    ld.add_action(control_vel)

    return ld