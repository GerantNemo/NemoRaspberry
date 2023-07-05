from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='sub_affichage',
            namespace='affichage',
            executable='sub_affichage',
            name='affichage',
            remappings=[
                ('/affichage/model/SousMarin/odometry', '/model/SousMarin/odometry'),
                ('/affichage/SousMarin/command/pos_vel', '/SousMarin/command/pos_vel'),
                ('/affichage/model/SousMarin/joint/CaissonPrinc_Moteur1/cmd_thrust', '/model/SousMarin/joint/CaissonPrinc_Moteur1/cmd_thrust'),
                ('/affichage/model/SousMarin/joint/CaissonPrinc_Moteur2/cmd_thrust', '/model/SousMarin/joint/CaissonPrinc_Moteur2/cmd_thrust'),
                ('/affichage/model/SousMarin/joint/CaissonPrinc_Moteur3/cmd_thrust', '/model/SousMarin/joint/CaissonPrinc_Moteur3/cmd_thrust'),
                ('/affichage/model/SousMarin/joint/CaissonPrinc_Moteur4/cmd_thrust', '/model/SousMarin/joint/CaissonPrinc_Moteur4/cmd_thrust'),
                ('/affichage/model/SousMarin/joint/CaissonPrinc_MoteurProf1/cmd_thrust', '/model/SousMarin/joint/CaissonPrinc_MoteurProf1/cmd_thrust'),
                ('/affichage/model/SousMarin/joint/CaissonPrinc_MoteurProf2/cmd_thrust', '/model/SousMarin/joint/CaissonPrinc_MoteurProf2/cmd_thrust'),
                ('/affichage/model/SousMarin/joint/CaissonPrinc_MoteurProf3/cmd_thrust', '/model/SousMarin/joint/CaissonPrinc_MoteurProf3/cmd_thrust'),
                ('/affichage/model/SousMarin/joint/CaissonPrinc_MoteurProf4/cmd_thrust', '/model/SousMarin/joint/CaissonPrinc_MoteurProf4/cmd_thrust')
            ]
        )
    ])