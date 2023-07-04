import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
   
   bridge_node = IncludeLaunchDescription(
      PythonLaunchDescriptionSource([os.path.join(
         get_package_share_directory('conversion_msgs'), 'launch'),
         '/sub_bridge_launch.py'])
      )
   #teleop_node = IncludeLaunchDescription(
   #   PythonLaunchDescriptionSource([os.path.join(
   #      get_package_share_directory('sub_teleop'), 'launch'),
   #      '/sub_teleop_launch.py'])
   #   )
   
   teleop_node_play = IncludeLaunchDescription(
      PythonLaunchDescriptionSource([os.path.join(
         get_package_share_directory('sub_teleop_play'), 'launch'),
         '/sub_teleop_play_launch.py'])
      )
   
   affichage_node = IncludeLaunchDescription(
      PythonLaunchDescriptionSource([os.path.join(
         get_package_share_directory('sub_affichage'), 'launch'),
         '/sub_affichage_launch.py'])
      )
   
   control_node = IncludeLaunchDescription(
      PythonLaunchDescriptionSource([os.path.join(
         get_package_share_directory('sub_control'), 'launch'),
         '/sub_control_launch.py'])
      )

   return LaunchDescription([
      bridge_node,
      #teleop_node,
      teleop_node_play,
      affichage_node,
      control_node
   ])