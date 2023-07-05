from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    return LaunchDescription([
        #Informations generales
        Node(
            package='ros_gz_bridge',
            #namespace='ros_gz_bridge',
            executable='parameter_bridge',
            name='bridge_keyboard',
            arguments=['/keyboard/keypress@std_msgs/msg/Int32@ignition.msgs.Int32']
        ),
        Node(
            package='ros_gz_bridge',
            #namespace='ros_gz_bridge',
            executable='parameter_bridge',
            name='bridge_clock',
            arguments=['/clock@rosgraph_msgs/msg/Clock@ignition.msgs.Clock']
        ),
        Node(
            package='ros_gz_bridge',
            #namespace='ros_gz_bridge',
            executable='parameter_bridge',
            name='bridge_gui_camera_pose',
            arguments=['/gui/camera/pose@geometry_msgs/msg/PoseArray@ignition.msgs.Pose_V']
        ),
        #Capteurs
        Node(
            package='ros_gz_bridge',
            #namespace='ros_gz_bridge',
            executable='parameter_bridge',
            name='bridge_imu',
            arguments=['/imu@sensor_msgs/msg/Imu@ignition.msgs.IMU']
        ),
        #Node(
        #    package='ros_gz_bridge',
        #    #namespace='ros_gz_bridge',
        #    executable='parameter_bridge',
        #    name='bridge_air_pressure',
        #    arguments=['/air_pressure@sensor_msgs/msg/FluidPressure@ignition.msgs.FluidPressure']
        #),
        Node(
            package='ros_gz_bridge',
            #namespace='ros_gz_bridge',
            executable='parameter_bridge',
            name='bridge_navsat',
            arguments=['/gps@sensor_msgs/msg/NavSatFix@ignition.msgs.NavSat']
        ),
        Node(
            package='ros_gz_bridge',
            #namespace='ros_gz_bridge',
            executable='parameter_bridge',
            name='bridge_magnetometer',
            arguments=['/magnetometer@sensor_msgs/msg/MagneticField@ignition.msgs.Magnetometer']
        ),
        Node(
            package='ros_gz_bridge',
            #namespace='ros_gz_bridge',
            executable='parameter_bridge',
            name='bridge_odometry',
            arguments=['/model/SousMarin/odometry@nav_msgs/msg/Odometry@ignition.msgs.Odometry']
        ),
        Node(
            package='ros_gz_bridge',
            #namespace='ros_gz_bridge',
            executable='parameter_bridge',
            name='bridge_odometry_w_cov',
            arguments=['/model/SousMarin/odometry_with_covariance@nav_msgs/msg/Odometry@ignition.msgs.OdometryWithCovariance']
        ),
        Node(
            package='ros_gz_bridge',
            #namespace='ros_gz_bridge',
            executable='parameter_bridge',
            name='bridge_gpu_lidar',
            arguments=['/gpu_lidar@sensor_msgs/msg/LaserScan@ignition.msgs.LaserScan']
        ),
        Node(
            package='ros_gz_bridge',
            #namespace='ros_gz_bridge',
            executable='parameter_bridge',
            name='bridge_gpu_lidar_points',
            arguments=['/gpu_lidar/points@sensor_msgs/msg/PointCloud2@ignition.msgs.PointCloudPacked']
        ),
        #Node(
        #    package='ros_gz_bridge',
        #    #namespace='ros_gz_bridge',
        #    executable='parameter_bridge',
        #    name='bridge_lidar_unid',
        #    arguments=['/lidar_unid@sensor_msgs/msg/LaserScan@ignition.msgs.LaserScan']
        #),
        #Node(
        #    package='ros_gz_bridge',
        #    #namespace='ros_gz_bridge',
        #    executable='parameter_bridge',
        #    name='bridge_lidar_unid_points',
        #    arguments=['/lidar_unid_points@sensor_msgs/msg/PointCloud2@ignition.msgs.PointCloudPacked']
        #),
        #Node(
        #    package='ros_gz_bridge',
        #    #namespace='ros_gz_bridge',
        #    executable='parameter_bridge',
        #    name='bridge_camera_under',
        #    arguments=['/camera_under@sensor_msgs/msg/Image@ignition.msgs.Image']
        #),
        Node(
            package='ros_gz_bridge',
            #namespace='ros_gz_bridge',
            executable='parameter_bridge',
            name='bridge_camera_front',
            arguments=['/camera_front@sensor_msgs/msg/Image@ignition.msgs.Image']
        ),
        #Actionneurs (pour le X3)
        Node(
            package='ros_gz_bridge',
            #namespace='ros_gz_bridge',
            executable='parameter_bridge',
            name='bridge_vel_mot1',
            arguments=['/model/SousMarin/joint/CaissonPrinc_Moteur1/cmd_thrust@std_msgs/msg/Float64@ignition.msgs.Double']
        ),
        Node(
            package='ros_gz_bridge',
            #namespace='ros_gz_bridge',
            executable='parameter_bridge',
            name='bridge_vel_mot2',
            arguments=['/model/SousMarin/joint/CaissonPrinc_Moteur2/cmd_thrust@std_msgs/msg/Float64@ignition.msgs.Double']
        ),
        Node(
            package='ros_gz_bridge',
            #namespace='ros_gz_bridge',
            executable='parameter_bridge',
            name='bridge_vel_mot3',
            arguments=['/model/SousMarin/joint/CaissonPrinc_Moteur3/cmd_thrust@std_msgs/msg/Float64@ignition.msgs.Double']
        ),
        Node(
            package='ros_gz_bridge',
            #namespace='ros_gz_bridge',
            executable='parameter_bridge',
            name='bridge_vel_mot4',
            arguments=['/model/SousMarin/joint/CaissonPrinc_Moteur4/cmd_thrust@std_msgs/msg/Float64@ignition.msgs.Double']
        ),
        Node(
            package='ros_gz_bridge',
            #namespace='ros_gz_bridge',
            executable='parameter_bridge',
            name='bridge_vel_motprof1',
            arguments=['/model/SousMarin/joint/CaissonPrinc_MoteurProf1/cmd_thrust@std_msgs/msg/Float64@ignition.msgs.Double']
        ),
        Node(
            package='ros_gz_bridge',
            #namespace='ros_gz_bridge',
            executable='parameter_bridge',
            name='bridge_vel_motprof2',
            arguments=['/model/SousMarin/joint/CaissonPrinc_MoteurProf2/cmd_thrust@std_msgs/msg/Float64@ignition.msgs.Double']
        ),
        Node(
            package='ros_gz_bridge',
            #namespace='ros_gz_bridge',
            executable='parameter_bridge',
            name='bridge_vel_motprof3',
            arguments=['/model/SousMarin/joint/CaissonPrinc_MoteurProf3/cmd_thrust@std_msgs/msg/Float64@ignition.msgs.Double']
        ),
        Node(
            package='ros_gz_bridge',
            #namespace='ros_gz_bridge',
            executable='parameter_bridge',
            name='bridge_vel_motprof4',
            arguments=['/model/SousMarin/joint/CaissonPrinc_MoteurProf4/cmd_thrust@std_msgs/msg/Float64@ignition.msgs.Double']
        )
    ])