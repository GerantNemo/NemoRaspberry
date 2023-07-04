import rclpy
from rclpy.node import Node

import time
from datetime import datetime
import numpy as np
from scipy.spatial.transform import Rotation as R

from std_msgs.msg import Float64
from std_msgs.msg import Int32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3

class Affichage():

    def __init__(self):

        self.t_odom = []
        self.t_cmd_pose = []
        self.t_cmd_vel = []
        self.t_cmd_F1 = []
        self.t_cmd_F2 = []
        self.t_cmd_F3 = []
        self.t_cmd_F4 = []
        self.t_cmd_F5 = []
        self.t_cmd_F6 = []
        self.t_cmd_F7 = []
        self.t_cmd_F8 = []

        self.mes_x = []
        self.mes_y = []
        self.mes_z = []

        self.mes_theta = []
        self.mes_phi = []
        self.mes_psi = []

        self.mes_vx = []
        self.mes_vy = []
        self.mes_vz = []

        self.mes_p = []
        self.mes_q = []
        self.mes_r = []

        self.cmd_x = []
        self.cmd_y = []
        self.cmd_z = []

        self.cmd_theta = []
        self.cmd_phi = []
        self.cmd_psi = []

        self.cmd_vx = []
        self.cmd_vy = []
        self.cmd_vz = []

        self.cmd_p = []
        self.cmd_q = []
        self.cmd_r = []

        self.F1 = []
        self.F2 = []
        self.F3 = []
        self.F4 = []
        self.F5 = []
        self.F6 = []
        self.F7 = []
        self.F8 = []

        self.enregistre = 0

        #self.file = '/home/aymeric/Documents/Fichiers_enregistres/enregistrement_gazebo' + str(datetime.now()) + ".csv"
        self.file = '/home/aymeric/Documents/Fichiers_enregistres/enregistrement_gazebo_2.csv'

    def register(self, node):
        
        taille_min = 1000
        
        tab_size = [self.t_odom, 
                    self.t_cmd_vel,
                    self.t_cmd_F1,
                    self.t_cmd_F2,
                    self.t_cmd_F3,
                    self.t_cmd_F4,
                    self.t_cmd_F5,
                    self.t_cmd_F6,
                    self.t_cmd_F7,
                    self.t_cmd_F8
                    ]
        
        taille_bonne = 1

        #node.get_logger().info('t_odom : %lf' %len(tab_size[0]))
        #node.get_logger().info('t_cmd_vel : %lf' %len(tab_size[1]))
        #node.get_logger().info('t_cmd_F1 : %lf' %len(tab_size[2]))
        #node.get_logger().info('t_cmd_F2 : %lf' %len(tab_size[3]))
        #node.get_logger().info('t_cmd_F3 : %lf' %len(tab_size[4]))
        #node.get_logger().info('t_cmd_F4 : %lf' %len(tab_size[5]))
        #node.get_logger().info('t_cmd_F5 : %lf' %len(tab_size[6]))
        #node.get_logger().info('t_cmd_F6 : %lf' %len(tab_size[7]))
        #node.get_logger().info('t_cmd_F7 : %lf' %len(tab_size[8]))
        #node.get_logger().info('t_cmd_F8 : %lf' %len(tab_size[9]))

        #Verification : tous les tableaux doivent avoir une longueur minimale (pour éviter les erreurs)
        for t in tab_size:
            if len(t) < taille_min + 20:
                taille_bonne = 0

        if self.enregistre == 1:
            node.get_logger().info('Fait')

        if taille_bonne == 1 and self.enregistre == 0:
            
            #node.get_logger().info('Condition remplie')

            #node.get_logger().info('t_odom : %lf' %tab_size[0][0])

            t_odom = np.array(self.t_odom)
            t_cmd_vel = np.array(self.t_cmd_vel)
            t_cmd_F1  = np.array(self.t_cmd_F1)
            t_cmd_F2  = np.array(self.t_cmd_F2)
            t_cmd_F3  = np.array(self.t_cmd_F3)
            t_cmd_F4  = np.array(self.t_cmd_F4)
            t_cmd_F5  = np.array(self.t_cmd_F5)
            t_cmd_F6  = np.array(self.t_cmd_F6)
            t_cmd_F7  = np.array(self.t_cmd_F7)
            t_cmd_F8  = np.array(self.t_cmd_F8)

            #node.get_logger().info('Check 1')

            mes_x = np.array(self.mes_x)
            mes_y = np.array(self.mes_y)
            mes_z = np.array(self.mes_z)

            #node.get_logger().info('Check 2')

            mes_theta = np.array(self.mes_theta)
            mes_phi = np.array(self.mes_phi)
            mes_psi = np.array(self.mes_psi)

            #node.get_logger().info('Check 3')

            mes_vx = np.array(self.mes_vx)
            mes_vy = np.array(self.mes_vy)
            mes_vz = np.array(self.mes_vz)

            #node.get_logger().info('Check 4')

            mes_p = np.array(self.mes_p)
            mes_q = np.array(self.mes_q)
            mes_r = np.array(self.mes_r)

            #node.get_logger().info('Check 5')

            #cmd_x = np.array(self.cmd_x)
            #cmd_y = np.array(self.cmd_y)
            #cmd_z = np.array(self.cmd_z)

            cmd_theta = np.array(self.cmd_theta)
            cmd_phi = np.array(self.cmd_phi)
            cmd_psi = np.array(self.cmd_psi)

            #node.get_logger().info('Check 6')

            cmd_vx = np.array(self.cmd_vx)
            cmd_vy = np.array(self.cmd_vy)
            cmd_vz = np.array(self.cmd_vz)

            #node.get_logger().info('Check 7')

            cmd_p = np.array(self.cmd_p)
            cmd_q = np.array(self.cmd_q)
            cmd_r = np.array(self.cmd_r)

            #node.get_logger().info('Check 8')

            F1 = np.array(self.F1)
            F2 = np.array(self.F2)
            F3 = np.array(self.F3)
            F4 = np.array(self.F4)
            F5 = np.array(self.F5)
            F6 = np.array(self.F6)
            F7 = np.array(self.F7)
            F8 = np.array(self.F8)

            #node.get_logger().info('Check 9')

            tab = [t_odom[:taille_min],
                    t_cmd_vel[:taille_min], 
                    t_cmd_F1[:taille_min],
                    t_cmd_F2[:taille_min], 
                    t_cmd_F3[:taille_min], 
                    t_cmd_F4[:taille_min],
                    t_cmd_F5[:taille_min], 
                    t_cmd_F6[:taille_min], 
                    t_cmd_F7[:taille_min], 
                    t_cmd_F8[:taille_min], 
                    mes_x[:taille_min], 
                    mes_y[:taille_min], 
                    mes_z[:taille_min],
                    ###cmd_x[:taille_min], 
                    ###cmd_y[:taille_min], 
                    ###cmd_z[:taille_min],
                    mes_theta[:taille_min], 
                    mes_phi[:taille_min], 
                    mes_psi[:taille_min], 
                    #cmd_theta[:taille_min], 
                    #cmd_phi[:taille_min], 
                    #cmd_psi[:taille_min], 
                    mes_vx[:taille_min], 
                    mes_vy[:taille_min], 
                    mes_vz[:taille_min], 
                    cmd_vx[:taille_min], 
                    cmd_vy[:taille_min], 
                    cmd_vz[:taille_min],
                    mes_p[:taille_min], 
                    mes_q[:taille_min], 
                    mes_r[:taille_min], 
                    cmd_p[:taille_min], 
                    cmd_q[:taille_min], 
                    cmd_r[:taille_min],
                    F1[:taille_min],
                    F2[:taille_min],
                    F3[:taille_min],
                    F4[:taille_min],
                    F5[:taille_min],
                    F6[:taille_min],
                    F7[:taille_min],
                    F8[:taille_min]
                   ]
            
            np.savetxt(self.file, tab, delimiter=",")

            self.enregistre = 1


class Enregistrement(Node):

    def __init__(self):
        super().__init__('affichage')
        
        self.pub_odom = self.create_publisher(Int32, '/verif_odom', 10)
        self.pub_vel = self.create_publisher(Int32, '/verif_vel', 10)
        self.pub_F1 = self.create_publisher(Int32, '/verif_F1', 10)
        self.pub_F2 = self.create_publisher(Int32, '/verif_F2', 10)
        self.pub_F3 = self.create_publisher(Int32, '/verif_F3', 10)
        self.pub_F4 = self.create_publisher(Int32, '/verif_F4', 10)
        self.pub_F5 = self.create_publisher(Int32, '/verif_F5', 10)
        self.pub_F6 = self.create_publisher(Int32, '/verif_F6', 10)
        self.pub_F7 = self.create_publisher(Int32, '/verif_F7', 10)
        self.pub_F8 = self.create_publisher(Int32, '/verif_F8', 10)

        self.sub_odometry = self.create_subscription(Odometry, '/model/SousMarin/odometry', self.enregistrement_odometry, 10)
        self.sub_cmd_vel = self.create_subscription(Odometry, '/SousMarin/command/pos_vel', self.enregistrement_cmd_vel, 10)

        self.sub_cmd_F1 = self.create_subscription(Float64, '/model/SousMarin/joint/CaissonPrinc_Moteur1/cmd_thrust', self.enregistrement_F1, 10)
        self.sub_cmd_F2 = self.create_subscription(Float64, '/model/SousMarin/joint/CaissonPrinc_Moteur2/cmd_thrust', self.enregistrement_F2, 10)
        self.sub_cmd_F3 = self.create_subscription(Float64, '/model/SousMarin/joint/CaissonPrinc_Moteur3/cmd_thrust', self.enregistrement_F3, 10)
        self.sub_cmd_F4 = self.create_subscription(Float64, '/model/SousMarin/joint/CaissonPrinc_Moteur4/cmd_thrust', self.enregistrement_F4, 10)
        self.sub_cmd_F5 = self.create_subscription(Float64, '/model/SousMarin/joint/CaissonPrinc_MoteurProf1/cmd_thrust', self.enregistrement_F5, 10)
        self.sub_cmd_F6 = self.create_subscription(Float64, '/model/SousMarin/joint/CaissonPrinc_MoteurProf2/cmd_thrust', self.enregistrement_F6, 10)
        self.sub_cmd_F7 = self.create_subscription(Float64, '/model/SousMarin/joint/CaissonPrinc_MoteurProf3/cmd_thrust', self.enregistrement_F7, 10)
        self.sub_cmd_F8 = self.create_subscription(Float64, '/model/SousMarin/joint/CaissonPrinc_MoteurProf4/cmd_thrust', self.enregistrement_F8, 10)

        #self.subscription  # prevent unused variable warning

        self.affichage = Affichage()

        #self.time_init = self.get_clock().now()
        self.time_init = datetime.now()

        self.t = 0


    def retour(self):
        return self.affichage
    
    def enregistrement_odometry(self, msg):
        
        t_1 = datetime.now() - self.time_init
        self.t = t_1.seconds + t_1.microseconds/1000000

        self.affichage.t_odom.append(self.t)

        self.affichage.mes_x.append(msg.pose.pose.position.x)
        self.affichage.mes_y.append(msg.pose.pose.position.y)
        self.affichage.mes_z.append(msg.pose.pose.position.z)

        r = R.from_quat([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        angles = r.as_euler('xyz')
        self.affichage.mes_theta.append(angles[0])
        self.affichage.mes_phi.append(angles[1])
        self.affichage.mes_psi.append(angles[2])

        self.affichage.mes_vx.append(msg.twist.twist.linear.x)
        self.affichage.mes_vy.append(msg.twist.twist.linear.y)
        self.affichage.mes_vz.append(msg.twist.twist.linear.z)

        self.affichage.mes_p.append(msg.twist.twist.angular.x)
        self.affichage.mes_q.append(msg.twist.twist.angular.y)
        self.affichage.mes_r.append(msg.twist.twist.angular.z)

        #self.get_logger().info('mes_x : %lf' %self.affichage.mes_x)
        #self.get_logger().info('mes_theta : %lf' %self.affichage.mes_theta)
        #self.get_logger().info('mes_vx : %lf' %self.affichage.mes_vx)
        #self.get_logger().info('mes_p : %lf' %self.affichage.mes_p)

        pub = Int32()
        pub.data = len(self.affichage.mes_x)
        self.pub_odom.publish(pub)

        self.affichage.register(self)

    def enregistrement_cmd_vel(self, msg):

        t_1 = datetime.now() - self.time_init
        self.t = t_1.seconds + t_1.microseconds/1000000
        
        self.affichage.t_cmd_vel.append(self.t)

        self.affichage.cmd_vx.append(msg.twist.twist.linear.x)
        self.affichage.cmd_vy.append(msg.twist.twist.linear.y)
        self.affichage.cmd_vz.append(msg.twist.twist.linear.z)

        self.affichage.cmd_p.append(msg.twist.twist.angular.x)
        self.affichage.cmd_q.append(msg.twist.twist.angular.y)
        self.affichage.cmd_r.append(msg.twist.twist.angular.z)

        pub = Int32()
        pub.data = len(self.affichage.cmd_vx)
        self.pub_vel.publish(pub)

    def enregistrement_F1(self, msg):

        t_1 = datetime.now() - self.time_init
        self.t = t_1.seconds + t_1.microseconds/1000000
        
        self.affichage.t_cmd_F1.append(self.t)

        self.affichage.F1.append(msg.data)

        pub = Int32()
        pub.data = len(self.affichage.F1)
        self.pub_F1.publish(pub)

    def enregistrement_F2(self, msg):

        t_1 = datetime.now() - self.time_init
        self.t = t_1.seconds + t_1.microseconds/1000000
        
        self.affichage.t_cmd_F2.append(self.t)

        self.affichage.F2.append(msg.data)

        pub = Int32()
        pub.data = len(self.affichage.F2)
        self.pub_F2.publish(pub)

    def enregistrement_F3(self, msg):

        t_1 = datetime.now() - self.time_init
        self.t = t_1.seconds + t_1.microseconds/1000000
        
        self.affichage.t_cmd_F3.append(self.t)

        self.affichage.F3.append(msg.data)

        pub = Int32()
        pub.data = len(self.affichage.F3)
        self.pub_F3.publish(pub)

    def enregistrement_F4(self, msg):

        t_1 = datetime.now() - self.time_init
        self.t = t_1.seconds + t_1.microseconds/1000000
        
        self.affichage.t_cmd_F4.append(self.t)

        self.affichage.F4.append(msg.data)

        pub = Int32()
        pub.data = len(self.affichage.F4)
        self.pub_F4.publish(pub)

    def enregistrement_F5(self, msg):

        t_1 = datetime.now() - self.time_init
        self.t = t_1.seconds + t_1.microseconds/1000000
        
        self.affichage.t_cmd_F5.append(self.t)

        self.affichage.F5.append(msg.data)

        pub = Int32()
        pub.data = len(self.affichage.F5)
        self.pub_F5.publish(pub)

    def enregistrement_F6(self, msg):

        t_1 = datetime.now() - self.time_init
        self.t = t_1.seconds + t_1.microseconds/1000000
        
        self.affichage.t_cmd_F6.append(self.t)

        self.affichage.F6.append(msg.data)

        pub = Int32()
        pub.data = len(self.affichage.F6)
        self.pub_F6.publish(pub)

    def enregistrement_F7(self, msg):

        t_1 = datetime.now() - self.time_init
        self.t = t_1.seconds + t_1.microseconds/1000000
        
        self.affichage.t_cmd_F7.append(self.t)

        self.affichage.F7.append(msg.data)

        pub = Int32()
        pub.data = len(self.affichage.F7)
        self.pub_F7.publish(pub)

    def enregistrement_F8(self, msg):

        t_1 = datetime.now() - self.time_init
        self.t = t_1.seconds + t_1.microseconds/1000000
        
        self.affichage.t_cmd_F8.append(self.t)

        self.affichage.F8.append(msg.data)

        pub = Int32()
        pub.data = len(self.affichage.F8)
        self.pub_F8.publish(pub)


def main(args=None):

    rclpy.init(args=args)

    enregistrement = Enregistrement()

    try:
        rclpy.spin(enregistrement)
    except KeyboardInterrupt:
        pass
    
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    
    enregistrement.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()