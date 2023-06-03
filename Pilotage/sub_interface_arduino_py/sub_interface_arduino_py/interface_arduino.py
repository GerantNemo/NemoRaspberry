import rclpy
from rclpy.node import Node
import serial

from std_msgs.msg import String
from trajectory_msgs import JointTrajectoryPoint


class InterfaceArduino(Node):

    def __init__(self):

        super().__init__('interface_arduino')
        self.subscription = self.create_subscription(JointTrajectoryPoint,'cmd/motors',self.listener_callback,10)
        self.subscription  # prevent unused variable warning

        self.ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
        self.ser.reset_input_buffer()

    def listener_callback(self, msg):
        
        header = "M"
        data = ""
        
        for i in range(8):

            cmds_str = "<" + header + "," + str(i) + "," + str(msg.velocities[i]) + ">"
            envoi = 0
            while envoi == 0:
                if self.ser.in_waiting > 0:
                    self.ser.write(cmds_str.encode('utf-8'))
                    envoi = 1

def main(args=None):
    rclpy.init(args=args)

    interface_arduino = InterfaceArduino()

    rclpy.spin(interface_arduino)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    interface_arduino.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()