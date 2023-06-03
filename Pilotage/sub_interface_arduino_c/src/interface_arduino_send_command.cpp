#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "Serial.hpp"
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

using namespace std::chrono_literals;

/* This example creates a subclass of Node and uses std::bind() to register a
* member function as a callback from the timer. */

class InterfaceArduino : public rclcpp::Node
{
  public:
    InterfaceArduino() : Node("interface_arduino"), count_(0)
    {
      cmd_velocity_sub_ = this->create_subscription<trajectory_msgs::msg::JointTrajectoryPoint>("cmd/velocity", 10, std::bind(&InterfaceArduino::SendCommandCallback, this, _1));
      
      //publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
      //timer_ = this->create_wall_timer(500ms, std::bind(&MinimalPublisher::timer_callback, this));

      Initialisation();
    }

    void Initialisation()
    {
        this->get_parameter("Port", port_name);
        this->get_parameter("Baud_Rate", baud_rate);
    
        if(!serial.open(port_name, baud_rate)){
            RCLCPP_ERROR(this->get_logger(), "couldn't open serial port, \"%s\"", port_name.c_str());
        }
    }

  private:

    void SendCommandCallback(const trajectory_msgs::msg::JointTrajectoryPoint traj_msg)
    {
        union{
	        double myDouble;
	        unsigned char myChars[sizeof(double)];
	    } data;
        
        /*std::ostringstream ostr;
        ostr << "Send data ";
        ostr << "[ ";
        for(int i = 0; i < static_cast<int>(data->data.size()); i++){
            if(i != 0) ostr << ", ";
            ostr << std::to_string(data->data[i]);
        }
        ostr << "]";
        RCLCPP_WARN(this->get_logger(), ostr.str());*/

        serial.write("A");
        for(int i=0; i<sizeof(traj_msg.velocities); i++)
        {
            data.myDouble = traj_msg.velocities[i];
            for(int k = 0; k < sizeof(double); k++ )
            {
                serial.write(data.myChars[k]);
            }
        }
    }

    /*void timer_callback()
    {
      auto message = std_msgs::msg::String();
      message.data = "Hello, world! " + std::to_string(count_++);
      RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
      publisher_->publish(message);
    }*/
    
   
    
    rclcpp::Subscription<trajectory_msgs::msg::JointTrajectoryPoint>::SharedPtr cmd_odometry_sub_;
     
    size_t count_;
    Serial serial; //Serial port instance
    std::string port_name;
    unsigned int baud_rate;
    unsigned char serial_byte;
    
    //rclcpp::TimerBase::SharedPtr timer_;
    //rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<InterfaceArduino>());
  rclcpp::shutdown();
  return 0;
}