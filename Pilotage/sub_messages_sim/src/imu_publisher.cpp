//#include "messages_sim/imu_publisher.h"

//#include "quad_control/parameters_ros.h"

#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "std_msgs/msg/string.hpp"

class IMU_Publisher : public rclcpp::Node
{

    public:
        IMU_Publisher() : Node("imu_publisher"), count_(0)
        {
            //Publisher
            publisher_ = this->create_publisher<sensor_msgs::msg::Imu>("quad/imu", 10);
            timer_ = this->create_wall_timer(std::chrono::milliseconds(100), std::bind(&IMU_Publisher::timer_callback, this));
        }

    private:
        void timer_callback()
        {
            //auto message = std_msgs::msg::String();
            auto message = sensor_msgs::msg::Imu();

            message.orientation.x = 0;
            message.orientation.y = 0;
            message.orientation.z = 0;
            message.orientation.w = 1;

            message.angular_velocity.x = 0;
            message.angular_velocity.y = 0;
            message.angular_velocity.z = 0;

            message.linear_acceleration.x = 0;
            message.linear_acceleration.y = 0;
            message.linear_acceleration.z = 0;

            //message.data = "Hello, world! " + std::to_string(count_++);
    
            //RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
    
            publisher_->publish(message);
        }

        rclcpp::TimerBase::SharedPtr timer_;
        rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr publisher_;
        size_t count_;

};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<IMU_Publisher>());
  rclcpp::shutdown();
  return 0;
}