//#include "messages_sim/imu_publisher.h"

//#include "quad_control/parameters_ros.h"

#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "std_msgs/msg/string.hpp"

class Odometry_Publisher : public rclcpp::Node
{

    public:
        Odometry_Publisher() : Node("odometry_publisher"), count_(0)
        {
            //Publisher
            publisher_ = this->create_publisher<nav_msgs::msg::Odometry>("sub/odometry", 10);
            timer_ = this->create_wall_timer(std::chrono::milliseconds(100), std::bind(&Odometry_Publisher::timer_callback, this));
        }

    private:
        void timer_callback()
        {
            //auto message = std_msgs::msg::String();
            auto message = nav_msgs::msg::Odometry();

            message.pose.pose.position.x = 0;
            message.pose.pose.position.y = 0;
            message.pose.pose.position.z = 0;

            message.pose.pose.orientation.x = 0;
            message.pose.pose.orientation.y = 0;
            message.pose.pose.orientation.z = 0;
            message.pose.pose.orientation.w = 1;

            message.twist.twist.linear.x = 0;
            message.twist.twist.linear.y = 0;
            message.twist.twist.linear.z = 0;

            message.twist.twist.angular.x = 0;
            message.twist.twist.angular.y = 0;
            message.twist.twist.angular.z = 0;

            //message.data = "Hello, world! " + std::to_string(count_++);
    
            //RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
    
            publisher_->publish(message);
        }

        rclcpp::TimerBase::SharedPtr timer_;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr publisher_;
        size_t count_;

};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Odometry_Publisher>());
  rclcpp::shutdown();
  return 0;
}