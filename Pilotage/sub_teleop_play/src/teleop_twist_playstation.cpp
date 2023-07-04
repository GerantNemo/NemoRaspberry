#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/int32.hpp"
#include <sensor_msgs/msg/joy.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include "nav_msgs/msg/odometry.hpp"

using namespace std::chrono_literals;

/* This example creates a subclass of Node and uses std::bind() to register a
* member function as a callback from the timer. */

class Teleop_Playstation : public rclcpp::Node
{
  public:
    Teleop_Playstation()
    : Node("teleop_twist_playstation"), count_(0)
    {
      joy_sub_ = this->create_subscription<sensor_msgs::msg::Joy>(
        "joy", rclcpp::SensorDataQoS(rclcpp::KeepLast(1)),
        std::bind(&Teleop_Playstation::JoyCallback, this, std::placeholders::_1));
      
      cmd_vel_twist_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/SousMarin/command/vel", 10);
      cmd_vel_odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/SousMarin/command/pos_vel", 10);
      cmd_vel_mode_pub_ = this->create_publisher<std_msgs::msg::Int32>("/SousMarin/command/mode", 10);

      timer_ = this->create_wall_timer(100ms, std::bind(&Teleop_Playstation::timer_callback, this));

      initialisation();
    }

  private:
    void initialisation()
    {
        //Declaration de parametres

        this->declare_parameter("axes.HG", 0);
        this->declare_parameter("axes.VG", 1);
        this->declare_parameter("axes.HD", 2);
        this->declare_parameter("axes.VD", 3);
        this->declare_parameter("axes.L2", 4);
        this->declare_parameter("axes.R2", 5);

        this->declare_parameter("buttons.croix", 0);
        this->declare_parameter("buttons.rond", 1);
        this->declare_parameter("buttons.carre", 2);
        this->declare_parameter("buttons.triangle", 3);
        this->declare_parameter("buttons.playstation", 5);
        this->declare_parameter("buttons.options", 6);
        this->declare_parameter("buttons.joy_G", 7);
        this->declare_parameter("buttons.L1", 9);
        this->declare_parameter("buttons.R1", 10);
        this->declare_parameter("buttons.Haut", 11);
        this->declare_parameter("buttons.Bas", 12);
        this->declare_parameter("buttons.Gauche", 13);
        this->declare_parameter("buttons.Droite", 14);
        this->declare_parameter("buttons.Centre", 15);

        this->declare_parameter("v_max.vx_max", 10.0);
        this->declare_parameter("v_max.vy_max", 10.0);
        this->declare_parameter("v_max.vz_max", 10.0);
        this->declare_parameter("v_max.psi_max", 10.0);

        //Recuperation des parametres

        this->get_parameter("axes.HG", HG);
        this->get_parameter("axes.VG", VG);
        this->get_parameter("axes.HD", HD);
        this->get_parameter("axes.VD", VD);
        this->get_parameter("axes.L2", L2);
        this->get_parameter("axes.R2", R2);

        this->get_parameter("buttons.croix", croix);
        this->get_parameter("buttons.rond", rond);
        this->get_parameter("buttons.carre", carre);
        this->get_parameter("buttons.triangle", triangle);
        this->get_parameter("buttons.playstation", playstation);
        this->get_parameter("buttons.options", options);
        this->get_parameter("buttons.joy_G", joy_G);
        this->get_parameter("buttons.L1", L1);
        this->get_parameter("buttons.R1", R1);
        this->get_parameter("buttons.Haut", Haut);
        this->get_parameter("buttons.Bas", Bas);
        this->get_parameter("buttons.Gauche", Gauche);
        this->get_parameter("buttons.Droite", Droite);
        this->get_parameter("buttons.Centre", Centre);

        this->get_parameter("v_max.vx_max", vx_max);
        this->get_parameter("v_max.vy_max", vy_max);
        this->get_parameter("v_max.vz_max", vz_max);
        this->get_parameter("v_max.psi_max", psi_max);

        recu = 0;
    }

    void timer_callback()
    {
      
      if(recu == 1)
      {  
        cmd_vel_mode.data = 1;
        
        cmd_vel.linear.x = 0.0;
        cmd_vel.linear.y = 0.0;
        cmd_vel.linear.z = 0.0;

        cmd_vel.angular.x = 0.0;
        cmd_vel.angular.y = 0.0;
        cmd_vel.angular.z = 0.0;

        cmd_vel.linear.z = vz_max * current_joy.axes.at(VG);
        cmd_vel.angular.z = psi_max * current_joy.axes.at(HG);
        cmd_vel.linear.x = vx_max * current_joy.axes.at(VD);
        cmd_vel.linear.y = vy_max * current_joy.axes.at(HD);

        if(abs(cmd_vel.linear.z) < 0.05)
        {
          cmd_vel.linear.z = vz_max * current_joy.buttons.at(Haut);
          if(abs(cmd_vel.linear.z) < 0.05)
          {
              cmd_vel.linear.z = -vz_max * current_joy.buttons.at(Bas);
          }
        }

        if(abs(cmd_vel.angular.z) < 0.05)
        {
          cmd_vel.angular.z = psi_max * current_joy.buttons.at(Gauche);
          if(abs(cmd_vel.angular.z) < 0.05)
          {
              cmd_vel.angular.z = -psi_max * current_joy.buttons.at(Droite);
          }
        }

        if(abs(cmd_vel.linear.x) < 0.05)
        {
          cmd_vel.linear.x = vx_max * current_joy.buttons.at(triangle);
          if(abs(cmd_vel.linear.x) < 0.05)
          {
              cmd_vel.linear.x = -vx_max * current_joy.buttons.at(croix);
          }
        }

        if(abs(cmd_vel.linear.y) < 0.05)
        {
          cmd_vel.linear.y = vy_max * current_joy.buttons.at(carre);
          if(abs(cmd_vel.linear.y) < 0.05)
          {
              cmd_vel.linear.y = -vy_max * current_joy.buttons.at(rond);
          }
        }

        cmd_vel_odom.twist.twist.linear.x = cmd_vel.linear.x;
        cmd_vel_odom.twist.twist.linear.y = cmd_vel.linear.y;
        cmd_vel_odom.twist.twist.linear.z = cmd_vel.linear.z;
        cmd_vel_odom.twist.twist.angular.x = cmd_vel.angular.x;
        cmd_vel_odom.twist.twist.angular.y = cmd_vel.angular.y;
        cmd_vel_odom.twist.twist.angular.z = cmd_vel.angular.z;

        //RCLCPP_INFO(this->get_logger(), "cmd_vel_x = %.2f, cmd_vel_y = %.2f, cmd_vel_z = %.2f", cmd_vel.linear.x, cmd_vel.linear.y, cmd_vel.linear.z);
        //RCLCPP_INFO(this->get_logger(), "cmd_vel_rot_x = %.2f, cmd_vel_rot_y = %.2f, cmd_vel_rot_z = %.2f", cmd_vel.angular.x, cmd_vel.angular.y, cmd_vel.angular.z);

        cmd_vel_twist_pub_->publish(cmd_vel);
        cmd_vel_odom_pub_->publish(cmd_vel_odom);
        cmd_vel_mode_pub_->publish(cmd_vel_mode);
      }
    }

    void JoyCallback(const sensor_msgs::msg::Joy& joy_msg)
    {
        current_joy = joy_msg;
        recu = 1;
    }

    //Subscribers and publishers
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_twist_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr cmd_vel_odom_pub_;
    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr cmd_vel_mode_pub_;
    
    //Messages
    sensor_msgs::msg::Joy current_joy;
    geometry_msgs::msg::Twist cmd_vel;
    nav_msgs::msg::Odometry cmd_vel_odom;
    std_msgs::msg::Int32 cmd_vel_mode;

    size_t count_;
    
    //Parametres
    int HG, VG, HD, VD;
    int L1, L2, R1, R2;
    int croix, rond, carre, triangle;
    int Haut, Bas, Gauche, Droite;
    int Centre;
    int playstation, options;
    int joy_G;

    int recu;

    float vx_max, vy_max, vz_max, psi_max;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Teleop_Playstation>());
  rclcpp::shutdown();
  return 0;
}