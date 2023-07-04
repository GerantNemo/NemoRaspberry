#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include <tf2_ros/transform_broadcaster.h>
#include <tf2/transform_datatypes.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int32.hpp"
#include "std_msgs/msg/string.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose.hpp>

using namespace std::chrono_literals;
using std::placeholders::_1;

//rate time : 50Hz

/* This example creates a subclass of Node and uses std::bind() to register a
* member function as a callback from the timer. */

class PoseControllerNode : public rclcpp::Node
{
  public:
    PoseControllerNode() : Node("pose_controller_node"), count_(0)
    {
        //0 : cmdes de position ; 1 : cmdes de vitesse
        cmd_mode_sub_ = this->create_subscription<std_msgs::msg::Int32>("/SousMarin/command/mode", 10, std::bind(&PoseControllerNode::PoseModeCallback, this, _1));
        cmd_position_sub_ = this->create_subscription<geometry_msgs::msg::Pose>("/SousMarin/command/pos", 10, std::bind(&PoseControllerNode::PoseControlCallback, this, _1));
        mes_odometry_sub_ = this->create_subscription<nav_msgs::msg::Odometry>("/model/SousMarin/odometry", 10, std::bind(&PoseControllerNode::PoseOdometryCallback, this, _1));
    
        timer_ = this->create_wall_timer(20ms, std::bind(&PoseControllerNode::timer_callback, this));
        cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/SousMarin/command/pos_vel", 10);

        InitializeParams();
    }

    void InitializeParams()
    {
        last_time = rclcpp::Clock{RCL_ROS_TIME}.now();

        //Declaration de parametres

        this->declare_parameter("x_PID.P", 1.0);
        this->declare_parameter("x_PID.I", 0.1);
        this->declare_parameter("x_PID.I_max", 1.0);
        this->declare_parameter("x_PID.D", 0.1);

        this->declare_parameter("y_PID.P", 1.0);
        this->declare_parameter("y_PID.I", 0.1);
        this->declare_parameter("y_PID.I_max", 1.0);
        this->declare_parameter("y_PID.D", 0.1);

        this->declare_parameter("z_PID.P", 1.0);
        this->declare_parameter("z_PID.I", 0.1);
        this->declare_parameter("z_PID.I_max", 1.0);
        this->declare_parameter("z_PID.D", 0.1);

        this->declare_parameter("roll_PID.P", 1.0);
        this->declare_parameter("roll_PID.I", 0.1);
        this->declare_parameter("roll_PID.I_max", 1.0);
        this->declare_parameter("roll_PID.D", 0.1);

        this->declare_parameter("pitch_PID.P", 1.0);
        this->declare_parameter("pitch_PID.I", 0.1);
        this->declare_parameter("pitch_PID.I_max", 0.1);
        this->declare_parameter("pitch_PID.D", 1.0);

        this->declare_parameter("yaw_PID.P", 1.0);
        this->declare_parameter("yaw_PID.I", 0.1);
        this->declare_parameter("yaw_PID.I_max", 1.0);
        this->declare_parameter("yaw_PID.D", 0.1);

        this->declare_parameter("vx_max", 5.0);
        this->declare_parameter("vy_max", 5.0);
        this->declare_parameter("vz_max", 5.0);

        this->declare_parameter("p_max", 1.57);
        this->declare_parameter("q_max", 1.57);
        this->declare_parameter("r_max", 1.57);

        //Recuperation des parametres
        
        this->get_parameter("x_PID.P", x_KP);
        this->get_parameter("x_PID.I", x_KI);
        this->get_parameter("x_PID.I_max", x_KI_max);
        this->get_parameter("x_PID.D", x_KD);

        this->get_parameter("y_PID.P", y_KP);
        this->get_parameter("y_PID.I", y_KI);
        this->get_parameter("y_PID.I_max", y_KI_max);
        this->get_parameter("y_PID.D", y_KD);

        this->get_parameter("z_PID.P", z_KP);
        this->get_parameter("z_PID.I", z_KI);
        this->get_parameter("z_PID.I_max", z_KI_max);
        this->get_parameter("z_PID.D", z_KD);

        this->get_parameter("roll_PID.P", roll_KP);
        this->get_parameter("roll_PID.I", roll_KI);
        this->get_parameter("roll_PID.I_max", roll_KI_max);
        this->get_parameter("roll_PID.D", roll_KD);

        this->get_parameter("pitch_PID.P", pitch_KP);
        this->get_parameter("pitch_PID.I", pitch_KI);
        this->get_parameter("pitch_PID.I_max", pitch_KI_max);
        this->get_parameter("pitch_PID.D", pitch_KD);

        this->get_parameter("yaw_PID.P", yaw_KP);
        this->get_parameter("yaw_PID.I", yaw_KI);
        this->get_parameter("yaw_PID.I_max", yaw_KI_max);
        this->get_parameter("yaw_PID.D", yaw_KD);

        this->get_parameter("vx_max", vx_max);
        this->get_parameter("vy_max", vy_max);
        this->get_parameter("vz_max", vz_max);

        this->get_parameter("p_max", p_max);
        this->get_parameter("q_max", q_max);
        this->get_parameter("r_max", r_max);

        x_er = 0;
        y_er = 0;
        z_er = 0;
        x_er_sum = 0;
        y_er_sum = 0;
        z_er_sum = 0;
        last_x = 0;
        last_y = 0;
        last_z = 0;
        vx_des = 0;
        vy_des = 0;
        vz_des = 0;

        roll_er = 0;
        pitch_er = 0;
        yaw_er = 0;
        roll_er_sum = 0;
        pitch_er_sum = 0;
        yaw_er_sum = 0;
        last_roll = 0;
        last_pitch = 0;
        last_yaw = 0;
        p_des = 0;
        q_des = 0;
        r_des = 0;
    }

    void CalculatePoseControl()
    {
        // Get simulator time
        sim_time = rclcpp::Clock{RCL_ROS_TIME}.now();
        dt = sim_time.seconds() - last_time.seconds(); //dt = (sim_time - last_time).toSec;
        last_time = sim_time;
        if (dt == 0.0) return;
        
        mode = cmd_mode.data;

        //RCLCPP_INFO(this->get_logger(), "pos_mode = %lf", mode);
        
        //Get position data
        mes_x = current_odometry.pose.pose.position.x;
        mes_y = current_odometry.pose.pose.position.y;
        mes_z = current_odometry.pose.pose.position.z;
        
        //Convert quaternion to Euler angles
        //tf2::quaternionMsgToTF(current_odometry.pose.pose.orientation, q);
        tf2::fromMsg(current_odometry.pose.pose.orientation, q);
        tf2::Matrix3x3(q).getRPY(mes_roll, mes_pitch, mes_yaw);

        cmd_x = cmd_pose.position.x;
        cmd_y = cmd_pose.position.y;
        cmd_z = cmd_pose.position.z;
        
        //tf2::quaternionMsgToTF(cmd_pose.orientation, q);
        tf2::fromMsg(cmd_pose.orientation, q);
        tf2::Matrix3x3(q).getRPY(cmd_roll, cmd_pitch, cmd_yaw);


        if(mode==0)
        {
          //X PID
          x_er = cmd_x - mes_x;
          if(abs(x_er) < x_KI_max){
          	x_er_sum = x_er_sum + x_er;
          }  
          cp = x_er * x_KP;
          ci = x_KI * dt * x_er_sum;
          cd = x_KD * (mes_x - last_x)/dt;
          vx_des = (cp + ci +cd);
          vx_des = limit(vx_des, (-1)*vx_max, vx_max);
          last_x = mes_x;

          //Y PID
          y_er = cmd_y - mes_y;
          if(abs(y_er) < y_KI_max){
          	y_er_sum = y_er_sum + y_er;
          }  
          cp = y_er * y_KP;
          ci = y_KI * dt * y_er_sum;
          cd = y_KD * (mes_y - last_y)/dt;
          vy_des = (cp + ci +cd);
          vy_des = limit(vy_des, (-1)*vy_max, vy_max);
          last_y = mes_y;

          //Z PID
          z_er = cmd_z - mes_z;
          if(abs(z_er) < z_KI_max){
          	z_er_sum = z_er_sum + z_er;
          }  
          cp = z_er * z_KP;
          ci = z_KI * dt * z_er_sum;
          cd = z_KD * (mes_z - last_z)/dt;
          vz_des = (cp + ci +cd);
          vz_des = limit(vz_des, (-1)*vz_max, vz_max);
          last_z = mes_z;

          //Roll PID
          roll_er = cmd_roll - mes_roll;
          if(abs(roll_er) < roll_KI_max){
          	roll_er_sum = roll_er_sum + roll_er;
          }  
          cp = roll_er * roll_KP;
          ci = roll_KI * dt * roll_er_sum;
          cd = roll_KD * (mes_roll - last_roll)/dt;
          p_des = (cp + ci + cd);
          p_des = limit(p_des, (-1)*p_max, p_max);
          last_roll = mes_roll;
        
          //Pitch PID
          pitch_er = cmd_pitch - mes_pitch;
          if(abs(pitch_er) < pitch_KI_max){
          	pitch_er_sum = pitch_er_sum + pitch_er;
          }  
          cp = pitch_er * pitch_KP;
          ci = pitch_KI * dt * pitch_er_sum;
          cd = pitch_KD * (mes_pitch - last_pitch)/dt;
          q_des = (cp + ci + cd);
          q_des = limit(q_des, (-1)*q_max, q_max);
          last_pitch = mes_pitch;
        
          //Yaw PID
          yaw_er = cmd_yaw - mes_yaw;
          if(abs(yaw_er) < yaw_KI_max){
          	yaw_er_sum = yaw_er_sum + yaw_er;
          }  
          cp = yaw_er * yaw_KP;
          ci = yaw_KI * dt * yaw_er_sum;
          cd = yaw_KD * (mes_yaw - last_yaw)/dt;
          r_des = (cp + ci + cd);
          r_des = limit(r_des, (-1)*r_max, r_max);
          last_yaw = mes_yaw;

          /*
          cmd_vel.linear.x = vx_des;
          cmd_vel.linear.y = vy_des;
          cmd_vel.linear.z = vz_des;
          cmd_vel.angular.x = p_des;
          cmd_vel.angular.y = q_des;
          cmd_vel.angular.z = r_des;
          */
        }

    }

    double limit( double in, double min, double max)
    {
    if(in < min){
      in = min;
    }
    if( in > max){
      in = max;
    }
    return in;
    }

    template <typename T> int sign(T val) 
    {
        return (T(0) < val) - (val < T(0));
    }

  private:

    void timer_callback()
    {
        CalculatePoseControl();
        cmd_vel_pub_->publish(cmd_vel);
    }
    
    void PoseModeCallback(const std_msgs::msg::Int32& int_msg)
    {
        cmd_mode = int_msg;
    }
    
    void PoseControlCallback(const geometry_msgs::msg::Pose& pose_msg)
    {
        cmd_pose = pose_msg;
    }

    void PoseOdometryCallback(const nav_msgs::msg::Odometry& odometry_msg)
    {
        current_odometry = odometry_msg;
    }

    //rclcpp::TimerBase::SharedPtr timer_;
    //rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    size_t count_;

    //Subscribers
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr cmd_mode_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr cmd_position_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr mes_odometry_sub_;
    
    //Publichers
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;

    //Messages
    std_msgs::msg::Int32 cmd_mode;
    nav_msgs::msg::Odometry current_odometry;
    geometry_msgs::msg::Pose cmd_pose;
    geometry_msgs::msg::Twist cmd_vel;

    //timer
    rclcpp::TimerBase::SharedPtr timer_;

    //General
    double mode;
 
    //General
    tf2::Quaternion q;
    double mes_x, mes_y, mes_z;
    double mes_roll, mes_pitch, mes_yaw;

    double cmd_x, cmd_y, cmd_z;
    double cmd_roll, cmd_pitch, cmd_yaw;

    rclcpp::Time last_time;
    rclcpp::Time sim_time;
    double dt;

    //Position Controller
    double x_er, y_er, z_er;
    double x_er_sum, y_er_sum, z_er_sum;
    double last_x, last_y, last_z;

    double roll_er, pitch_er, yaw_er;
    double roll_er_sum, pitch_er_sum, yaw_er_sum;
    double last_roll, last_pitch, last_yaw;

    double cp, ci, cd;

    double vx_des, vy_des, vz_des;
    double p_des, q_des, r_des;

    //Saturation
    double vx_max, vy_max, vz_max;
    double p_max, q_max, r_max;

    //X PID
    double x_KI_max;
    double x_KP;
    double x_KI;
    double x_KD;

    //Y PID
    double y_KI_max;
    double y_KP;
    double y_KI;
    double y_KD;

    //Z PID
    double z_KI_max;
    double z_KP;
    double z_KI;
    double z_KD;

    //roll PID
    double roll_KI_max;
    double roll_KP;
    double roll_KI;
    double roll_KD;

    //Y PID
    double pitch_KI_max;
    double pitch_KP;
    double pitch_KI;
    double pitch_KD;

    //Z PID
    double yaw_KI_max;
    double yaw_KP;
    double yaw_KI;
    double yaw_KD;

};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PoseControllerNode>());
  rclcpp::shutdown();
  return 0;
}