#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include <tf2_ros/transform_broadcaster.h>
#include <tf2/transform_datatypes.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
//#include <tf2/convert.h>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

//using namespace std::chrono_literals;
using std::placeholders::_1;

/* This example creates a subclass of Node and uses std::bind() to register a
* member function as a callback from the timer. */

class SimpleVelocityControllerNode : public rclcpp::Node
{
  public:
    SimpleVelocityControllerNode() : Node("simple_velocity_controller_node"), count_(0)
    {
        cmd_velocity_sub_ = this->create_subscription<geometry_msgs::msg::Twist>("cmd/velocity", 10, std::bind(&SimpleVelocityControllerNode::VelControlCallback, this, _1));
        //mes_odometry_sub_ = this->create_subscription<nav_msgs::msg::Odometry>("sub/odometry", 10, std::bind(&SimpleVelocityControllerNode::VelOdometryCallback, this, _1));
      
        cmd_motors_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectoryPoint>("cmd/motors", 10);

        InitializeParams();
    }

    void InitializeParams()
    {
        last_time = rclcpp::Clock{RCL_ROS_TIME}.now().seconds();

        this->get_parameter("alpha", alpha);
        this->get_parameter("L_m", L_m);
        this->get_parameter("L_d", L_d);
        this->get_parameter("Kd_m", Kd_m);
        this->get_parameter("Kd_d", Kd_d);
        this->get_parameter("rot_max_m", rot_max_m);
        this->get_parameter("rot_max_d", rot_max_d);
        this->get_parameter("PWM_0", PWM_0);
        this->get_parameter("PWM_max", PWM_min);
        this->get_parameter("PWM_min", PWM_max);
        
        this->get_parameter("vx_PID/P", vx_KP);
        this->get_parameter("vx_PID/I", vx_KI);
        this->get_parameter("vx_PID/I_max", vx_KI_max);
        this->get_parameter("vx_PID/D", vx_KD);

        this->get_parameter("vy_PID/P", vy_KP);
        this->get_parameter("vy_PID/I", vy_KI);
        this->get_parameter("vy_PID/I_max", vy_KI_max);
        this->get_parameter("vy_PID/D", vy_KD);

        this->get_parameter("vz_PID/P", vz_KP);
        this->get_parameter("vz_PID/I", vz_KI);
        this->get_parameter("vz_PID/I_max", vz_KI_max);
        this->get_parameter("vz_PID/D", vz_KD);

        this->get_parameter("p_PID/P", p_KP);
        this->get_parameter("p_PID/I", p_KI);
        this->get_parameter("p_PID/I_max", p_KI_max);
        this->get_parameter("p_PID/D", p_KD);

        this->get_parameter("q_PID/P", q_KP);
        this->get_parameter("q_PID/I", q_KI);
        this->get_parameter("q_PID/I_max", q_KI_max);
        this->get_parameter("q_PID/D", q_KD);

        this->get_parameter("r_PID/P", r_KP);
        this->get_parameter("r_PID/I", r_KI);
        this->get_parameter("r_PID/I_max", r_KI_max);
        this->get_parameter("r_PID/D", r_KD);

        this->get_parameter("Fx_max", Fx_max);
        this->get_parameter("Fy_max", Fy_max);
        this->get_parameter("Fz_max", Fz_max);

        this->get_parameter("Mx_max", Mx_max);
        this->get_parameter("My_max", My_max);
        this->get_parameter("Mz_max", Mz_max);

        vx_er = 0;
        vy_er = 0;
        vz_er = 0;
        vx_er_sum = 0;
        vy_er_sum = 0;
        vz_er_sum = 0;
        last_vx = 0;
        last_vy = 0;
        last_vz = 0;
        Fx_des = 0;
        Fy_des = 0;
        Fz_des = 0;

        p_er = 0;
        q_er = 0;
        r_er = 0;
        p_er_sum = 0;
        q_er_sum = 0;
        r_er_sum = 0;
        last_p = 0;
        last_q = 0;
        last_r = 0;
        Mx_des = 0;
        My_des = 0;
        Mz_des = 0;
    }

    void CalculateVelocityControl()
    {
        //Convert quaternion to Euler angles
        //tf2::quaternionMsgToTF(current_odometry.pose.pose.orientation, q);
        tf2::fromMsg(current_odometry.pose.pose.orientation, q);
        tf2::Matrix3x3(q).getRPY(mes_roll, mes_pitch, mes_yaw);
        
        // Get simulator time
        sim_time = rclcpp::Clock{RCL_ROS_TIME}.now().seconds();
        dt = sim_time - last_time; //dt = (sim_time - last_time).toSec();
        if (dt == 0.0) return;

        //Get position data
        mes_dist = current_odometry.twist.twist.linear.x;
        mes_angle = current_odometry.twist.twist.angular.x;

        double Fx_1, Fy_1;

        Fx_1 = 200*mes_dist;
        Fy_1 = 0;
        Mz_des = 200*mes_angle;

        Fz_des = 0;
        Mx_des = 0;
        My_des = 0;

        //x-y command

        F1 = (Fx_1 * cos(alpha) + Fy_1 * sin(alpha))/2 + Mz_des/(4*L_m);
        F2 = ((-1) * Fx_1 * sin(alpha) + Fy_1 * cos(alpha))/2 + Mz_des/(4*L_m);
        F3 = (Fx_1 * cos(alpha) + Fy_1 * sin(alpha))/2 - Mz_des/(4*L_m);
        F4 = ((-1) * Fx_1 * sin(alpha) + Fy_1 * cos(alpha))/2 - Mz_des/(4*L_m);

        F5 = Fz_des/4 + Mx_des/(4*L_d) + My_des/(4*L_d);
        F6 = Fz_des/4 + Mx_des/(4*L_d) - My_des/(4*L_d);
        F7 = Fz_des/4 - Mx_des/(4*L_d) - My_des/(4*L_d);
        F8 = Fz_des/4 - Mx_des/(4*L_d) + My_des/(4*L_d);

        cmd_motors.effort.clear();

        cmd_motors.effort.push_back(F1); //Moteur principal avant droit
        cmd_motors.effort.push_back(F2); //Moteur principal arriere droit
        cmd_motors.effort.push_back(F3); //Moteur principal arriere gauche
        cmd_motors.effort.push_back(F4); //Moteur principal avant gauche
        cmd_motors.effort.push_back(F5); //Moteur de profondeur avant droit
        cmd_motors.effort.push_back(F6); //Moteur de profondeur arriere droit
        cmd_motors.effort.push_back(F7); //Moteur de profondeur arriere gauche
        cmd_motors.effort.push_back(F8); //Moteur de profondeur avant gauche

        //Conversion in PWM
        speed_motor1 = (sqrt(F1/Kd_m) * sign(F1)) * (PWM_max-PWM_0)/rot_max_m + PWM_0;
        speed_motor1 = (sqrt(F2/Kd_m) * sign(F2)) * (PWM_max-PWM_0)/rot_max_m + PWM_0;
        speed_motor1 = (sqrt(F3/Kd_m) * sign(F3)) * (PWM_max-PWM_0)/rot_max_m + PWM_0;
        speed_motor1 = (sqrt(F4/Kd_m) * sign(F4)) * (PWM_max-PWM_0)/rot_max_m + PWM_0;
        speed_motor1 = (sqrt(F5/Kd_m) * sign(F5)) * (PWM_max-PWM_0)/rot_max_d + PWM_0;
        speed_motor1 = (sqrt(F6/Kd_m) * sign(F6)) * (PWM_max-PWM_0)/rot_max_d + PWM_0;
        speed_motor1 = (sqrt(F7/Kd_m) * sign(F7)) * (PWM_max-PWM_0)/rot_max_d + PWM_0;
        speed_motor1 = (sqrt(F8/Kd_m) * sign(F8)) * (PWM_max-PWM_0)/rot_max_d + PWM_0;

        cmd_motors.velocities.clear();

        cmd_motors.velocities.push_back(speed_motor1);
        cmd_motors.velocities.push_back(speed_motor2);
        cmd_motors.velocities.push_back(speed_motor3);
        cmd_motors.velocities.push_back(speed_motor4);
        cmd_motors.velocities.push_back(speed_motor5);
        cmd_motors.velocities.push_back(speed_motor6);
        cmd_motors.velocities.push_back(speed_motor7);
        cmd_motors.velocities.push_back(speed_motor8);
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

    void VelControlCallback(const geometry_msgs::msg::Twist& twist_msg)
    {
        cmd_velocity = twist_msg;
        CalculateVelocityControl();
        cmd_motors_pub_->publish(cmd_motors); 
    }

    void VelOdometryCallback(const nav_msgs::msg::Odometry& odometry_msg)
    {
        current_odometry = odometry_msg;
    }

    size_t count_;

    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_velocity_sub_;
    //rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr mes_odometry_sub_;
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectoryPoint>::SharedPtr cmd_motors_pub_;

    //nav_msgs::msg::Odometry current_odometry;
    geometry_msgs::msg::Twist cmd_velocity;
    trajectory_msgs::msg::JointTrajectoryPoint cmd_motors;

 
    //General
    tf2::Quaternion q;
    double mes_vx, mes_vy, mes_vz;
    double mes_p, mes_q, mes_r;

    double cmd_vx, cmd_vy, cmd_vz;
    double cmd_p, cmd_q, cmd_r;

    double last_time; //rclcpp::Time last_time;
    double sim_time; //rclcpp::Time sim_time;
    double dt;

    double mes_roll, mes_pitch, mes_yaw;

    //Motors
    double alpha; //Motor inclinaison
    double L_m, L_d; //Leverage arms
    double Kd_m, Kd_d; //Motors constant
    double rot_max_m, rot_max_d; //Max speed
    double PWM_0, PWM_max, PWM_min; //PWM limits

    //Position Controller
    double vx_er, vy_er, vz_er;
    double vx_er_sum, vy_er_sum, vz_er_sum;
    double last_vx, last_vy, last_vz;

    double p_er, q_er, r_er;
    double p_er_sum, q_er_sum, r_er_sum;
    double last_p, last_q, last_r;

    double cp, ci, cd;

    double Fx_des, Fy_des, Fz_des;
    double Mx_des, My_des, Mz_des;

    //Saturation
    double Fx_max, Fy_max, Fz_max;
    double Mx_max, My_max, Mz_max;

    //vx PID
    double vx_KI_max;
    double vx_KP;
    double vx_KI;
    double vx_KD;

    //vy PID
    double vy_KI_max;
    double vy_KP;
    double vy_KI;
    double vy_KD;

    //vz PID
    double vz_KI_max;
    double vz_KP;
    double vz_KI;
    double vz_KD;

    //p PID
    double p_KI_max;
    double p_KP;
    double p_KI;
    double p_KD;

    //q PID
    double q_KI_max;
    double q_KP;
    double q_KI;
    double q_KD;

    //r PID
    double r_KI_max;
    double r_KP;
    double r_KI;
    double r_KD;

    //Simple
    double mes_dist;
    double mes_angle;

    //Command
    double F1, F2, F3, F4, F5, F6, F7, F8;
    double speed_motor1, speed_motor2, speed_motor3, speed_motor4, speed_motor5, speed_motor6, speed_motor7, speed_motor8;

};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<VelocityControllerNode>());
  rclcpp::shutdown();
  return 0;
}