#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <cmath>

#include <tf2_ros/transform_broadcaster.h>
#include <tf2/transform_datatypes.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
//#include <tf2/convert.h>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int32.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/float64.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

using namespace std::chrono_literals;
using std::placeholders::_1;

//rate time : 50Hz

/* This example creates a subclass of Node and uses std::bind() to register a
* member function as a callback from the timer. */

class VelControllerNode : public rclcpp::Node
{
  public:
    VelControllerNode() : Node("vel_controller_node"), count_(0)
    {
        //0 : cmdes de position ; 1 : cmdes de vitesse
        cmd_mode_sub_ = this->create_subscription<std_msgs::msg::Int32>("/SousMarin/command/mode", 10, std::bind(&VelControllerNode::VelModeCallback, this, _1));
        cmd_vel_manuel_sub_ = this->create_subscription<nav_msgs::msg::Odometry>("/SousMarin/command/pos_vel", 10, std::bind(&VelControllerNode::VelControlManuelCallback, this, _1));
        cmd_vel_pos_sub_ = this->create_subscription<geometry_msgs::msg::Twist>("/SousMarin/command/pos_vel_pos", 10, std::bind(&VelControllerNode::VelControlPosCallback, this, _1));
        mes_odometry_sub_ = this->create_subscription<nav_msgs::msg::Odometry>("/model/SousMarin/odometry", 10, std::bind(&VelControllerNode::VelOdometryCallback, this, _1));
        
        timer_ = this->create_wall_timer(20ms, std::bind(&VelControllerNode::timer_callback, this));
        
        cmd_motor1_pub_ = this->create_publisher<std_msgs::msg::Float64>("/model/SousMarin/joint/CaissonPrinc_Moteur1/cmd_thrust", 10);
        cmd_motor2_pub_ = this->create_publisher<std_msgs::msg::Float64>("/model/SousMarin/joint/CaissonPrinc_Moteur2/cmd_thrust", 10);
        cmd_motor3_pub_ = this->create_publisher<std_msgs::msg::Float64>("/model/SousMarin/joint/CaissonPrinc_Moteur3/cmd_thrust", 10);
        cmd_motor4_pub_ = this->create_publisher<std_msgs::msg::Float64>("/model/SousMarin/joint/CaissonPrinc_Moteur4/cmd_thrust", 10);
        cmd_motorprof1_pub_ = this->create_publisher<std_msgs::msg::Float64>("/model/SousMarin/joint/CaissonPrinc_MoteurProf1/cmd_thrust", 10);
        cmd_motorprof2_pub_ = this->create_publisher<std_msgs::msg::Float64>("/model/SousMarin/joint/CaissonPrinc_MoteurProf2/cmd_thrust", 10);
        cmd_motorprof3_pub_ = this->create_publisher<std_msgs::msg::Float64>("/model/SousMarin/joint/CaissonPrinc_MoteurProf3/cmd_thrust", 10);
        cmd_motorprof4_pub_ = this->create_publisher<std_msgs::msg::Float64>("/model/SousMarin/joint/CaissonPrinc_MoteurProf4/cmd_thrust", 10);

        cmd_motors_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectoryPoint>("/SousMarin/command/pwm", 10);

        InitializeParams();
    }

    void InitializeParams()
    {
        last_time = rclcpp::Clock{RCL_ROS_TIME}.now();

        //Declaration de parametres

        this->declare_parameter("alpha", 0.79);
        this->declare_parameter("L_m", 1.0);
        this->declare_parameter("L_d", 0.5);
        this->declare_parameter("Kd_m", 0.4);
        this->declare_parameter("Kd_d", 0.4);
        this->declare_parameter("rot_max_m", 2000.0);
        this->declare_parameter("rot_max_d", 2000.0);
        this->declare_parameter("PWM_0", 1500.0);
        this->declare_parameter("PWM_max", 1100.0);
        this->declare_parameter("PWM_min", 1900.0);
        
        this->declare_parameter("vx_PID.P", 1.0);
        this->declare_parameter("vx_PID.I", 0.1);
        this->declare_parameter("vx_PID.I_max", 1.0);
        this->declare_parameter("vx_PID.D", 0.1);

        this->declare_parameter("vy_PID.P", 1.0);
        this->declare_parameter("vy_PID.I", 0.1);
        this->declare_parameter("vy_PID.I_max", 1.0);
        this->declare_parameter("vy_PID.D", 0.1);

        this->declare_parameter("vz_PID.P", 1.0);
        this->declare_parameter("vz_PID.I", 0.1);
        this->declare_parameter("vz_PID.I_max", 1.0);
        this->declare_parameter("vz_PID.D", 0.1);

        this->declare_parameter("p_PID.P", 1.0);
        this->declare_parameter("p_PID.I", 0.1);
        this->declare_parameter("p_PID.I_max", 1.0);
        this->declare_parameter("p_PID.D", 0.1);

        this->declare_parameter("q_PID.P", 1.0);
        this->declare_parameter("q_PID.I", 0.1);
        this->declare_parameter("q_PID.I_max", 1.0);
        this->declare_parameter("q_PID.D", 0.1);

        this->declare_parameter("r_PID.P", 1.0);
        this->declare_parameter("r_PID.I", 0.1);
        this->declare_parameter("r_PID.I_max", 1.0);
        this->declare_parameter("r_PID.D", 0.1);

        this->declare_parameter("Fx_max", 200.0);
        this->declare_parameter("Fy_max", 200.0);
        this->declare_parameter("Fz_max", 200.0);

        this->declare_parameter("Mx_max", 200.0);
        this->declare_parameter("My_max", 200.0);
        this->declare_parameter("Mz_max", 200.0);

        //Recuperation des parametres
        
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
        
        this->get_parameter("vx_PID.P", vx_KP);
        this->get_parameter("vx_PID.I", vx_KI);
        this->get_parameter("vx_PID.I_max", vx_KI_max);
        this->get_parameter("vx_PID.D", vx_KD);

        this->get_parameter("vy_PID.P", vy_KP);
        this->get_parameter("vy_PID.I", vy_KI);
        this->get_parameter("vy_PID.I_max", vy_KI_max);
        this->get_parameter("vy_PID.D", vy_KD);

        this->get_parameter("vz_PID.P", vz_KP);
        this->get_parameter("vz_PID.I", vz_KI);
        this->get_parameter("vz_PID.I_max", vz_KI_max);
        this->get_parameter("vz_PID.D", vz_KD);

        this->get_parameter("p_PID.P", p_KP);
        this->get_parameter("p_PID.I", p_KI);
        this->get_parameter("p_PID.I_max", p_KI_max);
        this->get_parameter("p_PID.D", p_KD);

        this->get_parameter("q_PID.P", q_KP);
        this->get_parameter("q_PID.I", q_KI);
        this->get_parameter("q_PID.I_max", q_KI_max);
        this->get_parameter("q_PID.D", q_KD);

        this->get_parameter("r_PID.P", r_KP);
        this->get_parameter("r_PID.I", r_KI);
        this->get_parameter("r_PID.I_max", r_KI_max);
        this->get_parameter("r_PID.D", r_KD);

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

        C = cos(alpha);
        L_b = sqrt(pow(L_m, 2) + pow(L_d, 2));
    }

    void CalculateVelControl()
    {
        // Get simulator time
        sim_time = rclcpp::Clock{RCL_ROS_TIME}.now();
        dt = sim_time.seconds() - last_time.seconds(); //dt = (sim_time - last_time).toSec();
        last_time = sim_time;
        if (dt == 0.0) return;
        
        //Get measured data (odometry)
        mode = cmd_mode.data;
        RCLCPP_INFO(this->get_logger(), "vel_mode = %lf", mode);

        //Get position data
        mes_vx = current_odometry.twist.twist.linear.x;
        mes_vy = current_odometry.twist.twist.linear.y;
        mes_vz = current_odometry.twist.twist.linear.z;

        //Convert quaternion to Euler angles
        //tf2::quaternionMsgToTF(current_odometry.pose.pose.orientation, q);
        //tf2::fromMsg(current_odometry.pose.pose.orientation, q);
        //tf2::Matrix3x3(q).getRPY(mes_roll, mes_pitch, mes_yaw);

        mes_p = current_odometry.twist.twist.angular.x;
        mes_q = current_odometry.twist.twist.angular.y;
        mes_r = current_odometry.twist.twist.angular.z;

        //vitesses (utilisation de vx, vy, vz en mode 1)
        if(mode == 0)
        {
          cmd_vx = cmd_vel_pose.linear.x;
          cmd_vy = cmd_vel_pose.linear.y;
          cmd_vz = cmd_vel_pose.linear.z;

          cmd_p = cmd_vel_pose.angular.x;
          cmd_q = cmd_vel_pose.angular.y;
          cmd_r = cmd_vel_pose.angular.z;
        }
        else
        {
          cmd_vx = cmd_vel_manuel.twist.twist.linear.x;
          cmd_vy = cmd_vel_manuel.twist.twist.linear.y;
          cmd_vz = cmd_vel_manuel.twist.twist.linear.z;

          cmd_p = cmd_vel_manuel.twist.twist.angular.x;
          cmd_q = cmd_vel_manuel.twist.twist.angular.y;
          cmd_r = cmd_vel_manuel.twist.twist.angular.z;
        }

        //vx PID
        vx_er = cmd_vx - mes_vx;
        if(abs(vx_er) < vx_KI_max){
        	vx_er_sum = vx_er_sum + vx_er;
        }  
        cp = vx_er * vx_KP;
        ci = vx_KI * dt * vx_er_sum;
        cd = vx_KD * (mes_vx - last_vx)/dt;
        Fx_des = (cp + ci +cd);
        Fx_des = limit(Fx_des, (-1)*Fx_max, Fx_max);
        last_vx = mes_vx;

        //RCLCPP_INFO(this->get_logger(), "cmd_vx = %lf ,mes_vx = %lf, vx_er = %lf ,vx_KP = %lf", cmd_vx, mes_vx, vx_er, vx_KP);
        //RCLCPP_INFO(this->get_logger(), "cp = %lf ,ci = %lf ,cd = %lf", cp, ci, cd);

        //vy PID
        vy_er = cmd_vy - mes_vy;
        if(abs(vy_er) < vy_KI_max){
        	vy_er_sum = vy_er_sum + vy_er;
        }  
        cp = vy_er * vy_KP;
        ci = vy_KI * dt * vy_er_sum;
        cd = vy_KD * (mes_vy - last_vy)/dt;
        Fy_des = (cp + ci +cd);
        Fy_des = limit(Fy_des, (-1)*Fy_max, Fy_max);
        last_vy = mes_vy;

        //vz PID
        vz_er = cmd_vz - mes_vz;
        if(abs(vz_er) < vz_KI_max){
        	vz_er_sum = vz_er_sum + vz_er;
        }  
        cp = vz_er * vz_KP;
        ci = vz_KI * dt * vz_er_sum;
        cd = vz_KD * (mes_vz - last_vz)/dt;
        Fz_des = (cp + ci +cd);
        Fz_des = limit(Fz_des, (-1)*Fz_max, Fz_max);
        last_vz = mes_vz;

        //Position and velocity defined in global frame

        //p PID
        p_er = cmd_p - mes_p;
        if(abs(p_er) < p_KI_max){
        	p_er_sum = p_er_sum + p_er;
        }  
        cp = p_er * p_KP;
        ci = p_KI * dt * p_er_sum;
        cd = p_KD * (mes_p - last_p)/dt;
        Mx_des = (cp + ci + cd);
        Mx_des = limit(Mx_des, (-1)*Mx_max, Mx_max);
        last_p = mes_p;
        
        //q PID
        q_er = cmd_q - mes_q;
        if(abs(q_er) < q_KI_max){
        	q_er_sum = q_er_sum + q_er;
        }  
        cp = q_er * q_KP;
        ci = q_KI * dt * q_er_sum;
        cd = q_KD * (mes_q - last_q)/dt;
        My_des = (cp + ci + cd);
        My_des = limit(My_des, (-1)*My_max, My_max);
        last_q = mes_q;
        
        //r PID
        r_er = cmd_r - mes_r;
        if(abs(r_er) < r_KI_max){
        	r_er_sum = r_er_sum + r_er;
        }  
        cp = r_er * r_KP;
        ci = r_KI * dt * r_er_sum;
        cd = r_KD * (mes_r - last_r)/dt;
        Mz_des = (cp + ci + cd);
        Mz_des = limit(Mz_des, (-1)*Mz_max, Mz_max);

        //Fx_des = 0;
        //Fy_des = 0;
        //Mz_des = -10;

        //Mixage

        F1 = (1.0/4.0) * (Fx_des/C + Fy_des/C + Mz_des/L_b); //Moteur principal avant droit
        F2 = (1.0/4.0) * (Fx_des/C - Fy_des/C + Mz_des/L_b); //Moteur principal arriere droit
        F3 = (1.0/4.0) * (Fx_des/C + Fy_des/C - Mz_des/L_b); //Moteur principal arriere gauche
        F4 = (1.0/4.0) * (Fx_des/C - Fy_des/C - Mz_des/L_b); //Moteur principal avant gauche

        F5 = (-1.0/4.0) * (Fz_des - Mx_des/L_d - My_des/L_d); //Moteur de profondeur avant droit
        F6 = (-1.0/4.0) * (Fz_des - Mx_des/L_d + My_des/L_d); //Moteur de profondeur arriere droit
        F7 = (-1.0/4.0) * (Fz_des + Mx_des/L_d + My_des/L_d); //Moteur de profondeur arriere gauche
        F8 = (-1.0/4.0) * (Fz_des + Mx_des/L_d - My_des/L_d); //Moteur de profondeur avant gauche

        F1 = 0.0; //20.0;
        F2 = 0.0; //20.0;
        F3 = 0.0; //20.0;
        F4 = 0.0; //20.0;
        
        //F5 = 0.0; //10.0;
        //F6 = 0.0; //10.0;
        //F7 = 0.0; //10.0;
        //F8 = 0.0; //10.0;

        cmd_motor1.data = F1;
        cmd_motor2.data = F2;
        cmd_motor3.data = F3;
        cmd_motor4.data = F4;
        cmd_motorprof1.data = F5;
        cmd_motorprof2.data = F6;
        cmd_motorprof3.data = F7;
        cmd_motorprof4.data = F8;
        
        //Conversion in PWM
        speed_motor1 = (sqrt(F1/Kd_m) * sign(F1)) * (PWM_max-PWM_0)/rot_max_m + PWM_0;
        speed_motor2 = (sqrt(F2/Kd_m) * sign(F2)) * (PWM_max-PWM_0)/rot_max_m + PWM_0;
        speed_motor3 = (sqrt(F3/Kd_m) * sign(F3)) * (PWM_max-PWM_0)/rot_max_m + PWM_0;
        speed_motor4 = (sqrt(F4/Kd_m) * sign(F4)) * (PWM_max-PWM_0)/rot_max_m + PWM_0;
        speed_motor5 = (sqrt(F5/Kd_m) * sign(F5)) * (PWM_max-PWM_0)/rot_max_d + PWM_0;
        speed_motor6 = (sqrt(F6/Kd_m) * sign(F6)) * (PWM_max-PWM_0)/rot_max_d + PWM_0;
        speed_motor7 = (sqrt(F7/Kd_m) * sign(F7)) * (PWM_max-PWM_0)/rot_max_d + PWM_0;
        speed_motor8 = (sqrt(F8/Kd_m) * sign(F8)) * (PWM_max-PWM_0)/rot_max_d + PWM_0;

        cmd_motors.velocities.clear();

        cmd_motors.velocities.push_back(speed_motor1);
        cmd_motors.velocities.push_back(speed_motor2);
        cmd_motors.velocities.push_back(speed_motor3);
        cmd_motors.velocities.push_back(speed_motor4);
        cmd_motors.velocities.push_back(speed_motor5);
        cmd_motors.velocities.push_back(speed_motor6);
        cmd_motors.velocities.push_back(speed_motor7);
        cmd_motors.velocities.push_back(speed_motor8);

        RCLCPP_INFO(this->get_logger(), "mes_vx = %lf ,mes_vy = %lf ,mes_vz = %lf", mes_vx, mes_vy, mes_vz);
        RCLCPP_INFO(this->get_logger(), "cmd_vx = %lf ,cmd_vy = %lf ,cmd_vz = %lf", cmd_vx, cmd_vy, cmd_vz);
        /*RCLCPP_INFO(this->get_logger(), "mes_p = %lf ,mes_q = %lf ,mes_r = %lf", mes_p, mes_q, mes_r);
        RCLCPP_INFO(this->get_logger(), "cmd_p = %lf ,cmd_q = %lf ,cmd_r = %lf", cmd_p, cmd_q, cmd_r);*/
        RCLCPP_INFO(this->get_logger(), "Fx_des = %lf ,Fy_des = %lf ,Fz_des = %lf", Fx_des, Fy_des, Fz_des);
        RCLCPP_INFO(this->get_logger(), "Mx_des = %lf ,My_des = %lf ,Mz_des = %lf", Mx_des, My_des, Mz_des);
        /*RCLCPP_INFO(this->get_logger(), "F1 = %lf ,F2 = %lf ,F3 = %lf ,F4 = %lf", F1, F2, F3, F4);
        RCLCPP_INFO(this->get_logger(), "F5 = %lf ,F6 = %lf ,F7 = %lf ,F8 = %lf", F5, F6, F7, F8);*/

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
        CalculateVelControl();

        cmd_motor1_pub_->publish(cmd_motor1);
        cmd_motor2_pub_->publish(cmd_motor2);
        cmd_motor3_pub_->publish(cmd_motor3);
        cmd_motor4_pub_->publish(cmd_motor4);

        cmd_motorprof1_pub_->publish(cmd_motorprof1);
        cmd_motorprof2_pub_->publish(cmd_motorprof2);
        cmd_motorprof3_pub_->publish(cmd_motorprof3);
        cmd_motorprof4_pub_->publish(cmd_motorprof4);

        cmd_motors_pub_->publish(cmd_motors);
    }
    
    void VelModeCallback(const std_msgs::msg::Int32& int_msg)
    {
        cmd_mode = int_msg;
    }
    
    void VelControlManuelCallback(const nav_msgs::msg::Odometry& odometry_msg)
    {
        cmd_vel_manuel = odometry_msg;
    }

    void VelControlPosCallback(const geometry_msgs::msg::Twist& twist_msg)
    {
        cmd_vel_pose = twist_msg;
    }

    void VelOdometryCallback(const nav_msgs::msg::Odometry& odometry_msg)
    {
        current_odometry = odometry_msg;
    }

    size_t count_;

    //Subscribers
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr cmd_mode_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr cmd_vel_manuel_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pos_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr mes_odometry_sub_;

    //Publishers
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr cmd_motor1_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr cmd_motor2_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr cmd_motor3_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr cmd_motor4_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr cmd_motorprof1_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr cmd_motorprof2_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr cmd_motorprof3_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr cmd_motorprof4_pub_;
    
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectoryPoint>::SharedPtr cmd_motors_pub_;

    //Messages
    std_msgs::msg::Int32 cmd_mode;
    nav_msgs::msg::Odometry current_odometry;
    nav_msgs::msg::Odometry cmd_vel_manuel;
    geometry_msgs::msg::Twist cmd_vel_pose;

    std_msgs::msg::Float64 cmd_motor1;
    std_msgs::msg::Float64 cmd_motor2;
    std_msgs::msg::Float64 cmd_motor3;
    std_msgs::msg::Float64 cmd_motor4;
    std_msgs::msg::Float64 cmd_motorprof1;
    std_msgs::msg::Float64 cmd_motorprof2;
    std_msgs::msg::Float64 cmd_motorprof3;
    std_msgs::msg::Float64 cmd_motorprof4;
    trajectory_msgs::msg::JointTrajectoryPoint cmd_motors;

    //timer
    rclcpp::TimerBase::SharedPtr timer_;

    //General
    double mode;

    //General
    tf2::Quaternion q;
    double mes_vx, mes_vy, mes_vz;
    double mes_p, mes_q, mes_r;

    double cmd_vx, cmd_vy, cmd_vz;
    double cmd_p, cmd_q, cmd_r;

    rclcpp::Time last_time;
    rclcpp::Time sim_time;
    double dt;

    double mes_roll, mes_pitch, mes_yaw;

    //Motors
    double alpha; //Motor inclinaison
    double C;
    double L_m, L_d, L_b; //Leverage arms
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

    //Command
    double F1, F2, F3, F4, F5, F6, F7, F8;
    double speed_motor1, speed_motor2, speed_motor3, speed_motor4, speed_motor5, speed_motor6, speed_motor7, speed_motor8;

};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<VelControllerNode>());
  rclcpp::shutdown();
  return 0;
}