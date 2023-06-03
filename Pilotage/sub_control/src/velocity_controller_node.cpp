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

class VelocityControllerNode : public rclcpp::Node
{
  public:
    VelocityControllerNode() : Node("velocity_controller_node"), count_(0)
    {
        cmd_velocity_sub_ = this->create_subscription<geometry_msgs::msg::Twist>("cmd/velocity", 10, std::bind(&VelocityControllerNode::VelControlCallback, this, _1));
        mes_odometry_sub_ = this->create_subscription<nav_msgs::msg::Odometry>("sub/odometry", 10, std::bind(&VelocityControllerNode::VelOdometryCallback, this, _1));
      
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
        mes_vx = current_odometry.twist.twist.linear.x;
        mes_vy = current_odometry.twist.twist.linear.y;
        mes_vz = current_odometry.twist.twist.linear.z;

        cmd_vx = cmd_velocity.linear.x;
        cmd_vy = cmd_velocity.linear.y;
        cmd_vz = cmd_velocity.linear.z;

        mes_p = current_odometry.twist.twist.angular.x;
        mes_q = current_odometry.twist.twist.angular.y;
        mes_r = current_odometry.twist.twist.angular.z;

        cmd_p = cmd_velocity.angular.x;
        cmd_q = cmd_velocity.angular.y;
        cmd_r = cmd_velocity.angular.z;

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

        //x-y command

        double Fx_1, Fy_1;

        //Conversion from global frame to body frame
        Fx_1 = Fx_des * cos(mes_yaw) + Fy_des * sin(mes_yaw);
        Fy_1 = (-1) * Fx_des * sin(mes_yaw) + Fy_des * cos(mes_yaw);

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
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr mes_odometry_sub_;
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectoryPoint>::SharedPtr cmd_motors_pub_;

    nav_msgs::msg::Odometry current_odometry;
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