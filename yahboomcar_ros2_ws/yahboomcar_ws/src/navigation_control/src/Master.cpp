// ============================================================
// Master.cpp — APEX Controller v2
//
// Migrated from PD on raw pixels to predictive controller
// with CTE + heading + curvature feedforward.
//
// Subscribes:
//   /lane_detection/control_ref (Vector3) — CTE, heading, curvature [-1,1]
//   /lane_detection/lane_state  (UInt8)   — 0=BOTH, 1=ONE, 2=NONE, 3=RECOVERING
//   /lane_detection/confidence  (Float32) — detection confidence [0,1]
//
// Publishes:
//   /cmd_vel (Twist) — steering + velocity
//
// Parameters (4+2):
//   kp_cte      — lateral correction
//   kp_heading  — angular alignment (replaces Kd)
//   kff         — curvature feedforward (anticipation)
//   max_speed   — max forward speed in straights
//   kv_curve    — speed reduction factor in curves
//   max_angular — steering clamp
// ============================================================

#include <cmath>
#include <algorithm>

#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <std_msgs/msg/u_int8.hpp>
#include <std_msgs/msg/float32.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/vector3.hpp>

class Master : public rclcpp::Node {
private:
    // Subscribers
    rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr ref_sub_;
    rclcpp::Subscription<std_msgs::msg::UInt8>::SharedPtr lane_state_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr confidence_sub_;

    // Publisher
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;

    // State from lane detection
    float cte_ = 0.0f;
    float heading_ = 0.0f;
    float curvature_ = 0.0f;
    float confidence_ = 0.0f;
    uint8_t lane_state_ = 0;  // 0=BOTH, 1=ONE, 2=NONE, 3=RECOVERING
    bool enabled_ = false;

    // === 4 main gains ===
    double kp_cte_;        // Lateral correction
    double kp_heading_;    // Angular alignment (natural damping)
    double kff_;           // Curvature feedforward
    double max_speed_;     // Max speed in straights

    // === Secondary ===
    double kv_curve_;      // Speed reduction in curves (0-1)
    double max_angular_;   // Steering clamp

public:
    Master() : Node("master_control") {
        // Main gains
        this->declare_parameter("kp_cte", 0.5);
        this->declare_parameter("kp_heading", 0.3);
        this->declare_parameter("kff", 0.2);
        this->declare_parameter("max_speed", 0.20);

        // Secondary
        this->declare_parameter("kv_curve", 0.5);
        this->declare_parameter("max_angular", 0.8);
        this->declare_parameter("enabled", false);

        load_params();

        param_cb_handle_ = this->add_on_set_parameters_callback(
            std::bind(&Master::on_param_change, this, std::placeholders::_1));

        // Subscribe to new lane detection signals
        ref_sub_ = this->create_subscription<geometry_msgs::msg::Vector3>(
            "/lane_detection/control_ref", 1,
            [this](const geometry_msgs::msg::Vector3::SharedPtr msg) {
                cte_ = msg->x;
                heading_ = msg->y;
                curvature_ = msg->z;
                control_loop();
            });

        lane_state_sub_ = this->create_subscription<std_msgs::msg::UInt8>(
            "/lane_detection/lane_state", 1,
            [this](const std_msgs::msg::UInt8::SharedPtr msg) {
                lane_state_ = msg->data;
            });

        confidence_sub_ = this->create_subscription<std_msgs::msg::Float32>(
            "/lane_detection/confidence", 1,
            [this](const std_msgs::msg::Float32::SharedPtr msg) {
                confidence_ = msg->data;
            });

        cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);

        RCLCPP_INFO(this->get_logger(),
            "Master v2: kp_cte=%.2f kp_head=%.2f kff=%.2f speed=%.2f",
            kp_cte_, kp_heading_, kff_, max_speed_);
    }

private:
    void load_params() {
        kp_cte_ = this->get_parameter("kp_cte").as_double();
        kp_heading_ = this->get_parameter("kp_heading").as_double();
        kff_ = this->get_parameter("kff").as_double();
        max_speed_ = this->get_parameter("max_speed").as_double();
        kv_curve_ = this->get_parameter("kv_curve").as_double();
        max_angular_ = this->get_parameter("max_angular").as_double();
        enabled_ = this->get_parameter("enabled").as_bool();
    }

    rcl_interfaces::msg::SetParametersResult on_param_change(
            const std::vector<rclcpp::Parameter> &params) {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        for (const auto &p : params) {
            const auto &name = p.get_name();
            if (name == "kp_cte") kp_cte_ = p.as_double();
            else if (name == "kp_heading") kp_heading_ = p.as_double();
            else if (name == "kff") kff_ = p.as_double();
            else if (name == "max_speed") max_speed_ = p.as_double();
            else if (name == "kv_curve") kv_curve_ = p.as_double();
            else if (name == "max_angular") max_angular_ = p.as_double();
            else if (name == "enabled") {
                enabled_ = p.as_bool();
                if (!enabled_) publish_stop();
            }
            RCLCPP_INFO(this->get_logger(), "Param '%s' = %s",
                        name.c_str(), p.value_to_string().c_str());
        }
        return result;
    }

    void publish_stop() {
        geometry_msgs::msg::Twist stop;
        cmd_vel_pub_->publish(stop);
    }

    // ============================================================
    // Control loop — runs on each control_ref message
    // ============================================================
    void control_loop() {
        if (!enabled_) {
            publish_stop();
            return;
        }

        // ----------------------------------------------------------
        // Steering: 3 parallel terms
        //   kp_cte * CTE:      "I'm offset, go back to center"
        //   kp_heading * head:  "I'm misaligned, straighten"
        //   kff * curvature:    "Curve ahead, pre-turn"
        // ----------------------------------------------------------
        double steer = kp_cte_ * cte_
                     + kp_heading_ * heading_
                     + kff_ * curvature_;

        // ----------------------------------------------------------
        // Scale by detection confidence
        // Low confidence = don't trust strong corrections
        // Floor at 0.3 so it never ignores everything
        // ----------------------------------------------------------
        float conf_scale = std::max(0.3f, confidence_);
        steer *= conf_scale;

        // ----------------------------------------------------------
        // Scale by lane state
        // The lane detector handles inertial nav internally;
        // here we just reduce aggressiveness
        // ----------------------------------------------------------
        switch (lane_state_) {
            case 0: break;                       // BOTH_LINES: full
            case 1: steer *= 0.85; break;        // ONE_LINE: slightly less
            case 2: steer *= 0.5; break;         // NO_LINES: conservative
            case 3: steer *= 0.7; break;         // RECOVERING: transition
        }

        steer = std::clamp(steer, -max_angular_, max_angular_);

        // ----------------------------------------------------------
        // Speed: reduce in curves and without lines
        // ----------------------------------------------------------
        double speed = max_speed_;

        // Curvature factor: more curve = slower
        speed *= (1.0 - kv_curve_ * std::abs(curvature_));

        // State factor
        if (lane_state_ == 2) speed *= 0.6;       // NO_LINES
        else if (lane_state_ == 3) speed *= 0.8;  // RECOVERING

        // Minimum to avoid stalling
        speed = std::max(0.05, speed);

        // ----------------------------------------------------------
        // Publish
        // ----------------------------------------------------------
        geometry_msgs::msg::Twist twist;
        twist.linear.x = speed;
        twist.angular.z = steer;
        cmd_vel_pub_->publish(twist);

        // Log every 30 frames
        static int log_cnt = 0;
        if (++log_cnt % 30 == 0) {
            RCLCPP_INFO(this->get_logger(),
                "[CTRL] cte=%.2f head=%.2f curv=%.2f -> az=%.3f vx=%.3f "
                "conf=%.1f state=%d",
                cte_, heading_, curvature_, steer, speed,
                confidence_, lane_state_);
        }
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Master>());
    rclcpp::shutdown();
    return 0;
}
