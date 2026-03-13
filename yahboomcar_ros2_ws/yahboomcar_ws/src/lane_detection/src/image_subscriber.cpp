#include <iostream>
#include <math.h>
#include <string>
#include <chrono>
#include <ctime>
#include <vector>
#include <algorithm>
#include <numeric>

// ROS 2 Headers
#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/int16.hpp>
#include <std_msgs/msg/u_int8.hpp>
#include <std_msgs/msg/float32.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>

// OpenCV Headers
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Dense>
#include <random>

using namespace cv;
using namespace std;
using std::placeholders::_1;

static const std::string OPENCV_WINDOW = "LANE DETECTION";

// Processing resolution — all detection runs at this size
static const int PROC_WIDTH  = 320;
static const int PROC_HEIGHT = 240;

class LineDetection : public rclcpp::Node {
private:
    // === ROS publishers & subscribers ===
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<std_msgs::msg::Int16>::SharedPtr center_pub;
    rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr angle_line_pub;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr confidence_pub_;
    rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr lane_state_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr control_ref_pub_;
    image_transport::Publisher debug_image_pub_;
    bool enable_web_view_;
    bool headless_;

    std_msgs::msg::Int16 center_message;
    std_msgs::msg::UInt8 angle_line_message;

    // === Timing ===
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed_seconds;
    stringstream ss;

    // === Image processing ===
    Mat frame;
    Mat img_edges;
    Mat birdeye_S_channel_;
    vector<Point> left_points, right_points;
    vector<Rect> sliding_window_rects_;
    float polyleft[3], polyright[3];
    float polyleft_last[3], polyright_last[3];
    float polyleft_smooth[3], polyright_smooth[3];
    bool has_smooth_left_, has_smooth_right_;
    Point2f Source[4];
    Point2f Destination[4];

    int center_cam;
    int center_lines;
    int distance_center;
    int angle;
    int fps;
    bool detected;
    int frames_since_detection_;
    Ptr<CLAHE> clahe_;

    // === Calibration state (Mejora 1A) ===
    enum class CalibState { CALIBRATING, RUNNING };
    CalibState calib_state_;
    int calib_frame_count_ = 0;
    int calib_frames_needed_;
    std::vector<float> calib_widths_;
    float calibrated_lane_width_ = 0.0f;
    int calib_white_votes_ = 0;
    int calib_color_votes_ = 0;
    bool lines_are_white_ = true;
    std::vector<float> calib_sat_values_;

    // === Illumination adaptation (Mejora 1B) ===
    float illum_ema_ = 0.0f;
    bool illum_has_history_ = false;
    float adaptive_c_base_;

    // === RANSAC inlier ratios (Mejora 7) ===
    float left_inlier_ratio_ = 0.0f;
    float right_inlier_ratio_ = 0.0f;

    // === Lane state machine (Mejora 8) ===
    enum class LaneState { BOTH_LINES, ONE_LINE, NO_LINES, RECOVERING };
    LaneState lane_state_ = LaneState::BOTH_LINES;
    int frames_no_lines_ = 0;
    int frames_recovering_ = 0;
    float last_good_angle_ = 90.0f;
    float last_good_distance_ = 0.0f;
    float angle_rate_ = 0.0f;
    float prev_angle_ = 90.0f;
    float last_confidence_ = 0.0f;

    // === Adaptive lane width (Mejora 5) ===
    float width_ema_ = 0.0f;
    bool has_width_history_ = false;

    // === Peak struct for histogram (Mejora 3) ===
    struct Peak {
        int position;
        int strength;
    };

    // === Tunable parameters ===
    int adaptive_block_;
    int adaptive_c_;
    double clahe_clip_;
    int sliding_margin_;
    int sliding_minpix_;
    int ransac_min_points_;
    int ransac_min_inliers_;
    int poly_search_min_;
    int nwindows_;
    double smooth_alpha_;
    int lane_width_px_;
    int persp_top_left_pct_;
    int persp_top_right_pct_;
    int persp_top_y_pct_;
    int persp_bot_left_pct_;
    int persp_bot_right_pct_;
    int persp_bot_y_pct_;
    // New tunable params
    double max_curvature_radius_;
    int use_color_filter_;
    int sat_max_white_;
    int recovery_blend_frames_;
    int max_inertial_frames_;
    double inertial_straighten_rate_;

    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;

    static std::mt19937& get_rng() {
        static std::mt19937 gen(std::random_device{}());
        return gen;
    }

public:
    LineDetection() : Node("line_detection") {
        // --- Static parameters ---
        this->declare_parameter("enable_web_view", true);
        this->declare_parameter("headless", true);
        enable_web_view_ = this->get_parameter("enable_web_view").as_bool();
        headless_ = this->get_parameter("headless").as_bool();

        // --- Existing tunable parameters ---
        this->declare_parameter("adaptive_block", 51);
        this->declare_parameter("adaptive_c", -25);
        this->declare_parameter("clahe_clip", 3.0);
        this->declare_parameter("sliding_margin", 40);
        this->declare_parameter("sliding_minpix", 15);
        this->declare_parameter("ransac_min_points", 30);
        this->declare_parameter("ransac_min_inliers", 8);
        this->declare_parameter("poly_search_min", 30);
        this->declare_parameter("nwindows", 9);
        this->declare_parameter("smooth_alpha", 0.6);
        this->declare_parameter("lane_width_px", 90);
        this->declare_parameter("persp_top_left_pct", 20);
        this->declare_parameter("persp_top_right_pct", 80);
        this->declare_parameter("persp_top_y_pct", 60);
        this->declare_parameter("persp_bot_left_pct", 0);
        this->declare_parameter("persp_bot_right_pct", 100);
        this->declare_parameter("persp_bot_y_pct", 75);

        // --- New tunable parameters ---
        this->declare_parameter("max_curvature_radius", 0.0);
        this->declare_parameter("use_color_filter", 0);
        this->declare_parameter("sat_max_white", 60);
        this->declare_parameter("recovery_blend_frames", 10);
        this->declare_parameter("max_inertial_frames", 90);
        this->declare_parameter("inertial_straighten_rate", 0.02);
        this->declare_parameter("calib_frames_needed", 0);

        load_tunable_params();

        // Calibration init
        calib_frames_needed_ = this->get_parameter("calib_frames_needed").as_int();
        calib_state_ = (calib_frames_needed_ > 0) ? CalibState::CALIBRATING : CalibState::RUNNING;
        adaptive_c_base_ = (float)adaptive_c_;

        // Register param callback
        param_cb_handle_ = this->add_on_set_parameters_callback(
            std::bind(&LineDetection::on_parameter_change, this, std::placeholders::_1));

        // Publishers — existing (backward compatible)
        center_pub = this->create_publisher<std_msgs::msg::Int16>("/distance_center_line", 10);
        angle_line_pub = this->create_publisher<std_msgs::msg::UInt8>("/angle_line_now", 10);

        // Publishers — new
        confidence_pub_ = this->create_publisher<std_msgs::msg::Float32>(
            "/lane_detection/confidence", 10);
        lane_state_pub_ = this->create_publisher<std_msgs::msg::UInt8>(
            "/lane_detection/lane_state", 10);
        control_ref_pub_ = this->create_publisher<geometry_msgs::msg::Vector3>(
            "/lane_detection/control_ref", 10);

        if (enable_web_view_) {
            debug_image_pub_ = image_transport::create_publisher(this, "/lane_detection/debug_image");
            RCLCPP_INFO(this->get_logger(), "Web view ENABLED");
        }

        // Subscriber
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/ascamera_hp60c/camera_publisher/rgb0/image", 10,
            std::bind(&LineDetection::LineDetectionCb, this, _1));

        if (!headless_) {
            cv::namedWindow(OPENCV_WINDOW, WINDOW_KEEPRATIO);
        }

        // Init state
        distance_center = 0;
        center_cam = 0;
        center_lines = 0;
        angle = 90;
        fps = 0;
        detected = false;
        frames_since_detection_ = 0;
        has_smooth_left_ = false;
        has_smooth_right_ = false;

        memset(polyleft, 0, sizeof(polyleft));
        memset(polyright, 0, sizeof(polyright));
        memset(polyleft_last, 0, sizeof(polyleft_last));
        memset(polyright_last, 0, sizeof(polyright_last));
        memset(polyleft_smooth, 0, sizeof(polyleft_smooth));
        memset(polyright_smooth, 0, sizeof(polyright_smooth));

        clahe_ = createCLAHE(clahe_clip_, Size(8, 8));

        if (calib_state_ == CalibState::CALIBRATING) {
            RCLCPP_INFO(this->get_logger(), "Starting calibration (%d frames)...", calib_frames_needed_);
        }
    }

    void load_tunable_params() {
        adaptive_block_ = this->get_parameter("adaptive_block").as_int();
        adaptive_c_ = this->get_parameter("adaptive_c").as_int();
        clahe_clip_ = this->get_parameter("clahe_clip").as_double();
        sliding_margin_ = this->get_parameter("sliding_margin").as_int();
        sliding_minpix_ = this->get_parameter("sliding_minpix").as_int();
        ransac_min_points_ = this->get_parameter("ransac_min_points").as_int();
        ransac_min_inliers_ = this->get_parameter("ransac_min_inliers").as_int();
        poly_search_min_ = this->get_parameter("poly_search_min").as_int();
        nwindows_ = this->get_parameter("nwindows").as_int();
        smooth_alpha_ = this->get_parameter("smooth_alpha").as_double();
        lane_width_px_ = this->get_parameter("lane_width_px").as_int();
        persp_top_left_pct_ = this->get_parameter("persp_top_left_pct").as_int();
        persp_top_right_pct_ = this->get_parameter("persp_top_right_pct").as_int();
        persp_top_y_pct_ = this->get_parameter("persp_top_y_pct").as_int();
        persp_bot_left_pct_ = this->get_parameter("persp_bot_left_pct").as_int();
        persp_bot_right_pct_ = this->get_parameter("persp_bot_right_pct").as_int();
        persp_bot_y_pct_ = this->get_parameter("persp_bot_y_pct").as_int();
        // New params
        max_curvature_radius_ = this->get_parameter("max_curvature_radius").as_double();
        use_color_filter_ = this->get_parameter("use_color_filter").as_int();
        sat_max_white_ = this->get_parameter("sat_max_white").as_int();
        recovery_blend_frames_ = this->get_parameter("recovery_blend_frames").as_int();
        max_inertial_frames_ = this->get_parameter("max_inertial_frames").as_int();
        inertial_straighten_rate_ = this->get_parameter("inertial_straighten_rate").as_double();

        if (adaptive_block_ % 2 == 0) adaptive_block_++;
        if (adaptive_block_ < 3) adaptive_block_ = 3;
    }

    rcl_interfaces::msg::SetParametersResult on_parameter_change(
        const std::vector<rclcpp::Parameter> &params) {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        for (const auto &p : params) {
            if (p.get_name() == "clahe_clip") {
                clahe_clip_ = p.as_double();
                clahe_ = createCLAHE(clahe_clip_, Size(8, 8));
            } else if (p.get_name() == "adaptive_block") {
                adaptive_block_ = p.as_int();
                if (adaptive_block_ % 2 == 0) adaptive_block_++;
                if (adaptive_block_ < 3) adaptive_block_ = 3;
            } else if (p.get_name() == "adaptive_c") {
                adaptive_c_ = p.as_int();
                adaptive_c_base_ = (float)adaptive_c_;
            } else if (p.get_name() == "sliding_margin") {
                sliding_margin_ = p.as_int();
            } else if (p.get_name() == "sliding_minpix") {
                sliding_minpix_ = p.as_int();
            } else if (p.get_name() == "ransac_min_points") {
                ransac_min_points_ = p.as_int();
            } else if (p.get_name() == "ransac_min_inliers") {
                ransac_min_inliers_ = p.as_int();
            } else if (p.get_name() == "poly_search_min") {
                poly_search_min_ = p.as_int();
            } else if (p.get_name() == "nwindows") {
                nwindows_ = p.as_int();
            } else if (p.get_name() == "smooth_alpha") {
                smooth_alpha_ = p.as_double();
            } else if (p.get_name() == "lane_width_px") {
                lane_width_px_ = p.as_int();
            } else if (p.get_name() == "persp_top_left_pct") {
                persp_top_left_pct_ = p.as_int();
            } else if (p.get_name() == "persp_top_right_pct") {
                persp_top_right_pct_ = p.as_int();
            } else if (p.get_name() == "persp_top_y_pct") {
                persp_top_y_pct_ = p.as_int();
            } else if (p.get_name() == "persp_bot_left_pct") {
                persp_bot_left_pct_ = p.as_int();
            } else if (p.get_name() == "persp_bot_right_pct") {
                persp_bot_right_pct_ = p.as_int();
            } else if (p.get_name() == "persp_bot_y_pct") {
                persp_bot_y_pct_ = p.as_int();
            } else if (p.get_name() == "max_curvature_radius") {
                max_curvature_radius_ = p.as_double();
            } else if (p.get_name() == "use_color_filter") {
                use_color_filter_ = p.as_int();
            } else if (p.get_name() == "sat_max_white") {
                sat_max_white_ = p.as_int();
            } else if (p.get_name() == "recovery_blend_frames") {
                recovery_blend_frames_ = p.as_int();
            } else if (p.get_name() == "max_inertial_frames") {
                max_inertial_frames_ = p.as_int();
            } else if (p.get_name() == "inertial_straighten_rate") {
                inertial_straighten_rate_ = p.as_double();
            }
            RCLCPP_INFO(this->get_logger(), "Parameter '%s' updated", p.get_name().c_str());
        }
        return result;
    }

    ~LineDetection() {
        if (!headless_) {
            cv::destroyWindow(OPENCV_WINDOW);
        }
    }

    // ============================================================
    // Main callback
    // ============================================================
    void LineDetectionCb(const sensor_msgs::msg::Image::SharedPtr msg) {
        set_start_time();
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            frame = cv_ptr->image;
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        static bool logged_size = false;
        if (!logged_size) {
            RCLCPP_INFO(this->get_logger(), "Frame size: %dx%d -> processing at %dx%d",
                        frame.rows, frame.cols, PROC_HEIGHT, PROC_WIDTH);
            logged_size = true;
        }

        Mat original_frame = frame.clone();

        // Resize for fast processing
        Mat proc_frame;
        cv::resize(frame, proc_frame, Size(PROC_WIDTH, PROC_HEIGHT));

        // Setup perspective points
        float tl = PROC_WIDTH * persp_top_left_pct_ / 100.0f;
        float tr = PROC_WIDTH * persp_top_right_pct_ / 100.0f;
        float ty = PROC_HEIGHT * persp_top_y_pct_ / 100.0f;
        float bl = PROC_WIDTH * persp_bot_left_pct_ / 100.0f;
        float br = PROC_WIDTH * persp_bot_right_pct_ / 100.0f;
        float by = PROC_HEIGHT * persp_bot_y_pct_ / 100.0f;

        Source[0] = Point2f(tl, ty);
        Source[1] = Point2f(tr, ty);
        Source[2] = Point2f(bl, by);
        Source[3] = Point2f(br, by);

        Destination[0] = Point2f(bl, 0);
        Destination[1] = Point2f(br, 0);
        Destination[2] = Point2f(bl, PROC_HEIGHT);
        Destination[3] = Point2f(br, PROC_HEIGHT);

        Mat invertedPerspectiveMatrix;
        Perspective(proc_frame, invertedPerspectiveMatrix);
        img_edges = Threshold(proc_frame);

        Mat lane_mask = Mat::zeros(proc_frame.size(), proc_frame.type());

        if (calib_state_ == CalibState::CALIBRATING) {
            calibration_step(img_edges);
            angle = 90;
            distance_center = 0;
        } else {
            locate_lanes(img_edges, proc_frame);
            draw_lines(lane_mask);
        }

        // === BUILD COMPOSITE OUTPUT ===
        Mat warped_mask_small;
        warpPerspective(lane_mask, warped_mask_small, invertedPerspectiveMatrix,
                        Size(PROC_WIDTH, PROC_HEIGHT));

        int out_h = original_frame.rows;
        int out_w = original_frame.cols;

        // LEFT PANEL: camera view with lane overlay + trapezoid
        Mat warped_mask;
        cv::resize(warped_mask_small, warped_mask, Size(out_w, out_h));
        addWeighted(original_frame, 1.0, warped_mask, 1.0, 0, frame);

        // Draw perspective trapezoid
        {
            float sx = (float)out_w / PROC_WIDTH;
            float sy = (float)out_h / PROC_HEIGHT;
            Point pts[4] = {
                Point((int)(tl * sx), (int)(ty * sy)),
                Point((int)(tr * sx), (int)(ty * sy)),
                Point((int)(br * sx), (int)(by * sy)),
                Point((int)(bl * sx), (int)(by * sy))
            };
            for (int i = 0; i < 4; i++)
                line(frame, pts[i], pts[(i + 1) % 4], Scalar(0, 255, 255), 2);
        }

        // FPS + angle
        ss.str(" ");
        ss.clear();
        ss << "[ANG]:" << angle << " [FPS]:" << fps;
        putText(frame, ss.str(), Point2f(2, 20), 0, 0.7, Scalar(255, 0, 0), 2);

        // State + confidence overlay
        if (calib_state_ == CalibState::CALIBRATING) {
            char cal_buf[64];
            snprintf(cal_buf, sizeof(cal_buf), "CALIBRATING %d/%d",
                     calib_frame_count_, calib_frames_needed_);
            putText(frame, cal_buf, Point2f(2, 45), 0, 0.6, Scalar(0, 200, 255), 2);
            int progress_w = (int)(200.0f * calib_frame_count_ / calib_frames_needed_);
            rectangle(frame, Point(2, 55), Point(2 + progress_w, 65), Scalar(0, 200, 255), -1);
        } else {
            const char* state_names[] = {"BOTH", "ONE", "NONE", "RECOVER"};
            char state_buf[64];
            snprintf(state_buf, sizeof(state_buf), "[STATE]:%s [CONF]:%.0f%%",
                     state_names[(int)lane_state_], last_confidence_ * 100);
            putText(frame, state_buf, Point2f(2, 45), 0, 0.5, Scalar(255, 128, 0), 1);

            // Illumination bar
            int bar_w = (int)(illum_ema_ * 200);
            Scalar bar_color;
            if (illum_ema_ > 0.15f)
                bar_color = Scalar(0, 0, 255);
            else if (illum_ema_ < 0.03f)
                bar_color = Scalar(255, 0, 0);
            else
                bar_color = Scalar(0, 255, 0);
            rectangle(frame, Point(10, out_h - 30),
                      Point(10 + bar_w, out_h - 20), bar_color, -1);
            char illum_buf[64];
            snprintf(illum_buf, sizeof(illum_buf), "C=%d dens=%.1f%%",
                     adaptive_c_, illum_ema_ * 100);
            putText(frame, illum_buf, Point(10, out_h - 35),
                    FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);
        }

        // RIGHT PANEL: bird-eye threshold
        Mat birdeye_bgr;
        cvtColor(img_edges, birdeye_bgr, COLOR_GRAY2BGR);
        for (const auto& r : sliding_window_rects_) {
            rectangle(birdeye_bgr, r, Scalar(255, 255, 255), 1);
        }
        addWeighted(birdeye_bgr, 1.0, lane_mask, 0.8, 0, birdeye_bgr);
        Mat birdeye_panel;
        cv::resize(birdeye_bgr, birdeye_panel, Size(out_w, out_h), 0, 0, INTER_NEAREST);
        putText(birdeye_panel, "BIRD-EYE THRESHOLD", Point(8, 24),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);

        // Concatenate
        Mat composite;
        hconcat(frame, birdeye_panel, composite);

        if (enable_web_view_) {
            sensor_msgs::msg::Image::SharedPtr debug_msg =
                cv_bridge::CvImage(msg->header, "bgr8", composite).toImageMsg();
            debug_image_pub_.publish(*debug_msg);
        }

        if (!headless_) {
            resizeWindow(OPENCV_WINDOW, composite.cols, composite.rows);
            cv::imshow(OPENCV_WINDOW, composite);
            cv::waitKey(1);
        }

        // Publish backward-compatible messages
        center_message.data = distance_center;
        center_pub->publish(center_message);
        angle_line_message.data = angle;
        angle_line_pub->publish(angle_line_message);

        set_end_time();
        fps = FPS_subscriber();
        RCLCPP_INFO(this->get_logger(), "[FPS]: %i ", fps);
    }

    // ============================================================
    // Timing
    // ============================================================
    void set_start_time() { start = std::chrono::system_clock::now(); }
    void set_end_time()   { end = std::chrono::system_clock::now(); }

    int FPS_subscriber() {
        elapsed_seconds = end - start;
        double t_ms = elapsed_seconds.count() * 1000.0;
        return (t_ms > 0) ? (int)(1000.0 / t_ms) : 0;
    }

    // ============================================================
    // Perspective transform
    // ============================================================
    void Perspective(Mat &frame, Mat &invertedPerspectiveMatrix) {
        Mat matrixPerspective = getPerspectiveTransform(Source, Destination);
        warpPerspective(frame, frame, matrixPerspective, Size(frame.cols, frame.rows));
        invert(matrixPerspective, invertedPerspectiveMatrix);
    }

    // ============================================================
    // Threshold — with color filter (Mejora 6) and illumination
    // adaptation (Mejora 1B)
    // ============================================================
    Mat Threshold(Mat frame) {
        Mat frameHLS;
        cvtColor(frame, frameHLS, COLOR_BGR2HLS);

        std::vector<Mat> channels;
        split(frameHLS, channels);
        Mat L = channels[1];
        birdeye_S_channel_ = channels[2];

        Mat L_clahe;
        clahe_->apply(L, L_clahe);

        Mat binary;
        adaptiveThreshold(L_clahe, binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
                          THRESH_BINARY, adaptive_block_, adaptive_c_);

        Mat kernel_open = getStructuringElement(MORPH_RECT, Size(3, 3));
        Mat kernel_close = getStructuringElement(MORPH_RECT, Size(5, 5));
        morphologyEx(binary, binary, MORPH_OPEN, kernel_open);
        morphologyEx(binary, binary, MORPH_CLOSE, kernel_close);

        // Color filter (Mejora 6): keep only low-saturation (white) pixels
        if (use_color_filter_) {
            Mat white_mask;
            inRange(birdeye_S_channel_, 0, sat_max_white_, white_mask);
            binary = binary & white_mask;
        }

        // Illumination adaptation (Mejora 1B)
        int white_count = cv::countNonZero(binary);
        update_illumination(white_count, binary.total());

        return binary;
    }

    // ============================================================
    // Illumination adaptation (Mejora 1B)
    // Conservative: slow drift (±0.1/frame), wide dead zone (2%-20%),
    // tight clamp (-35 to -15). Only reacts to extreme lighting.
    // ============================================================
    void update_illumination(int white_pixels, size_t total_pixels) {
        float density = (float)white_pixels / (float)total_pixels;

        if (!illum_has_history_) {
            illum_ema_ = density;
            illum_has_history_ = true;
        } else {
            illum_ema_ = 0.05f * density + 0.95f * illum_ema_;
        }

        // Only adapt in extreme conditions, slow drift
        if (illum_ema_ > 0.20f) {
            // Too bright — make threshold more demanding
            adaptive_c_base_ = std::max(-35.0f, adaptive_c_base_ - 0.1f);
        } else if (illum_ema_ < 0.02f) {
            // Too dark — make threshold more permissive
            adaptive_c_base_ = std::min(-15.0f, adaptive_c_base_ + 0.1f);
        }
        // Between 2%-20%: don't touch — normal range

        adaptive_c_ = (int)adaptive_c_base_;
    }

    // ============================================================
    // Robust peak detection (Mejora 3)
    // ============================================================
    std::vector<Peak> find_peaks(const std::vector<int>& histogram,
                                  int min_distance, int min_height) {
        std::vector<Peak> peaks;
        int n = histogram.size();

        for (int i = 1; i < n - 1; i++) {
            if (histogram[i] > min_height &&
                histogram[i] >= histogram[i - 1] &&
                histogram[i] >= histogram[i + 1]) {

                bool is_dominant = true;
                for (auto it = peaks.begin(); it != peaks.end(); ) {
                    if (std::abs(it->position - i) < min_distance) {
                        if (histogram[i] > it->strength) {
                            it = peaks.erase(it);
                        } else {
                            is_dominant = false;
                            break;
                        }
                    } else {
                        ++it;
                    }
                }
                if (is_dominant) {
                    peaks.push_back({i, histogram[i]});
                }
            }
        }

        std::sort(peaks.begin(), peaks.end(),
                  [](const Peak& a, const Peak& b) { return a.strength > b.strength; });
        return peaks;
    }

    // ============================================================
    // Histogram — with robust peak detection (Mejora 3)
    // ============================================================
    int *Histogram(Mat &img) {
        static int LanePosition[2];
        vector<int> histogramLane(img.cols, 0);

        int start_row = img.rows * 2 / 3;
        for (int c = 0; c < img.cols; c++) {
            for (int r = start_row; r < img.rows; r++) {
                if (img.at<uchar>(r, c) > 0) {
                    histogramLane[c]++;
                }
            }
        }

        // Smooth histogram
        vector<int> smoothed(img.cols, 0);
        int hw = 5;
        for (int c = hw; c < img.cols - hw; c++) {
            int sum = 0;
            for (int k = -hw; k <= hw; k++) sum += histogramLane[c + k];
            smoothed[c] = sum;
        }

        // Robust peak detection
        int min_distance = img.cols / 4;
        int min_height = 5;
        auto peaks = find_peaks(smoothed, min_distance, min_height);

        if (peaks.size() >= 2) {
            LanePosition[0] = std::min(peaks[0].position, peaks[1].position);
            LanePosition[1] = std::max(peaks[0].position, peaks[1].position);
        } else if (peaks.size() == 1) {
            float w = (calibrated_lane_width_ > 0) ? calibrated_lane_width_ : (float)lane_width_px_;
            if (peaks[0].position < img.cols / 2) {
                LanePosition[0] = peaks[0].position;
                LanePosition[1] = peaks[0].position + (int)w;
            } else {
                LanePosition[1] = peaks[0].position;
                LanePosition[0] = peaks[0].position - (int)w;
            }
        } else {
            LanePosition[0] = img.cols / 4;
            LanePosition[1] = img.cols * 3 / 4;
        }

        return LanePosition;
    }

    // ============================================================
    // Lane location
    // ============================================================
    void locate_lanes(Mat &img, Mat &out_img) {
        (void)out_img;
        if (detected && frames_since_detection_ < 5) {
            search_around_poly(img);
        } else {
            sliding_window_search(img);
        }
    }

    void search_around_poly(Mat &img) {
        left_points.clear();
        right_points.clear();
        sliding_window_rects_.clear();
        int margin = sliding_margin_;

        int nstripes = nwindows_;
        int stripe_h = img.rows / nstripes;

        for (int s = 0; s < nstripes; s++) {
            int y_low = s * stripe_h;
            int y_high = std::min((s + 1) * stripe_h, img.rows);
            int r_mid = (y_low + y_high) / 2;

            int lc = (int)(polyleft_smooth[0] + polyleft_smooth[1] * r_mid + polyleft_smooth[2] * r_mid * r_mid);
            int rc = (int)(polyright_smooth[0] + polyright_smooth[1] * r_mid + polyright_smooth[2] * r_mid * r_mid);

            int ll = std::max(0, lc - margin);
            int lh = std::min(img.cols, lc + margin);
            int rl = std::max(0, rc - margin);
            int rh = std::min(img.cols, rc + margin);

            sliding_window_rects_.push_back(Rect(ll, y_low, lh - ll, y_high - y_low));
            sliding_window_rects_.push_back(Rect(rl, y_low, rh - rl, y_high - y_low));
        }

        for (int r = 0; r < img.rows; r++) {
            int left_center = (int)(polyleft_smooth[0] + polyleft_smooth[1] * r + polyleft_smooth[2] * r * r);
            int right_center = (int)(polyright_smooth[0] + polyright_smooth[1] * r + polyright_smooth[2] * r * r);

            int left_low  = std::max(0, left_center - margin);
            int left_high = std::min(img.cols - 1, left_center + margin);
            int right_low  = std::max(0, right_center - margin);
            int right_high = std::min(img.cols - 1, right_center + margin);

            for (int c = left_low; c < left_high; c++) {
                if (img.at<uchar>(r, c) > 0) {
                    left_points.push_back(Point(r, c));
                }
            }
            for (int c = right_low; c < right_high; c++) {
                if (img.at<uchar>(r, c) > 0) {
                    right_points.push_back(Point(r, c));
                }
            }
        }

        if ((int)left_points.size() < poly_search_min_ || (int)right_points.size() < poly_search_min_) {
            detected = false;
        }
    }

    void sliding_window_search(Mat &img) {
        left_points.clear();
        right_points.clear();
        sliding_window_rects_.clear();

        int nwindows = nwindows_;
        int window_height = img.rows / nwindows;
        int margin = sliding_margin_;
        int minpix = sliding_minpix_;

        int *locate_histogram = Histogram(img);
        int leftx_current = locate_histogram[0];
        int rightx_current = locate_histogram[1];

        for (int window = 0; window < nwindows; window++) {
            int win_y_low  = img.rows - (window + 1) * window_height;
            int win_y_high = img.rows - window * window_height;

            int win_xleft_low   = std::max(0, leftx_current - margin);
            int win_xleft_high  = std::min(img.cols, leftx_current + margin);
            int win_xright_low  = std::max(0, rightx_current - margin);
            int win_xright_high = std::min(img.cols, rightx_current + margin);

            sliding_window_rects_.push_back(Rect(win_xleft_low, win_y_low,
                win_xleft_high - win_xleft_low, win_y_high - win_y_low));
            sliding_window_rects_.push_back(Rect(win_xright_low, win_y_low,
                win_xright_high - win_xright_low, win_y_high - win_y_low));

            win_y_low  = std::max(0, win_y_low);
            win_y_high = std::min(img.rows, win_y_high);

            int mean_leftx = 0, mean_rightx = 0;
            int count_left = 0, count_right = 0;

            for (int r = win_y_low; r < win_y_high; r++) {
                for (int c = win_xleft_low; c < win_xleft_high; c++) {
                    if (img.at<uchar>(r, c) > 0) {
                        left_points.push_back(Point(r, c));
                        mean_leftx += c;
                        count_left++;
                    }
                }
                for (int c = win_xright_low; c < win_xright_high; c++) {
                    if (img.at<uchar>(r, c) > 0) {
                        right_points.push_back(Point(r, c));
                        mean_rightx += c;
                        count_right++;
                    }
                }
            }

            if (count_left >= minpix) {
                leftx_current = mean_leftx / count_left;
            }
            if (count_right >= minpix) {
                rightx_current = mean_rightx / count_right;
            }
        }
    }

    // ============================================================
    // Polynomial regression with RANSAC
    // Modified: outputs inlier_ratio + curvature validation (Mejora 4,7)
    // ============================================================
    bool regression_left() {
        if ((int)left_points.size() < ransac_min_points_) {
            RCLCPP_DEBUG(this->get_logger(), "LEFT: only %d pts (need %d)",
                         (int)left_points.size(), ransac_min_points_);
            left_inlier_ratio_ = 0.0f;
            return false;
        }
        bool ok = polyfit_ransac(left_points, polyleft, 150, 12.0, &left_inlier_ratio_);
        if (!ok) {
            ok = linear_fit(left_points, polyleft);
            if (ok) {
                left_inlier_ratio_ = 0.5f;
                RCLCPP_INFO(this->get_logger(), "LEFT: RANSAC fail, linear fallback OK (%d pts)",
                            (int)left_points.size());
            }
        }
        if (ok) {
            // Sanity check: bottom of image
            float bot_col = polyleft[0] + polyleft[1] * (PROC_HEIGHT - 1) +
                           polyleft[2] * (PROC_HEIGHT - 1) * (PROC_HEIGHT - 1);
            if (bot_col < -50 || bot_col > PROC_WIDTH + 50) {
                RCLCPP_INFO(this->get_logger(), "LEFT: rejected (bot_col=%.0f out of range)", bot_col);
                ok = false;
            }
        }
        // Curvature validation (Mejora 4)
        if (ok) {
            ok = validate_curvature(polyleft, "LEFT");
        }
        return ok;
    }

    bool regression_right() {
        if ((int)right_points.size() < ransac_min_points_) {
            RCLCPP_DEBUG(this->get_logger(), "RIGHT: only %d pts (need %d)",
                         (int)right_points.size(), ransac_min_points_);
            right_inlier_ratio_ = 0.0f;
            return false;
        }
        bool ok = polyfit_ransac(right_points, polyright, 150, 12.0, &right_inlier_ratio_);
        if (!ok) {
            ok = linear_fit(right_points, polyright);
            if (ok) {
                right_inlier_ratio_ = 0.5f;
                RCLCPP_INFO(this->get_logger(), "RIGHT: RANSAC fail, linear fallback OK (%d pts)",
                            (int)right_points.size());
            }
        }
        if (ok) {
            float bot_col = polyright[0] + polyright[1] * (PROC_HEIGHT - 1) +
                           polyright[2] * (PROC_HEIGHT - 1) * (PROC_HEIGHT - 1);
            if (bot_col < -50 || bot_col > PROC_WIDTH + 50) {
                RCLCPP_INFO(this->get_logger(), "RIGHT: rejected (bot_col=%.0f out of range)", bot_col);
                ok = false;
            }
        }
        if (ok) {
            ok = validate_curvature(polyright, "RIGHT");
        }
        return ok;
    }

    bool linear_fit(const std::vector<Point>& points, float* coeffs) {
        if (points.size() < 2) return false;
        float sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
        int n = points.size();
        for (const auto& p : points) {
            float row = (float)p.x;
            float col = (float)p.y;
            sum_x += row; sum_y += col;
            sum_xy += row * col; sum_xx += row * row;
        }
        float denom = n * sum_xx - sum_x * sum_x;
        if (std::abs(denom) < 1e-6f) return false;
        coeffs[1] = (n * sum_xy - sum_x * sum_y) / denom;
        coeffs[0] = (sum_y - coeffs[1] * sum_x) / n;
        coeffs[2] = 0.0f;
        return true;
    }

    bool polyfit_ransac(const std::vector<Point>& points, float* coeffs,
                        int iterations, double threshold,
                        float* out_inlier_ratio = nullptr) {
        if (points.size() < 3) return false;

        // Subsample if too many points — keeps RANSAC fast on Jetson
        const int MAX_RANSAC_PTS = 500;
        const std::vector<Point>* pts_ptr = &points;
        std::vector<Point> sampled;
        if ((int)points.size() > MAX_RANSAC_PTS) {
            sampled.reserve(MAX_RANSAC_PTS);
            auto& gen = get_rng();
            std::uniform_int_distribution<> dis(0, (int)points.size() - 1);
            for (int i = 0; i < MAX_RANSAC_PTS; i++) {
                sampled.push_back(points[dis(gen)]);
            }
            pts_ptr = &sampled;
        }

        const auto& pts = *pts_ptr;
        int n = pts.size();
        std::vector<int> best_inliers;
        std::vector<int> current_inliers;
        current_inliers.reserve(n);
        bool found_model = false;

        auto& gen = get_rng();
        std::uniform_int_distribution<> dis(0, n - 1);

        // 80 iterations is sufficient (>99.9% chance with 50%+ inliers)
        int iters = std::min(iterations, 80);
        for (int k = 0; k < iters; ++k) {
            int i0 = dis(gen), i1 = dis(gen), i2 = dis(gen);
            if (i0 == i1 || i1 == i2 || i0 == i2) continue;

            Eigen::Matrix3f A;
            Eigen::Vector3f b;
            const Point* p3[3] = {&pts[i0], &pts[i1], &pts[i2]};
            for (int i = 0; i < 3; ++i) {
                float row = (float)p3[i]->x;
                A(i, 0) = 1.0f;
                A(i, 1) = row;
                A(i, 2) = row * row;
                b(i) = (float)p3[i]->y;
            }

            Eigen::Vector3f x = A.colPivHouseholderQr().solve(b);

            current_inliers.clear();
            for (int i = 0; i < n; ++i) {
                float row = (float)pts[i].x;
                float y_pred = x(0) + x(1) * row + x(2) * row * row;
                if (std::abs(pts[i].y - y_pred) < threshold) {
                    current_inliers.push_back(i);
                }
            }

            if (current_inliers.size() > best_inliers.size()) {
                best_inliers = current_inliers;
                found_model = true;
            }
        }

        if (!found_model || (int)best_inliers.size() < ransac_min_inliers_) return false;

        if (out_inlier_ratio) {
            *out_inlier_ratio = (float)best_inliers.size() / (float)n;
        }

        // Refit on all inliers
        Eigen::MatrixXf A_all(best_inliers.size(), 3);
        Eigen::VectorXf b_all(best_inliers.size());
        for (size_t i = 0; i < best_inliers.size(); ++i) {
            int idx = best_inliers[i];
            float row = (float)pts[idx].x;
            A_all(i, 0) = 1.0f;
            A_all(i, 1) = row;
            A_all(i, 2) = row * row;
            b_all(i) = (float)pts[idx].y;
        }

        Eigen::Vector3f final_coeffs = A_all.colPivHouseholderQr().solve(b_all);
        coeffs[0] = final_coeffs(0);
        coeffs[1] = final_coeffs(1);
        coeffs[2] = final_coeffs(2);

        return true;
    }

    // ============================================================
    // Curvature validation (Mejora 4)
    // ============================================================
    bool validate_curvature(float* coeffs, const char* side) {
        float second_deriv = 2.0f * coeffs[2];
        if (std::abs(second_deriv) > 1e-6f) {
            float mid_row = PROC_HEIGHT / 2.0f;
            float first_deriv = coeffs[1] + 2.0f * coeffs[2] * mid_row;
            float radius = std::pow(1.0f + first_deriv * first_deriv, 1.5f)
                          / std::abs(second_deriv);
            if (radius < max_curvature_radius_) {
                RCLCPP_INFO(this->get_logger(),
                    "%s: curvature rejected R=%.1f px < min %.1f",
                    side, radius, max_curvature_radius_);
                return false;
            }
        }
        return true;
    }

    // ============================================================
    // Temporal smoothing — EMA
    // ============================================================
    void smooth_polynomial(float* raw, float* smooth, bool& has_history) {
        float alpha = (float)smooth_alpha_;
        if (!has_history) {
            for (int i = 0; i < 3; i++) smooth[i] = raw[i];
            has_history = true;
        } else {
            for (int i = 0; i < 3; i++) {
                smooth[i] = alpha * raw[i] + (1.0f - alpha) * smooth[i];
            }
        }
    }

    // ============================================================
    // Calibration (Mejora 1A)
    // ============================================================
    void calibration_step(Mat& birdeye_binary) {
        // Measure width between histogram peaks
        int* peaks = Histogram(birdeye_binary);
        float width = (float)(peaks[1] - peaks[0]);
        if (width > 20 && width < PROC_WIDTH * 0.8f) {
            calib_widths_.push_back(width);
        }

        // Collect saturation values of detected pixels
        for (int r = PROC_HEIGHT / 2; r < PROC_HEIGHT; r += 4) {
            for (int c = 0; c < PROC_WIDTH; c += 4) {
                if (birdeye_binary.at<uchar>(r, c) > 0) {
                    float s = (float)birdeye_S_channel_.at<uchar>(r, c);
                    calib_sat_values_.push_back(s);
                    if (s > 40) {
                        calib_color_votes_++;
                    } else {
                        calib_white_votes_++;
                    }
                }
            }
        }

        calib_frame_count_++;
        if (calib_frame_count_ >= calib_frames_needed_) {
            finalize_calibration();
        }
    }

    void finalize_calibration() {
        // Lane width: median
        if (!calib_widths_.empty()) {
            std::sort(calib_widths_.begin(), calib_widths_.end());
            calibrated_lane_width_ = calib_widths_[calib_widths_.size() / 2];
            lane_width_px_ = (int)calibrated_lane_width_;
        }

        // Saturation threshold: mean + 2*std
        if (!calib_sat_values_.empty()) {
            float mean_s = 0, var_s = 0;
            for (float s : calib_sat_values_) mean_s += s;
            mean_s /= calib_sat_values_.size();
            for (float s : calib_sat_values_) var_s += (s - mean_s) * (s - mean_s);
            var_s = std::sqrt(var_s / calib_sat_values_.size());
            sat_max_white_ = (int)(mean_s + 2.0f * var_s);
            sat_max_white_ = std::clamp(sat_max_white_, 30, 120);
        }

        lines_are_white_ = (calib_white_votes_ > calib_color_votes_);
        calib_state_ = CalibState::RUNNING;

        RCLCPP_INFO(this->get_logger(),
            "CALIBRATION COMPLETE: lane_width=%d px, sat_max=%d, lines=%s",
            lane_width_px_, sat_max_white_, lines_are_white_ ? "white" : "colored");
    }

    // ============================================================
    // Lane state machine (Mejora 8)
    // ============================================================
    void update_lane_state(bool left_ok, bool right_ok) {
        LaneState prev_state = lane_state_;

        if (left_ok && right_ok) {
            if (prev_state == LaneState::NO_LINES ||
                prev_state == LaneState::ONE_LINE) {
                lane_state_ = LaneState::RECOVERING;
                frames_recovering_ = 0;
            } else if (prev_state == LaneState::RECOVERING) {
                frames_recovering_++;
                if (frames_recovering_ >= recovery_blend_frames_) {
                    lane_state_ = LaneState::BOTH_LINES;
                }
            } else {
                lane_state_ = LaneState::BOTH_LINES;
            }
            frames_no_lines_ = 0;

        } else if (left_ok || right_ok) {
            lane_state_ = LaneState::ONE_LINE;
            frames_no_lines_ = 0;

        } else {
            lane_state_ = LaneState::NO_LINES;
            frames_no_lines_++;
        }
    }

    void navigate_inertial(Mat &img) {
        float target = 90.0f;
        float blend = std::min(1.0f,
            frames_no_lines_ * (float)inertial_straighten_rate_);
        float inertial_angle = last_good_angle_ * (1.0f - blend)
                             + target * blend;

        // Maintain turning rate briefly if entering no-lines during a curve
        if (frames_no_lines_ < 15) {
            inertial_angle += angle_rate_ * (15 - frames_no_lines_);
        }

        angle = (int)std::clamp(inertial_angle, 45.0f, 135.0f);
        distance_center = 0;
        center_lines = center_cam;

        // Visual indicators
        putText(img, "INERTIAL NAV", Point(10, img.rows - 20),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 165, 255), 2);

        float fuel = 1.0f - (float)frames_no_lines_ / max_inertial_frames_;
        fuel = std::max(0.0f, fuel);
        int bar_width = (int)(fuel * 100);
        rectangle(img, Point(10, img.rows - 10),
                  Point(10 + bar_width, img.rows - 5),
                  Scalar(0, (int)(fuel * 255), (int)((1-fuel) * 255)), -1);
    }

    void blend_recovery(float new_angle, float new_distance) {
        float t = (float)frames_recovering_ / recovery_blend_frames_;
        t = t * t * (3.0f - 2.0f * t);  // Smoothstep

        angle = (int)(last_good_angle_ * (1.0f - t) + new_angle * t);
        distance_center = (int)(last_good_distance_ * (1.0f - t)
                              + new_distance * t);
    }

    // ============================================================
    // Adaptive lane width (Mejora 5)
    // ============================================================
    void update_adaptive_width(float* dl, float* dr, Mat &img) {
        float total_w = 0;
        float weights[] = {0.5f, 0.3f, 0.2f};
        float rows[] = {(float)(img.rows - 1), (float)(img.rows / 2), 0.0f};

        for (int i = 0; i < 3; i++) {
            float r = rows[i];
            float lc = dl[0] + dl[1]*r + dl[2]*r*r;
            float rc = dr[0] + dr[1]*r + dr[2]*r*r;
            total_w += (rc - lc) * weights[i];
        }

        if (!has_width_history_) {
            width_ema_ = total_w;
            has_width_history_ = true;
        } else {
            width_ema_ = 0.7f * total_w + 0.3f * width_ema_;
        }
    }

    float get_effective_width() {
        if (has_width_history_) return width_ema_;
        if (calibrated_lane_width_ > 0) return calibrated_lane_width_;
        return (float)lane_width_px_;
    }

    // ============================================================
    // Confidence computation (Mejora 7)
    // ============================================================
    float compute_confidence(bool left_ok, bool right_ok,
                              int n_left_pts, int n_right_pts) {
        float conf = 0.0f;

        // Factor 1: inlier ratio (40%)
        float ir = 0.0f;
        int count = 0;
        if (left_ok)  { ir += left_inlier_ratio_;  count++; }
        if (right_ok) { ir += right_inlier_ratio_; count++; }
        if (count > 0) ir /= count;
        conf += ir * 0.4f;

        // Factor 2: point count (30%)
        float pt_score = std::min(1.0f, (n_left_pts + n_right_pts) / 400.0f);
        conf += pt_score * 0.3f;

        // Factor 3: both detected (30%)
        conf += (left_ok && right_ok) ? 0.3f : 0.15f;

        // Multiply by state factor
        conf *= state_confidence_factor();

        return std::clamp(conf, 0.0f, 1.0f);
    }

    float state_confidence_factor() {
        switch (lane_state_) {
            case LaneState::BOTH_LINES:  return 1.0f;
            case LaneState::ONE_LINE:    return 0.7f;
            case LaneState::NO_LINES: {
                float fuel = 1.0f - (float)frames_no_lines_ / max_inertial_frames_;
                return std::max(0.1f, 0.5f * fuel);
            }
            case LaneState::RECOVERING: {
                float t = (float)frames_recovering_ / recovery_blend_frames_;
                return 0.5f + 0.5f * t;
            }
        }
        return 0.0f;
    }

    // ============================================================
    // Control reference publishing (Mejora 10)
    // ============================================================
    void publish_control_ref(float* dl, float* dr) {
        // CTE: cross-track error at bottom (near robot)
        float bot = PROC_HEIGHT - 1;
        float colL_bot = dl[0] + dl[1] * bot + dl[2] * bot * bot;
        float colR_bot = dr[0] + dr[1] * bot + dr[2] * bot * bot;
        float center_lines_bot = (colL_bot + colR_bot) / 2.0f;
        float half_width = (float)lane_width_px_ / 2.0f;
        float cte = ((float)center_cam - center_lines_bot) / half_width;
        cte = std::clamp(cte, -1.0f, 1.0f);

        // Heading error
        float row_near = PROC_HEIGHT - 1;
        float row_far  = PROC_HEIGHT * 0.3f;
        float center_near = 0.5f * (dl[0] + dl[1]*row_near + dl[2]*row_near*row_near
                                  + dr[0] + dr[1]*row_near + dr[2]*row_near*row_near);
        float center_far  = 0.5f * (dl[0] + dl[1]*row_far + dl[2]*row_far*row_far
                                  + dr[0] + dr[1]*row_far + dr[2]*row_far*row_far);
        float lane_dx = center_far - center_near;
        float lane_dy = row_far - row_near;
        float lane_angle = atan2f(lane_dx, -lane_dy);
        float heading_error = std::clamp(-lane_angle / 0.7854f, -1.0f, 1.0f);

        // Curvature anticipada (negated: positive curvature = turn right)
        float avg_p2 = 0.5f * (dl[2] + dr[2]);
        float curvature = std::clamp(-avg_p2 / 0.003f, -1.0f, 1.0f);

        geometry_msgs::msg::Vector3 ref;
        ref.x = cte;
        ref.y = heading_error;
        ref.z = curvature;
        control_ref_pub_->publish(ref);
    }

    // ============================================================
    // Draw helpers
    // ============================================================
    void draw_and_compute_steering(Mat &img, float* dl, float* dr,
                                    bool left_inferred, bool right_inferred) {
        float columnL, columnR, row;
        Scalar colorL = left_inferred  ? Scalar(255, 255, 0) : Scalar(0, 255, 0);
        Scalar colorR = right_inferred ? Scalar(255, 255, 0) : Scalar(0, 255, 0);

        for (row = img.rows - 1; row >= 0; row -= 4) {
            columnR = dr[0] + dr[1] * row + dr[2] * row * row;
            columnL = dl[0] + dl[1] * row + dl[2] * row * row;
            circle(img, cv::Point((int)columnR, (int)row), 2, colorR, 2);
            circle(img, cv::Point((int)columnL, (int)row), 2, colorL, 2);
        }

        center_lines = (int)((columnR + columnL) / 2);
        distance_center = center_cam - center_lines;

        if (distance_center == 0) {
            angle = 90;
        } else {
            float angle_to_mid_radian = atan2f((float)(img.rows - 1),
                                                (float)(center_lines - center_cam));
            angle = (int)(angle_to_mid_radian * 57.295779f);
            if (angle < 0) angle = -angle;
            if (angle > 90 && angle < 180) angle = 180 - angle;
        }
    }

    void draw_lanes_dim(Mat &img, float* dl, float* dr) {
        float columnL, columnR, row;
        for (row = img.rows - 1; row >= 0; row -= 4) {
            columnR = dr[0] + dr[1] * row + dr[2] * row * row;
            columnL = dl[0] + dl[1] * row + dl[2] * row * row;
            circle(img, cv::Point((int)columnR, (int)row), 2, Scalar(0, 150, 0), 2);
            circle(img, cv::Point((int)columnL, (int)row), 2, Scalar(0, 150, 0), 2);
        }
        center_lines = (int)((columnR + columnL) / 2);
    }

    // ============================================================
    // Draw lines and compute steering — with state machine
    // ============================================================
    void draw_lines(Mat &img) {
        // Save point counts before clearing
        int n_left_pts = (int)left_points.size();
        int n_right_pts = (int)right_points.size();

        bool find_line_right = regression_right();
        bool find_line_left = regression_left();

        // Diagnostics
        static int diag_counter = 0;
        if (++diag_counter % 30 == 0) {
            float lb = polyleft[0] + polyleft[1] * (PROC_HEIGHT-1) +
                      polyleft[2] * (PROC_HEIGHT-1)*(PROC_HEIGHT-1);
            float rb = polyright[0] + polyright[1] * (PROC_HEIGHT-1) +
                      polyright[2] * (PROC_HEIGHT-1)*(PROC_HEIGHT-1);
            RCLCPP_INFO(this->get_logger(),
                "[DIAG] L_pts=%d R_pts=%d L_ok=%d R_ok=%d L_bot=%.0f R_bot=%.0f",
                n_left_pts, n_right_pts, find_line_left, find_line_right, lb, rb);
        }

        // Sanity check: right > left
        if (find_line_left && find_line_right) {
            float lb = polyleft[0] + polyleft[1] * (img.rows-1) +
                      polyleft[2] * (img.rows-1)*(img.rows-1);
            float rb = polyright[0] + polyright[1] * (img.rows-1) +
                      polyright[2] * (img.rows-1)*(img.rows-1);
            if (rb < lb + 20) {
                RCLCPP_INFO(this->get_logger(),
                    "SANITY: right(%.0f) < left(%.0f)+20, rejecting right", rb, lb);
                find_line_right = false;
            }
        }

        // Temporal smoothing
        if (find_line_left)
            smooth_polynomial(polyleft, polyleft_smooth, has_smooth_left_);
        if (find_line_right)
            smooth_polynomial(polyright, polyright_smooth, has_smooth_right_);

        // Update lane state machine (Mejora 8)
        update_lane_state(find_line_left, find_line_right);

        // Update detection for locate_lanes
        detected = (lane_state_ == LaneState::BOTH_LINES ||
                    lane_state_ == LaneState::RECOVERING);
        if (detected) frames_since_detection_ = 0;
        else frames_since_detection_++;

        // Track angle rate before modifying angle
        angle_rate_ = (float)angle - prev_angle_;
        prev_angle_ = (float)angle;

        // Clear points and set center
        right_points.clear();
        left_points.clear();
        center_cam = (img.cols / 2) - 5;

        // Choose smoothed or last polynomials
        float* draw_left  = has_smooth_left_  ? polyleft_smooth  : polyleft_last;
        float* draw_right = has_smooth_right_ ? polyright_smooth : polyright_last;

        // State-dependent processing
        float inferred_left[3], inferred_right[3];
        bool left_inferred = false, right_inferred = false;
        bool have_valid_polys = false;

        switch (lane_state_) {
            case LaneState::BOTH_LINES: {
                // Adaptive width (Mejora 5)
                update_adaptive_width(draw_left, draw_right, img);
                // Draw and compute steering
                draw_and_compute_steering(img, draw_left, draw_right, false, false);
                // Save as last good
                last_good_angle_ = (float)angle;
                last_good_distance_ = (float)distance_center;
                have_valid_polys = true;
                break;
            }

            case LaneState::ONE_LINE: {
                float ew = get_effective_width();
                if (find_line_left && !find_line_right) {
                    inferred_right[0] = draw_left[0] + ew;
                    inferred_right[1] = draw_left[1];
                    inferred_right[2] = draw_left[2];
                    draw_right = inferred_right;
                    right_inferred = true;
                    smooth_polynomial(inferred_right, polyright_smooth, has_smooth_right_);
                } else if (find_line_right && !find_line_left) {
                    inferred_left[0] = draw_right[0] - ew;
                    inferred_left[1] = draw_right[1];
                    inferred_left[2] = draw_right[2];
                    draw_left = inferred_left;
                    left_inferred = true;
                    smooth_polynomial(inferred_left, polyleft_smooth, has_smooth_left_);
                }
                draw_and_compute_steering(img, draw_left, draw_right,
                                          left_inferred, right_inferred);
                last_good_angle_ = (float)angle;
                last_good_distance_ = (float)distance_center;
                have_valid_polys = true;
                break;
            }

            case LaneState::NO_LINES: {
                draw_lanes_dim(img, polyleft_last, polyright_last);
                navigate_inertial(img);
                have_valid_polys = false;
                break;
            }

            case LaneState::RECOVERING: {
                // Both lines detected during recovery — draw normally then blend
                draw_and_compute_steering(img, draw_left, draw_right, false, false);
                float new_angle = (float)angle;
                float new_dist = (float)distance_center;
                blend_recovery(new_angle, new_dist);
                have_valid_polys = true;
                break;
            }
        }

        // Center line
        line(img, Point(center_lines, 0), Point(center_cam, img.rows - 1),
             Scalar(0, 0, 255), 2);

        // Save last polynomials
        for (int k = 0; k < 3; k++) {
            polyleft_last[k] = draw_left[k];
            polyright_last[k] = draw_right[k];
        }

        // Publish control reference (Mejora 10)
        if (have_valid_polys) {
            publish_control_ref(draw_left, draw_right);
        } else {
            geometry_msgs::msg::Vector3 ref;
            ref.x = 0.0; ref.y = 0.0; ref.z = 0.0;
            control_ref_pub_->publish(ref);
        }

        // Publish confidence (Mejora 7)
        last_confidence_ = compute_confidence(find_line_left, find_line_right,
                                               n_left_pts, n_right_pts);
        std_msgs::msg::Float32 conf_msg;
        conf_msg.data = last_confidence_;
        confidence_pub_->publish(conf_msg);

        // Publish lane state (Mejora 8)
        std_msgs::msg::UInt8 state_msg;
        state_msg.data = static_cast<uint8_t>(lane_state_);
        lane_state_pub_->publish(state_msg);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LineDetection>());
    rclcpp::shutdown();
    return 0;
}
