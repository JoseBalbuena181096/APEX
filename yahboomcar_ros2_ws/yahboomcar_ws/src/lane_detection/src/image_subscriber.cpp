#include <iostream>
#include <math.h>
#include <string>
#include <chrono>
#include <ctime>
#include <vector>

// ROS 2 Headers
#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/int16.hpp>
#include <std_msgs/msg/u_int8.hpp>
#include <sensor_msgs/msg/image.hpp>
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

// ============================================================
// Processing resolution — all detection runs at this size
// for speed on Jetson Nano. Results are scaled back to original.
// ============================================================
static const int PROC_WIDTH  = 320;
static const int PROC_HEIGHT = 240;

class LineDetection : public rclcpp::Node {
private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<std_msgs::msg::Int16>::SharedPtr center_pub;
    rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr angle_line_pub;
    image_transport::Publisher debug_image_pub_;
    bool enable_web_view_;
    bool headless_;

    std_msgs::msg::Int16 center_message;
    std_msgs::msg::UInt8 angle_line_message;

    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed_seconds;
    stringstream ss;

    /* image variables */
    Mat frame;
    Mat img_edges;
    vector<Point> left_points, right_points;
    vector<Rect> sliding_window_rects_;  // for visualization
    float polyleft[3], polyright[3];
    float polyleft_last[3], polyright_last[3];
    // Smoothed polynomial coefficients (EMA)
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

    // Pre-allocated CLAHE
    Ptr<CLAHE> clahe_;

    // === Tunable parameters (adjustable at runtime via ROS2 params) ===
    int adaptive_block_;      // adaptiveThreshold block size (must be odd)
    int adaptive_c_;          // adaptiveThreshold C constant (negative = brighter)
    double clahe_clip_;       // CLAHE clip limit
    int sliding_margin_;      // sliding window half-width in pixels
    int sliding_minpix_;      // min pixels to recenter window
    int ransac_min_points_;   // min points for polynomial regression
    int ransac_min_inliers_;  // min inliers for RANSAC model to be accepted
    int poly_search_min_;     // min points in search_around_poly to keep tracking
    int nwindows_;            // number of sliding windows
    double smooth_alpha_;     // EMA smoothing factor (0=all history, 1=all new)
    int lane_width_px_;       // Lane width in bird's-eye pixels (geometric constraint)
    // Perspective source trapezoid (% of image, 0-100)
    int persp_top_left_pct_;  // top-left X as % of width
    int persp_top_right_pct_; // top-right X as % of width
    int persp_top_y_pct_;     // top Y as % of height
    int persp_bot_left_pct_;  // bottom-left X as % of width
    int persp_bot_right_pct_; // bottom-right X as % of width
    int persp_bot_y_pct_;     // bottom Y as % of height

    // Parameter callback handle
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;

    static std::mt19937& get_rng() {
        static std::mt19937 gen(std::random_device{}());
        return gen;
    }

public:
    LineDetection() : Node("line_detection") {
        // --- Static parameters (set once at launch) ---
        this->declare_parameter("enable_web_view", true);
        this->declare_parameter("headless", true);
        enable_web_view_ = this->get_parameter("enable_web_view").as_bool();
        headless_ = this->get_parameter("headless").as_bool();

        // --- Tunable parameters (adjustable at runtime from web UI) ---
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
        this->declare_parameter("persp_top_y_pct", 70);
        this->declare_parameter("persp_bot_left_pct", 0);
        this->declare_parameter("persp_bot_right_pct", 100);
        this->declare_parameter("persp_bot_y_pct", 85);

        load_tunable_params();

        // Register callback for live parameter updates
        param_cb_handle_ = this->add_on_set_parameters_callback(
            std::bind(&LineDetection::on_parameter_change, this, std::placeholders::_1));

        // Publishers
        center_pub = this->create_publisher<std_msgs::msg::Int16>("/distance_center_line", 1000);
        angle_line_pub = this->create_publisher<std_msgs::msg::UInt8>("/angle_line_now", 1000);

        if (enable_web_view_) {
            debug_image_pub_ = image_transport::create_publisher(this, "/lane_detection/debug_image");
            RCLCPP_INFO(this->get_logger(), "Web view ENABLED - publishing to /lane_detection/debug_image");
        } else {
            RCLCPP_INFO(this->get_logger(), "Web view DISABLED");
        }

        // Subscriber
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/ascamera_hp60c/camera_publisher/rgb0/image",
            10,
            std::bind(&LineDetection::LineDetectionCb, this, _1));

        if (!headless_) {
            cv::namedWindow(OPENCV_WINDOW, WINDOW_KEEPRATIO);
        }

        distance_center = 0;
        center_cam = 0;
        center_lines = 0;
        angle = 0;
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

        // Keep original for display overlay
        Mat original_frame = frame.clone();

        // --- Resize for fast processing ---
        Mat proc_frame;
        cv::resize(frame, proc_frame, Size(PROC_WIDTH, PROC_HEIGHT));

        // Setup perspective points from tunable parameters
        float tl = PROC_WIDTH * persp_top_left_pct_ / 100.0f;
        float tr = PROC_WIDTH * persp_top_right_pct_ / 100.0f;
        float ty = PROC_HEIGHT * persp_top_y_pct_ / 100.0f;
        float bl = PROC_WIDTH * persp_bot_left_pct_ / 100.0f;
        float br = PROC_WIDTH * persp_bot_right_pct_ / 100.0f;
        float by = PROC_HEIGHT * persp_bot_y_pct_ / 100.0f;

        Source[0] = Point2f(tl, ty);             // Top Left
        Source[1] = Point2f(tr, ty);             // Top Right
        Source[2] = Point2f(bl, by);    // Bottom Left
        Source[3] = Point2f(br, by);    // Bottom Right

        Destination[0] = Point2f(bl, 0);
        Destination[1] = Point2f(br, 0);
        Destination[2] = Point2f(bl, PROC_HEIGHT);
        Destination[3] = Point2f(br, PROC_HEIGHT);

        Mat invertedPerspectiveMatrix;
        Perspective(proc_frame, invertedPerspectiveMatrix);
        img_edges = Threshold(proc_frame);
        locate_lanes(img_edges, proc_frame);

        // Draw lines on a mask at processing resolution
        Mat lane_mask = Mat::zeros(proc_frame.size(), proc_frame.type());
        draw_lines(lane_mask);

        // Warp mask back to original perspective (still at proc resolution)
        Mat warped_mask_small;
        warpPerspective(lane_mask, warped_mask_small, invertedPerspectiveMatrix,
                        Size(PROC_WIDTH, PROC_HEIGHT));

        // === BUILD COMPOSITE OUTPUT: camera (left) | bird-eye threshold (right) ===
        int out_h = original_frame.rows;  // 480
        int out_w = original_frame.cols;  // 640

        // LEFT PANEL: camera view with lane overlay + trapezoid
        Mat warped_mask;
        cv::resize(warped_mask_small, warped_mask, Size(out_w, out_h));
        addWeighted(original_frame, 1.0, warped_mask, 1.0, 0, frame);

        // Draw perspective trapezoid on camera view
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

        ss.str(" ");
        ss.clear();
        ss << "[ANG]:" << angle << " [FPS]:" << fps;
        putText(frame, ss.str(), Point2f(2, 20), 0, 0.7, Scalar(255, 0, 0), 2);

        // RIGHT PANEL: bird-eye threshold (full size) with sliding windows + lanes
        Mat birdeye_bgr;
        cvtColor(img_edges, birdeye_bgr, COLOR_GRAY2BGR);
        // Draw sliding window rectangles (like reference image)
        for (const auto& r : sliding_window_rects_) {
            rectangle(birdeye_bgr, r, Scalar(255, 255, 255), 1);
        }
        // Overlay detected lane points (green/cyan dots)
        addWeighted(birdeye_bgr, 1.0, lane_mask, 0.8, 0, birdeye_bgr);
        // Scale to same height as camera view
        Mat birdeye_panel;
        cv::resize(birdeye_bgr, birdeye_panel, Size(out_w, out_h), 0, 0, INTER_NEAREST);
        putText(birdeye_panel, "BIRD-EYE THRESHOLD", Point(8, 24),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);

        // Concatenate side by side: [camera | bird-eye]
        Mat composite;
        hconcat(frame, birdeye_panel, composite);

        // Publish composite for web view
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

        center_message.data = distance_center;
        center_pub->publish(center_message);
        angle_line_message.data = angle;
        angle_line_pub->publish(angle_line_message);

        set_end_time();
        fps = FPS_subscriber();
        RCLCPP_INFO(this->get_logger(), "[FPS]: %i ", fps);
    }

    void set_start_time() {
        start = std::chrono::system_clock::now();
    }

    void set_end_time() {
        end = std::chrono::system_clock::now();
    }

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
    // THRESHOLD — The core improvement
    //
    // Strategy: adaptiveThreshold on CLAHE-enhanced L channel.
    //   - CLAHE normalizes local contrast (handles shadows/highlights)
    //   - adaptiveThreshold finds locally bright pixels (the white lines)
    //     relative to their neighborhood — NO global threshold to tune.
    //   - Morphological cleanup removes small noise dots.
    //
    // This replaces: Otsu + Sobel + saturation mask (too noisy).
    // One well-tuned signal > three noisy signals combined.
    // ============================================================
    Mat Threshold(Mat frame) {
        Mat frameHLS;
        cvtColor(frame, frameHLS, COLOR_BGR2HLS);

        // Extract L channel (Lightness)
        std::vector<Mat> channels;
        split(frameHLS, channels);
        Mat L = channels[1];

        // CLAHE: equalize contrast locally (handles uneven lighting)
        Mat L_clahe;
        clahe_->apply(L, L_clahe);

        // Adaptive threshold: finds pixels that are BRIGHTER than their
        // local neighborhood. Block size 51 (~16% of 320px width) defines
        // "local". C=-25 means pixel must be 25 units brighter than average.
        // This auto-adapts to ANY lighting condition.
        Mat binary;
        adaptiveThreshold(L_clahe, binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
                          THRESH_BINARY, adaptive_block_, adaptive_c_);

        // Morphological cleanup: remove small noise, connect nearby fragments
        Mat kernel_open = getStructuringElement(MORPH_RECT, Size(3, 3));
        Mat kernel_close = getStructuringElement(MORPH_RECT, Size(5, 5));
        morphologyEx(binary, binary, MORPH_OPEN, kernel_open);   // remove noise dots
        morphologyEx(binary, binary, MORPH_CLOSE, kernel_close); // connect lane fragments

        return binary;
    }

    // ============================================================
    // Histogram — find initial lane positions
    //
    // Improved: uses bottom third (strongest signal near robot),
    // applies Gaussian smoothing to histogram to avoid noise peaks,
    // and validates that peaks have meaningful signal strength.
    // ============================================================
    int *Histogram(Mat &img) {
        static int LanePosition[2];
        vector<int> histogramLane(img.cols, 0);

        // Use bottom third for strongest lane signal (closer to robot)
        int start_row = img.rows * 2 / 3;
        for (int c = 0; c < img.cols; c++) {
            for (int r = start_row; r < img.rows; r++) {
                if (img.at<uchar>(r, c) > 0) {
                    histogramLane[c]++;
                }
            }
        }

        // Smooth histogram with a simple box filter (window=11)
        // This prevents noise spikes from being chosen as lane positions
        vector<int> smoothed(img.cols, 0);
        int hw = 5; // half-window
        for (int c = hw; c < img.cols - hw; c++) {
            int sum = 0;
            for (int k = -hw; k <= hw; k++) sum += histogramLane[c + k];
            smoothed[c] = sum;
        }

        // Find peaks in left half and right half
        auto LeftPtr = max_element(smoothed.begin(), smoothed.begin() + img.cols / 2);
        LanePosition[0] = distance(smoothed.begin(), LeftPtr);
        auto RightPtr = max_element(smoothed.begin() + img.cols / 2, smoothed.end());
        LanePosition[1] = distance(smoothed.begin(), RightPtr);

        // If a peak is too weak (< 3 pixels), use a reasonable default
        int min_peak = 3;
        if (*LeftPtr < min_peak) LanePosition[0] = img.cols / 4;
        if (*RightPtr < min_peak) LanePosition[1] = img.cols * 3 / 4;

        return LanePosition;
    }

    // ============================================================
    // Lane location: sliding window or search around prior
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
        sliding_window_rects_.clear();  // Clear old rects
        int margin = sliding_margin_;

        // Visualize search bands as rectangles (one per "stripe")
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

        // If we lost too many points, force full search next frame
        if ((int)left_points.size() < poly_search_min_ || (int)right_points.size() < poly_search_min_) {
            detected = false;
        }
    }

    // ============================================================
    // Sliding window search — IMPROVED
    //   - margin=60 (was 20) — much wider search window
    //   - minpix=15 (was 40) — recenter earlier
    //   - Clamps all bounds correctly
    // ============================================================
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

            // Store window rects for visualization
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
    //   - Minimum points: 30 (was 80)
    //   - Iterations: 40 (was 80) — faster
    //   - Cleaned up redundant matrix assignments
    // ============================================================
    bool regression_left() {
        if ((int)left_points.size() < ransac_min_points_) {
            RCLCPP_DEBUG(this->get_logger(), "LEFT: only %d pts (need %d)",
                         (int)left_points.size(), ransac_min_points_);
            return false;
        }
        bool ok = polyfit_ransac(left_points, polyleft, 150, 12.0);
        if (!ok) {
            ok = linear_fit(left_points, polyleft);
            if (ok) RCLCPP_INFO(this->get_logger(), "LEFT: RANSAC fail, linear fallback OK (%d pts)", (int)left_points.size());
        }
        if (ok) {
            // Sanity check: evaluate at bottom of image
            float bot_col = polyleft[0] + polyleft[1] * (PROC_HEIGHT - 1) + polyleft[2] * (PROC_HEIGHT - 1) * (PROC_HEIGHT - 1);
            if (bot_col < -50 || bot_col > PROC_WIDTH + 50) {
                RCLCPP_INFO(this->get_logger(), "LEFT: rejected (bot_col=%.0f out of range)", bot_col);
                ok = false;
            }
        }
        return ok;
    }

    bool regression_right() {
        if ((int)right_points.size() < ransac_min_points_) {
            RCLCPP_DEBUG(this->get_logger(), "RIGHT: only %d pts (need %d)",
                         (int)right_points.size(), ransac_min_points_);
            return false;
        }
        bool ok = polyfit_ransac(right_points, polyright, 150, 12.0);
        if (!ok) {
            ok = linear_fit(right_points, polyright);
            if (ok) RCLCPP_INFO(this->get_logger(), "RIGHT: RANSAC fail, linear fallback OK (%d pts)", (int)right_points.size());
        }
        if (ok) {
            // Sanity check: evaluate at bottom of image
            float bot_col = polyright[0] + polyright[1] * (PROC_HEIGHT - 1) + polyright[2] * (PROC_HEIGHT - 1) * (PROC_HEIGHT - 1);
            if (bot_col < -50 || bot_col > PROC_WIDTH + 50) {
                RCLCPP_INFO(this->get_logger(), "RIGHT: rejected (bot_col=%.0f out of range)", bot_col);
                ok = false;
            }
        }
        return ok;
    }

    // Simple linear regression fallback for sparse data
    // Fits col = p0 + p1*row (sets p2=0)
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
        coeffs[1] = (n * sum_xy - sum_x * sum_y) / denom;  // slope
        coeffs[0] = (sum_y - coeffs[1] * sum_x) / n;        // intercept
        coeffs[2] = 0.0f;  // no quadratic term
        return true;
    }

    bool polyfit_ransac(const std::vector<Point>& points, float* coeffs,
                        int iterations = 40, double threshold = 8.0) {
        if (points.size() < 3) return false;

        int n = points.size();
        std::vector<int> best_inliers;
        bool found_model = false;

        auto& gen = get_rng();
        std::uniform_int_distribution<> dis(0, n - 1);

        for (int k = 0; k < iterations; ++k) {
            // Sample 3 random points
            int i0 = dis(gen), i1 = dis(gen), i2 = dis(gen);
            if (i0 == i1 || i1 == i2 || i0 == i2) continue;

            // Fit: col = p0 + p1*row + p2*row^2
            // point.x = row, point.y = col
            Eigen::Matrix3f A;
            Eigen::Vector3f b;
            const Point* pts[3] = {&points[i0], &points[i1], &points[i2]};
            for (int i = 0; i < 3; ++i) {
                float row = (float)pts[i]->x;
                A(i, 0) = 1.0f;
                A(i, 1) = row;
                A(i, 2) = row * row;
                b(i) = (float)pts[i]->y;
            }

            Eigen::Vector3f x = A.colPivHouseholderQr().solve(b);

            // Count inliers
            std::vector<int> current_inliers;
            current_inliers.reserve(n);
            for (int i = 0; i < n; ++i) {
                float row = (float)points[i].x;
                float y_pred = x(0) + x(1) * row + x(2) * row * row;
                if (std::abs(points[i].y - y_pred) < threshold) {
                    current_inliers.push_back(i);
                }
            }

            if (current_inliers.size() > best_inliers.size()) {
                best_inliers = current_inliers;
                found_model = true;
            }
        }

        if (!found_model || (int)best_inliers.size() < ransac_min_inliers_) return false;

        // Refit on all inliers
        Eigen::MatrixXf A_all(best_inliers.size(), 3);
        Eigen::VectorXf b_all(best_inliers.size());
        for (size_t i = 0; i < best_inliers.size(); ++i) {
            int idx = best_inliers[i];
            float row = (float)points[idx].x;
            A_all(i, 0) = 1.0f;
            A_all(i, 1) = row;
            A_all(i, 2) = row * row;
            b_all(i) = (float)points[idx].y;
        }

        Eigen::Vector3f final_coeffs = A_all.colPivHouseholderQr().solve(b_all);

        coeffs[0] = final_coeffs(0);
        coeffs[1] = final_coeffs(1);
        coeffs[2] = final_coeffs(2);

        return true;
    }

    // ============================================================
    // Temporal smoothing — Exponential Moving Average
    //
    // Prevents lines from jumping between frames.
    // alpha = 0.6 means 60% new frame, 40% history.
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
    // Draw lines and compute steering
    // ============================================================
    void draw_lines(Mat &img) {
        float columnL, columnR;
        float row;
        float angle_to_mid_radian;

        bool find_line_right = regression_right();
        bool find_line_left = regression_left();

        // Log diagnostics periodically
        static int diag_counter = 0;
        if (++diag_counter % 30 == 0) {
            float lb = polyleft[0] + polyleft[1] * (PROC_HEIGHT-1) + polyleft[2] * (PROC_HEIGHT-1)*(PROC_HEIGHT-1);
            float rb = polyright[0] + polyright[1] * (PROC_HEIGHT-1) + polyright[2] * (PROC_HEIGHT-1)*(PROC_HEIGHT-1);
            RCLCPP_INFO(this->get_logger(), "[DIAG] L_pts=%d R_pts=%d L_ok=%d R_ok=%d L_bot=%.0f R_bot=%.0f margin=%d",
                        (int)left_points.size(), (int)right_points.size(),
                        find_line_left, find_line_right, lb, rb, sliding_margin_);
        }

        // Sanity check: right must be to the right of left at image bottom
        if (find_line_left && find_line_right) {
            float lb = polyleft[0] + polyleft[1] * (img.rows-1) + polyleft[2] * (img.rows-1)*(img.rows-1);
            float rb = polyright[0] + polyright[1] * (img.rows-1) + polyright[2] * (img.rows-1)*(img.rows-1);
            if (rb < lb + 20) {
                // Right line is to the left of or overlapping left line — reject right
                RCLCPP_INFO(this->get_logger(), "SANITY: right(%.0f) < left(%.0f)+20, rejecting right", rb, lb);
                find_line_right = false;
            }
        }

        // Apply temporal smoothing
        if (find_line_left)
            smooth_polynomial(polyleft, polyleft_smooth, has_smooth_left_);
        if (find_line_right)
            smooth_polynomial(polyright, polyright_smooth, has_smooth_right_);

        // Update detection status — require BOTH lanes for tracking mode
        // (if only one is found, keep sliding window to rediscover the other)
        if (find_line_left && find_line_right) {
            detected = true;
            frames_since_detection_ = 0;
        } else {
            // Even if one lane found, force sliding window to keep searching
            detected = false;
            frames_since_detection_++;
        }

        right_points.clear();
        left_points.clear();
        center_cam = (img.cols / 2) - 5;

        // Choose which coefficients to draw with (smoothed when available)
        float* draw_left  = has_smooth_left_  ? polyleft_smooth  : polyleft_last;
        float* draw_right = has_smooth_right_ ? polyright_smooth : polyright_last;

        // =============================================================
        // GEOMETRIC CONSTRAINT (KITcar-style)
        // If only one lane is detected, INFER the other by offsetting
        // the detected polynomial by lane_width_px_. This uses the
        // REAL curve shape, not a flat guess.
        // =============================================================
        float inferred_left[3], inferred_right[3];
        bool left_inferred = false, right_inferred = false;

        if (find_line_left && !find_line_right) {
            // Infer right = left + lane_width (same curve, shifted right)
            inferred_right[0] = draw_left[0] + lane_width_px_;
            inferred_right[1] = draw_left[1];
            inferred_right[2] = draw_left[2];
            draw_right = inferred_right;
            right_inferred = true;
            // Also smooth the inferred right so it becomes the "last known"
            smooth_polynomial(inferred_right, polyright_smooth, has_smooth_right_);
        } else if (find_line_right && !find_line_left) {
            // Infer left = right - lane_width (same curve, shifted left)
            inferred_left[0] = draw_right[0] - lane_width_px_;
            inferred_left[1] = draw_right[1];
            inferred_left[2] = draw_right[2];
            draw_left = inferred_left;
            left_inferred = true;
            smooth_polynomial(inferred_left, polyleft_smooth, has_smooth_left_);
        }

        // Now we always have both lines (detected or inferred)
        bool have_both = find_line_left || find_line_right;

        if (have_both) {
            for (row = img.rows - 1; row >= 0; row -= 4) {
                columnR = draw_right[0] + draw_right[1] * row + draw_right[2] * row * row;
                columnL = draw_left[0]  + draw_left[1]  * row + draw_left[2]  * row * row;
                // Green = detected, Cyan = inferred
                Scalar colorL = left_inferred  ? Scalar(255, 255, 0) : Scalar(0, 255, 0);
                Scalar colorR = right_inferred ? Scalar(255, 255, 0) : Scalar(0, 255, 0);
                circle(img, cv::Point((int)columnR, (int)row), 2, colorR, 2);
                circle(img, cv::Point((int)columnL, (int)row), 2, colorL, 2);
            }
            center_lines = (int)((columnR + columnL) / 2);
            distance_center = center_cam - center_lines;

            if (distance_center == 0) {
                angle = 90;
            } else {
                angle_to_mid_radian = atan2f((float)(img.rows - 1),
                                             (float)(center_lines - center_cam));
                angle = (int)(angle_to_mid_radian * 57.295779f);
                if (angle < 0) angle = -angle;
                if (angle > 90 && angle < 180) angle = 180 - angle;
            }

            line(img, Point(center_lines, 0), Point(center_cam, img.rows - 1), Scalar(0, 0, 255), 2);

            for (int k = 0; k < 3; k++) {
                polyleft_last[k] = draw_left[k];
                polyright_last[k] = draw_right[k];
            }

        } else {
            // No lines found at all — use last known (dimmer)
            for (row = img.rows - 1; row >= 0; row -= 4) {
                columnR = polyright_last[0] + polyright_last[1] * row + polyright_last[2] * row * row;
                columnL = polyleft_last[0]  + polyleft_last[1]  * row + polyleft_last[2]  * row * row;
                circle(img, cv::Point((int)columnR, (int)row), 2, Scalar(0, 150, 0), 2);
                circle(img, cv::Point((int)columnL, (int)row), 2, Scalar(0, 150, 0), 2);
            }
            center_lines = (int)((columnR + columnL) / 2);
            distance_center = center_cam - center_lines;
            if (distance_center == 0) {
                angle = 90;
            } else {
                angle_to_mid_radian = atan2f((float)(img.rows - 1),
                                             (float)(center_lines - center_cam));
                angle = (int)(angle_to_mid_radian * 57.295779f);
                if (angle < 0) angle = -angle;
                if (angle > 90 && angle < 180) angle = 180 - angle;
            }
        }

        line(img, Point(center_lines, 0), Point(center_cam, img.rows - 1), Scalar(0, 0, 255), 2);
    }

    void compute_angle(Mat &img, float col_top, float col_bot) {
        if (std::abs(col_top - col_bot) < 1.0f) {
            angle = 90;
        } else {
            float angle_to_mid_radian = atan2f((float)(img.rows - 1), col_top - col_bot);
            angle = (int)(angle_to_mid_radian * 57.295779f);
            if (angle < 0) angle = -angle;
            if (angle > 90 && angle < 180) angle = 180 - angle;
        }
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LineDetection>());
    rclcpp::shutdown();
    return 0;
}
