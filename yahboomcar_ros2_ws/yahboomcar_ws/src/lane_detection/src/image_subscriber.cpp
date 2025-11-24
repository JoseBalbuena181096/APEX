#include <iostream>
#include <math.h>
#include <string>
#include <chrono>
#include <ctime>
#include <vector>

// ROS 2 Headers
#include <rclcpp/rclcpp.hpp>
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

class LineDetection : public rclcpp::Node {
private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<std_msgs::msg::Int16>::SharedPtr center_pub;
    rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr angle_line_pub;
    
    std_msgs::msg::Int16 center_message;
    std_msgs::msg::UInt8 angle_line_message;
    
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed_seconds;
    stringstream ss;
    
    /*variable image*/
    Mat frame;
    Mat img_edges;
    Mat img_lines;
    Mat img_perspective;
    vector<Point> left_points, right_points;
    vector<Point> left_line, right_line, center_line;
    float polyleft[3], polyright[3];
    float polyleft_last[3], polyright_last[3];
    Point2f Source[4];
    Point2f Destination[4];
    
    int center_cam;
    int center_lines;
    int distance_center;
    int angle;
    int fps;
    bool detected; // Flag to track if lines were detected in previous frame

public:
    // Constructor
    LineDetection() : Node("line_detection") {
        // Publishers
        center_pub = this->create_publisher<std_msgs::msg::Int16>("/distance_center_line", 1000);
        angle_line_pub = this->create_publisher<std_msgs::msg::UInt8>("/angle_line_now", 1000);

        // Subscriber
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/ascamera_hp60c/camera_publisher/rgb0/image",
            10,
            std::bind(&LineDetection::LineDetectionCb, this, _1));

        cv::namedWindow(OPENCV_WINDOW, WINDOW_KEEPRATIO);
        
        distance_center = 0.0;
        center_cam = 0;
        center_lines = 0;
        angle = 0;
        fps = 0;
        detected = false;
    }

    // Destructor
    ~LineDetection() {
        cv::destroyWindow(OPENCV_WINDOW);
    }

    // Callback funtion
    void LineDetectionCb(const sensor_msgs::msg::Image::SharedPtr msg) {
        set_start_time();
        cv_bridge::CvImagePtr cv_ptr;
        try {
            // Reverted to BGR8 as standard for OpenCV
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            frame = cv_ptr->image; 
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        // Initialize points dynamically based on actual frame size
        // Source: Trapezoid (Bottom corners to Middle ~28% or 2/7)
        // This is wider than 1/3 (33%) but narrower than 1/4 (25%)
        Source[0] = Point2f(frame.cols * 2 / 7, frame.rows / 2);      // Top Left
        Source[1] = Point2f(frame.cols * 5 / 7, frame.rows / 2);      // Top Right
        Source[2] = Point2f(0, frame.rows);                           // Bottom Left
        Source[3] = Point2f(frame.cols, frame.rows);                  // Bottom Right
        
        // Destination: Full image rectangle for uniform view
        Destination[0] = Point2f(frame.cols * 2 / 7, 0);
        Destination[1] = Point2f(frame.cols * 5 / 7, 0);
        Destination[2] = Point2f(frame.cols * 2 / 7, frame.rows);
        Destination[3] = Point2f(frame.cols * 5 / 7, frame.rows);
        
        // Removed resize_image to use full resolution
        // resize_image(frame, 0.5);
        
        std::cout << "Frame size: " << frame.rows << "x" << frame.cols << std::endl;
        Mat copy_frame = frame.clone();
        Mat empty_frame = Mat::zeros(frame.rows, frame.cols, frame.type());
        
        Mat invertedPerspectiveMatrix;
        Perspective(frame, invertedPerspectiveMatrix);
        img_edges = Threshold(frame);
        locate_lanes(img_edges, frame);
        // Create a black mask to draw lines on
        Mat lane_mask = Mat::zeros(frame.size(), frame.type());
        draw_lines(lane_mask);
        
        // Warp the lane mask back to original perspective
        Mat warped_mask;
        warpPerspective(lane_mask, warped_mask, invertedPerspectiveMatrix, Size(frame.cols, frame.rows));
        
        // Combine the original frame with the warped lines
        // addWeighted(src1, alpha, src2, beta, gamma, dst)
        addWeighted(copy_frame, 1.0, warped_mask, 1.0, 0, frame);
        
        ss.str(" ");
        ss.clear();
        ss << "[ANG]:" << angle << " [FPS]:" << fps;
        putText(frame, ss.str(), Point2f(2, 20), 0, 1, Scalar(255, 0, 0), 2);
        
        // Show image
        resizeWindow(OPENCV_WINDOW, frame.cols, frame.rows);
        //draw_bird_eye_line(frame, Source, Destination);
        cv::imshow(OPENCV_WINDOW, frame);
        cv::waitKey(1);
        
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
        float t = elapsed_seconds.count();
        return (t > 0) ? (int)(1 / t) : 0;
    }

    // Method to rotate image
    void rotate(Mat &src, double angle = 180.0) {
        Point2f pt(src.cols / 2., src.rows / 2.);
        Mat r = getRotationMatrix2D(pt, angle, 1.0);
        warpAffine(src, src, r, Size(src.cols, src.rows));
    }

    void Perspective(Mat &frame, Mat &invertedPerspectiveMatrix) {
        Mat matrixPerspective(2, 4, CV_32FC1);
        matrixPerspective = Mat::zeros(frame.rows, frame.cols, frame.type());
        matrixPerspective = getPerspectiveTransform(Source, Destination);
        warpPerspective(frame, frame, matrixPerspective, Size(frame.cols, frame.rows));
        invert(matrixPerspective, invertedPerspectiveMatrix);
    }

    void draw_bird_eye_line(Mat &frame, Point2f *Source, Point2f *Destination) {
        line(frame, Source[0], Source[1], Scalar(0, 0, 255), 2);
        line(frame, Source[1], Source[3], Scalar(0, 0, 255), 2);
        line(frame, Source[2], Source[3], Scalar(0, 0, 255), 2);
        line(frame, Source[2], Source[0], Scalar(0, 0, 255), 2);
        line(frame, Destination[0], Destination[1], Scalar(0, 255, 0), 2);
        line(frame, Destination[1], Destination[3], Scalar(0, 255, 0), 2);
        line(frame, Destination[2], Destination[3], Scalar(0, 255, 0), 2);
        line(frame, Destination[2], Destination[0], Scalar(0, 255, 0), 2);
    }

    Mat Threshold(Mat frame) {
        Mat frameHLS, frameGray;
        // Convert to HLS color space for better robustness to lighting
        cvtColor(frame, frameHLS, COLOR_BGR2HLS);
        
        // Extract L channel (Lightness) and S channel (Saturation)
        std::vector<Mat> channels;
        split(frameHLS, channels);
        Mat L = channels[1];
        Mat S = channels[2];

        // White detection: High Lightness
        Mat white_mask;
        // Adaptive thresholding for white (robust to lighting changes)
        // Using a high threshold for L channel to pick up bright white lines
        // You can also use inRange with HLS ranges for white: L > 200, S < 50 roughly
        inRange(frameHLS, Scalar(0, 200, 0), Scalar(180, 255, 255), white_mask);
        
        // Optional: Combine with Sobel edge detection if needed, but color/lightness is often enough for white lines
        
        // Apply Gaussian Blur to reduce noise
        GaussianBlur(white_mask, white_mask, Size(5, 5), 0);
        
        return white_mask;
    }

    void resize_image(Mat &input, float alpha = 0.15) {
        cv::resize(input, input, Size(input.cols * alpha, input.rows * alpha));
    }

    int *Histogram(Mat &img) {
        // Create histogram with the length of the width of the frame
        // Note: Returning pointer to static local variable is not thread-safe but preserved from original code logic
        vector<int> histogramLane;
        static int LanePosition[2];
        int init_row, end_row;
        Mat ROILane;
        Mat frame;
        init_row = img.rows * 2 / 3;
        end_row = img.rows / 3 - 1;
        img.copyTo(frame);
        for (int i = 0; i < img.cols; i++) {
            // Region interest
            ROILane = frame(Rect(i, init_row, 1, end_row));
            // Normal values
            divide(255, ROILane, ROILane);
            // add the value
            histogramLane.push_back((int)(sum(ROILane)[0]));
        }
        // Find line left
        vector<int>::iterator LeftPtr;
        LeftPtr = max_element(histogramLane.begin(), histogramLane.begin() + img.cols / 2);
        LanePosition[0] = distance(histogramLane.begin(), LeftPtr);
        // find line right
        vector<int>::iterator RightPtr;
        RightPtr = max_element(histogramLane.begin() + (img.cols / 2) + 1, histogramLane.end());
        LanePosition[1] = distance(histogramLane.begin(), RightPtr);
        return LanePosition;
    }

    void locate_lanes(Mat &img, Mat &out_img) {
        (void)out_img;
        
        // If we detected lines previously, try to search around them first
        if (detected) {
            search_around_poly(img);
        } else {
            // Otherwise, perform a full sliding window search
            sliding_window_search(img);
        }
    }

    void search_around_poly(Mat &img) {
        left_points.clear();
        right_points.clear();
        int margin = 50; // Margin around previous polynomial to search

        for (int r = 0; r < img.rows; r++) {
            // Calculate center x for this row based on previous polynomial
            // x = a*y^2 + b*y + c (where y is row, x is col)
            // poly[0]=c, poly[1]=b, poly[2]=a
            int left_center = polyleft[0] + polyleft[1] * r + polyleft[2] * r * r;
            int right_center = polyright[0] + polyright[1] * r + polyright[2] * r * r;

            int left_low = left_center - margin;
            int left_high = left_center + margin;
            int right_low = right_center - margin;
            int right_high = right_center + margin;

            // Clamp to image bounds
            left_low = std::max(0, left_low);
            left_high = std::min(img.cols - 1, left_high);
            right_low = std::max(0, right_low);
            right_high = std::min(img.cols - 1, right_high);

            for (int c = left_low; c < left_high; c++) {
                if (img.at<uchar>(r, c) > 0) {
                    left_points.push_back(Point(r, c)); // Point(row, col) as per original convention
                }
            }
            for (int c = right_low; c < right_high; c++) {
                if (img.at<uchar>(r, c) > 0) {
                    right_points.push_back(Point(r, c));
                }
            }
        }
        
        // If we lost too many points, force a full search next time
        if (left_points.size() < 50 || right_points.size() < 50) {
            detected = false;
        }
    }

    void sliding_window_search(Mat &img) {
        left_points.clear();
        right_points.clear();
        int nwindows, margin, minpix;
        int win_y_low = 0, win_y_high = 0, win_xleft_low = 0, win_xleft_high = 0, win_xright_low = 0, win_xright_high = 0;
        int leftx_current, rightx_current;
        int mean_leftx, mean_rightx, count_left, count_right;
        int *locate_histogram;
        uchar now_left_point;
        uchar now_right_point;
        Point add_left_point, add_right_point;
        nwindows = 9;
        int window_height = img.rows / nwindows;
        locate_histogram = Histogram(img);
        leftx_current = locate_histogram[0];
        rightx_current = locate_histogram[1];
        // Set the width of the windows +/- margin
        margin = 20;
        minpix = 40;
        // Set minimum number of pixels found to recenter window
        for (int window = 0; window < nwindows; window++) {
            mean_leftx = 0;
            mean_rightx = 0;
            count_left = 0;
            count_right = 0;
            win_y_low = img.rows - (window + 1) * window_height;
            win_y_high = img.rows - window * window_height;
            win_xleft_low = leftx_current - margin;
            win_xleft_high = leftx_current + margin;
            win_xright_low = rightx_current - margin;
            win_xright_high = rightx_current + margin;
            if (win_xleft_low < 0)
                win_xleft_low = 0 + 1;
            if (win_xright_high >= img.cols - 1)
                win_xright_high = img.cols - 1;
            if (win_y_high >= img.rows - 1)
                win_y_high = img.rows - 1;
            if (win_y_low <= 0)
                win_y_low = 1;
            for (int r = win_y_low; r < win_y_high; r++) {
                for (int cl = win_xleft_low + 1; cl < win_xleft_high; cl++) {
                    now_left_point = img.at<uchar>(r, cl);
                    if (now_left_point > 0) {
                        add_left_point.x = r;
                        add_left_point.y = cl;
                        left_points.push_back(add_left_point);
                        mean_leftx += add_left_point.y;
                        count_left++;
                    }
                }
                for (int cr = win_xright_low + 1; cr < win_xright_high; cr++) {
                    now_right_point = img.at<uchar>(r, cr);
                    if (now_right_point > 0) {
                        add_right_point.x = r;
                        add_right_point.y = cr;
                        right_points.push_back(add_right_point);
                        mean_rightx += add_right_point.y;
                        count_right++;
                    }
                }
            }
            if (count_left >= minpix) {
                mean_leftx /= count_left;
                leftx_current = mean_leftx;
            }
            if (count_right >= minpix) {
                mean_rightx /= count_right;
                rightx_current = mean_rightx;
            }
        }
    }

    bool regression_left() {
        if (left_points.size() < 80) // Lowered threshold to keep "Search from Prior" active more often
            return false;
        // Reduce iterations to 80 for speed (default was 100)
        return polyfit_ransac(left_points, polyleft, 80); 
    }

    bool regression_right() {
        if (right_points.size() < 80)
            return false;
        return polyfit_ransac(right_points, polyright, 80);
    }

    // RANSAC Polynomial Fitting using Eigen
    bool polyfit_ransac(const std::vector<Point>& points, float* coeffs, int iterations = 100, double threshold = 10.0) {
        if (points.size() < 3) return false;

        int n = points.size();
        std::vector<int> best_inliers;
        Eigen::Vector3f best_model;
        bool found_model = false;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, n - 1);

        for (int k = 0; k < iterations; ++k) {
            // 1. Sample 3 random points
            std::vector<Point> sample;
            for (int i = 0; i < 3; ++i) {
                sample.push_back(points[dis(gen)]);
            }

            // 2. Fit quadratic model to sample: y = ax^2 + bx + c
            // Matrix form: X * [a, b, c]^T = Y
            Eigen::Matrix3f A;
            Eigen::Vector3f b;
            for (int i = 0; i < 3; ++i) {
                A(i, 0) = sample[i].y * sample[i].y; // x^2 (using y as independent var like original code?)
                // WAIT: Original code used 'row' as independent variable? 
                // In draw_lines: column = poly[0] + poly[1]*row + poly[2]*row^2
                // So ROW (y-coordinate) is the independent variable X in the polynomial equation x = f(y).
                // Let's stick to that convention: x = a*y^2 + b*y + c
                // So coeffs are [c, b, a] in standard notation?
                // Original code: column = poly[0] + poly[1]*row + poly[2]*row^2
                // poly[0] is constant (c), poly[1] is linear (b), poly[2] is quadratic (a)
                
                A(i, 0) = 1.0;
                A(i, 1) = sample[i].x; // ROW is x in code logic? 
                // Let's check draw_lines again:
                // columnR = polyright[0] + polyright[1] * (row) + polyright[2] * (row * row);
                // row is the Y coordinate of the image. column is X.
                // So we are fitting X = f(Y). Independent var is Y.
                
                A(i, 0) = 1.0;
                A(i, 1) = sample[i].x; // Wait, sample[i].x is ROW?
                // In locate_lanes: add_left_point.x = r (row); add_left_point.y = cl (col);
                // So point.x is ROW (Y), point.y is COL (X).
                // CONFUSING NAMING IN ORIGINAL CODE!
                // Let's verify:
                // locate_lanes: add_left_point.x = r; (row index)
                //               add_left_point.y = cl; (col index)
                // So point.x = Y_image, point.y = X_image.
                
                // Equation: X_image = p0 + p1*Y_image + p2*Y_image^2
                // point.y = p0 + p1*point.x + p2*point.x^2
                
                A(i, 0) = 1.0;
                A(i, 1) = sample[i].x;
                A(i, 2) = sample[i].x * sample[i].x;
                b(i) = sample[i].y;
            }

            // Solve A * x = b
            Eigen::Vector3f x = A.colPivHouseholderQr().solve(b);

            // 3. Count inliers
            std::vector<int> current_inliers;
            for (int i = 0; i < n; ++i) {
                float y_pred = x(0) + x(1) * points[i].x + x(2) * points[i].x * points[i].x;
                float error = std::abs(points[i].y - y_pred);
                if (error < threshold) {
                    current_inliers.push_back(i);
                }
            }

            // 4. Update best model
            if (current_inliers.size() > best_inliers.size()) {
                best_inliers = current_inliers;
                best_model = x;
                found_model = true;
            }
        }

        if (!found_model || best_inliers.size() < 10) return false;

        // 5. Refit on all inliers for better precision
        Eigen::MatrixXf A_all(best_inliers.size(), 3);
        Eigen::VectorXf b_all(best_inliers.size());
        for (size_t i = 0; i < best_inliers.size(); ++i) {
            int idx = best_inliers[i];
            A_all(i, 0) = 1.0;
            A_all(i, 1) = points[idx].x;
            A_all(i, 2) = points[idx].x * points[idx].x;
            b_all(i) = points[idx].y;
        }

        // Least squares solution: (A^T A)^-1 A^T b
        Eigen::Vector3f final_coeffs = A_all.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b_all);

        coeffs[0] = final_coeffs(0);
        coeffs[1] = final_coeffs(1);
        coeffs[2] = final_coeffs(2);

        return true;
    }

    // Removed old solve_system as it is replaced by polyfit_ransac with Eigen

    void draw_lines(Mat &img) {
        float columnL, columnL_aux;
        float columnR, columnR_aux;
        float row;
        bool find_line_left;
        bool find_line_right;
        float angle_to_mid_radian;
        
        find_line_right = regression_right();
        find_line_left = regression_left();
        
        // Update detected status
        if (find_line_left && find_line_right) {
            detected = true;
        } else {
            detected = false;
        }

        right_points.clear();
        left_points.clear();
        center_cam = (img.cols / 2) - 5;
        if (find_line_left && find_line_right) {
            for (row = img.rows - 1; row >= 0; row -= 8) {
                columnR = polyright[0] + polyright[1] * (row) + polyright[2] * (row * row);
                circle(img, cv::Point(columnR, row), cvRound((double)4 / 2), Scalar(0, 255, 0), 2);
                columnL = polyleft[0] + polyleft[1] * (row) + polyleft[2] * (row * row);
                circle(img, cv::Point(columnL, row), cvRound((double)4 / 2), Scalar(0, 255, 0), 2);
            }
            center_lines = (columnR + columnL) / 2;
            if (columnR == columnL) {
                if (center_lines < center_cam)
                    distance_center = center_cam - 0;
                else
                    distance_center = center_cam - img.cols;
                return;
            }
            distance_center = center_cam - center_lines;
            if (distance_center == 0)
                angle = 90;
            else {
                angle_to_mid_radian = atan(static_cast<float>(0 - (img.rows - 1)) / static_cast<float>(center_lines - center_cam));
                angle = static_cast<int>(angle_to_mid_radian * 57.295779);
                if (angle < 0 && angle > (0 - 90))
                    angle = (0 - 1) * (angle);
                else if (angle > 0 && angle < 90)
                    angle = 180 - angle;
            }
            line(img, Point(center_lines, 0), Point(center_cam, (img.rows - 1)), Scalar(0, 0, 255), 2);
            for (int k = 0; k < 3; k++) {
                polyleft_last[k] = polyleft[k];
                polyright_last[k] = polyright[k];
            }
        } else if (find_line_left) {
            for (row = img.rows - 1; row >= 0; row -= 8) {
                columnL = polyleft[0] + polyleft[1] * (row) + polyleft[2] * (row * row);
                circle(img, cv::Point(columnL, row), cvRound((double)4 / 2), Scalar(0, 255, 0), 2);
            }
            // columnL = polyleft[0] + polyleft[1]*(0.0)+polyleft[2]*(0.0);
            columnL = polyleft[0];
            columnL_aux = polyleft[0] + polyleft[1] * static_cast<float>(img.rows - 1) + polyleft[2] * ((img.rows - 1) * (img.rows - 1));
            if (columnL_aux == columnL) {
                angle = 90;
                center_lines = columnL_aux;
            } else {
                angle_to_mid_radian = atan(static_cast<float>(0 - (img.rows - 1)) / static_cast<float>(columnL - columnL_aux));
                angle = static_cast<int>(angle_to_mid_radian * 57.295779);
                if (angle < 0 && angle > (0 - 90))
                    angle = (0 - 1) * (angle);
                else if (angle > 0 && angle < 90)
                    angle = 180 - angle;
                if (angle < 90) {
                    angle_to_mid_radian = (float)(angle)*0.0174533;
                    center_lines = center_cam + (int)(360.0 * cos(angle_to_mid_radian));
                } else {
                    angle_to_mid_radian = (float)(180 - angle) * 0.0174533;
                    center_lines = center_cam - (int)(360.0 * cos(angle_to_mid_radian));
                }
            }
            distance_center = center_cam - center_lines;

            for (int k = 0; k < 3; k++)
                polyleft_last[k] = polyleft[k];

            if (center_lines <= center_cam)
                distance_center = center_cam - 0;
            else
                distance_center = center_cam - img.cols;

        } else if (find_line_right) {
            for (row = img.rows - 1; row >= 0; row -= 8) {
                columnR = polyright[0] + polyright[1] * (row) + polyright[2] * (row * row);
                circle(img, cv::Point(columnR, row), cvRound((double)4 / 2), Scalar(0, 255, 0), 2);
            }
            // columnR = polyright[0] + polyright[1]*(0.0)+polyright[2]*(0.0);
            columnR = polyright[0];
            columnR_aux = polyright[0] + polyright[1] * static_cast<float>(img.rows - 1) + polyright[2] * static_cast<float>((img.rows - 1) * (img.rows - 1));
            if (columnR_aux == columnR) {
                angle = 90;
                center_lines = columnR_aux;
            } else {
                angle_to_mid_radian = atan(static_cast<float>(0 - (img.rows - 1)) / static_cast<float>(columnR - columnR_aux));
                angle = static_cast<int>(angle_to_mid_radian * 57.295779);
                if (angle < 0 && angle > (0 - 90))
                    angle = (0 - 1) * (angle);
                else if (angle > 0 && angle < 90)
                    angle = 180 - angle;
                if (angle < 90) {
                    angle_to_mid_radian = angle * 0.0174533;
                    center_lines = center_cam + (int)(360.0 * cos(angle_to_mid_radian));
                } else {
                    angle_to_mid_radian = (float)(180 - angle) * 0.0174533;
                    center_lines = center_cam - (int)(360.0 * cos(angle_to_mid_radian));
                }
            }
            distance_center = center_cam - center_lines;

            for (int k = 0; k < 3; k++)
                polyright[k] = polyright_last[k];

            if (center_lines <= center_cam)
                distance_center = center_cam - 0;
            else
                distance_center = center_cam - img.cols;
        }
        if (!find_line_left && !find_line_right) {
            for (row = img.rows - 1; row >= 0; row -= 8) {
                columnR = polyright_last[0] + polyright_last[1] * (row) + polyright_last[2] * (row * row);
                circle(img, cv::Point(columnR, row), cvRound((double)4 / 2), Scalar(0, 255, 0), 2);
                columnL = polyleft_last[0] + polyleft_last[1] * (row) + polyleft_last[2] * (row * row);
                circle(img, cv::Point(columnL, row), cvRound((double)4 / 2), Scalar(0, 255, 0), 2);
            }
            center_lines = (columnR + columnL) / 2;
            if (columnR == columnL) {
                if (center_lines < center_cam)
                    distance_center = center_cam - 0;
                else
                    distance_center = center_cam - img.cols;
                return;
            }
            distance_center = center_cam - center_lines;
            if (distance_center == 0)
                angle = 90;
            else {
                angle_to_mid_radian = atan(static_cast<float>(0 - (img.rows - 1)) / static_cast<float>(center_lines - center_cam));
                angle = static_cast<int>(angle_to_mid_radian * 57.295779);
                if (angle < 0 && angle > (0 - 90))
                    angle = (0 - 1) * (angle);
                else if (angle > 0 && angle < 90)
                    angle = 180 - angle;
            }
        }
        // line(img,Point(center_cam,(img.rows/4)),Point(center_cam,(img.rows*3/4)),Scalar(0,255,0),2);
        line(img, Point(center_lines, 0), Point(center_cam, (img.rows - 1)), Scalar(0, 0, 255), 2);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LineDetection>());
    rclcpp::shutdown();
    return 0;
}