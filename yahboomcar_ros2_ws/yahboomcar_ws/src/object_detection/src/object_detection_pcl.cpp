#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl_conversions/pcl_conversions.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cmath>
#include <chrono>

static const std::string OPENCV_WINDOW = "OBJECT DETECTION PCL";

class ObjectDetectionPCL : public rclcpp::Node
{
private:
    // ROS2 components
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
    
    // Parameters (configurable)
    float range_max_;
    float range_min_;
    float initial_angle_;
    float end_angle_;
    
    // Clustering parameters
    double cluster_tolerance_;
    int min_cluster_size_;
    int max_cluster_size_;
    
    // Filtering parameters
    bool use_voxel_filter_;
    float voxel_leaf_size_;
    bool use_outlier_filter_;
    int outlier_mean_k_;
    double outlier_stddev_;
    
    // Visualization
    cv::Mat image_;
    const int img_size_ = 800;
    const int img_center_ = 400;
    const float pixels_per_meter_ = 100.0;
    
    // Performance monitoring
    bool show_fps_;
    std::chrono::steady_clock::time_point last_time_;
    double fps_;
    
    // Color palette for clusters (stable colors)
    std::vector<cv::Scalar> color_palette_;

public:
    ObjectDetectionPCL() : Node("object_detection_pcl")
    {
        // Declare and get parameters
        this->declare_parameter("range_max", 16.0);
        this->declare_parameter("range_min", 0.1);
        this->declare_parameter("initial_angle", 0.0);
        this->declare_parameter("end_angle", 360.0);
        
        this->declare_parameter("cluster_tolerance", 0.15);
        this->declare_parameter("min_cluster_size", 3);
        this->declare_parameter("max_cluster_size", 10000);
        
        this->declare_parameter("use_voxel_filter", false);
        this->declare_parameter("voxel_leaf_size", 0.02);
        this->declare_parameter("use_outlier_filter", true);
        this->declare_parameter("outlier_mean_k", 10);
        this->declare_parameter("outlier_stddev", 0.5);
        
        this->declare_parameter("show_fps", true);
        
        // Get parameters
        range_max_ = this->get_parameter("range_max").as_double();
        range_min_ = this->get_parameter("range_min").as_double();
        initial_angle_ = this->get_parameter("initial_angle").as_double();
        end_angle_ = this->get_parameter("end_angle").as_double();
        
        cluster_tolerance_ = this->get_parameter("cluster_tolerance").as_double();
        min_cluster_size_ = this->get_parameter("min_cluster_size").as_int();
        max_cluster_size_ = this->get_parameter("max_cluster_size").as_int();
        
        use_voxel_filter_ = this->get_parameter("use_voxel_filter").as_bool();
        voxel_leaf_size_ = this->get_parameter("voxel_leaf_size").as_double();
        use_outlier_filter_ = this->get_parameter("use_outlier_filter").as_bool();
        outlier_mean_k_ = this->get_parameter("outlier_mean_k").as_int();
        outlier_stddev_ = this->get_parameter("outlier_stddev").as_double();
        
        show_fps_ = this->get_parameter("show_fps").as_bool();
        
        // Initialize subscribers and publishers
        scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10, std::bind(&ObjectDetectionPCL::laserCallback, this, std::placeholders::_1));
        
        cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/detected_objects_cloud", 10);
        
        // Initialize color palette
        initializeColorPalette();
        
        // Initialize FPS tracking
        last_time_ = std::chrono::steady_clock::now();
        fps_ = 0.0;
        
        RCLCPP_INFO(this->get_logger(), "Object Detection PCL Node Initialized");
        RCLCPP_INFO(this->get_logger(), "Range: [%.2f, %.2f] m", range_min_, range_max_);
        RCLCPP_INFO(this->get_logger(), "Cluster tolerance: %.3f m", cluster_tolerance_);
        RCLCPP_INFO(this->get_logger(), "Min/Max cluster size: %d/%d points", min_cluster_size_, max_cluster_size_);
    }
    
    ~ObjectDetectionPCL()
    {
        cv::destroyWindow(OPENCV_WINDOW);
    }

private:
    void initializeColorPalette()
    {
        // Pre-generate distinct colors for up to 20 clusters
        color_palette_.reserve(20);
        for (int i = 0; i < 20; ++i)
        {
            int hue = (i * 180 / 20) % 180;
            cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 255, 255));
            cv::Mat bgr;
            cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
            color_palette_.push_back(cv::Scalar(bgr.at<cv::Vec3b>(0, 0)[0], 
                                                bgr.at<cv::Vec3b>(0, 0)[1], 
                                                bgr.at<cv::Vec3b>(0, 0)[2]));
        }
    }
    
    void laserCallback(const sensor_msgs::msg::LaserScan::SharedPtr scan)
    {
        auto start_time = std::chrono::steady_clock::now();
        
        // Convert LaserScan to PCL point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = laserScanToPointCloud(scan);
        
        if (cloud->points.empty())
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "No points in range");
            return;
        }
        
        // Apply filters
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud = applyFilters(cloud);
        
        // Perform clustering
        std::vector<pcl::PointIndices> cluster_indices;
        performClustering(filtered_cloud, cluster_indices);
        
        // Extract cluster information
        std::vector<ClusterInfo> clusters;
        extractClusterInfo(filtered_cloud, cluster_indices, clusters);
        
        // Visualize
        visualizeClusters(filtered_cloud, cluster_indices, clusters);
        
        // Publish results
        publishClusterInfo(clusters);
        
        // Calculate and display FPS
        if (show_fps_)
        {
            updateFPS(start_time);
        }
    }
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr laserScanToPointCloud(
        const sensor_msgs::msg::LaserScan::SharedPtr& scan)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        
        // Pre-allocate memory for efficiency
        size_t estimated_points = scan->ranges.size() / 2;
        cloud->points.reserve(estimated_points);
        
        float angle = scan->angle_min;
        
        for (size_t i = 0; i < scan->ranges.size(); ++i, angle += scan->angle_increment)
        {
            float range = scan->ranges[i];
            
            // Check if point is within valid range
            if (range < range_min_ || range > range_max_ || std::isnan(range) || std::isinf(range))
            {
                continue;
            }
            
            // Convert angle to degrees for angle filtering
            float angle_deg = angle * 180.0 / M_PI;
            if (angle_deg < 0) angle_deg += 360.0;
            
            // Check if angle is within ROI
            bool in_roi = false;
            if (initial_angle_ >= end_angle_)
            {
                in_roi = (angle_deg >= initial_angle_ || angle_deg <= end_angle_);
            }
            else
            {
                in_roi = (angle_deg >= initial_angle_ && angle_deg <= end_angle_);
            }
            
            if (!in_roi) continue;
            
            // Convert polar to cartesian (robot frame: x forward, y left)
            pcl::PointXYZ point;
            point.x = range * std::cos(angle);
            point.y = range * std::sin(angle);
            point.z = 0.0;
            
            cloud->points.push_back(point);
        }
        
        cloud->width = cloud->points.size();
        cloud->height = 1;
        cloud->is_dense = false;
        
        return cloud;
    }
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr applyFilters(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud = cloud;
        
        // Apply voxel grid filter if enabled (for dense point clouds)
        if (use_voxel_filter_ && cloud->points.size() > 1000)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_filtered(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
            voxel_filter.setInputCloud(filtered_cloud);
            voxel_filter.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
            voxel_filter.filter(*voxel_filtered);
            filtered_cloud = voxel_filtered;
        }
        
        // Apply statistical outlier removal if enabled
        if (use_outlier_filter_ && filtered_cloud->points.size() > static_cast<size_t>(outlier_mean_k_))
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr outlier_filtered(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
            sor.setInputCloud(filtered_cloud);
            sor.setMeanK(outlier_mean_k_);
            sor.setStddevMulThresh(outlier_stddev_);
            sor.filter(*outlier_filtered);
            filtered_cloud = outlier_filtered;
        }
        
        return filtered_cloud;
    }
    
    void performClustering(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
        std::vector<pcl::PointIndices>& cluster_indices)
    {
        if (cloud->points.size() < static_cast<size_t>(min_cluster_size_))
        {
            return;
        }
        
        // Create KdTree for efficient nearest neighbor search
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        tree->setInputCloud(cloud);
        
        // Perform Euclidean clustering
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(cluster_tolerance_);
        ec.setMinClusterSize(min_cluster_size_);
        ec.setMaxClusterSize(max_cluster_size_);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(cluster_indices);
    }
    
    struct ClusterInfo
    {
        pcl::PointXYZ centroid;
        pcl::PointXYZ min_point;
        pcl::PointXYZ max_point;
        size_t num_points;
        float distance_to_robot;
        float angle_to_robot;
    };
    
    void extractClusterInfo(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
        const std::vector<pcl::PointIndices>& cluster_indices,
        std::vector<ClusterInfo>& clusters)
    {
        clusters.clear();
        clusters.reserve(cluster_indices.size());
        
        for (const auto& indices : cluster_indices)
        {
            ClusterInfo info;
            
            // Create cluster point cloud
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>());
            cluster_cloud->points.reserve(indices.indices.size());
            
            for (const auto& idx : indices.indices)
            {
                cluster_cloud->points.push_back(cloud->points[idx]);
            }
            
            // Calculate centroid
            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(*cluster_cloud, centroid);
            info.centroid.x = centroid[0];
            info.centroid.y = centroid[1];
            info.centroid.z = centroid[2];
            
            // Get bounding box
            pcl::getMinMax3D(*cluster_cloud, info.min_point, info.max_point);
            
            // Store number of points
            info.num_points = indices.indices.size();
            
            // Calculate distance and angle to robot (at origin)
            info.distance_to_robot = std::sqrt(info.centroid.x * info.centroid.x + 
                                              info.centroid.y * info.centroid.y);
            info.angle_to_robot = std::atan2(info.centroid.y, info.centroid.x) * 180.0 / M_PI;
            if (info.angle_to_robot < 0) info.angle_to_robot += 360.0;
            
            clusters.push_back(info);
        }
    }
    
    void visualizeClusters(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
        const std::vector<pcl::PointIndices>& cluster_indices,
        const std::vector<ClusterInfo>& clusters)
    {
        // Create fresh image
        image_ = cv::Mat::zeros(img_size_, img_size_, CV_8UC3);
        
        // Draw reference circles and axes
        cv::circle(image_, cv::Point(img_center_, img_center_), 50, cv::Scalar(255, 0, 0), 2);
        cv::circle(image_, cv::Point(img_center_, img_center_), 
                   static_cast<int>(range_max_ * pixels_per_meter_), cv::Scalar(100, 100, 100), 1);
        cv::line(image_, cv::Point(img_center_, 0), cv::Point(img_center_, img_size_), 
                 cv::Scalar(255, 0, 0), 1);
        cv::line(image_, cv::Point(0, img_center_), cv::Point(img_size_, img_center_), 
                 cv::Scalar(255, 0, 0), 1);
        
        // Draw robot position
        cv::circle(image_, cv::Point(img_center_, img_center_), 4, cv::Scalar(0, 255, 0), -1);
        
        // Draw clustered points
        for (size_t i = 0; i < cluster_indices.size(); ++i)
        {
            cv::Scalar color = color_palette_[i % color_palette_.size()];
            
            for (const auto& idx : cluster_indices[i].indices)
            {
                cv::Point2f img_point = worldToImage(cloud->points[idx]);
                cv::circle(image_, img_point, 2, color, -1);
            }
            
            // Draw centroid
            cv::Point2f centroid_img = worldToImage(clusters[i].centroid);
            cv::circle(image_, centroid_img, 5, color, 2);
            cv::circle(image_, centroid_img, 3, cv::Scalar(255, 255, 255), -1);
            
            // Draw line from robot to centroid
            cv::line(image_, cv::Point(img_center_, img_center_), centroid_img, 
                     cv::Scalar(0, 0, 255), 1);
            
            // Draw bounding box
            cv::Point2f min_img = worldToImage(clusters[i].min_point);
            cv::Point2f max_img = worldToImage(clusters[i].max_point);
            cv::rectangle(image_, min_img, max_img, color, 1);
            
            // Add cluster label
            std::string label = "C" + std::to_string(i) + " (" + 
                               std::to_string(clusters[i].num_points) + "pts)";
            cv::putText(image_, label, 
                       cv::Point(centroid_img.x + 10, centroid_img.y - 10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
        }
        
        // Add info text
        std::string info_text = "Clusters: " + std::to_string(cluster_indices.size()) + 
                               " | Points: " + std::to_string(cloud->points.size());
        cv::putText(image_, info_text, cv::Point(10, 25),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        
        // Add FPS if enabled
        if (show_fps_)
        {
            std::string fps_text = "FPS: " + std::to_string(static_cast<int>(fps_));
            cv::putText(image_, fps_text, cv::Point(10, 50),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        }
        
        // Display
        cv::imshow(OPENCV_WINDOW, image_);
        cv::waitKey(1);
    }
    
    cv::Point2f worldToImage(const pcl::PointXYZ& point)
    {
        // Convert from world coordinates (meters) to image coordinates (pixels)
        // Robot frame: x forward, y left -> Image: x right, y down
        float img_x = img_center_ - (point.y * pixels_per_meter_);
        float img_y = img_center_ - (point.x * pixels_per_meter_);
        
        // Clamp to image bounds
        img_x = std::max(0.0f, std::min(static_cast<float>(img_size_ - 1), img_x));
        img_y = std::max(0.0f, std::min(static_cast<float>(img_size_ - 1), img_y));
        
        return cv::Point2f(img_x, img_y);
    }
    
    void publishClusterInfo(const std::vector<ClusterInfo>& clusters)
    {
        // Log cluster information
        for (size_t i = 0; i < clusters.size(); ++i)
        {
            RCLCPP_INFO(this->get_logger(), 
                       "Cluster %zu: x=%.2f, y=%.2f, dist=%.2f, angle=%.1f°, points=%zu",
                       i, clusters[i].centroid.x, clusters[i].centroid.y,
                       clusters[i].distance_to_robot, clusters[i].angle_to_robot,
                       clusters[i].num_points);
        }
    }
    
    void updateFPS(const std::chrono::steady_clock::time_point& start_time)
    {
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // Calculate instantaneous FPS
        double instant_fps = 1000.0 / duration.count();
        
        // Apply exponential moving average for smoothing
        fps_ = 0.9 * fps_ + 0.1 * instant_fps;
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ObjectDetectionPCL>();
    RCLCPP_INFO(node->get_logger(), "Object Detection PCL node running...");
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
