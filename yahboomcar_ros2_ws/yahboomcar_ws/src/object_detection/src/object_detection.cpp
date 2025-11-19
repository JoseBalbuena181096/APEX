#include <rclcpp/rclcpp.hpp>
#include <signal.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cmath>
#include <vector>
#include <map>
#include <iostream>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <math.h>

#define UNCLASSIFIED -1
#define CORE_POINT 1
#define BORDER_POINT 2
#define NOISE -2
#define SUCCESS 0
#define FAILURE -3



static const std::string OPENCV_WINDOW = "OBJECT DETECTION";
typedef std::vector<cv::Point2f> vector_point;
typedef std::vector<vector_point> obstacles;
typedef std::vector<cv::Point> Cluster;

// Global Parameters
bool DEBUG = true;
float RANGE = 16.0;
float RANGE_MIN = 0.1;
float INITIAL_RANGE_ANGLE = 0.0;
float END_RANGE_ANGLE = 360.0;


class PointC
{
    public:
        float x, y;  // X, Y position
        int clusterID;  // clustered ID
        PointC(float x, float y){
            this->x = x;
            this->y = y;
            this->clusterID = UNCLASSIFIED;
        }
};


class DBSCAN {
public: 
    std::vector<PointC> m_points;
    DBSCAN(unsigned int minPts, float eps,const std::vector<cv::Point> &points_){
        m_minPoints = minPts;
        m_epsilon = eps;
        for (auto point : points_){
            m_points.push_back({(float)(point.x),(float)(point.y)}); 
        }
        m_pointSize =  m_points.size();
    }
    ~DBSCAN(){}

    int run();
    std:: vector<int> calculateCluster(PointC point);
    int expandCluster(PointC point, int clusterID);
    inline double calculateDistance(PointC pointCore, PointC pointTarget);
    int getTotalPointSize() {return m_pointSize;}
    int getMinimumClusterSize() {return m_minPoints;}
    int getEpsilonSize() {return m_epsilon;}
    void getCluster(std::map<int, std::vector<cv::Point>>&);
    
private:
    unsigned int m_pointSize;
    unsigned int m_minPoints;
    float m_epsilon;
};

int DBSCAN::run()
{
    int clusterID = 0;
    std::vector<PointC>::iterator iter;
    for(iter = m_points.begin(); iter != m_points.end(); ++iter)
    {
        if ( iter->clusterID == UNCLASSIFIED )
        {
            if ( expandCluster(*iter, clusterID) != FAILURE )
            {
                clusterID += 1;
            }
        }
    }

    return 0;
}

int DBSCAN::expandCluster(PointC point, int clusterID)
{    
    std::vector<int> clusterSeeds = calculateCluster(point);

    if ( clusterSeeds.size() < m_minPoints )
    {
        point.clusterID = NOISE;
        return FAILURE;
    }
    else
    {
        int index = 0, indexCorePoint = 0;
        std::vector<int>::iterator iterSeeds;
        for( iterSeeds = clusterSeeds.begin(); iterSeeds != clusterSeeds.end(); ++iterSeeds)
        {
            m_points.at(*iterSeeds).clusterID = clusterID;
            if (m_points.at(*iterSeeds).x == point.x && m_points.at(*iterSeeds).y == point.y)
            {
                indexCorePoint = index;
            }
            ++index;
        }
        clusterSeeds.erase(clusterSeeds.begin()+indexCorePoint);

        for(std::vector<int>::size_type i = 0, n = clusterSeeds.size(); i < n; ++i )
        {
            std::vector<int> clusterNeighors = calculateCluster(m_points.at(clusterSeeds[i]));

            if ( clusterNeighors.size() >= m_minPoints )
            {
                std::vector<int>::iterator iterNeighors;
                for ( iterNeighors = clusterNeighors.begin(); iterNeighors != clusterNeighors.end(); ++iterNeighors )
                {
                    if ( m_points.at(*iterNeighors).clusterID == UNCLASSIFIED || m_points.at(*iterNeighors).clusterID == NOISE )
                    {
                        if ( m_points.at(*iterNeighors).clusterID == UNCLASSIFIED )
                        {
                            clusterSeeds.push_back(*iterNeighors);
                            n = clusterSeeds.size();
                        }
                        m_points.at(*iterNeighors).clusterID = clusterID;
                    }
                }
            }
        }

        return SUCCESS;
    }
}

std::vector<int> DBSCAN::calculateCluster(PointC point)
{
    int index = 0;
    std::vector<PointC>::iterator iter;
    std::vector<int> clusterIndex;
    for( iter = m_points.begin(); iter != m_points.end(); ++iter)
    {
        if ( calculateDistance(point, *iter) <= m_epsilon )
        {
            clusterIndex.push_back(index);
        }
        index++;
    }
    return clusterIndex;
}

inline double DBSCAN::calculateDistance(PointC pointCore, PointC pointTarget )
{
    return sqrt((pointCore.x - pointTarget.x)*(pointCore.x - pointTarget.x)+(pointCore.y - pointTarget.y)*(pointCore.y - pointTarget.y));
}

void DBSCAN::getCluster(std::map<int, std::vector<cv::Point>> &clusters_points){
    for (auto point: m_points){
        clusters_points[point.clusterID].push_back(cv::Point(int(point.x), int(point.y)));
    }
}


class ObjectDetection : public rclcpp::Node
{
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub;
    sensor_msgs::msg::LaserScan newscan;
    vector_point scan_points;
    std::vector<cv::Point> points_;
    int MINIMUM_POINTS;         // minimum number of cluster
    int EPSILON;                // distance for clustering
    std::map<int,std::vector<cv::Point>> clusters_points;
    std::vector<cv::Point> points_centroids;
    cv::Mat image;
    public:

    ObjectDetection() : Node("object_detection"){
        MINIMUM_POINTS = 1;
        EPSILON = 95.0;
        scan_sub = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10, std::bind(&ObjectDetection::laser_msg_Callback, this, std::placeholders::_1)); 
    }

        //Destructor
        ~ObjectDetection(){
          	cv::destroyWindow(OPENCV_WINDOW);
        }
    void laser_msg_Callback(const sensor_msgs::msg::LaserScan::SharedPtr scan){
        // Local variables
        std::vector<float> ROI;
        std::vector<float> ranges;
        std::vector<float> intensities;
        vector_point scan_points;

        // Fill the ranges vector
        for (float i = 0.0; i <= 360.0; i++){
            ranges.push_back(std::numeric_limits<float>::infinity());
            intensities.push_back(std::numeric_limits<float>::infinity());  
        }

        // Range to be considered
        if (INITIAL_RANGE_ANGLE >= END_RANGE_ANGLE){
            float i = INITIAL_RANGE_ANGLE;
            while (i<360){
                ROI.push_back(i);
                i += 1.0;
            }
            i = 0.0;
            while (i < END_RANGE_ANGLE){
                ROI.push_back(i);
                i += 1.0;
            }
        }
        else{
            float i = INITIAL_RANGE_ANGLE;
            while (i < END_RANGE_ANGLE){
                ROI.push_back(i);
                i += 1.0;
            }
        }

        // Takes points within a certian distance range
        for (std::vector<float>::iterator i = ROI.begin();i != ROI.end(); i++){ 
            if(scan->ranges[*i] <= RANGE &&  scan->ranges[*i] >= RANGE_MIN)
            {
                ranges[*i] = scan->ranges[*i];
                intensities[*i] = scan->intensities[*i];
                scan_points.push_back(
                cv::Point2f(*i, scan->ranges[*i]));
           }
        }
        image = cv::Mat::zeros( 800, 800, CV_8UC3 );
        get_object_points(scan_points);
        circle(image, cv::Point( 400, 400 ), 50.0, cv::Scalar( 255, 0, 0), 2);
        circle(image, cv::Point(400 , 400),cvRound((double)4/ 2), cv::Scalar(0, 255, 0),2);
        line(image, cv::Point(400,0), cv::Point(400,800), cv::Scalar(255, 0, 0), 2); 
        line(image, cv::Point(0,400), cv::Point(800,400), cv::Scalar(255, 0, 0), 2); 
        //std::cout << "Number clusters " << clusters_points.size() << std::endl;
        

        if (points_centroids.size() > 0){
            for (auto point : points_centroids){
                geometry_msgs::msg::Point point_msg;
                line(image, cv::Point(400,400), point, cv::Scalar(0, 0, 255), 2);                
                transform_point(point, point_msg);
                point_msg.z = 0;
                //std::cout << "x: "<< point.x << std::endl; 
                //std::cout << "y: "<< point.y << std::endl; 
                RCLCPP_INFO(this->get_logger(), "x: %.2f, y: %.2f", point_msg.x, point_msg.y);
            }
        }
        resize_image(image);
        cv::resizeWindow(OPENCV_WINDOW, image.cols, image.rows);
        imshow(OPENCV_WINDOW,image);
        cv::waitKey(15);
    }

    void get_object_points(const vector_point& scan_points){
        std::vector <PointC> points_dbscan;
        points_.clear();
        clusters_points.clear();
        float x, y, r;
        for (auto point : scan_points){
            r = point.y * 100.0;
            x = (r)*cos(static_cast<float>(180-point.x)*3.14159/180.0);
            y = (r)*sin(static_cast<float>(180-point.x)*3.14159/180.0);
            x = 400.0 + x;
            x = x < 800 ? x : 800;
            x = x >= 0 ? x : 0; 
            y = 400.0 + y;
            y = y < 800 ? y : 800;
            y = y >= 0 ? y : 0; 
            points_.push_back(cv::Point(x, y));
        }
        DBSCAN dbScan(MINIMUM_POINTS,EPSILON,points_);
        dbScan.run();
        dbScan.getCluster(clusters_points);
        show_clusters(image, clusters_points);
        get_centroids_objects(clusters_points);
    }

    void transform_point(cv::Point &point, geometry_msgs::msg::Point &point_msg){
        float adjacent_leg, opposite_leg;
        adjacent_leg = point.y - 400; 
        opposite_leg = point.x - 400;
        adjacent_leg = (adjacent_leg == 0.0) ? 0.000001 : adjacent_leg;
        if (point.x >= 0 && point.x <= 400 && point.y >= 0 && point.y <= 400){
            point_msg.y =  atan(opposite_leg / adjacent_leg)* 57.295779;
        }
        else if  (point.x >= 0 && point.x <= 400 && point.y >= 400 && point.y <= 800){
            point_msg.y =  180.0 + atan(opposite_leg / adjacent_leg)* 57.295779;
        }
        else if  (point.x >= 400 && point.x <= 800 && point.y >= 400 && point.y <= 800){
            point_msg.y =  180.0 + atan(opposite_leg / adjacent_leg)* 57.295779;
        }
        else if  (point.x >= 400 && point.x <= 800 && point.y >= 0 && point.y <= 400){
            point_msg.y =  360.0 + atan(opposite_leg / adjacent_leg)* 57.295779;
        }
        if (point_msg.y < 0){
            point_msg.y = (-1) * point_msg.y;
        }
        point_msg.x = cv::norm(point - cv::Point(400,400));
    }

    void resize_image(cv::Mat &input,float alpha=1.0){
		    cv::resize(input,input,cv::Size(input.cols*alpha,input.rows*alpha));
	}

    void show_clusters(cv::Mat &frame,const std::map<int,std::vector<cv::Point>> &clusteredPoints_){
            int colors[3];
            for (auto cluster : clusteredPoints_){
                colors[0] = static_cast<int>(rand() % 255);
                colors[1] = static_cast<int>(rand() % 255);
                colors[2] = static_cast<int>(rand() % 255);
                for (auto point : cluster.second){
                    circle(frame, cv::Point(point.x , point.y),cvRound((double)4/ 2), cv::Scalar(colors[0], colors[1], colors[2]),2);
                }
                //std::cout << cluster.second.size()<<std::endl;
            }
        }

    void  get_centroids_objects(const std::map<int,std::vector<cv::Point>> &clusteredPoints_){
        points_centroids.clear();
        int x , y, n;
        for (auto cluster : clusteredPoints_){
            x = y = 0;
            n = cluster.second.size();
            for (auto point : cluster.second){
                x += point.x; 
                y += point.y;
            }
            points_centroids.push_back(cv::Point(x / n , y / n));
        }
    }

};


int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ObjectDetection>();
    RCLCPP_INFO(node->get_logger(), "Object detection node running...");
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}