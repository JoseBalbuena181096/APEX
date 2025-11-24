cmake_minimum_required(VERSION 3.8)
project(lane_detection)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(actionlib_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(OpenCV 4.10 REQUIRED)
find_package(Eigen3 REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

add_executable(image_subscriber src/image_subscriber.cpp)
target_include_directories(image_subscriber PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR})
target_link_libraries(image_subscriber ${OpenCV_LIBS})
target_compile_features(image_subscriber PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
ament_target_dependencies(
  image_subscriber
  "rclcpp"
  "rclpy"
  "std_msgs"
  "sensor_msgs"
  "cv_bridge"
  "image_transport"
  "actionlib_msgs"

)

install(TARGETS image_subscriber
  DESTINATION lib/${PROJECT_NAME})



ament_package()
