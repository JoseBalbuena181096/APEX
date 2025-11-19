cmake_minimum_required(VERSION 3.8)
project(object_detection)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

add_executable(object_detection src/object_detection.cpp)
ament_target_dependencies(object_detection
  rclcpp
  sensor_msgs
  geometry_msgs
  pcl_conversions
)
target_link_libraries(object_detection
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
) 

add_executable(object_detection_pcl src/object_detection_pcl.cpp)
ament_target_dependencies(object_detection_pcl
  rclcpp
  sensor_msgs
  geometry_msgs
  pcl_conversions
)
target_link_libraries(object_detection_pcl
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
)

install(TARGETS
  object_detection
  object_detection_pcl
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()