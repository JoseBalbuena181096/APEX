cmake_minimum_required(VERSION 3.8)
project(navigation_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3 -DNDEBUG)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

add_executable(master src/Master.cpp)
target_compile_features(master PUBLIC cxx_std_17)
ament_target_dependencies(master rclcpp std_msgs geometry_msgs)

install(TARGETS master
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch)

install(PROGRAMS scripts/control_tuning_server.py
  DESTINATION share/${PROJECT_NAME}/scripts)

ament_package()
