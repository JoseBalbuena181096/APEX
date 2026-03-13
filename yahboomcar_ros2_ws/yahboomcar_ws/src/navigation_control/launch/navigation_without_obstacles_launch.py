#!/usr/bin/env python3
"""
navigation_without_obstacles_launch.py

Launches the full lane-following pipeline WITHOUT obstacle detection:
  1. Lane detection (C++ node) — processes camera, publishes control signals
  2. Web video server (Python) — MJPEG debug view on port 8081
  3. Master control (C++) — predictive controller with CTE + heading + curvature
  4. Control tuning server (Python) — web UI on port 8082 for live tuning

Prerequisites:
  - Camera must be running (ros2 launch yahboomcar_depth camera_app.launch.py)
  - Chassis driver must be running (ros2 run yahboomcar_bringup Ackman_driver_A1)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Flags
    enable_web_view = LaunchConfiguration('web', default='true')
    headless = LaunchConfiguration('headless', default='true')

    # Script paths
    lane_pkg_dir = get_package_share_directory('lane_detection')
    ctrl_pkg_dir = get_package_share_directory('navigation_control')
    web_server_script = os.path.join(lane_pkg_dir, 'scripts', 'web_video_server.py')
    control_tuning_script = os.path.join(ctrl_pkg_dir, 'scripts', 'control_tuning_server.py')

    return LaunchDescription([
        DeclareLaunchArgument(
            'web',
            default_value='true',
            description='Enable web video server on port 8081'),

        DeclareLaunchArgument(
            'headless',
            default_value='true',
            description='Run without GUI windows'),

        # --- Perception: Lane detection ---
        Node(
            package='lane_detection',
            executable='image_subscriber',
            name='line_detection',
            parameters=[{
                'enable_web_view': enable_web_view,
                'headless': headless,
            }],
            output='screen'),

        # --- Debug: Lane detection web view (port 8081) ---
        ExecuteProcess(
            cmd=['python3', web_server_script],
            condition=IfCondition(enable_web_view),
            output='screen'),

        # --- Control: Master v2 predictive controller ---
        Node(
            package='navigation_control',
            executable='master',
            name='master_control',
            parameters=[{
                'kp_cte': 0.5,
                'kp_heading': 0.3,
                'kff': 0.2,
                'max_speed': 0.20,
                'kv_curve': 0.5,
                'max_angular': 0.8,
                'enabled': False,
            }],
            output='screen'),

        # --- Tuning: Control parameter web UI (port 8082) ---
        ExecuteProcess(
            cmd=['python3', control_tuning_script],
            condition=IfCondition(enable_web_view),
            output='screen'),
    ])
