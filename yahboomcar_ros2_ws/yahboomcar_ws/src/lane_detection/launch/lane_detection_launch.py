#!/usr/bin/env python3

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

    # Path to the web server script
    pkg_dir = get_package_share_directory('lane_detection')
    web_server_script = os.path.join(pkg_dir, 'scripts', 'web_video_server.py')

    return LaunchDescription([
        DeclareLaunchArgument(
            'web',
            default_value='true',
            description='Enable web video server on port 8081 (true/false)'),

        DeclareLaunchArgument(
            'headless',
            default_value='true',
            description='Run without GUI window (true for SSH, false for local display)'),

        # Lane detection C++ node
        Node(
            package='lane_detection',
            executable='image_subscriber',
            name='line_detection',
            parameters=[{
                'enable_web_view': enable_web_view,
                'headless': headless,
            }],
            output='screen'),

        # Web video server (only if web:=true)
        ExecuteProcess(
            cmd=['python3', web_server_script],
            condition=IfCondition(enable_web_view),
            output='screen'),
    ])
