# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

APEX is a ROS2-based autonomous navigation system for the Yahboom ROSMASTER-A1 robot (Jetson Nano). It implements a two-stage perception pipeline: LiDAR-based object detection and camera-based lane detection.

## Build Commands

```bash
# Build all packages in the main workspace
cd /home/jose/APEX/yahboomcar_ros2_ws/yahboomcar_ws
colcon build

# Build a single package
colcon build --packages-select object_detection
colcon build --packages-select lane_detection

# Build the LiDAR driver (separate workspace)
cd /home/jose/APEX/yahboomcar_ros2_ws/software/library_ws
colcon build --packages-select sllidar_ros2

# Source the workspace after building
source install/setup.bash
```

## Running Nodes

```bash
# LiDAR driver (Slamtec C1)
ros2 launch sllidar_ros2 sllidar_c1_launch.py

# Object detection (PCL-based, recommended)
ros2 run object_detection object_detection_pcl

# Object detection (custom DBSCAN implementation)
ros2 run object_detection object_detection

# Lane detection
ros2 run lane_detection image_subscriber
```

## Architecture

### Workspace Layout
- `yahboomcar_ros2_ws/yahboomcar_ws/src/` — Main application packages
- `yahboomcar_ros2_ws/software/library_ws/src/` — External library packages (LiDAR driver)

### Data Flow
```
Slamtec C1 LiDAR → sllidar_ros2 (/scan LaserScan)
  → object_detection_pcl → /detected_objects_cloud (PointCloud2)

Camera → lane_detection (image_subscriber) → steering control
```

### Packages

**object_detection** — LiDAR point cloud processing with DBSCAN clustering via PCL. Pipeline: polar→cartesian conversion, range/angle filtering, voxel grid downsampling, statistical outlier removal, euclidean cluster extraction. Two executables: `object_detection_pcl` (PCL-based, preferred) and `object_detection` (custom implementation). Note: missing `package.xml`.

**lane_detection** — Camera-based lane detection using OpenCV 4.10 with perspective transformation, edge detection, and line fitting. C++17.

**sllidar_ros2** — Slamtec LiDAR ROS2 driver (external). Serial connection at `/dev/rplidar`, 460800 baud. Has launch files per LiDAR model.

### Key ROS Parameters (object_detection_pcl)
- `range_min`/`range_max`: 0.1m / 16.0m
- `cluster_tolerance`: 0.15m
- `min_cluster_size`/`max_cluster_size`: 3 / 10000
- `voxel_leaf_size`: 0.02m
- `outlier_mean_k`/`outlier_stddev`: 10 / 0.5

## Hardware Target

Yahboom ROSMASTER-A1 with Jetson Nano. SSH access: `ssh jetson@<robot-ip>` (default password: `yahboom`).
