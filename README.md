# furrow_follower

A ROS2 node for autonomous robot navigation that follows furrows (crop rows) using computer vision and depth sensing.

## Overview

This repository contains two versions of a furrow-following robot controller that uses RGB and depth camera data to:
- Navigate autonomously along crop furrows
- Detect and count crops using color-based vision
- Control robot movement with proportional steering

## Files

### [scripts/furrow_following_crop_counter.py](scripts/furrow_following_crop_counter.py)
**Original version** - Valley-based furrow following
- Follows the **lowest points** (valleys) in depth data to stay in the furrow
- Includes crop detection and counting using green color detection
- Saves debug images and depth masks
- Linear speed: 0.4 m/s

### [scripts/furrow_following_crop_counter_v2.py](scripts/furrow_following_crop_counter_v2.py)
**Enhanced version** - Ridge-based furrow following
- Follows the **highest points** (ridges) in the center region for more stable navigation
- Enhanced depth visualization with colormap
- Temporal filtering for smoother movement
- Reduced linear speed: 0.2 m/s for better control
- Additional debug visualization features

## Key Features

- **Synchronized RGB + Depth Processing**: Uses ZED2 camera data
- **Real-time Control**: Publishes velocity commands for robot movement
- **Visual Debugging**: Saves timestamped debug images for analysis
- **Crop Detection**: Counts green vegetation using HSV color filtering
- **ROS2 Integration**: Compatible with ROS2 robotics framework

## Hardware Requirements

- Robot with differential drive (Robotnik base)
- ZED2 stereo camera for RGB and depth data
- ROS2 environment

## Usage

Run either version depending on your furrow following strategy:

```bash
# For valley-following (original)
ros2 run furrow_follower furrow_following_crop_counter.py

# For ridge-following (enhanced)
ros2 run furrow_follower furrow_following_crop_counter_v2.py
```

Debug images are automatically saved to `~/furrow_debug_images/`