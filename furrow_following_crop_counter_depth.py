import os
import cv2
import numpy as np
from datetime import datetime

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

import message_filters
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist


class FurrowFollowerCropCounter(Node):

    def __init__(self) -> None:
        super().__init__("furrow_follower_crop_counter")

        self.debug_image_dir = os.path.expanduser("~/furrow_debug_images")
        os.makedirs(self.debug_image_dir, exist_ok=True)

        # Publishers and Bridge
        self.debug_image_pub = self.create_publisher(Image, "/debug_image", 10)
        self.depth_vis_pub = self.create_publisher(Image, "/depth_visualization", 10)
        self.cmd_vel_pub = self.create_publisher(
            Twist, "/robot/robotnik_base_control/cmd_vel", 10
        )
        self.bridge = CvBridge()

        # Image subscribers (synchronized)
        color_sub = message_filters.Subscriber(
            self, Image, "/robot/zed2/zed_node/rgb/image_rect_color"
        )
        depth_sub = message_filters.Subscriber(
            self, Image, "/robot/zed2/zed_node/depth/depth_registered"
        )
        ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], 10, 0.1)
        ts.registerCallback(self.image_callback)

        # Parameters
        self.Kp = 0.02
        self.linear_speed = 0.4
        self.crop_count = 0
        self.highest_point_memory = None  # To store highest point position for filtering
        
        # Depth visualization parameters
        self.min_depth_vis = 0.3  # Minimum depth for visualization (meters)
        self.max_depth_vis = 1.2  # Maximum depth for visualization (meters)

        self.get_logger().info("Furrow Follower & Crop Counter Node Started - HIGHEST POINT FOLLOWING MODE")

    def image_callback(self, rgb_msg: Image, depth_msg: Image) -> None:
        try:
            # Convert to OpenCV format
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        # Focus on lower half (assumed furrow region)
        h, w = depth_image.shape
        roi_rgb = rgb_image[h // 2 :, :]
        roi_depth = depth_image[h // 2 :, :]

        # Pre-process depth image - remove invalid values and limit range
        valid_mask = np.isfinite(roi_depth) & (roi_depth > 0.05) & (roi_depth < 3.0)
        processed_depth = np.copy(roi_depth)
        processed_depth[~valid_mask] = np.inf  # Set invalid areas to infinity to ignore them

        # === FIND HIGHEST POINT (MINIMUM DEPTH) IN CENTER REGION ===
        # Define center region (middle third of the image)
        center_start = w // 3
        center_end = 2 * w // 3
        center_region_depth = processed_depth[:, center_start:center_end]
        
        # Find the highest point (minimum depth value) in the center region
        if np.any(np.isfinite(center_region_depth)):
            # Find minimum depth position in center region
            finite_mask = np.isfinite(center_region_depth)
            if np.any(finite_mask):
                min_depth_value = np.min(center_region_depth[finite_mask])
                min_positions = np.where(center_region_depth == min_depth_value)
                
                if len(min_positions[0]) > 0:
                    # Take the first occurrence of minimum depth
                    highest_row = min_positions[0][0]
                    highest_col_in_center = min_positions[1][0]
                    # Convert back to full image coordinates
                    highest_col = center_start + highest_col_in_center
                    
                    self.get_logger().info(f"Highest point found at col: {highest_col}, depth: {min_depth_value:.3f}m")
                    
                    # Apply temporal filtering for smoothness
                    if self.highest_point_memory is None:
                        self.highest_point_memory = highest_col
                    else:
                        # Smooth filtering (80% previous, 20% current)
                        self.highest_point_memory = 0.8 * self.highest_point_memory + 0.2 * highest_col
                    
                    target_center = self.highest_point_memory
                else:
                    self.get_logger().warn("No valid highest point found in center region")
                    return
            else:
                self.get_logger().warn("No finite depth values in center region")
                return
        else:
            self.get_logger().warn("No valid depth data in center region")
            return

        # Calculate error from center of image
        # Positive error means the highest point is to the right of center (robot should turn left)
        # Negative error means the highest point is to the left of center (robot should turn right)
        error = (w / 2) - target_center

        # Control Robot with proportional control
        twist = Twist()
        twist.linear.x = self.linear_speed
        
        # Apply proportional control to angular velocity based on error
        # Limit maximum correction for smooth driving
        max_correction = 0.5  # Limit maximum angular velocity
        twist.angular.z = max(min(self.Kp * error, max_correction), -max_correction)
        
        self.cmd_vel_pub.publish(twist)
        
        # === VISUALIZATION ===
        # Create debug visualization
        debug_image = roi_rgb.copy()
        
        # Draw the center region boundaries
        cv2.line(debug_image, (center_start, 0), (center_start, debug_image.shape[0]), (255, 255, 0), 1)  # Cyan
        cv2.line(debug_image, (center_end, 0), (center_end, debug_image.shape[0]), (255, 255, 0), 1)  # Cyan
        
        # Draw the highest point
        if target_center is not None and not np.isnan(target_center):
            cv2.circle(debug_image, (int(target_center), highest_row), 5, (0, 0, 255), -1)  # Red circle
            
            # Draw the target center line
            cv2.line(
                debug_image,
                (int(target_center), 0),
                (int(target_center), debug_image.shape[0]),
                (255, 0, 0),  # Blue line
                2,
            )
        
        # Draw robot center line for reference
        cv2.line(
            debug_image,
            (w // 2, 0),
            (w // 2, debug_image.shape[0]),
            (0, 255, 0),  # Green line
            1,
        )

        # Create enhanced depth visualization
        depth_vis = np.zeros((roi_depth.shape[0], roi_depth.shape[1], 3), dtype=np.uint8)
        
        vis_mask = np.isfinite(roi_depth) & (roi_depth >= self.min_depth_vis) & (roi_depth <= self.max_depth_vis)
        
        if np.any(vis_mask):
            depth_range = self.max_depth_vis - self.min_depth_vis
            normalized = np.zeros_like(roi_depth)
            clipped_depth = np.clip(roi_depth, self.min_depth_vis, self.max_depth_vis)
            normalized[vis_mask] = 255 - ((clipped_depth[vis_mask] - self.min_depth_vis) / depth_range * 255)
            
            depth_vis = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)
            depth_vis[~vis_mask] = [0, 0, 0]
            
            # Add depth range info
            cv2.putText(
                depth_vis,
                f"Depth Range: {self.min_depth_vis:.2f}m - {self.max_depth_vis:.2f}m",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        
        # Draw center region boundaries on depth visualization
        cv2.line(depth_vis, (center_start, 0), (center_start, depth_vis.shape[0]), (255, 255, 0), 1)
        cv2.line(depth_vis, (center_end, 0), (center_end, depth_vis.shape[0]), (255, 255, 0), 1)
        
        # Draw highest point and center line on depth visualization
        if target_center is not None and not np.isnan(target_center):
            cv2.circle(depth_vis, (int(target_center), highest_row), 5, (0, 0, 255), -1)
            cv2.line(
                depth_vis,
                (int(target_center), 0),
                (int(target_center), depth_vis.shape[0]),
                (255, 255, 255),
                2,
            )

        # Add velocity info to debug image
        rotation_direction = "Straight"
        if twist.angular.z > 0.05:
            rotation_direction = "Turning Left"
        elif twist.angular.z < -0.05:
            rotation_direction = "Turning Right"

        text_lines = [
            f"Linear Velocity: {twist.linear.x:.2f} m/s",
            f"Angular Velocity: {twist.angular.z:.2f} rad/s",
            f"Direction: {rotation_direction}",
            f"HIGHEST POINT FOLLOWING MODE",
            f"Target Position: {int(target_center) if target_center else 'N/A'}",
            f"Error: {error:.1f} pixels"
        ]

        y_offset = 20
        for line in text_lines:
            cv2.putText(
                debug_image,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            y_offset += 25

        # Save debug images to disk with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Save RGB debug image
        filename = f"debug_{timestamp}.jpg"
        filepath = os.path.join(self.debug_image_dir, filename)
        cv2.imwrite(filepath, debug_image)
        
        # Save depth visualization
        depth_vis_filename = f"depth_vis_{timestamp}.jpg"
        depth_vis_filepath = os.path.join(self.debug_image_dir, depth_vis_filename)
        cv2.imwrite(depth_vis_filepath, depth_vis)

        # Convert and publish debug images
        debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding="bgr8")
        debug_msg.header.stamp = self.get_clock().now().to_msg()
        self.debug_image_pub.publish(debug_msg)
        
        depth_vis_msg = self.bridge.cv2_to_imgmsg(depth_vis, encoding="bgr8")
        depth_vis_msg.header.stamp = self.get_clock().now().to_msg()
        self.depth_vis_pub.publish(depth_vis_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FurrowFollowerCropCounter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
