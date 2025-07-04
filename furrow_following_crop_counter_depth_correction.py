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

        self.debug_image_pub = self.create_publisher(Image, "/debug_image", 10)
        self.depth_vis_pub = self.create_publisher(Image, "/depth_visualization", 10)
        self.cmd_vel_pub = self.create_publisher(
            Twist, "/robot/robotnik_base_control/cmd_vel", 10
        )
        self.bridge = CvBridge()

        color_sub = message_filters.Subscriber(
            self, Image, "/robot/zed2/zed_node/rgb/image_rect_color"
        )
        depth_sub = message_filters.Subscriber(
            self, Image, "/robot/zed2/zed_node/depth/depth_registered"
        )
        ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], 10, 0.1)
        ts.registerCallback(self.image_callback)

        self.Kp = 0.02
        self.linear_speed = 0.4
        self.crop_count = 0
        self.highest_point_memory = None
        
        self.min_depth_vis = 0.3
        self.max_depth_vis = 1.2

        self.get_logger().info("Furrow Follower & Crop Counter Node Started - HIGHEST POINT FOLLOWING MODE")

    def image_callback(self, rgb_msg: Image, depth_msg: Image) -> None:
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        h, w = depth_image.shape
        roi_rgb = rgb_image[h // 2 :, :]
        roi_depth = depth_image[h // 2 :, :]

        valid_mask = np.isfinite(roi_depth) & (roi_depth > 0.05) & (roi_depth < 3.0)
        processed_depth = np.copy(roi_depth)
        processed_depth[~valid_mask] = np.inf

        # === AJUSTE POR DESVIACIÓN DE CÁMARA ===
        # Región de búsqueda (desplazada hacia la derecha)
        shift_pixels = 30
        center_start = w // 3 + shift_pixels
        center_end = 2 * w // 3 + shift_pixels
        center_region_depth = processed_depth[:, center_start:center_end]

        if np.any(np.isfinite(center_region_depth)):
            finite_mask = np.isfinite(center_region_depth)
            if np.any(finite_mask):
                min_depth_value = np.min(center_region_depth[finite_mask])
                min_positions = np.where(center_region_depth == min_depth_value)
                
                if len(min_positions[0]) > 0:
                    highest_row = min_positions[0][0]
                    highest_col_in_center = min_positions[1][0]
                    highest_col = center_start + highest_col_in_center
                    
                    self.get_logger().info(f"Highest point found at col: {highest_col}, depth: {min_depth_value:.3f}m")
                    
                    if self.highest_point_memory is None:
                        self.highest_point_memory = highest_col
                    else:
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

        # Error calculado respecto al centro REAL del robot
        robot_center = (w // 2) - shift_pixels  # Centro del robot está a la izquierda
        error = robot_center - target_center

        twist = Twist()
        twist.linear.x = self.linear_speed
        max_correction = 0.5
        twist.angular.z = max(min(self.Kp * error, max_correction), -max_correction)
        
        self.cmd_vel_pub.publish(twist)

        debug_image = roi_rgb.copy()
        cv2.line(debug_image, (center_start, 0), (center_start, debug_image.shape[0]), (255, 255, 0), 1)
        cv2.line(debug_image, (center_end, 0), (center_end, debug_image.shape[0]), (255, 255, 0), 1)

        if target_center is not None and not np.isnan(target_center):
            cv2.circle(debug_image, (int(target_center), highest_row), 5, (0, 0, 255), -1)
            cv2.line(debug_image, (int(target_center), 0), (int(target_center), debug_image.shape[0]), (255, 0, 0), 2)
        
        cv2.line(debug_image, (w // 2, 0), (w // 2, debug_image.shape[0]), (0, 255, 0), 1)

        depth_vis = np.zeros((roi_depth.shape[0], roi_depth.shape[1], 3), dtype=np.uint8)
        vis_mask = np.isfinite(roi_depth) & (roi_depth >= self.min_depth_vis) & (roi_depth <= self.max_depth_vis)
        
        if np.any(vis_mask):
            depth_range = self.max_depth_vis - self.min_depth_vis
            normalized = np.zeros_like(roi_depth)
            clipped_depth = np.clip(roi_depth, self.min_depth_vis, self.max_depth_vis)
            normalized[vis_mask] = 255 - ((clipped_depth[vis_mask] - self.min_depth_vis) / depth_range * 255)
            depth_vis = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)
            depth_vis[~vis_mask] = [0, 0, 0]
            cv2.putText(depth_vis, f"Depth Range: {self.min_depth_vis:.2f}m - {self.max_depth_vis:.2f}m", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw center region boundaries on depth visualization
        # CORREGIR: usar coordenadas relativas a la ROI, no a la imagen completa
        roi_center_start = center_start  # Ya están ajustadas para la ROI
        roi_center_end = center_end      # Ya están ajustadas para la ROI
        
        cv2.line(depth_vis, (roi_center_start, 0), (roi_center_start, depth_vis.shape[0]), (255, 255, 0), 2)
        cv2.line(depth_vis, (roi_center_end, 0), (roi_center_end, depth_vis.shape[0]), (255, 255, 0), 2)
        
        # Draw robot center line on depth visualization (compensated position)
        # Si la cámara está 5cm a la derecha del robot, el centro del robot está a la izquierda
        robot_center_pixel = int(w // 2)  # Centro exacto de la imagen
        cv2.line(
            depth_vis,
            (robot_center_pixel, 0),
            (robot_center_pixel, depth_vis.shape[0]),
            (0, 255, 0),  # Green line
            2,
        )
        
        # Draw camera center line for reference
        cv2.line(
            depth_vis,
            (w // 2, 0),
            (w // 2, depth_vis.shape[0]),
            (128, 128, 128),  # Gray line - centro de la cámara
            1,
        )

        if target_center is not None and not np.isnan(target_center):
            cv2.circle(depth_vis, (int(target_center), highest_row), 5, (0, 0, 255), -1)
            cv2.line(depth_vis, (int(target_center), 0), (int(target_center), depth_vis.shape[0]), (255, 255, 255), 2)

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
            cv2.putText(debug_image, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filepath = os.path.join(self.debug_image_dir, f"debug_{timestamp}.jpg")
        cv2.imwrite(filepath, debug_image)

        depth_vis_filepath = os.path.join(self.debug_image_dir, f"depth_vis_{timestamp}.jpg")
        cv2.imwrite(depth_vis_filepath, depth_vis)

        debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding="bgr8")
        debug_msg.header.stamp = self.get_clock().now().to_msg()
        self.debug_image_pub.publish(debug_msg)

        depth_vis_msg = self.bridge.cv2_to_imgmsg(depth_vis, encoding="bgr8")
        depth_vis_msg.header.stamp = self.get_clock().now().to_msg()
        self.depth_vis_pub.publish(depth_vis_msg)

        # Draw additional reference lines for better visualization
        # Center line of the ROI
        cv2.line(depth_vis, (w // 2, 0), (w // 2, depth_vis.shape[0]), (128, 128, 128), 1)
        
        # Search region boundaries (amarillo más grueso)
        cv2.line(depth_vis, (center_start, 0), (center_start, depth_vis.shape[0]), (0, 255, 255), 3)
        cv2.line(depth_vis, (center_end, 0), (center_end, depth_vis.shape[0]), (0, 255, 255), 3)
        
        # Add text labels
        cv2.putText(depth_vis, "Search Region", (center_start + 5, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


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
