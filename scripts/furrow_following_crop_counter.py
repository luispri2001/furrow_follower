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

        self.get_logger().info("Furrow Follower & Crop Counter Node Started")

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

        # Furrow Following Logic (center of lowest points per column)
        smoothed_depth = cv2.GaussianBlur(roi_depth, (5, 5), 0)
        valley_positions = np.argmin(smoothed_depth, axis=0)
        center_line = np.mean(
            np.where(valley_positions < np.median(valley_positions) + 10)[0]
        )
        error = (w / 2) - center_line

        # Control Robot
        twist = Twist()
        twist.linear.x = self.linear_speed
        twist.angular.z = self.Kp * error
        self.cmd_vel_pub.publish(twist)

        # Crop Detection
        hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_BGR2HSV)
        lower_green = np.array([25, 30, 30])
        upper_green = np.array([95, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Clean mask
        kernel = np.ones((5, 5), np.uint8)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        crop_count_frame = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 200:
                x, y, w, h = cv2.boundingRect(cnt)
                center_x = x + w // 2
                center_y = y + h // 2

                depth_value = roi_depth[center_y, center_x]
                if 0.05 < depth_value < 1.5:
                    crop_count_frame += 1

        self.crop_count += crop_count_frame
        self.get_logger().info(
            f"Crops Detected in Frame: {crop_count_frame} | Total: {self.crop_count}"
        )

        # Draw furrow center line
        debug_image = roi_rgb.copy()
        cv2.line(
            debug_image,
            (int(center_line), 0),
            (int(center_line), debug_image.shape[0]),
            (255, 0, 0),
            2,
        )  # Blue line

        # Draw detected crops
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 200:
                x, y, w, h = cv2.boundingRect(cnt)
                center_x = x + w // 2
                center_y = y + h // 2
                depth_value = roi_depth[center_y, center_x]
                if 0.05 < depth_value < 1.5:
                    cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(debug_image, (center_x, center_y), 3, (0, 0, 255), -1)

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

        # Save debug image to disk with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"debug_{timestamp}.jpg"
        filepath = os.path.join(self.debug_image_dir, filename)
        cv2.imwrite(filepath, debug_image)

        # Create a binary mask: valid depth = 255, invalid = 0
        depth_mask = np.where((roi_depth > 0.1) & (roi_depth < 3.0), 255, 0).astype(
            np.uint8
        )

        # Draw center line in white (or use 128 if you want a distinguishable mid-gray line)
        if center_line is not None and not np.isnan(center_line):
            cv2.line(
                depth_mask,
                (int(center_line), 0),
                (int(center_line), depth_mask.shape[0]),
                128,
                2,
            )  # Line value 128 to differentiate from binary mask

        # Save depth mask as PNG
        depth_mask_filename = f"depth_mask_{timestamp}.png"
        depth_mask_filepath = os.path.join(self.debug_image_dir, depth_mask_filename)
        cv2.imwrite(depth_mask_filepath, depth_mask)

        # Convert and publish debug image
        debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding="bgr8")
        debug_msg.header.stamp = self.get_clock().now().to_msg()
        self.debug_image_pub.publish(debug_msg)


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
