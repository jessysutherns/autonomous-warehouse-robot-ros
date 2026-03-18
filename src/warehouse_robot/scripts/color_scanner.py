# Autonomous Warehouse Robot Controller
# Author: Jessica Sutherns
# Description:
# This node performs object detection (HSV), navigation control,
# and pick-and-place operations using a robotic arm in ROS.

# #!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
from std_srvs.srv import Empty

class ColorScanner:
    def __init__(self):
        rospy.init_node('color_scanner', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.arm_pub = rospy.Publisher('/arm_controller/command', Float64, queue_size=10, latch=True)
        self.move_cmd = Twist()
        self.blue_pixels = 0
        self.x_pos = -2.0
        self.y_pos = 0.0
        self.cube_seen = False
        self.rate = rospy.Rate(50)
        self.is_picked_up = False
        self.is_delivered = False
        self.cube_actual_x = 2.0
        self.drop_off_x = 1.6  # Changed to 1.6
        self.stop_x = 1.4      # Back off to 1.4
        self.last_time = rospy.get_time()
        self.pre_pickup_pixels = 0
        rospy.loginfo("ColorScanner initialized. Waiting 6s for cube spawn...")
        rospy.sleep(6)
        self.reset_simulation()

    def reset_simulation(self):
        try:
            rospy.wait_for_service('/gazebo/reset_world')
            reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
            reset_world()
            rospy.loginfo("Gazebo world reset requested.")
            self.x_pos = -2.0
            self.y_pos = 0.0
            self.last_time = rospy.get_time()
            self.cube_seen = False
            self.is_picked_up = False
            self.is_delivered = False
            rospy.sleep(1)
            rospy.loginfo("Reset complete, starting motion.")
        except rospy.ServiceException as e:
            rospy.logerr(f"Gazebo reset failed: {e}")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([140, 255, 255])
            mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
            self.blue_pixels = cv2.countNonZero(mask)
            percentage = (self.blue_pixels / mask.size) * 100
            distance_to_cube = abs(self.x_pos - self.cube_actual_x)
            self.cube_seen = 5000 < self.blue_pixels < 2500000 and distance_to_cube < 3.5
            if self.cube_seen:
                rospy.loginfo(f"Detected cube at x={self.x_pos:.3f} with {self.blue_pixels} pixels")
            elif self.blue_pixels <= 5000:
                rospy.loginfo(f"Cube not visible at x={self.x_pos:.3f}, too few pixels: {self.blue_pixels}")
            elif self.blue_pixels >= 2500000:
                rospy.loginfo(f"Excessive blue pixels at x={self.x_pos:.3f}: {self.blue_pixels}")
            hsv_mean = np.mean(hsv_image, axis=(0, 1))
            h, w = hsv_image.shape[:2]
            center_patch = hsv_image[h//4:3*h//4, w//4:3*w//4]
            center_mean = np.mean(center_patch, axis=(0, 1))
            rospy.loginfo(f"Image size: {cv_image.shape}, Blue pixels: {self.blue_pixels}, Percentage: {percentage:.2f}%, Cube seen: {self.cube_seen}, X-Distance: {distance_to_cube:.2f}m, HSV mean: {hsv_mean}, Center HSV mean: {center_mean}")
        except Exception as e:
            rospy.logerr(f"Error in image_callback: {e}")

    def move_arm(self, position):
        rospy.loginfo(f"Attempting to move arm to {position} radians")
        for _ in range(10):
            self.arm_pub.publish(position)
            rospy.sleep(0.2)
        rospy.loginfo("Arm move command sequence completed")

    def run(self):
        while not rospy.is_shutdown():
            current_time = rospy.get_time()
            dt = current_time - self.last_time
            new_x_pos = self.x_pos + self.move_cmd.linear.x * dt
            if self.is_picked_up and new_x_pos < self.drop_off_x:
                self.x_pos = self.drop_off_x
            elif self.is_delivered and new_x_pos < self.stop_x:
                self.x_pos = self.stop_x
            else:
                self.x_pos = new_x_pos
            self.y_pos += self.move_cmd.angular.z * dt
            self.last_time = current_time
            distance_to_cube = abs(self.x_pos - self.cube_actual_x)
            distance_to_drop = abs(self.x_pos - self.drop_off_x)
            distance_to_stop = abs(self.x_pos - self.stop_x)
            self.move_cmd.angular.z = 0.0

            rospy.loginfo(f"Tracked x={self.x_pos:.3f}, y={self.y_pos:.3f}, dt={dt:.3f}, X-Distance to cube: {distance_to_cube:.2f}m, X-Distance to drop: {distance_to_drop:.2f}m")

            if self.x_pos < self.drop_off_x and not self.is_picked_up and not self.is_delivered:
                self.move_cmd.linear.x = 0.06
                rospy.loginfo("Moving forward (fast) to pickup")
            elif self.cube_seen and distance_to_cube > 0.03 and not self.is_picked_up and not self.is_delivered:
                self.move_cmd.linear.x = 0.02
                rospy.loginfo("Moving forward (slow) to pickup")
            elif distance_to_cube <= 0.03 and not self.is_picked_up and not self.is_delivered:
                self.move_cmd.linear.x = 0.0
                rospy.loginfo("Stopped within 0.03m of cube")
                self.vel_pub.publish(self.move_cmd)
                self.pre_pickup_pixels = self.blue_pixels
                rospy.sleep(0.5)
                self.move_arm(0.8)  # Grip
                rospy.sleep(5)     # Longer grip time
                self.move_arm(1.5)  # Higher lift
                rospy.sleep(2)
                if self.blue_pixels < self.pre_pickup_pixels * 0.7:
                    self.is_picked_up = True
                    rospy.loginfo(f"Picked up cube at x={self.x_pos:.3f}, pixels dropped from {self.pre_pickup_pixels} to {self.blue_pixels}")
                else:
                    rospy.logwarn(f"Failed to pick up cube at x={self.x_pos:.3f}, pixels: {self.pre_pickup_pixels} to {self.blue_pixels}")
                    self.is_picked_up = False
            elif self.is_picked_up and distance_to_drop > 0.03 and not self.is_delivered:
                self.move_cmd.linear.x = -0.06
                self.move_arm(1.5)  # Hold higher
                rospy.loginfo(f"Moving to drop-off at x={self.drop_off_x}, distance remaining: {distance_to_drop:.2f}m")
                for _ in range(10):
                    self.vel_pub.publish(self.move_cmd)
                    rospy.sleep(0.1)
            elif self.is_picked_up and distance_to_drop <= 0.03 and not self.is_delivered:
                self.move_cmd.linear.x = 0.0
                rospy.loginfo(f"Reached drop-off at x={self.drop_off_x:.3f}")
                self.vel_pub.publish(self.move_cmd)
                rospy.sleep(0.5)
                self.move_arm(0.0)  # Release
                rospy.sleep(15)    # Long delay
                self.move_cmd.linear.x = 0.0
                for _ in range(10):  # Ensure stop
                    self.vel_pub.publish(self.move_cmd)
                    rospy.sleep(0.1)
                self.move_cmd.linear.x = -0.04  # Back to x=1.4
                rospy.loginfo(f"Moving past drop-off to x={self.stop_x}, distance remaining: {distance_to_stop:.2f}m")
                for _ in range(50):  # 5s at -0.04 = 0.2m
                    self.vel_pub.publish(self.move_cmd)
                    rospy.sleep(0.1)
                self.move_cmd.linear.x = 0.0
                for _ in range(10):  # Ensure stop
                    self.vel_pub.publish(self.move_cmd)
                    rospy.sleep(0.1)
                self.move_arm(2.5)  # Max lift
                rospy.sleep(2)
                self.is_delivered = True
                self.x_pos = self.stop_x
                rospy.loginfo(f"Dropped cube at x={self.drop_off_x:.3f}, stopped at x={self.x_pos:.3f}")
            elif self.is_delivered:
                self.move_cmd.linear.x = 0.0
                self.move_arm(2.5)
                self.x_pos = self.stop_x
                rospy.sleep(0.1)
                rospy.loginfo("Task complete, stopped past drop-off")
                break

            self.vel_pub.publish(self.move_cmd)
            rospy.loginfo(f"Published cmd_vel: linear.x={self.move_cmd.linear.x}, angular.z={self.move_cmd.angular.z}")
            self.rate.sleep()

if __name__ == '__main__':
    try:
        scanner = ColorScanner()
        scanner.run()
    except rospy.ROSInterruptException:
        pass