#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import rospy
from geometry_msgs.msg import Twist


class LaneFollower:
    """Follows the lane using PD control on the detected road center"""
    
    def __init__(self, detection_mode="asphalt", kp=0.052, kd=0.0, speed=1.5, skip_bottom=230, roi_height=80):
        # Detection mode (asphalt, grass, etc.)
        self.detection_mode = detection_mode
        
        # Image cropping and center finding
        self.skip_bottom = skip_bottom
        self.roi_height = roi_height

        # PD control gains
        self.kp = kp
        self.kd = kd
        self.speed = speed
        self.max_turn_speed = 4.0
        
        # Memory for line tracking
        self.last_center = None
        self.frames_lost = 0
        self.number_of_allowed_missed_frames = 6
        self.prev_error = 0.0
        self.prev_time = None
    
    # Main entry to driving
    def get_command(self, img, center_bias=0.0):
        """Get driving command from image
        
        center_bias: -0.1 = aim left, 0.0 = aim center, 0.1 = aim right
        """
        # Find road center
        road_center, img_width = self.find_road_center(img)
        
        # Calculate target position
        target_x = img_width * (0.5 + center_bias)
        
        # Generate movement command
        twist = self.calculate_movement(road_center, target_x, img_width)

        return twist
    
    # easy function to switch between roads were driving on
    def find_road_center(self, img):
        """Find the center of the road - method depends on detection_mode"""
        if self.detection_mode == "asphalt":
            return self._detect_asphalt_road(img)
        elif self.detection_mode == "grass":
            return self._detect_grass_road(img)
        else:
            return self._detect_asphalt_road(img)  # Default
    
    def _detect_asphalt_road(self, img):
        """Detect grey asphalt road using HSV"""
        h, w = img.shape[:2]
        
        # Look at a box in the bottom half of image (skip very bottom)
        skip_bottom = 230
        roi_height = 80
        
        y_bottom = h - skip_bottom
        y_top = max(0, y_bottom - roi_height)
        
        if y_top >= y_bottom:
            return None, w
        
        roi = img[y_top:y_bottom, :]
        
        # Convert to HSV and find grey road
        hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        h_channel, s_channel, v_channel = cv.split(hsv)
        
        # Grey road = low saturation, medium brightness
        low_sat_mask = cv.inRange(s_channel, 0, 60)
        mid_val_mask = cv.inRange(v_channel, 50, 200)
        
        road_mask = cv.bitwise_and(low_sat_mask, mid_val_mask)
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        road_mask = cv.morphologyEx(road_mask, cv.MORPH_OPEN, kernel)
        road_mask = cv.morphologyEx(road_mask, cv.MORPH_CLOSE, kernel)
        
        # Find center of road
        moments = cv.moments(road_mask)
        
        if moments["m00"] > 1000:  # Enough pixels detected
            center_x = moments["m10"] / moments["m00"]
        else:
            center_x = None
        
        # Show debug view
        self._show_debug(img, y_top, y_bottom, center_x)
        
        return center_x, w
    
    def _detect_grass_road(self, img):
        # TODO: 
        return self._detect_asphalt_road(img)
    
    def _show_debug(self, img, y_top, y_bottom, center_x):
        """Show debug visualization"""
        h, w = img.shape[:2]
        vis = img.copy()
        
        # Draw ROI box
        cv.rectangle(vis, (0, y_top), (w-1, y_bottom-1), (0, 0, 255), 2)
        
        # Draw detected center
        if center_x is not None:
            y_mid = (y_top + y_bottom) // 2
            cv.circle(vis, (int(center_x), y_mid), 8, (0, 0, 255), -1)
        
        # Draw image center line
        cv.line(vis, (w//2, 0), (w//2, h-1), (255, 0, 0), 1)
        
        cv.imshow("lane_debug", vis)
        cv.waitKey(1)
    

    def calculate_movement(self, center_x, target_x, img_width):
        """Calculate Twist command using PD control"""
        twist = Twist()
        
        # Update memory
        if center_x is not None:
            self.last_center = center_x
            self.frames_lost = 0
        else:
            self.frames_lost += 1
        
        # Case 1: We can see the line (or recently saw it)
        if self.last_center is not None and self.frames_lost <= self.number_of_allowed_missed_frames:
            error = self.last_center - target_x
            
            # Calculate derivative (rate of change of error)
            now = rospy.Time.now().to_sec()
            
            if self.prev_time is None or now <= self.prev_time:
                d_error = 0.0
            else:
                dt = now - self.prev_time
                if dt > 0.0001:
                    d_error = (error - self.prev_error) / dt
                else:
                    d_error = 0.0
            
            # PD control
            turn_speed = self.kp * error + self.kd * d_error
            
            # Limit turn speed
            turn_speed = np.clip(turn_speed, -self.max_turn_speed, self.max_turn_speed)
            
            twist.linear.x = self.speed
            twist.angular.z = -turn_speed  # Negative because error is backwards
            
            self.prev_error = error
            self.prev_time = now
        
        # Case 2: Lost the line - search for it
        else:
            if self.last_center is not None:
                # Spin toward where the line should be
                error = self.last_center - target_x
                direction = -1.0 if error > 0 else 1.0
                
                twist.linear.x = self.speed * 0.5
                twist.angular.z = direction * self.max_turn_speed * 0.5
            else:
                # No idea where line is - just spin
                twist.linear.x = 0.0
                twist.angular.z = 1.0
        
        return twist