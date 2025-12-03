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

         # Memory for grass road (to smooth gaps)
        self.last_grass_center = None
        self.grass_frames_lost = 0
        self.max_grass_lost_frames = 15  # Remember position for 3 frames
    
    # Main entry to driving
    def get_command(self, img, center_bias=0.0, speed = None):
        """Get driving command from image
        
        center_bias: -0.1 = aim left, 0.0 = aim center, 0.1 = aim right
        """
        # Find road center
        road_center, img_width = self.find_road_center(img)
        

        # custom speed
        use_speed = speed if speed is not None else self.speed

        # Calculate target position
        target_x = img_width * (0.5 + center_bias)
        
        # Generate movement command
        twist = self.calculate_movement(road_center, target_x, img_width, use_speed)

        return twist
    
    # easy function to switch between roads were driving on
    def find_road_center(self, img):
        """Find the center of the road - method depends on detection_mode"""
        #rospy.loginfo(f"Detecting road in mode: {self.detection_mode}")
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
        """Detect white lines on grass - intelligently follow with adaptive offset"""
        h, w = img.shape[:2]
        
        skip_bottom = 200
        roi_height = 170
        
        y_bottom = h - skip_bottom
        y_top = max(0, y_bottom - roi_height)
        
        if y_top >= y_bottom:
            return None, w
        
        roi = img[y_top:y_bottom, :]
        roi = cv.medianBlur(roi, 5)
        
        hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        
        lower_white = np.array([0, 18, 178])
        upper_white = np.array([159, 87, 255])
        
        white_mask = cv.inRange(hsv, lower_white, upper_white)
        
        kernel = np.ones((5, 5), np.uint8)
        white_mask = cv.morphologyEx(white_mask, cv.MORPH_OPEN, kernel)
        white_mask = cv.morphologyEx(white_mask, cv.MORPH_CLOSE, kernel)
        
        # Find all contours
        contours, _ = cv.findContours(white_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        filtered_mask = np.zeros_like(white_mask)

        min_contour_area = 600
        max_contour_area = 20000          # reject huge blobs (tune)
        min_height = 0.35 * roi_height    # a bit more forgiving than 0.5
        max_lane_width_frac = 0.45        # lane should not span > ~45% of image width

        left_candidates = []
        right_candidates = []

        for cnt in contours:
            area = cv.contourArea(cnt)
            if area < min_contour_area or area > max_contour_area:
                continue

            x, y, cw, ch = cv.boundingRect(cnt)

            # Reject super-wide blobs (road patches, horizon bands, etc.)
            if cw > max_lane_width_frac * w:
                continue

            if ch < min_height:
                continue

            cx = x + cw / 2.0  # contour center x

            rospy.loginfo(
                f"[GRASS DEBUG] contour passed area filter: "
                f"area={area:.1f}, x={x}, y={y}, w={cw}, h={ch}, cx={cx:.1f}"
            )

            # Explicitly separate left vs right half of the image
            if cx < w / 2.0:
                left_candidates.append(cnt)
            else:
                right_candidates.append(cnt)

        # Pick at most 1 lane per side (the largest by area)
        line_contours = []

        if left_candidates:
            left_candidates = sorted(left_candidates, key=cv.contourArea, reverse=True)
            line_contours.append(left_candidates[0])

        if right_candidates:
            right_candidates = sorted(right_candidates, key=cv.contourArea, reverse=True)
            line_contours.append(right_candidates[0])

        # Draw only these "lane" contours into filtered_mask
        filtered_mask[:] = 0
        for cnt in line_contours:
            cv.drawContours(filtered_mask, [cnt], -1, 255, -1)



        # Hole filling pass
        contours_filled, _ = cv.findContours(filtered_mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours_filled)):
            cv.drawContours(filtered_mask, contours_filled, i, 255, -1)
        
        # Find ALL lane centers
        contours_final, _ = cv.findContours(filtered_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        lane_centers = []
        
        for cnt in contours_final:
            M = cv.moments(cnt)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                lane_centers.append(cx)
        
        # SMART OFFSET CALCULATION
        if len(lane_centers) >= 2:
            # Can see BOTH lanes - drive exactly between them
            lane_centers.sort()
            left_lane = lane_centers[0]
            right_lane = lane_centers[-1]
            
            # Center is exactly between the two lanes
            center_x = (left_lane + right_lane) / 2.0
            
        elif len(lane_centers) == 1:
            # Can see only ONE lane - offset based on lane width estimate
            lane_pos = lane_centers[0]
            img_center = w / 2.0
            
            # Estimate lane width (typical road lane spacing)
            estimated_lane_width = 200  # pixels (tune this if needed)
            
            if lane_pos < img_center:
                # Following LEFT lane - offset RIGHT by half lane width
                center_x = lane_pos + 3.0*(estimated_lane_width / 2.0)
            else:
                # Following RIGHT lane - offset LEFT by half lane width
                center_x = lane_pos - 3.0*(estimated_lane_width / 2.0)
            
        else:
            # Lost both lanes - use memory
            self.grass_frames_lost += 1
            
            if self.grass_frames_lost <= self.max_grass_lost_frames and self.last_grass_center is not None:
                center_x = self.last_grass_center
            else:
                center_x = None
            
            cv.imshow("white_line_mask", filtered_mask)
            cv.waitKey(1)
            self._show_debug(img, y_top, y_bottom, center_x)
            return center_x, w
        
        # Update memory
        self.last_grass_center = center_x
        self.grass_frames_lost = 0
        
        cv.imshow("white_line_mask", filtered_mask)
        cv.waitKey(1)
        self._show_debug(img, y_top, y_bottom, center_x)
        
        return center_x, w
    
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
    

    def calculate_movement(self, center_x, target_x, img_width, speed=None):
        """Calculate Twist command using PD control"""
        twist = Twist()

        v_cmd = speed if speed is not None else self.speed
        
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
            
            twist.linear.x = v_cmd
            twist.angular.z = -turn_speed  # Negative because error is backwards
            
            self.prev_error = error
            self.prev_time = now
        
        # Case 2: Lost the line - search for it
        else:
            if self.last_center is not None:
                # Spin toward where the line should be
                error = self.last_center - target_x
                direction = -1.0 if error > 0 else 1.0
                
                twist.linear.x = v_cmd * 0.5
                twist.angular.z = direction * self.max_turn_speed * 0.5
            else:
                # No idea where line is - just spin
                twist.linear.x = 0.0
                twist.angular.z = 1.0
        
        return twist