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
        self.max_grass_lost_frames = 15

        # Memory for loop left-line following
        self.last_left_line_x = None
        self.left_line_frames_lost = 0
        self.max_left_line_lost_frames = 10
    
    # Main entry to driving
    def get_command(self, img, center_bias=0.0, speed=None):
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
    
    def get_command_loop(self, img, speed=None):
        """
        Special command for IN_LOOP state:
        Detect the LEFT white line and drive with robot center ON that line.
        This keeps the robot on the inside edge of the clockwise loop.
        """
        h, w = img.shape[:2]
        
        # Find the left white line position using friend's contour method
        left_line_x = self._detect_left_white_line_contour(img)
        
        use_speed = speed if speed is not None else self.speed
        
        # Update memory
        if left_line_x is not None:
            self.last_left_line_x = left_line_x
            self.left_line_frames_lost = 0
        else:
            self.left_line_frames_lost += 1
        
        # Use memory if we recently saw the line
        effective_left_x = left_line_x
        if left_line_x is None and self.left_line_frames_lost <= self.max_left_line_lost_frames:
            effective_left_x = self.last_left_line_x
            if effective_left_x is not None:
                rospy.loginfo_throttle(0.3, f"[LOOP] Using memory: left_x={effective_left_x:.0f}, frames_lost={self.left_line_frames_lost}")
        
        twist = Twist()
        
        if effective_left_x is not None:
            # Target: robot center should be ON the left line
            # So we want left_line_x to be at image center
            target_x = w / 2.0
            error = effective_left_x - target_x
            
            # PD control
            now = rospy.Time.now().to_sec()
            
            if self.prev_time is None or now <= self.prev_time:
                d_error = 0.0
            else:
                dt = now - self.prev_time
                if dt > 0.0001:
                    d_error = (error - self.prev_error) / dt
                else:
                    d_error = 0.0
            
            turn_speed = self.kp * error + self.kd * d_error
            turn_speed = np.clip(turn_speed, -self.max_turn_speed, self.max_turn_speed)
            
            twist.linear.x = use_speed
            twist.angular.z = -turn_speed
            
            self.prev_error = error
            self.prev_time = now
            
            rospy.loginfo_throttle(0.5, f"[LOOP] Left line at x={effective_left_x:.0f}, target={target_x:.0f}, error={error:.1f}, turn={-turn_speed:.2f}")
        else:
            # Lost the line - turn left to find it (since we're following left line)
            rospy.loginfo_throttle(0.3, f"[LOOP] Lost left line for {self.left_line_frames_lost} frames - searching LEFT")
            twist.linear.x = use_speed * 0.3
            twist.angular.z = 1.2  # Turn left more aggressively to find the left line
        
        return twist
    
    def _detect_left_white_line_contour(self, img):
        """
        Detect left white line using contour method (based on friend's code).
        Finds the leftmost bright contour and returns its center x-coordinate.
        """
        h, w = img.shape[:2]
        
        # Use same ROI as asphalt detection
        y_bottom = h - self.skip_bottom
        y_top = max(0, y_bottom - self.roi_height)
        
        if y_top >= y_bottom:
            return None
        
        roi = img[y_top:y_bottom, :]
        
        # Convert to HSV and make white mask (low saturation, high value)
        hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        _, s_channel, v_channel = cv.split(hsv)
        
        low_sat_mask = cv.inRange(s_channel, 0, 60)
        high_val_mask = cv.inRange(v_channel, 180, 255)
        mask = cv.bitwise_and(low_sat_mask, high_val_mask)
        
        # Clean mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        
        # Find contours and choose leftmost qualifying contour
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            self._show_loop_debug_simple(img, y_top, y_bottom, mask, None, "No contours found")
            return None
        
        min_contour_area = 300  # Lowered from 500 to catch smaller lines
        leftmost_contour = None
        leftmost_x = float('inf')
        
        # Find the leftmost contour with sufficient area
        for c in contours:
            area = cv.contourArea(c)
            if area < min_contour_area:
                continue
            
            x, y, bw, bh = cv.boundingRect(c)
            if x < leftmost_x:
                leftmost_x = x
                leftmost_contour = (x, y, bw, bh)
        
        if leftmost_contour is None:
            self._show_loop_debug_simple(img, y_top, y_bottom, mask, None, 
                                         f"No contours > {min_contour_area} area")
            return None
        
        x, y, bw, bh = leftmost_contour
        center_x_roi = float(x + bw / 2.0)
        
        debug_info = f"Leftmost contour at x={x}, center={center_x_roi:.0f}, w={bw}, h={bh}"
        #self._show_loop_debug_simple(img, y_top, y_bottom, mask, center_x_roi, debug_info)
        
        return center_x_roi
    
    def _detect_left_white_line_improved(self, img):
        """
        Column-wise brightness analysis to find white lane lines.
        This is the most robust approach for finding thin vertical white lines.
        """
        h, w = img.shape[:2]
        
        skip_bottom = self.skip_bottom
        roi_height = self.roi_height
        
        y_bottom = h - skip_bottom
        y_top = max(0, y_bottom - roi_height)
        
        if y_top >= y_bottom:
            return None
        
        roi = img[y_top:y_bottom, :]
        roi_h, roi_w = roi.shape[:2]
        
        # Convert to grayscale
        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        
        # Apply blur
        gray_blur = cv.GaussianBlur(gray, (5, 5), 0)
        
        # Create a simple brightness threshold mask
        _, bright_mask = cv.threshold(gray_blur, 160, 255, cv.THRESH_BINARY)
        
        # Calculate the brightness profile (sum of white pixels in each column)
        column_sums = np.sum(bright_mask, axis=0) / 255.0  # Normalize
        
        # Smooth the column profile using simple moving average
        kernel_size = 7
        kernel = np.ones(kernel_size) / kernel_size
        column_sums_smooth = np.convolve(column_sums, kernel, mode='same')
        
        # Find peaks (local maxima) in the brightness profile
        # These correspond to white lines
        peaks = []
        threshold = np.mean(column_sums_smooth) + 0.5 * np.std(column_sums_smooth)
        
        for i in range(5, len(column_sums_smooth) - 5):
            if column_sums_smooth[i] > threshold:
                # Check if it's a local maximum
                if column_sums_smooth[i] >= column_sums_smooth[i-1] and \
                   column_sums_smooth[i] >= column_sums_smooth[i+1]:
                    peaks.append((i, column_sums_smooth[i]))
        
        if not peaks:
            self._show_loop_debug_simple(img, y_top, y_bottom, bright_mask, None, 
                                         f"No peaks found, max_brightness={np.max(column_sums_smooth):.1f}")
            return None
        
        # Filter peaks: only consider those on the LEFT half
        img_center_x = w / 2.0
        left_peaks = [(x, brightness) for x, brightness in peaks if x < img_center_x]
        
        if not left_peaks:
            self._show_loop_debug_simple(img, y_top, y_bottom, bright_mask, None, 
                                         f"No left peaks, total_peaks={len(peaks)}")
            return None
        
        # Pick the LEFTMOST bright peak (closest to left edge)
        left_peaks.sort(key=lambda x: x[0])
        left_line_x = left_peaks[0][0]
        
        debug_info = f"Found {len(left_peaks)} left peaks at x={[int(p[0]) for p in left_peaks[:3]]}, picked x={left_line_x:.0f}"
        #self._show_loop_debug_simple(img, y_top, y_bottom, bright_mask, left_line_x, debug_info)
        
        return left_line_x
    
    def _detect_left_white_line_backup(self, img):
        """
        ULTRA-AGGRESSIVE BACKUP: Uses Canny edge detection + Hough Lines
        to find vertical white lines. This is the nuclear option.
        """
        h, w = img.shape[:2]
        
        y_bottom = h - self.skip_bottom
        y_top = max(0, y_bottom - self.roi_height)
        
        if y_top >= y_bottom:
            return None
        
        roi = img[y_top:y_bottom, :]
        roi_h, roi_w = roi.shape[:2]
        
        # Convert to grayscale
        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        
        # Strong blur
        gray_blur = cv.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold for VERY bright pixels only
        _, bright_mask = cv.threshold(gray_blur, 140, 255, cv.THRESH_BINARY)
        
        # Apply Canny edge detection on bright areas
        edges = cv.Canny(bright_mask, 50, 150)
        
        # Use Hough Lines to find vertical lines
        lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=30, maxLineGap=10)
        
        if lines is None:
            return None
        
        # Filter for VERTICAL lines (small angle)
        left_lines = []
        img_center_x = w / 2.0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle
            if x2 - x1 == 0:
                angle = 90
            else:
                angle = abs(np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi)
            
            # Must be nearly vertical (angle > 60 degrees)
            if angle < 60:
                continue
            
            # Get x position (average of endpoints)
            avg_x = (x1 + x2) / 2.0
            
            # Only left half
            if avg_x < img_center_x:
                line_length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                left_lines.append((avg_x, line_length, line))
        
        if not left_lines:
            return None
        
        # Pick the LONGEST vertical line on the left
        left_lines.sort(key=lambda x: x[1], reverse=True)
        return left_lines[0][0]
    
    def _show_loop_debug_simple(self, img, y_top, y_bottom, white_mask, left_line_x, debug_info=""):
        """Simplified debug visualization for loop line following"""
        h, w = img.shape[:2]
        vis = img.copy()
        
        # Draw ROI box
        cv.rectangle(vis, (0, y_top), (w-1, y_bottom-1), (255, 0, 255), 2)
        
        y_mid = (y_top + y_bottom) // 2
        
        # Draw left line position (GREEN - this is what we follow)
        if left_line_x is not None:
            cv.circle(vis, (int(left_line_x), y_mid), 10, (0, 255, 0), -1)
            cv.putText(vis, "LEFT LINE", (int(left_line_x) - 40, y_mid - 20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw image center line (where robot center is)
        cv.line(vis, (w//2, y_top), (w//2, y_bottom), (0, 0, 255), 2)
        cv.putText(vis, "ROBOT CENTER", (w//2 - 60, y_top - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw debug info
        cv.putText(vis, debug_info, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv.imshow("loop_debug", vis)
        cv.imshow("loop_white_mask", white_mask)
        cv.waitKey(1)

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
        max_contour_area = 20000
        min_height = 0.35 * roi_height
        max_lane_width_frac = 0.45

        left_candidates = []
        right_candidates = []

        for cnt in contours:
            area = cv.contourArea(cnt)
            if area < min_contour_area or area > max_contour_area:
                continue

            x, y, cw, ch = cv.boundingRect(cnt)

            if cw > max_lane_width_frac * w:
                continue

            if ch < min_height:
                continue

            cx = x + cw / 2.0

            if cx < w / 2.0:
                left_candidates.append(cnt)
            else:
                right_candidates.append(cnt)

        line_contours = []

        if left_candidates:
            left_candidates = sorted(left_candidates, key=cv.contourArea, reverse=True)
            line_contours.append(left_candidates[0])

        if right_candidates:
            right_candidates = sorted(right_candidates, key=cv.contourArea, reverse=True)
            line_contours.append(right_candidates[0])

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
            lane_centers.sort()
            left_lane = lane_centers[0]
            right_lane = lane_centers[-1]
            center_x = (left_lane + right_lane) / 2.0
            
        elif len(lane_centers) == 1:
            lane_pos = lane_centers[0]
            img_center = w / 2.0
            estimated_lane_width = 200
            
            if lane_pos < img_center:
                center_x = lane_pos + 3.0*(estimated_lane_width / 2.0)
            else:
                center_x = lane_pos - 3.0*(estimated_lane_width / 2.0)
            
        else:
            self.grass_frames_lost += 1
            
            if self.grass_frames_lost <= self.max_grass_lost_frames and self.last_grass_center is not None:
                center_x = self.last_grass_center
            else:
                center_x = None
            
            #cv.imshow("white_line_mask", filtered_mask)
            cv.waitKey(1)
            #self._show_debug(img, y_top, y_bottom, center_x)
            return center_x, w
        
        self.last_grass_center = center_x
        self.grass_frames_lost = 0
        
        #cv.imshow("white_line_mask", filtered_mask)
        cv.waitKey(1)
        #self._show_debug(img, y_top, y_bottom, center_x)
        
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