#!/usr/bin/env python3
import cv2 as cv
import numpy as np
import rospy
from geometry_msgs.msg import Twist

class TunnelNavigator:
    """Tunnel navigator - handles tunnel traversal and mountain climb"""
    
    def __init__(self):
        # Navigation phases
        # 0=search, 1=approach, 2=drive_past, 3=rotate_to_sign, 4=rotate_right_after_sign, 
        # 5=traverse_tunnel, 6=mountain_climb, 7=complete
        self.phase = 0
        self.frames_in_phase = 0
        
        # Tunnel detection parameters (for initial approach)
        self.tunnel_lower = np.array([4, 105, 146])
        self.tunnel_upper = np.array([16, 181, 200])
        
        # Green path detection for INSIDE tunnel (from reference code)
        self.green_path_lower = np.array([58, 120, 100])
        self.green_path_upper = np.array([110, 202, 160])
        
        # Brown brick detection (exit condition)
        self.brick_lower = np.array([8, 34, 165])
        self.brick_upper = np.array([172, 187, 239])
        self.brick_threshold = 1000  # Exit tunnel when bricks drop below this
        
        # Mountain white line detection (from reference code)
        # These will be adjusted dynamically based on brightness
        self.mountain_white_upper = np.array([190, 255, 255])
        
        # Pink stripe detection (for reference, not main navigation)
        self.pink_lower = np.array([51, 107, 187])
        self.pink_upper = np.array([249, 255, 255])
        
        # Search pattern
        self.search_direction = 1
        
        # Detection stability
        self.last_tunnel_info = None
        self.frames_without_tunnel = 0
        self.max_frames_without_tunnel = 20
        self.reached_hilltop = False
        
        # PD control for tunnel centering
        self.prev_tunnel_error = 0.0
        self.prev_tunnel_time = None
        self.tunnel_kp = 8.0
        self.tunnel_kd = 0.0
        
        # Completion flag
        self.complete = False
        
        # Post-sign rotation counter
        self.post_sign_rotation_frames = 0
        self.post_sign_rotation_duration = 30
        
        # Mountain climb - track last known left line position
        self.last_left = 0
        
        # Tunnel traversal timing (backup exit condition)
        self.tunnel_start_time = None
        self.tunnel_timeout = 10.0  # seconds max not in tunnel
    
    def navigate(self, img, sign_captured=False):
        """Main navigation function - returns (cmd, complete)
        
        Args:
            img: Current camera image
            sign_captured: True when main controller has captured the sign (sign 6 for mountain)
        """
        cmd = Twist()
        self.frames_in_phase += 1
        
        # Detect tunnel in current frame (for phases 0-1)
        tunnel_info = self.detect_tunnel(img)
        
        # UPDATE DETECTION MEMORY
        if tunnel_info is not None:
            self.last_tunnel_info = tunnel_info
            self.frames_without_tunnel = 0
        else:
            self.frames_without_tunnel += 1
            if self.frames_without_tunnel <= self.max_frames_without_tunnel and self.last_tunnel_info is not None:
                tunnel_info = self.last_tunnel_info
        
        # ===== PHASE 0: SEARCH FOR TUNNEL =====
        if self.phase == 0:
            cmd = self._phase_search(img, tunnel_info)
        
        # ===== PHASE 1: APPROACH TUNNEL AND CLIMB HILL =====
        elif self.phase == 1:
            cmd = self._phase_approach(img, tunnel_info)
        
        # ===== PHASE 2: DRIVE PAST TUNNEL =====
        elif self.phase == 2:
            cmd = self._phase_drive_past(img)
        
        # ===== PHASE 3: ROTATE LEFT TO FIND SIGN =====
        elif self.phase == 3:
            cmd = self._phase_rotate_to_sign(img, sign_captured)
        
        # ===== PHASE 4: ROTATE RIGHT AFTER SIGN CAPTURE =====
        elif self.phase == 4:
            cmd = self._phase_rotate_right_after_sign(img)
        
        # ===== PHASE 5: TRAVERSE TUNNEL (GREEN PATH FOLLOWING) =====
        elif self.phase == 5:
            cmd = self._phase_traverse_tunnel(img)
        
        # ===== PHASE 6: MOUNTAIN CLIMB (LEFT WHITE LINE FOLLOWING) =====
        elif self.phase == 6:
            cmd = self._phase_mountain_climb(img, sign_captured)
        
        # ===== PHASE 7: COMPLETE =====
        elif self.phase == 7:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.complete = True
        
        return cmd, self.complete
    
    # ========== PHASE IMPLEMENTATIONS ==========
    
    def _phase_search(self, img, tunnel_info):
        """Phase 0: Search for tunnel and center it"""
        cmd = Twist()
        
        if tunnel_info is None:
            rospy.loginfo_throttle(1.0, "[TUNNEL] Searching for tunnel...")
            cmd.linear.x = 0.0
            cmd.angular.z = 1.2 * self.search_direction
            return cmd
        
        # Tunnel visible - center it
        center_x_norm = tunnel_info['center_x']
        target_x_norm = 0.5
        error = center_x_norm - target_x_norm
        
        now = rospy.Time.now().to_sec()
        if self.prev_tunnel_time is None or now <= self.prev_tunnel_time:
            d_error = 0.0
        else:
            dt = now - self.prev_tunnel_time
            if dt > 0.0001:
                d_error = (error - self.prev_tunnel_error) / dt
            else:
                d_error = 0.0
        
        turn_speed = self.tunnel_kp * error + self.tunnel_kd * d_error
        turn_speed = np.clip(turn_speed, -2.0, 2.0)
        
        self.prev_tunnel_error = error
        self.prev_tunnel_time = now
        
        rospy.loginfo_throttle(0.5, f"[TUNNEL] Centering - x={center_x_norm:.2f}, error={error:.3f}")
        
        if abs(error) < 0.35:
            rospy.loginfo("[TUNNEL] Phase 0 → Phase 1: Tunnel centered, approaching")
            self.phase = 1
            self.frames_in_phase = 0
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            return cmd
        
        cmd.linear.x = 0.0
        cmd.angular.z = -turn_speed
        return cmd
    
    def _phase_approach(self, img, tunnel_info):
        """Phase 1: Drive toward tunnel and up the hill"""
        cmd = Twist()
        
        if tunnel_info is None:
            if not self.reached_hilltop:
                rospy.loginfo_throttle(1.0, f"[TUNNEL] Lost visual while climbing (frame {self.frames_in_phase}), continuing straight...")
                cmd.linear.x = 1.0
                cmd.angular.z = 0.0
                return cmd
            else:
                # Reached hilltop, now drive past it
                rospy.loginfo("[TUNNEL] Phase 1 → Phase 2: At hilltop, driving past tunnel")
                self.phase = 2
                self.frames_in_phase = 0
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                return cmd
        
        center_x_norm = tunnel_info['center_x']
        area = tunnel_info['area']
        
        # Check if we've reached the hilltop (tunnel area suddenly increases)
        if area > 3500 and not self.reached_hilltop:
            rospy.loginfo("="*60)
            rospy.loginfo(f"[TUNNEL] REACHED HILLTOP! Area={area:.0f}")
            rospy.loginfo("="*60)
            self.reached_hilltop = True
            
            # Immediately transition to drive past phase
            self.phase = 2
            self.frames_in_phase = 0
            return cmd
        
        # PD control to keep tunnel centered while approaching
        target_x_norm = 0.5  # Keep tunnel in CENTER
        error = center_x_norm - target_x_norm
        
        now = rospy.Time.now().to_sec()
        if self.prev_tunnel_time is None or now <= self.prev_tunnel_time:
            d_error = 0.0
        else:
            dt = now - self.prev_tunnel_time
            if dt > 0.0001:
                d_error = (error - self.prev_tunnel_error) / dt
            else:
                d_error = 0.0
        
        # PD control - same gains as before
        turn_speed = self.tunnel_kp * error + self.tunnel_kd * d_error
        turn_speed = np.clip(turn_speed, -2.5, 2.5)
        
        self.prev_tunnel_error = error
        self.prev_tunnel_time = now
        
        # DEBUG INFO
        rospy.loginfo_throttle(0.5, f"[PHASE 1] frame={self.frames_in_phase}, area={area:.0f}, x={center_x_norm:.2f}, "
                             f"error={error:.3f}, turn={turn_speed:.2f}")
        
        cmd.linear.x = 1.0
        cmd.angular.z = -turn_speed
        return cmd
    
    def _phase_drive_past(self, img):
        """Phase 2: Drive straight forward DOWN the hill and well past the tunnel"""
        cmd = Twist()
        
        # Gradually increasing right turn as we go down the hill
        # Start at -0.25, end at -0.40 over 200 frames
        max_frames = 207
        min_turn = -0.29      # Starting angular.z
        max_turn = -0.42      # Final angular.z (right turn)
        
        # Linear interpolation from min_turn to max_turn
        progress = min(self.frames_in_phase / max_frames, 1.0)
        current_turn = min_turn + (max_turn - min_turn) * progress
        
        # DEBUG INFO
        rospy.loginfo_throttle(0.5, f"[PHASE 2] Driving down hill - frame {self.frames_in_phase}/{max_frames}, "
                             f"angular.z={current_turn:.2f}")
        
        cmd.linear.x = 1.2
        cmd.angular.z = current_turn
        
        if self.frames_in_phase > max_frames:
            rospy.loginfo("="*60)
            rospy.loginfo("[PHASE 2→3] Driven well past, now rotating left to find sign")
            rospy.loginfo("="*60)
            self.phase = 3
            self.frames_in_phase = 0
            
            # Reset tunnel detection memory
            self.last_tunnel_info = None
            self.frames_without_tunnel = 0
        
        return cmd
    
    def _phase_rotate_to_sign(self, img, sign_captured):
        """Phase 3: Rotate left until sign is visible and can be captured by main controller"""
        cmd = Twist()
        
        # Check if main controller has captured the sign
        if sign_captured:
            rospy.loginfo("="*60)
            rospy.loginfo("[PHASE 3→4] Sign captured! Starting right rotation")
            rospy.loginfo("="*60)
            self.phase = 4
            self.frames_in_phase = 0
            self.post_sign_rotation_frames = 0
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            return cmd
        
        # Keep rotating left to find sign
        rospy.loginfo_throttle(0.5, f"[PHASE 3] Rotating left to find sign - frame {self.frames_in_phase}")
        cmd.linear.x = 0.0
        cmd.angular.z = 1.2  # Rotate left
        
        # Safety timeout
        if self.frames_in_phase > 200:
            rospy.logwarn("[PHASE 3] Rotation timeout, proceeding to traverse tunnel")
            self.phase = 5
            self.frames_in_phase = 0
            self.tunnel_start_time = rospy.get_time()
        
        return cmd
    
    def _phase_rotate_right_after_sign(self, img):
        """Phase 4: Rotate right for 30 frames after sign capture"""
        cmd = Twist()
        
        rospy.loginfo_throttle(0.5, f"[PHASE 4] Rotating right - frame {self.post_sign_rotation_frames}/{self.post_sign_rotation_duration}")
        
        cmd.linear.x = 0.0
        cmd.angular.z = -1.3  # Rotate right
        
        self.post_sign_rotation_frames += 1
        
        if self.post_sign_rotation_frames >= self.post_sign_rotation_duration:
            rospy.loginfo("="*60)
            rospy.loginfo("[PHASE 4→5] Right rotation complete, entering tunnel")
            rospy.loginfo("="*60)
            self.phase = 5
            self.frames_in_phase = 0
            self.tunnel_start_time = rospy.get_time()
        
        return cmd
    
    def _phase_traverse_tunnel(self, img):
        """Phase 5: Follow tunnel by detecting and centering between walls"""
        cmd = Twist()
        
        h, w = img.shape[:2]

        top_roi = img[:int(0.4 * h), :]
        avg_brightness = np.mean(cv.cvtColor(top_roi, cv.COLOR_BGR2GRAY))
    
        rospy.loginfo_throttle(1.0, f"[PHASE 5] avg_brightness={avg_brightness:.1f}") 
        
        # Check exit condition
        if avg_brightness > 100 and self.tunnel_start_time and (rospy.get_time() - self.tunnel_start_time) > self.tunnel_timeout:
            rospy.loginfo("="*60)
            rospy.loginfo(f"[PHASE 5→6] EXITED TUNNEL! brightness={avg_brightness:.1f}")
            rospy.loginfo("="*60)
            self.phase = 6
            self.frames_in_phase = 0
            self.last_left = 0
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            return cmd
    
        # Focus on bottom half where walls are clearer
        roi_top = int(0.5 * h)
        roi = img[roi_top:, :]
        roi_h, roi_w = roi.shape[:2]
        
        # Convert to grayscale for edge detection
        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        
        # Use Canny edge detection to find wall boundaries
        edges = cv.Canny(gray, 50, 150)
        
        # Apply morphology to connect edges
        kernel = np.ones((5, 5), np.uint8)
        edges = cv.dilate(edges, kernel, iterations=1)
        
        cv.imshow("tunnel_edges", edges)
        cv.waitKey(1)
        
        # Split frame into left and right halves
        mid = roi_w // 2
        left_half = edges[:, :mid]
        right_half = edges[:, mid:]
        
        # Find the rightmost edge in left half (left wall)
        # and leftmost edge in right half (right wall)
        left_wall_x = None
        right_wall_x = None
        
        # Sample from bottom quarter of ROI (most reliable view)
        sample_row = int(0.75 * roi_h)
        
        # Find left wall (rightmost white pixel in left half)
        left_row = left_half[sample_row, :]
        left_edges = np.where(left_row > 0)[0]
        if len(left_edges) > 0:
            left_wall_x = left_edges[-1]  # Rightmost edge in left half
        
        # Find right wall (leftmost white pixel in right half)
        right_row = right_half[sample_row, :]
        right_edges = np.where(right_row > 0)[0]
        if len(right_edges) > 0:
            right_wall_x = mid + right_edges[0]  # Leftmost edge in right half
        
        # Calculate target: midpoint between walls
        if left_wall_x is not None and right_wall_x is not None:
            # Both walls detected - center between them
            wall_center = (left_wall_x + right_wall_x) / 2.0
            tunnel_width = right_wall_x - left_wall_x
            
            # Calculate error (where we should be vs center)
            target = wall_center
            current = roi_w / 2.0
            error = (target - current) / (roi_w / 2.0)
            
            # Stronger turn response
            turn_speed = 3.0 * error
            turn_speed = np.clip(turn_speed, -1.5, 1.5)
            
            cmd.linear.x = 0.8
            cmd.angular.z = turn_speed
            
            rospy.loginfo_throttle(0.3, 
                f"[PHASE 5] left={left_wall_x}, right={right_wall_x}, "
                f"width={tunnel_width:.0f}, target={target:.0f}, error={error:.3f}, turn={turn_speed:.2f}")
            
            # Debug visualization
            debug_img = roi.copy()
            if left_wall_x is not None:
                cv.circle(debug_img, (int(left_wall_x), sample_row), 8, (0, 255, 0), -1)
            if right_wall_x is not None:
                cv.circle(debug_img, (int(right_wall_x), sample_row), 8, (0, 0, 255), -1)
            cv.circle(debug_img, (int(wall_center), sample_row), 10, (255, 255, 0), -1)
            cv.line(debug_img, (int(current), 0), (int(current), roi_h), (255, 0, 0), 2)
            cv.putText(debug_img, f"error={error:.2f} turn={turn_speed:.2f}", (10, 30),
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv.imshow("tunnel_wall_detection", debug_img)
            cv.waitKey(1)
            
        elif left_wall_x is not None:
            # Only left wall detected - stay away from it
            rospy.loginfo_throttle(0.5, f"[PHASE 5] Only left wall at {left_wall_x}")
            cmd.linear.x = 0.6
            cmd.angular.z = -1.25  # Turn right away from left wall
            
        elif right_wall_x is not None:
            # Only right wall detected - stay away from it
            rospy.loginfo_throttle(0.5, f"[PHASE 5] Only right wall at {right_wall_x}")
            cmd.linear.x = 0.6
            cmd.angular.z = 1.25  # Turn left away from right wall
            
        else:
            # No walls detected - drive slowly straight
            rospy.logwarn_throttle(0.5, "[PHASE 5] No walls detected!")
            cmd.linear.x = 0.4
            cmd.angular.z = 0.0
        
        return cmd
    
    def _phase_mountain_climb(self, img, sign_captured):
        """Phase 6: Climb mountain following LEFT white line (original approach, slowed down)"""
        cmd = Twist()
        
        h, w = img.shape[:2]
        
        # Adaptive blue threshold (from the guide)
        # Sample bottom 20% of image for average blue channel
        bottom_roi = img[int(0.8 * h):, :]
        avg_B = np.mean(bottom_roi[:, :, 0])  # Blue channel average
        
        # Adaptive threshold for white lines
        threshold_shift = 26.5  # Original value
        blue_threshold = int(avg_B + threshold_shift)
        
        # Create mask: pixels where blue channel exceeds threshold
        blue_channel = img[:, :, 0]
        green_channel = img[:, :, 1]
        red_channel = img[:, :, 2]
        
        # White lines have high values in all channels
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[(blue_channel > blue_threshold) & 
             (green_channel > blue_threshold - 20) & 
             (red_channel > blue_threshold - 30)] = 255
        
        # Focus on bottom portion of image (road area)
        road_roi_top = int(0.5 * h)
        mask[:road_roi_top, :] = 0  # Ignore top half
        
        # Clean up mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv.erode(mask, kernel, iterations=2)
        mask = cv.dilate(mask, kernel, iterations=2)
        mask = self.remove_small_contours(mask, 500)
        
        # Debug visualization
        cv.imshow("mountain_adaptive_mask", mask)
        cv.waitKey(1)
        
        # Find left edge of white line at 15% from bottom (original approach)
        sample_row = int(0.85 * h)
        try:
            left_raw = np.where(mask[sample_row, :] == 255)[0][0]
        except:
            left_raw = 0
        
        left = left_raw  # Start with raw detection
        
        # Reject detections too close to the left edge (likely noise)
        if left < 50:
            left = 0
        
        # Avoid detecting right line as left line
        if left > 0.35 * w:
            left = 0
        
        # Reject sudden large jumps (likely false detection)
        # If we had a good detection recently, new detection shouldn't jump too far
        if left > 0 and self.last_left > 0:
            jump = abs(left - self.last_left)
            if jump > 100:  # More than 100 pixels jump is suspicious
                rospy.logwarn_throttle(0.5, f"[PHASE 6] Rejecting jump: {self.last_left} -> {left_raw} (jump={jump})")
                left = 0  # Reject this detection
        
        # Track consecutive frames without line detection
        if not hasattr(self, 'frames_without_line'):
            self.frames_without_line = 0
        if not hasattr(self, 'recovery_direction'):
            self.recovery_direction = 1  # 1 = left, -1 = right
        if not hasattr(self, 'recovery_rotation_frames'):
            self.recovery_rotation_frames = 0
        
        if left == 0:
            self.frames_without_line += 1
        else:
            self.frames_without_line = 0
            self.recovery_rotation_frames = 0  # Reset rotation counter when line found
            self.recovery_direction = 1  # Reset to default left rotation
            self.last_left = left
        
        # RECOVERY MODE: If we've lost the line for too long, stop and rotate to find it
        if self.frames_without_line > 15:
            self.recovery_rotation_frames += 1
            
            # Switch direction after ~90 degrees (~50 frames at wz=1.2)
            frames_per_direction = 50
            
            if self.recovery_rotation_frames > frames_per_direction:
                self.recovery_direction *= -1  # Switch direction
                self.recovery_rotation_frames = 0
                rospy.loginfo(f"[PHASE 6] Switching recovery direction to {'LEFT' if self.recovery_direction > 0 else 'RIGHT'}")
            
            direction_str = "left" if self.recovery_direction > 0 else "right"
            rospy.logwarn_throttle(0.5, f"[PHASE 6] LINE LOST! Rotating {direction_str} to recover... (lost for {self.frames_without_line} frames, rot_frames={self.recovery_rotation_frames})")
            
            cmd.linear.x = 0.0
            cmd.angular.z = 1.2 * self.recovery_direction  # Rotate in current direction
            
            # Debug visualization
            debug_img = img.copy()
            cv.putText(debug_img, f"RECOVERY MODE - Rotating {direction_str}", (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv.putText(debug_img, f"Lost for {self.frames_without_line} frames", (10, 60), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv.imshow("mountain_debug", debug_img)
            cv.waitKey(1)
            
            return cmd
        
        # Calculate target position (left line + offset to drive in lane center)
        loc = int(left + (0.42 * w))  # Increased offset to position car more to the right
        error = (loc - w // 2) / (w // 2)
        
        # SLOWED DOWN control (original was vx=1.5, wz=-20*error)
        vx = 0.6 * (1 - 0.6 * abs(error))  # Slower base speed
        vx = max(vx, 0.4)  # Minimum speed
        
        wz = -4.0 * error  # Much gentler than -20
        wz = np.clip(wz, -1.2, 1.2)  # Limit turn rate
        
        cmd.linear.x = vx
        cmd.angular.z = wz
        
        rospy.loginfo_throttle(0.3, f"[PHASE 6] frame={self.frames_in_phase}, left={left}, loc={loc}, "
                             f"error={error:.3f}, avg_B={avg_B:.1f}, thresh={blue_threshold}, "
                             f"vx={vx:.2f}, wz={wz:.2f}")
        
        # Debug visualization
        debug_img = img.copy()
        cv.circle(debug_img, (loc, int(0.8 * h)), 10, (0, 0, 255), -1)  # Target point (red)
        cv.circle(debug_img, (left, sample_row), 8, (0, 255, 0), -1)    # Detected left edge (green)
        cv.line(debug_img, (w // 2, 0), (w // 2, h), (255, 0, 0), 2)    # Center line (blue)
        cv.line(debug_img, (0, sample_row), (w, sample_row), (255, 255, 0), 1)  # Sample row
        cv.putText(debug_img, f"avg_B={avg_B:.0f} thresh={blue_threshold} left={left}", (10, 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv.imshow("mountain_debug", debug_img)
        cv.waitKey(1)
        
        return cmd
    
    # ========== DETECTION FUNCTIONS ==========
    
    def detect_tunnel(self, img):
        """Detect tunnel entrance using HSV"""
        h, w = img.shape[:2]
        roi_height = int(0.6 * h)
        roi = img[0:roi_height, :]
        
        hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, self.tunnel_lower, self.tunnel_upper)
        
        kernel = np.ones((9, 9), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        
        cv.imshow("tunnel_mask", mask)
        cv.waitKey(1)
        
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        largest = max(contours, key=cv.contourArea)
        area = cv.contourArea(largest)
        
        if area < 300:
            return None
        
        rect = cv.minAreaRect(largest)
        (center_x_roi, center_y_roi), (rect_w, rect_h), angle = rect
        
        if rect_w < rect_h:
            rect_w, rect_h = rect_h, rect_w
            angle = angle + 90
        
        while angle > 90:
            angle -= 180
        while angle < -90:
            angle += 180
        
        center_x_norm = center_x_roi / w
        center_y_norm = center_y_roi / roi_height
        
        return {
            'center_x': center_x_norm,
            'center_y': center_y_norm,
            'width': rect_w,
            'height': rect_h,
            'angle': angle,
            'area': area
        }
    
    def count_brown_bricks(self, img):
        """Count brown brick pixels in the image"""
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        brick_mask = cv.inRange(hsv, self.brick_lower, self.brick_upper)
        
        # Count white pixels (brick pixels)
        brick_count = cv.countNonZero(brick_mask)
        
        cv.imshow("brown_brick_mask", brick_mask)
        cv.waitKey(1)
        
        return brick_count
    
    def keep_k_contours(self, mask, k):
        """Keep only the k largest contours in the mask"""
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask
        
        areas = [cv.contourArea(c) for c in contours]
        sorted_indices = np.argsort(areas)[::-1]
        
        # Zero out all but top k contours
        for i in range(k, len(contours)):
            cv.drawContours(mask, contours, sorted_indices[i], 0, -1)
        
        return mask
    
    def remove_small_contours(self, mask, min_area):
        """Remove contours smaller than min_area"""
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            if w * h < min_area:
                cv.drawContours(mask, [contour], -1, 0, -1)
        return mask