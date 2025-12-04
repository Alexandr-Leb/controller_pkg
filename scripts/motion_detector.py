#!/usr/bin/env python3

import cv2 as cv
import numpy as np
import rospy


class MotionDetector:
    """
    Detects motion using frame differencing.
    Used to detect pedestrians at crosswalks.
    
    This version requires seeing SIGNIFICANT motion FIRST before declaring "clear".
    """
    
    def __init__(self, min_area=2000, still_frames_needed=15, motion_frames_needed=5):
        """
        Args:
            min_area: minimum changed pixels to count as motion (lowered to 2000)
            still_frames_needed: consecutive still frames before "clear" (15 frames ~0.5s)
            motion_frames_needed: consecutive motion frames to confirm pedestrian (5 frames)
        """
        self.reference = None
        self.min_area = min_area
        self.still_frames_needed = still_frames_needed
        self.motion_frames_needed = motion_frames_needed
        
        self.still_counter = 0
        self.motion_counter = 0
        
        # Track if we've CONFIRMED motion (pedestrian detected)
        self.has_seen_motion = False
        
        # Frame counter for debugging and timeout
        self.frame_count = 0
        
        # Maximum frames to wait (safety timeout) - about 10 seconds at 30fps
        self.max_wait_frames = 300
    
    def reset(self):
        """Reset the detector state"""
        rospy.loginfo("[MOTION] === RESET === Starting fresh pedestrian detection")
        self.reference = None
        self.still_counter = 0
        self.motion_counter = 0
        self.has_seen_motion = False
        self.frame_count = 0
    
    def check_motion(self, frame):
        """
        Check if there is motion in the frame.
        
        Returns:
            True  — keep waiting
            False — clear to go
        """
        self.frame_count += 1
        
        # SAFETY TIMEOUT: If we've waited too long, just go
        if self.frame_count > self.max_wait_frames:
            rospy.logwarn(f"[MOTION] *** TIMEOUT *** Waited {self.frame_count} frames, proceeding anyway!")
            return False
        
        # Convert to grayscale and blur
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (21, 21), 0)
        
        # First frame becomes reference
        if self.reference is None:
            self.reference = gray
            self.still_counter = 0
            self.motion_counter = 0
            rospy.loginfo("[MOTION] Frame 1: Captured reference frame")
            return True
        
        # Calculate frame difference
        diff = cv.absdiff(self.reference, gray)
        _, thresh = cv.threshold(diff, 25, 255, cv.THRESH_BINARY)
        
        # Morphology to clean up
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
        thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
        
        motion_pixels = int(cv.countNonZero(thresh))
        
        # Debug visualization
        #cv.imshow("motion_diff", thresh)
        cv.waitKey(1)
        
        has_motion_this_frame = motion_pixels > self.min_area
        
        # Log every 10th frame to reduce spam (but still show important events)
        if self.frame_count % 10 == 0 or has_motion_this_frame or self.has_seen_motion:
            rospy.loginfo(f"[MOTION] Frame {self.frame_count}: pixels={motion_pixels}, "
                          f"motion={has_motion_this_frame}, "
                          f"motion_cnt={self.motion_counter}/{self.motion_frames_needed}, "
                          f"still_cnt={self.still_counter}/{self.still_frames_needed}, "
                          f"CONFIRMED={self.has_seen_motion}")
        
        if has_motion_this_frame:
            # Motion detected
            self.motion_counter += 1
            self.still_counter = 0
            self.reference = gray  # Update reference
            
            # Check if enough consecutive motion frames to confirm pedestrian
            if self.motion_counter >= self.motion_frames_needed and not self.has_seen_motion:
                self.has_seen_motion = True
                rospy.loginfo("="*50)
                rospy.loginfo("[MOTION] *** PEDESTRIAN CONFIRMED ***")
                rospy.loginfo("="*50)
            
            return True  # Keep waiting
        
        else:
            # No motion this frame
            self.motion_counter = 0
            self.still_counter += 1
            
            # Update reference periodically
            if self.still_counter % 20 == 0:
                self.reference = gray
            
            # Can we declare clear?
            if self.has_seen_motion and self.still_counter >= self.still_frames_needed:
                rospy.loginfo("="*50)
                rospy.loginfo(f"[MOTION] *** CLEAR *** Pedestrian passed (still for {self.still_counter} frames)")
                rospy.loginfo("="*50)
                return False  # CLEAR TO GO!
            
            return True  # Keep waiting