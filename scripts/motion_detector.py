#!/usr/bin/env python3
import cv2 as cv
import numpy as np


class MotionDetector:
    def __init__(self, min_area=2500, still_frames_needed=8):
        """
        min_area: minimum number of changed pixels to count as motion
        still_frames_needed: how many consecutive still frames before concluding "clear"
        """
        self.reference = None
        self.min_area = min_area
        self.still_frames_needed = still_frames_needed
        self.still_counter = 0

    def reset(self):
        self.reference = None
        self.still_counter = 0

    def check_motion(self, frame):
        """
        Returns:
            True  — movement detected
            False — no movement
        """

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (7, 7), 0)

        # First captured frame becomes reference — treat as NO motion to avoid immediate spam
        if self.reference is None:
            self.reference = gray
            self.still_counter = 0
            return False  # do not report motion on the very first reference frame

        # Frame difference
        diff = cv.absdiff(self.reference, gray)
        _, thresh = cv.threshold(diff, 25, 255, cv.THRESH_BINARY)

        # Morphology to remove noise / small specks
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
        thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

        motion_pixels = int(cv.countNonZero(thresh))

        # If substantial motion, reset still counter and update reference
        if motion_pixels > self.min_area:
            self.still_counter = 0
            # update reference to current frame to avoid repeatedly detecting same motion blob
            self.reference = gray
            return True  # MOTION PRESENT

        # No substantial motion this frame
        self.still_counter += 1

        # If we've seen many still frames, consider the scene stable and update reference
        if self.still_counter >= self.still_frames_needed:
            self.reference = gray  # refresh reference to current clean frame

        # Only report "no motion" once we've had enough consecutive still frames
        return self.still_counter < self.still_frames_needed
