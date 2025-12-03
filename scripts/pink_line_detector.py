#!/usr/bin/env python3

import cv2 as cv
import numpy as np


class StripeDetector:
    """Detects red crosswalk stripes and pink markers"""
    
    def __init__(self):
        # ROI settings
        self.roi_height = 150
        self.skip_bottom = 120
        self.min_red_pixels = 1500
        self.min_pink_pixels = 1500
        
        # Tracking
        self.was_on_red = False
        self.was_on_pink = False
        self.stripe_count_red = 0
        self.stripe_count_pink = 0

    def crop_frame_to_roi(self, img):
        """Crops the image to a chosen region of interest"""
        h, w = img.shape[:2]
        
        y_bottom = h - self.skip_bottom
        y_top = max(0, y_bottom - self.roi_height)
        
        if y_top >= y_bottom:
            return None
        
        roi = img[y_top:y_bottom, :]
        
        return roi

    def check_red_stripe(self, img):
        """Check if we're over a red stripe
        
        Returns:
            on_red: True if currently over red
            crossed_second: True when we've crossed the second stripe
        """
        
        roi = self.crop_frame_to_roi(img)

        if roi is None:
            return False, False
        
        hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        
        # Red wraps around hue=0
        mask1 = cv.inRange(hsv, np.array([0, 120, 80]), np.array([10, 255, 255]))
        mask2 = cv.inRange(hsv, np.array([170, 120, 80]), np.array([180, 255, 255]))
        
        mask = cv.bitwise_or(mask1, mask2)
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        
        on_red = False
        if cv.countNonZero(mask) > self.min_red_pixels:
            on_red = True
        
        crossed_second = False
        
        if on_red and not self.was_on_red:
            if self.stripe_count_red == 0:
                self.stripe_count_red = 1
            elif self.stripe_count_red == 1:
                self.stripe_count_red = 2
                crossed_second = True
        
        self.was_on_red = on_red
        
        return on_red, crossed_second
    
    def check_pink_stripe(self, img):
        """Check if we're over a pink stripe"""
        roi = self.crop_frame_to_roi(img)

        if roi is None:
            return False, False
        
        hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        
        lower_pink = np.array([135, 80, 80])
        upper_pink = np.array([175, 255, 255])
        
        mask = cv.inRange(hsv, lower_pink, upper_pink)
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        
        on_pink = cv.countNonZero(mask) > self.min_pink_pixels
        
        crossed_pink = False
        
        # Fire pulse on rising edge (off->on)
        if on_pink and not self.was_on_pink:
            self.stripe_count_pink += 1
            crossed_pink = True
        
        self.was_on_pink = on_pink
        
        return on_pink, crossed_pink