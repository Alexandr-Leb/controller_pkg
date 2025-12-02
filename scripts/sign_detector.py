#!/usr/bin/env python3

import cv2 as cv
import numpy as np


class SignDetector:
    """Detects blue-backed signs and warps them to rectangular images"""
    
    def __init__(self):
        self.min_area_ratio = 0.005  # Minimum sign size relative to image
        self.output_size = (600, 400)  # Width x Height of homographed sign
    
    def find_sign(self, img, use_light_blue=False):
        """Find the largest blue sign in the image
        
        use_light_blue: If True, look for purple/light blue 
                       If False, look for dark saturated blue 
        
        Returns dict with sign info, or None if no sign found
        """
        h, w = img.shape[:2]
        
        # Define search area and minimum size
        if use_light_blue:
            y_start = int(0.35 * h)
            min_area = 0.0003 * h * w
        else:
            y_start = int(0.25 * h)
            min_area = 0.003 * h * w
        
        roi = img[y_start:h, :]
        
        # Convert to HSV
        hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        
        # Choose color range
        if use_light_blue:
            # Purple/blue-violet range (above sky blue hue ~110)
            lower = np.array([120, 80, 120])
            upper = np.array([155, 255, 255])
        else:
            # Dark saturated blue
            lower = np.array([100, 120, 80])
            upper = np.array([130, 255, 255])
        
        mask = cv.inRange(hsv, lower, upper)
        
        # Clean mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        
        # Find blobs
        contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Filter candidates
        candidates = []
        
        for c in contours:
            area = cv.contourArea(c)
            
            if area < min_area:
                continue
            
            # Get position
            x, y, box_w, box_h = cv.boundingRect(c)
            center_x = x + box_w/2
            center_y = y_start + (y + box_h/2)
            
            # For loop sign it must be in lower-left of image idk
            # IF OTHER LIGHT BLUE SIGNS BREAK LATER LOOK AT CHANGING THIS FIRST
            if use_light_blue:
                if center_y < 0.5 * h:  # Not in bottom half
                    continue
                if center_x > 0.6 * w:  # Not on left side
                    continue
            
            candidates.append((c, area))
        
        if not candidates:
            return None
        
        # Get largest candidate
        largest_cnt, area = max(candidates, key=lambda x: x[1])
        
        # Get 4 corners
        perimeter = cv.arcLength(largest_cnt, True)
        approx = cv.approxPolyDP(largest_cnt, 0.02 * perimeter, True)
        
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
        else:
            # Fallback: use minimum area rectangle
            rect = cv.minAreaRect(largest_cnt)
            quad = cv.boxPoints(rect).astype(np.float32)
        
        # Shift coordinates from ROI to full image
        quad[:, 1] += y_start
        
        # Calculate bounding box
        x_coords = quad[:, 0]
        y_coords = quad[:, 1]
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        center_x = x_coords.mean()
        center_y = y_coords.mean()
        
        # Filter weird shapes (too skinny/tall)
        aspect = max(bbox_w, bbox_h) / max(min(bbox_w, bbox_h), 1.0)
        if aspect > 6.0:
            return None
        
        return {
            "quad": quad,
            "bbox": (x_min, x_max, y_min, y_max),
            "center_x": float(center_x),
            "center_y": float(center_y),
            "bbox_width": float(bbox_w),
            "bbox_height": float(bbox_h),
            "frame_width": w,
            "frame_height": h,
            "area": float(area)
        }
    
    def warp_sign(self, img, quad):
        """Warp the sign to a rectangular image"""
        # Order corners: top-left, top-right, bottom-right, bottom-left
        quad = self._order_corners(quad)
        
        # Destination corners
        out_w, out_h = self.output_size
        dst = np.array([
            [0, 0],
            [out_w-1, 0],
            [out_w-1, out_h-1],
            [0, out_h-1]
        ], dtype=np.float32)
        
        # Calculate homography
        H, _ = cv.findHomography(quad, dst)
        
        if H is None:
            return None
        
        # Warp image
        warped = cv.warpPerspective(img, H, (out_w, out_h))
        
        return warped
    
    def _order_corners(self, pts):
        """Order 4 points as: top-left, top-right, bottom-right, bottom-left"""
        pts = np.array(pts, dtype=np.float32)
        
        # Sum and difference to find corners
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).flatten()
        
        top_left = pts[np.argmin(s)]
        bottom_right = pts[np.argmax(s)]
        top_right = pts[np.argmin(diff)]
        bottom_left = pts[np.argmax(diff)]
        
        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)