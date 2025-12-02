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
    
    def warp_gray_square(self, warped_img):
        """
        Detect and crop to the gray/white clueboard background
        by thresholding low-saturation pixels.
        """

        hsv = cv.cvtColor(warped_img, cv.COLOR_BGR2HSV)

        h, s, v = cv.split(hsv)

        # Gray box = low saturation but reasonably bright
        gray_mask = cv.inRange(s, 0, 60) & cv.inRange(v, 80, 255)

        # Remove tiny specks
        kernel = np.ones((5, 5), np.uint8)
        gray_mask = cv.morphologyEx(gray_mask, cv.MORPH_CLOSE, kernel)
        gray_mask = cv.morphologyEx(gray_mask, cv.MORPH_OPEN, kernel)

        # Find bounding box of the gray region
        ys, xs = np.where(gray_mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            print("⚠ Could not find gray region")
            return warped_img

        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        # Crop to gray board region
        cropped = warped_img[y1:y2, x1:x2]

        # FINAL SHRINK: trim a small fraction from each edge to avoid residual blue
        SHRINK_FRAC = 0.02  # 2% trim on each side (tweak as needed)
        Hc, Wc = cropped.shape[:2]
        cx = int(Wc * SHRINK_FRAC)
        cy = int(Hc * SHRINK_FRAC)

        # Defensive checks
        if cx * 2 >= Wc or cy * 2 >= Hc:
            return cropped

        try:
            inner = cropped[cy:Hc - cy, cx:Wc - cx]
            if inner.size == 0:
                return cropped
            # resize back to original cropped size so downstream code sees same dims
            final = cv.resize(inner, (Wc, Hc), interpolation=cv.INTER_LINEAR)
            return final
        except Exception:
            return cropped

    
    def extract_letters_only(self, gray_img):
        """
        Extract ONLY blue letters from the clueboard.
        No blur, no smoothing, no distortion.
        Returns a binary mask with white letters on black background.
        """

        # Convert to HSV (even if already warped grayscale)
        hsv = cv.cvtColor(gray_img, cv.COLOR_BGR2HSV)

        # Pure blue letter mask
        lower_blue = np.array([105, 120, 40])
        upper_blue = np.array([135, 255, 255])

        mask = cv.inRange(hsv, lower_blue, upper_blue)

       

        return mask
    
    def _resize_letter_for_cnn(self, crop, out_w, out_h):
        """Resizes a cropped letter to CNN size while keeping aspect ratio."""
        h, w = crop.shape

        # Scale with preserved aspect ratio
        scale = min(out_w / float(w), out_h / float(h))
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize letter
        resized = cv.resize(crop, (new_w, new_h), interpolation=cv.INTER_NEAREST)

        # Create padded output canvas
        canvas = np.zeros((out_h, out_w), dtype=np.uint8)
        x_offset = (out_w - new_w) // 2
        y_offset = (out_h - new_h) // 2

        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        return canvas

    
    def segment_words(self, letter_mask):
        H, W = letter_mask.shape

        # Split horizontally
        top_region = letter_mask[0:int(0.45 * H), :]
        bottom_region = letter_mask[int(0.45 * H):H, :]

        bottom_offset = int(0.45 * H)

        # Extract letters separately
        top_raw = self._segment_letters_region(top_region, y_offset=0)
        bottom_raw = self._segment_letters_region(bottom_region, y_offset=bottom_offset)

        # Sort left→right
        top_raw.sort(key=lambda t: t[0])
        bottom_raw.sort(key=lambda t: t[0])

        # Convert into word groups
        top_words = self.group_letters_into_words(top_raw)
        bottom_words = self.group_letters_into_words(bottom_raw)

        return top_words, bottom_words
    
    def _segment_letters_region(self, region_mask, y_offset):
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(region_mask)

        letters = []

        H_reg, W_reg = region_mask.shape[:2]
        # Fractional padding to keep black space above/below letters (tweakable)
        V_PAD_FRAC = 0.25   # pad 25% of letter height above and below
        H_PAD_FRAC = 0.08   # small horizontal pad to avoid tight crop

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]

            if area < 50:
                continue

            # compute padding in pixels (clamp to reasonable limits)
            pad_y = max(2, int(h * V_PAD_FRAC))
            pad_x = max(2, int(w * H_PAD_FRAC))

            y0 = max(0, y - pad_y)
            y1 = min(H_reg, y + h + pad_y)
            x0 = max(0, x - pad_x)
            x1 = min(W_reg, x + w + pad_x)

            crop = region_mask[y0:y1, x0:x1]

            # Resize for CNN (preserves black margins due to padding)
            resized = self._resize_letter_for_cnn(crop, 200, 150)

            # Store absolute X for word grouping (use x0 relative to region)
            abs_x = x0

            letters.append((abs_x, resized))

        return letters
    
    def group_letters_into_words(self, letter_components):
        if len(letter_components) == 0:
            return []

        xs = [c[0] for c in letter_components]
        gaps = []

        for i in range(len(xs)-1):
            gaps.append(xs[i+1] - xs[i])

        median_gap = np.median(gaps)
        space_threshold = median_gap * 1.8   # tuning constant

        words = []
        current = []

        for i, comp in enumerate(letter_components):
            current.append(comp[1])

            if i == len(gaps):
                break

            if gaps[i] > space_threshold:
                words.append(current)
                current = []

        if current:
            words.append(current)

        return words








