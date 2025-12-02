# #!/usr/bin/env python3

# import cv2 as cv
# import numpy as np
# from geometry_msgs.msg import Twist
# import rospy


# # --- image-processing constants  ---
# SKIP = 300
# NUMBER_OF_ROWS_TO_ANALZYE = 120



# class LaneFollower:
#     """
#     A class containing all of the different lane detection algorithms well be using
#     """

#     def __init__(self, kp=15.0, kd=0.0, forward_speed=2.0, max_ang=45.0):
#         self.kp = kp 
#         self.kd = kd
#         self.forward_speed = forward_speed
#         self.max_ang = max_ang

#         self.last_midpoint = None # for drawing and smoothing
#         self.last_mid = None # for pid control
#         self.misses = 0
#         self.prev_e = 0.0
#         self.prev_t = None

#         self.one_contour_orientation = None  # will be +1 / -1 when we see one contour
#         self.one_contour_wh = None 

    
#     def find_road_midpoint_asphalt(self, cv_image):
#         H, W = cv_image.shape[:2]

#         # Cut the image to a region of interest
#         y0 = H - (SKIP + NUMBER_OF_ROWS_TO_ANALZYE)
#         y1 = H - SKIP
#         if y0 < 0:
#             y0 = 0
        
#         roi = cv_image[y0:y1, 0:W]

#         # Make a mask for white road sides
#         hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
#         h_chan, s_chan, v_chan = cv.split(hsv)

#         ## Isolate for low saturation and bright
#         sat_mask = cv.inRange(s_chan, 0, 60) # low saturation
#         val_mask = cv.inRange(v_chan, 180, 255) # bright

#         mask_road = cv.bitwise_and(sat_mask, val_mask)

#         # Add morphology to clean up the mask into solid lane blobs
#         kernel = np.ones((5, 5), np.uint8)
#         mask_white = cv.morphologyEx(mask_road, cv.MORPH_OPEN, kernel)
#         mask_white = cv.morphologyEx(mask_white, cv.MORPH_CLOSE, kernel)

#         # Find the contours after the mask
#         contours, hierarchy  = cv.findContours(mask_white, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

#         # Isloate for the 2 biggest contours
#         contours = sorted(contours, key=cv.contourArea, reverse=True)
#         largest_contours = contours[:2]

#         # Filter for line like contours
#         line_contours = []
#         line_contour_xbounds = []
#         line_contour_ybounds = []
#         for c in largest_contours:
#             x,y,w,h = cv.boundingRect(c)
#             contour_area = h*w
            
#             if contour_area < 200:
#                 continue

#             if w > 70.0 * h:
#                 continue

#             rect = cv.minAreaRect(c)
#             angle = abs(rect[2])

#             if angle < 25 or angle > 155:
#                 continue

#             line_contours.append(c)
#             line_contour_xbounds.append((x, x+w))
#             line_contour_ybounds.append((y, y+h))
                

#         # Reset lean info
#         self.one_contour_orientation = None
#         self.one_contour_wh = None


#         # Find midpoint between the lanes
#         midpointx, midpointy = None, None
#         if len(line_contour_xbounds) >= 2:
#             # Sort by x-center so we reliably get left and right lanes
#             lane_centers = []
#             for i, (x0, x1) in enumerate(line_contour_xbounds):
#                 xc = 0.5 * (x0 + x1)
#                 lane_centers.append((xc, i))

#             lane_centers.sort(key=lambda t: t[0])  # sort by x-center

#             left_idx = lane_centers[0][1]
#             right_idx = lane_centers[1][1]

#             x_left0, x_left1 = line_contour_xbounds[left_idx]
#             x_right0, x_right1 = line_contour_xbounds[right_idx]

#             # Horizontal midpoint between inner edges of the lanes
#             x_first_lane_right = x_left1
#             x_second_lane_left = x_right0
#             midpointx = 0.5 * (x_first_lane_right + x_second_lane_left)

#             # Vertical midpoint of, say, the left lane's bounding box
#             y0_left, y1_left = line_contour_ybounds[left_idx]
#             midpointy = 0.5 * (y0_left + y1_left)

#         # in the case it finds only one contour
#         elif len(line_contours) == 1:
#             c = line_contours[0]
#             x, y, w, h = cv.boundingRect(c)
#             pts = c[:, 0, :]               # shape (N, 2) -> [x, y]
#             mid_x = x + w / 2.0

#             left_pts = pts[pts[:, 0] < mid_x]
#             right_pts = pts[pts[:, 0] >= mid_x]

#             if len(left_pts) > 0 and len(right_pts) > 0:
#                 avg_y_left = np.mean(left_pts[:, 1])
#                 avg_y_right = np.mean(right_pts[:, 1])

#                 # lean > 0 means right half is lower/higher than left – sign may need flipping
#                 lean = avg_y_right - avg_y_left
#                 orientation = np.sign(lean)

#                 # store for controller
#                 self.one_contour_orientation = orientation
#                 self.one_contour_wh = (float(w), float(h))

        
#         # Smoothening out midpoint transition for two contours
#         if midpointx is not None and midpointy is not None:
#             if self.last_midpoint is None:
#                 self.last_midpoint = np.array([midpointx, midpointy], dtype=float)
#             else:
#                 alpha = 0.2 # for smooth transitions from midpoints
#                 meas = np.array([midpointx, midpointy], dtype=float)
#                 self.last_midpoint = (1-alpha) * self.last_midpoint + alpha * meas
            
#         # Extract only the x midpoint for twist calculations
#         if self.last_midpoint is not None:
#             midpoint_x = float(self.last_midpoint[0])  
#         else:
#             midpoint_x = None

#         # drawing each component
#         img_contours = cv_image.copy()
#         cv.rectangle(img_contours, (0, y0), (W - 1, y1 - 1), (0, 0, 255), 2)

#         shifted = []
#         for c in line_contours:
#             c2 = c.copy()
#             c2[:, 0, 1] += y0
#             shifted.append(c2)  
            
#         if self.last_midpoint is not None:
#             draw_x, draw_y = self.last_midpoint
            
#             center = (int(draw_x), int(draw_y + y0))
#             cv.circle(img_contours, center, 5, (0, 0, 255), -1)



#         cv.drawContours(img_contours, shifted, -1, (0, 255, 0), 2)
#         cv.imshow("lane_debug", img_contours)
#         cv.waitKey(1)

#         return midpoint_x, W




    
#     def compute_twist_from_midpoint(self, midpoint_x, image_width):
#         twist = Twist()

#         # some tunable constants
#         DEAD_BAND = 0.02          # for 2-contour case
#         TURN_SLOW = 0.7           # speed factor in gentle turns
#         TURN_SLOW_HARD = 0.3      # speed factor in 1-contour case
#         ONE_CONTOUR_GAIN = 0.8    # how hard to steer from w/h

#         # --- 2-contour case: we have a valid midpoint ---
#         if midpoint_x is not None:
#             self.misses = 0
#             self.last_mid = midpoint_x

#             x_ref = image_width / 2.0
#             error = (x_ref - midpoint_x) / x_ref  # normalized to ~[-1, 1]

#             if abs(error) < DEAD_BAND:
#                 # go straight
#                 twist.linear.x = self.forward_speed
#                 twist.angular.z = 0.0
#             else:
#                 # gentle P-turn
#                 twist.linear.x = self.forward_speed * TURN_SLOW
#                 ang = self.kp * error
#                 ang = max(-self.max_ang, min(self.max_ang, ang))
#                 twist.angular.z = ang

#             return twist

#         # --- 1-contour case: no midpoint, but we have lean info ---
#         if self.one_contour_orientation is not None and self.one_contour_wh is not None:
#             w, h = self.one_contour_wh
#             if h <= 0:
#                 # degenerate, fallback to stop
#                 twist.linear.x = 0.0
#                 twist.angular.z = 0.0
#                 return twist

#             aspect = w / h         # this is the w/h from the report
#             orientation = self.one_contour_orientation  # +1 or -1

#             # orientation * aspect is our "error"
#             error = orientation * aspect * ONE_CONTOUR_GAIN

#             twist.linear.x = self.forward_speed * TURN_SLOW_HARD
#             ang = self.kp * error
#             ang = max(-self.max_ang, min(self.max_ang, ang))
#             twist.angular.z = ang
#             return twist

#         # --- nothing usable: lost the road -> stop ---
#         twist.linear.x = 0.0
#         twist.angular.z = 0.0
#         return twist

        


    
#     def process_image_beginning_road(self, cv_image):
#         twist = Twist()
#         midpoint, W = self.find_road_midpoint_asphalt(cv_image)
#         twist = self.compute_twist_from_midpoint(midpoint, W)
#         return twist

