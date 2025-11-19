#!/usr/bin/env python3


import rospy
import cv2
import numpy as np
import rosnode
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

# --- image-processing constants (same as your original node) ---
SKIP = 250                    # rows to skip from the bottom
THRESH = 127                # threshold
ROWS_TO_AVG = 10            # neighbour rows to average for midpoint
NUMBER_OF_MISSED_FRAMES = 6 # hysteresis on "line lost"


class TimeTrialsLineFollower:
    def __init__(self):
        self.bridge = CvBridge()

        # Topics for B1
        img_topic = rospy.get_param("~image_topic",
                                    "/B1/rrbot/camera1/image_raw")
        cmd_topic = rospy.get_param("~cmd_vel_topic",
                                    "/B1/cmd_vel")

        self.image_sub = rospy.Subscriber( img_topic, Image, self.callback, queue_size=1)
        self.cmd_pub = rospy.Publisher(cmd_topic, Twist, queue_size=10)
        self.score_pub = rospy.Publisher("/score_tracker", String, queue_size=1)

        # Tunables (mirroring your old node)
        self.forward_speed = rospy.get_param("~forward_speed", 2.0)
        self.kp           = rospy.get_param("~kp", 0.052)
        self.kd           = rospy.get_param("~kd", 0.0)
        self.max_ang      = rospy.get_param("~max_ang", 5.0)
        self.n_rows       = rospy.get_param("~n_rows", 80)

        # State for hysteresis + derivative
        self.last_mid = None
        self.misses   = 0
        self.prev_e   = 0.0
        self.prev_t   = None

        # Time-trials timing
        self.duration   = rospy.get_param("~duration", 30.0)  # seconds
        self.start_time = rospy.Time.now()
        self.run_active = False
        self.stopped    = False

        # Start the score timer immediately
        self.start_run()

    # ---------- score-tracker / timing helpers ----------

    def start_run(self):
        """Send start message to /score_tracker and mark run as active."""
        timeout_t = rospy.Time.now() + rospy.Duration(2.0)
        while (self.score_pub.get_num_connections() == 0 and
               rospy.Time.now() < timeout_t and
               not rospy.is_shutdown()):
            rospy.sleep(0.1)

        msg = String()
        msg.data = "TeamRed,multi21,0,WHATEVER"
        self.score_pub.publish(msg)
        rospy.loginfo("Sent START to /score_tracker")

        self.start_time = rospy.Time.now()
        self.run_active = True

    def stop_run(self):
        """Send stop message, stop robot, and kill camera node."""
        if self.stopped:
            return

        self.run_active = False

        # zero velocity
        stop_twist = Twist()
        self.cmd_pub.publish(stop_twist)

        # stop timer
        msg = String()
        msg.data = "TeamRed,multi21,-1,WHATEVER"
        self.score_pub.publish(msg)
        rospy.loginfo("Sent STOP to /score_tracker")

        rospy.sleep(3.0)
        try:
            rosnode.kill_nodes(['/B1_camera_view'])
        except rosnode.ROSNodeException as e:
            rospy.logwarn("Failed to kill /B1_camera_view: %s", e)

        self.stopped = True

    # ---------------- main callback ----------------

    def callback(self, data):
        # If we’ve already finished, ignore images
        if self.stopped or not self.run_active:
            return

        # Enforce time limit
        elapsed = (rospy.Time.now() - self.start_time).to_sec()
        if elapsed >= self.duration:
            self.stop_run()
            return

        # Convert image
        
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

        # Live param refresh (so you can tune via rosparam)
        self.kp           = rospy.get_param("~kp", self.kp)
        self.kd           = rospy.get_param("~kd", self.kd)
        self.forward_speed = rospy.get_param("~forward_speed", self.forward_speed)
        self.max_ang      = rospy.get_param("~max_ang", self.max_ang)
        self.n_rows       = rospy.get_param("~n_rows", self.n_rows)

        H, W = cv_image.shape[:2]

        # Crop ROI at bottom: [H-(SKIP+n_rows) : H-SKIP]
        y0 = H - (SKIP + self.n_rows)
        y1 = H - SKIP
        if y0 < 0:
            y0 = 0
        if y1 <= y0:
            twist = Twist()
            self.cmd_pub.publish(twist)
            return

        cutFrame = cv_image[y0:y1, 0:W]
        if cutFrame.size == 0:
            twist = Twist()
            self.cmd_pub.publish(twist)
            return

        # use HSV to distinguish grey road from green grass + white curb
        hsv = cv2.cvtColor(cutFrame, cv2.COLOR_BGR2HSV)
        H_roi, W_roi = hsv.shape[:2]

        # split channels
        h_chan, s_chan, v_chan = cv2.split(hsv)

        # rough ranges for "greyish road":
        # - low saturation (not green)
        # - medium value (not bright white curb, not dark trees)
        # these numbers will likely need a bit of tuning
        sat_mask  = cv2.inRange(s_chan, 0, 60)       # 0..60 out of 255
        val_mask  = cv2.inRange(v_chan, 50, 200)     # 50..200 out of 255

        mask_road = cv2.bitwise_and(sat_mask, val_mask)

        # clean up mask a bit
        kernel = np.ones((5, 5), np.uint8)
        mask_road = cv2.morphologyEx(mask_road, cv2.MORPH_OPEN, kernel)
        mask_road = cv2.morphologyEx(mask_road, cv2.MORPH_CLOSE, kernel)

        # compute centroid of road mask
        M = cv2.moments(mask_road)

        averageMidpoint = None
        if M["m00"] > 1000:  # area threshold so noise doesn't count
            cx_local = M["m10"] / M["m00"]  # in ROI coordinates
            averageMidpoint = int(cx_local)
            found = True
        else:
            found = False


        x_ref = W / 2.0  # image center

        # hysteresis on midpoint
        if averageMidpoint is not None:
            self.last_mid = int(averageMidpoint)
            self.misses = 0
        else:
            self.misses += 1

        twist = Twist()

        if (self.last_mid is not None) and (self.misses <= NUMBER_OF_MISSED_FRAMES):
            # we still have a line (or recent memory of it)
            e = float(self.last_mid - x_ref)
            t = rospy.Time.now().to_sec()

            # derivative term
            if (self.prev_t is None) or (t <= self.prev_t):
                e_d = 0.0
            else:
                dt = t - self.prev_t
                if dt < 1e-4:
                    e_d = 0.0
                else:
                    e_d = (e - self.prev_e) / dt

            if not np.isfinite(e_d):
                e_d = 0.0

            omega = self.kp * e + self.kd * e_d
            if not np.isfinite(omega):
                omega = 0.0
            omega = float(np.clip(omega, -self.max_ang, self.max_ang))

            twist.linear.x  = self.forward_speed
            twist.angular.z = -omega

            self.prev_e = e
            self.prev_t = t

        else:
            # Lost the line for too many frames: spin with reduced speed
            if self.last_mid is not None:
                e = float(self.last_mid - x_ref)
                direction = -np.sign(e)
                spin_rate = 0.5

                omega_lost = direction * (spin_rate * self.max_ang)
                v_lost     = self.forward_speed * (1.0 - spin_rate)

                twist.linear.x  = v_lost
                twist.angular.z = omega_lost
            else:
                twist.linear.x  = 0.0
                twist.angular.z = 1.0

        # Optional debug overlay 
        if rospy.get_param("~debug", True):
            vis = cv_image.copy()
            # draw ROI rectangle
            cv2.rectangle(vis, (0, y0), (W-1, y1), (0, 0, 255), 2)
            if self.last_mid is not None:
                y_draw = y0 + H // 2
                cv2.circle(vis, (int(self.last_mid), int(y_draw)), 16, (0, 0, 255), -1)
            cv2.line(vis, (int(x_ref), 0), (int(x_ref), H), (255, 0, 0), 1)
            cv2.imshow("lane_debug", vis)
            cv2.waitKey(1)

        self.cmd_pub.publish(twist)


if __name__ == "__main__":
    rospy.init_node('time_trials_move')
    node = TimeTrialsLineFollower()
    rospy.spin()
