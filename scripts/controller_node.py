#!/usr/bin/env python3

import sys
import threading
import cv2 as cv
import rospy
import rosnode

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from std_msgs.msg import String

from controller_state_machine import StateMachine
from line_follower import LaneFollower
from pink_line_detector import StripeDetector
from sign_detector import SignDetector
from tunnel_navigrator import TunnelNavigator
from letter_classifier import LetterClassifier
from motion_detector import MotionDetector





class ControllerNode:
    """Main robot controller - handles lane following, signs, and tunnel navigation"""
    
    def __init__(self):
        rospy.init_node("controller_node")

        self.bridge = CvBridge()

        # Topics (read params first)
        image_topic = rospy.get_param("~image_topic", "/B1/pi_camera/image_raw")
        cmd_topic = rospy.get_param("~cmd_vel_topic", "/B1/cmd_vel")

        # Publishers (safe to create early)
        self.cmd_pub = rospy.Publisher(cmd_topic, Twist, queue_size=10)
        self.score_pub = rospy.Publisher("/score_tracker", String, queue_size=1)

        # Main components
        self.state_machine = StateMachine()
        self.lane_follower = LaneFollower(detection_mode="asphalt")
        self.stripe_detector = StripeDetector()
        self.sign_detector = SignDetector()
        self.tunnel_navigator = TunnelNavigator()
        
        # Run status
        self.run_active = False
        self.stopped = False

        # Pedestrian / crosswalk state (defaults)
        self.crosswalk_active = False
        self.crosswalk_cleared = False
        self.last_red_y = None

        # Boost forward parameters (used when crossing is clear)
        self.boost_active = False
        self.boost_end_time = rospy.Time(0)
        self.boost_duration = rospy.get_param("~ped_boost_duration", 0.8)
        self.boost_speed = rospy.get_param("~ped_boost_speed", 1.0)

        # Motion-based pedestrian detector (frame-diff)
        self.motion_detector = MotionDetector(min_area=1800, still_frames_needed=10)

        # New: track when we first enter the IN_LOOP drive mode
        self.in_loop_start_time = None

        # Sign tracking - one sign per phase
        self.sign_colors = {0: "dark_blue", 1: "dark_blue", 2: "light_blue", 3: "dark_blue", 4: "dark_blue"}
        self.sign_captured = {0: False, 1: False, 2: False, 3: False, 4: False}
        
        # Sign tracking
        self.sign_colors = {0: "dark_blue", 1: "dark_blue", 2: "light_blue", 3: "dark_blue", 4: "dark_blue", 5: "dark_blue", 6: "dark_blue", 7: "dark_blue"}
        self.sign_captured = {0: False, 1: False, 2: False, 3: False, 4: False, 5: False, 6: False, 7: False}
        self.current_phase = None
        self.sign_cooldown_time = rospy.Time(0)
        self.clue_board_number = 0  # To track clue board submissions

        # Loop distance tracking
        self.loop_distance = 0.0
        self.last_time = None
        self.loop_complete = False

        # Start keyboard listener
        self.key_thread = threading.Thread(target=self.listen_for_stop, daemon=True)
        self.key_thread.start()

        # Start CNN classifier
        self.classifier = LetterClassifier("/home/fizzer/ros_ws/src/controller_pkg/Util/conv_model(2).tflite")

        # Finally subscribe to the image topic AFTER all attributes are initialized
        self.image_sub = rospy.Subscriber(image_topic, Image, self.image_callback, queue_size=1)
    
    # ========== Keyboard Control ==========
    def listen_for_stop(self):
        """Background thread - press 'q' + Enter to stop"""
        rospy.loginfo("Press 'q' + Enter to stop the robot")
        while not rospy.is_shutdown():
            key = sys.stdin.read(1)
            if key and key.lower() == 'q':
                rospy.loginfo("Stopping robot...")
                self.stop_run()
    
    # ========== Start/Stop Functions ==========
    def start_run(self):
        """Send START message to score tracker"""
        timeout = rospy.Time.now() + rospy.Duration(2.0)
        while self.score_pub.get_num_connections() == 0 and rospy.Time.now() < timeout:
            rospy.sleep(0.1)
        
        msg = String()
        msg.data = "TeamRed,multi21,0,WHATEVER"
        self.score_pub.publish(msg)
        rospy.loginfo("Started run")
        
        self.run_active = True
    
    def stop_run(self):
        """Send STOP message and halt robot"""
        if self.stopped:
            return
        
        self.run_active = False
        
        stop_cmd = Twist()
        self.cmd_pub.publish(stop_cmd)
        
        msg = String()
        msg.data = "TeamRed,multi21,-1,WHATEVER"
        self.score_pub.publish(msg)
        rospy.loginfo("Stopped run")
        
        rospy.sleep(3.0)
        
        try:
            rosnode.kill_nodes(["/B1_camera_view"])
        except:
            pass
        
        self.stopped = True
    
    # ========== Main Image Processing Loop ==========
    def image_callback(self, image_msg):
        """Called every frame - this is the main control loop"""
        img = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

        now = rospy.Time.now()
        cmd = Twist()

        # --- Detect red stripe ---
        on_red, crossed_crosswalk, red_y = self.stripe_detector.check_red_stripe(img)

        # ---------- RED LINE STOP ----------
        if on_red and not self.crosswalk_active and not self.crosswalk_cleared:
            rospy.loginfo("Crosswalk: STOPPING at red line")
            self.crosswalk_active = True
            self.crosswalk_cleared = False
            self.motion_detector.reset()   # start fresh
            cmd.linear.x = 0
            cmd.angular.z = 0
            self.cmd_pub.publish(cmd)
            return


        # ---------- If stopped at red line, check motion ----------
        if self.crosswalk_active and not self.crosswalk_cleared:

            H, W = img.shape[:2]
            y1 = int(H * 2/8)
            y2 = int(H * 5/8)

            movement_region = img[y1:y2, :]

            moving = self.motion_detector.check_motion(movement_region)

            if moving:
                rospy.loginfo_throttle(1.0, "Crosswalk: MOVEMENT detected → HOLD")
                cmd.linear.x = 0
                cmd.angular.z = 0
                self.cmd_pub.publish(cmd)
                return
            else:
                rospy.loginfo("Crosswalk: CLEAR → BOOST AND GO")
                self.crosswalk_cleared = True
                self.crosswalk_active = False

                # BOOST SPEED
                boost = Twist()
                boost.linear.x = 1.0    # max speed
                boost.angular.z = 0.0
                self.cmd_pub.publish(boost)
                rospy.sleep(0.8)        # short burst across crosswalk

                return  # next callback resumes normal driving



        on_pink, crossed_pink_marker = self.stripe_detector.check_pink_stripe(img)

        current_state = self.state_machine.state

        # PRE-TUNNEL FORWARD DRIVE (after capturing sign 5, before tunnel search)
        if self.pre_tunnel_forward:
            cmd = Twist()
            cmd.linear.x = 1.0
            cmd.angular.z = 0.0
            
            self.pre_tunnel_frames += 1
            
            if self.pre_tunnel_frames >= self.pre_tunnel_duration:
                self.pre_tunnel_forward = False
                rospy.loginfo("[PRE-TUNNEL] Forward drive complete, starting tunnel navigation")
            
            self.cmd_pub.publish(cmd)
            return

        # TUNNEL NAVIGATION STATE
        if current_state == StateMachine.START_GRASS_NO_ROAD:
            # Determine which sign to look for based on tunnel navigator phase
            tunnel_phase = self.tunnel_navigator.phase
            
            # Phase 3 = rotating to find sign at tunnel entrance
            if tunnel_phase == 3:
                tunnel_sign_captured = self.sign_captured[6]
                cmd, complete = self.tunnel_navigator.navigate(img, sign_captured=tunnel_sign_captured)
                
                # Try to capture sign 6
                if not self.sign_captured[6]:
                    sign_cmd = self.try_capture_sign(img, cmd, phase=6)
                    if sign_cmd.linear.x != cmd.linear.x or sign_cmd.angular.z != cmd.angular.z:
                        cmd = sign_cmd
            else:
                # Other tunnel phases
                cmd, complete = self.tunnel_navigator.navigate(img, sign_captured=self.sign_captured[6])
            
            self.cmd_pub.publish(cmd)
            return

        # CLIMBING MOUNTAIN STATE (Looking for Sign 7)
        if current_state == StateMachine.CLIMBING_MOUNTAIN:
            # Continue using tunnel navigator for white line following (Phase 6 logic)
            cmd, complete = self.tunnel_navigator.navigate(img, sign_captured=True)

            # Look for Sign 7 (Final Sign)
            if not self.sign_captured[7]:
                captured = self.try_capture_sign_mountain(img, phase=7)
                
                if captured:
                    # Sign 7 captured - stop the run!
                    rospy.loginfo("="*70)
                    rospy.loginfo("FINAL SIGN 7 CAPTURED! STOPPING RUN.")
                    rospy.loginfo("="*70)
                    
                    stop_cmd = Twist()
                    stop_cmd.linear.x = 0.0
                    stop_cmd.angular.z = 0.0
                    self.cmd_pub.publish(stop_cmd)
                    
                    self.stop_run()
                    return

            self.cmd_pub.publish(cmd)
            return
        
        # Update state machine
        events = {
            "crossed_crosswalk": crossed_crosswalk,
            "sign1_captured": self.sign_captured[1],
            "loop_complete": self.loop_complete,
            "crossed_first_pink": (
                crossed_pink_marker and current_state == StateMachine.AFTER_LOOP
            ),
            "grass_sign_done": self.sign_captured[4] and (now > self.sign_cooldown_time),
            "bridge_sign_done": self.sign_captured[5] and (now > self.sign_cooldown_time),
            "tunnel_exited": self.tunnel_navigator.phase >= 6
        }

        actions = self.state_machine.update(events)
        
        # Start timer on first frame
        if actions["start_timer"] and not self.run_active:
            self.start_run()
        
        # Get movement command
        
        if self.run_active:
            drive_mode = actions["drive_mode"]
            self.current_phase = actions["sign_phase"]

            # Switch lane follower detection mode based on drive mode
            if drive_mode == "grass_road":
                self.lane_follower.detection_mode = "grass"
            else:
                self.lane_follower.detection_mode = "asphalt"

            # CHECK FOR RECOVERY MANEUVER FIRST
            if self.grass_sign_recovery and self.recovery_frames < self.recovery_duration:
                cmd.linear.x = 0.0
                cmd.angular.z = -1.8
                
                self.recovery_frames += 1
                
                if self.recovery_frames >= self.recovery_duration:
                    self.grass_sign_recovery = False

                self.cmd_pub.publish(cmd)
                return
            
            # Lane following with different biases for different sections
            try:
                if drive_mode == "normal":
                    cmd = self.lane_follower.get_command(img, center_bias=0.0)
                
                elif drive_mode == "after_crosswalk":
                    cmd = self.lane_follower.get_command(img, center_bias=0.0)
                
                elif drive_mode == "in_loop":
                    # On entering IN_LOOP record start time
                    if self.in_loop_start_time is None:
                        self.in_loop_start_time = rospy.Time.now()

                    # First 3 seconds: use standard lane follower with left bias, then switch
                    elapsed = rospy.Time.now() - self.in_loop_start_time
                    if elapsed < rospy.Duration(3.0):
                        cmd = self.lane_follower.get_command(img, center_bias=0.035)
                    else:
                        # After 3s switch to left-contour follower
                        cmd = self.lane_follower.get_command_left_contour(img, center_bias=0.035)
                
                elif drive_mode == "exit_loop":
                    # Bias left to take exit
                    cmd = self.lane_follower.get_command(img, center_bias=-0.15)

            elif drive_mode == "grass_road":
                cmd = self.lane_follower.get_command(img, center_bias=0.0)
                cmd.linear.x = 1.0

            # Try to capture sign if we haven't yet in this phase
            if self.current_phase is not None and not self.sign_captured[self.current_phase]:
                cmd = self.try_capture_sign(img, cmd)
            
            # Track distance while in loop
            if drive_mode == "in_loop":
                self.update_loop_distance(cmd)
            else:
                self.loop_distance = 0.0
                self.last_time = None
        
        self.cmd_pub.publish(cmd)
    
    # ========== Loop Distance Tracking ==========
    def update_loop_distance(self, cmd):
        """Track how far we've driven in the loop"""
        now = rospy.Time.now().to_sec()
        
        if self.last_time is None:
            self.last_time = now
        else:
            dt = now - self.last_time
            if dt > 0:
                self.loop_distance += abs(cmd.linear.x) * dt
            self.last_time = now
        
        # Check if we've done one lap 
        if self.loop_distance >= 13.0:
            self.loop_complete = True
    
    # ========== Sign Detection and Capture ==========
    def classify_word_groups(self, groups):
        """
        groups: list of lists of letter images
                Example: [ [L1,L2,L3], [L4,L5] ]
        Returns: list of strings
                Example: ["BIG", "HOUSE"]
        """
        words = []
        for word in groups:
            chars = [self.classifier.predict_letter(L) for L in word]
            words.append("".join(chars))
        return words


    def try_capture_sign(self, img, cmd):
    def try_capture_sign(self, img, cmd, phase=None):
        """Align with sign and capture it"""
        now = rospy.Time.now()
        
        if now < self.sign_cooldown_time:
            return cmd
        
        # Use provided phase or current phase
        target_phase = phase if phase is not None else self.current_phase
        
        if target_phase is None:
            return cmd
        
        use_light_blue = (self.sign_colors[target_phase] == "light_blue")
        
        sign = self.sign_detector.find_sign(img, use_light_blue=use_light_blue)
        
        if sign is None:
            return cmd
        
        width = sign["frame_width"]
        sign_x = sign["center_x"]
        sign_width = sign["bbox_width"]
        
        x_pos = sign_x / width
        size = sign_width / width
        
        cmd.linear.x = 0.0
        
        # Center the sign horizontally
        if x_pos < 0.35:
            cmd.angular.z = 0.8
            return cmd
        
        if x_pos > 0.65:
            cmd.angular.z = -0.8
            return cmd
        
        # Check if it's big enough
        if size < 0.2:
            cmd.linear.x = 0.4
            cmd.angular.z = 0.0
            return cmd
        
        # Capture it
        warped = self.sign_detector.warp_sign(img, sign["quad"])

        if warped is not None:
            phase_name = f"sign_phase_{target_phase}"
            cv.imshow(phase_name, warped)
            cv.waitKey(1)

            # 1 — remove blue border → get gray interior board
            gray_board = self.sign_detector.warp_gray_square(warped)
            if gray_board is None:
                rospy.loginfo("gray warp failed")
                return cmd

            cv.imshow("gray_board", gray_board)
            cv.waitKey(1)

            # 2 — extract only blue letters
            letter_mask = self.sign_detector.extract_letters_only(gray_board)
            cv.imshow("letter_mask", letter_mask)
            cv.waitKey(1)

            # 3 — segment into TOP words + BOTTOM words
            top_words, bottom_words = self.sign_detector.segment_words(letter_mask)

            # DEBUG visualization
            for wi, word in enumerate(top_words):
                for li, L in enumerate(word):
                    cv.imshow(f"TOP_word{wi}_letter{li}", L)
                    cv.waitKey(1)

            for wi, word in enumerate(bottom_words):
                for li, L in enumerate(word):
                    cv.imshow(f"BOT_word{wi}_letter{li}", L)
                    cv.waitKey(1)
            # 4 — classify letters with TFLite
            top_strings = self.classify_word_groups(top_words)
            bottom_strings = self.classify_word_groups(bottom_words)

            # 5 — send to score tracker
            self.clue_board_number += 1
            msg = String()
            bottom_text = " ".join(bottom_strings) if bottom_strings else ""
            msg.data = f"TeamRed,multi21,{self.clue_board_number},{bottom_text}"
            self.score_pub.publish(msg)

            # Print out results
            rospy.loginfo(f"{self.clue_board_number},{bottom_text}")
            rospy.loginfo(f"TOP WORDS: {top_strings}")
            rospy.loginfo(f"BOTTOM WORDS: {bottom_strings}")


        # mark the sign captured
        self.sign_captured[self.current_phase] = True
        self.sign_cooldown_time = now + rospy.Duration(3)
        cmd.linear.x = 0
        cmd.angular.z = 0
            rospy.loginfo(f"Captured sign in phase {target_phase}")
        
        self.sign_captured[target_phase] = True
        
        # If this is the grass sign (phase 4), trigger recovery maneuver
        if target_phase == 4:
            self.grass_sign_recovery = True
            self.recovery_frames = 0
        
        # If this is the bridge sign (phase 5), trigger pre-tunnel forward drive
        if target_phase == 5:
            rospy.loginfo("[SIGN 5] Captured! Starting pre-tunnel forward drive")
            self.pre_tunnel_forward = True
            self.pre_tunnel_frames = 0
        
        self.sign_cooldown_time = now + rospy.Duration(3.0)
        
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        
        return cmd

    def try_capture_sign_mountain(self, img, phase=7):
        """Try to capture sign during mountain climb - returns True if captured"""
        now = rospy.Time.now()
        
        if now < self.sign_cooldown_time:
            return False
        
        target_phase = phase
        
        use_light_blue = (self.sign_colors[target_phase] == "light_blue")
        
        sign = self.sign_detector.find_sign(img, use_light_blue=use_light_blue)
        
        if sign is None:
            return False
        
        width = sign["frame_width"]
        sign_x = sign["center_x"]
        sign_width = sign["bbox_width"]
        
        x_pos = sign_x / width
        size = sign_width / width
        
        rospy.loginfo_throttle(0.5, f"[MOUNTAIN SIGN 7] Detected! x_pos={x_pos:.2f}, size={size:.3f}")
        
        # Capture when sign is big enough and reasonably centered
        if size >= 0.12 and 0.2 < x_pos < 0.8:
            # Capture it
            warped = self.sign_detector.warp_sign(img, sign["quad"])
            
            if warped is not None:
                phase_name = f"sign_phase_{target_phase}"
                cv.imshow(phase_name, warped)
                cv.waitKey(1)
                rospy.loginfo("="*60)
                rospy.loginfo(f"[MOUNTAIN] CAPTURED SIGN {target_phase}!")
                rospy.loginfo("="*60)
            
            self.sign_captured[target_phase] = True
            self.sign_cooldown_time = now + rospy.Duration(3.0)
            
            return True
        
        return False


if __name__ == "__main__":
    node = ControllerNode()
    rospy.spin()