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
    """Main robot controller - handles lane following, signs, pedestrian detection, and tunnel navigation"""
    
    def __init__(self):
        rospy.init_node("controller_node")
        
        self.bridge = CvBridge()
        
        # ROS topics
        image_topic = rospy.get_param("~image_topic", "/B1/pi_camera/image_raw")
        cmd_topic = rospy.get_param("~cmd_vel_topic", "/B1/cmd_vel")
        
        # Publishers (create before subscriber)
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

        self.run_start_time = None
        self.run_max_duration = 240
        
        # ========== Pedestrian / Crosswalk Detection ==========
        self.crosswalk_active = False
        self.crosswalk_cleared = False
        
        self.motion_detector = MotionDetector(
            min_area=2000,
            still_frames_needed=15,
            motion_frames_needed=5
        )
        
        # Boost parameters
        self.boost_duration = rospy.get_param("~ped_boost_duration", 0.5)
        self.boost_speed = rospy.get_param("~ped_boost_speed", 1.0)
        
        # ========== Sign tracking ==========
        self.sign_colors = {0: "dark_blue", 1: "dark_blue", 2: "light_blue", 3: "dark_blue", 4: "dark_blue", 5: "dark_blue", 6: "dark_blue", 7: "dark_blue"}
        self.sign_captured = {0: False, 1: False, 2: False, 3: False, 4: False, 5: False, 6: False, 7: False}
        self.current_phase = None
        self.sign_cooldown_time = rospy.Time(0)
        self.clue_board_number = 0
        
        # ========== Loop distance tracking ==========
        self.loop_distance = 0.0
        self.last_time = None
        self.loop_complete = False
        self.in_loop_start_time = None

        # ========== GRASS SIGN RECOVERY (PHASE 4) ==========
        # States: "inactive", "pausing", "turning"
        self.grass_recovery_state = "inactive"
        self.grass_recovery_start_time = None
        self.grass_recovery_pause_duration = 1.0  # 1 second pause before turning
        self.grass_recovery_turn_frames = 0
        self.grass_recovery_turn_duration = 15  # frames of turning

        # ========== Pre-tunnel forward drive ==========
        self.pre_tunnel_forward = False
        self.pre_tunnel_frames = 0
        self.pre_tunnel_duration = 20

        # Start keyboard listener
        self.key_thread = threading.Thread(target=self.listen_for_stop, daemon=True)
        self.key_thread.start()

        # Start CNN classifier
        self.classifier = LetterClassifier("/home/fizzer/ros_ws/src/controller_pkg/Util/conv_model(2).tflite")

        # Subscribe AFTER all init
        self.image_sub = rospy.Subscriber(image_topic, Image, self.image_callback, queue_size=1)
        
        rospy.loginfo("="*60)
        rospy.loginfo("ControllerNode initialized")
        rospy.loginfo("="*60)

    
    # ========== Keyboard Control ==========
    def listen_for_stop(self):
        rospy.loginfo("Press 'q' + Enter to stop the robot")
        while not rospy.is_shutdown():
            key = sys.stdin.read(1)
            if key and key.lower() == 'q':
                rospy.loginfo("Stopping robot...")
                self.stop_run()
    
    # ========== Start/Stop Functions ==========
    def start_run(self):
        timeout = rospy.Time.now() + rospy.Duration(2.0)
        while self.score_pub.get_num_connections() == 0 and rospy.Time.now() < timeout:
            rospy.sleep(0.1)
        
        msg = String()
        msg.data = "TeamRed,multi21,0,WHATEVER"
        self.score_pub.publish(msg)
        rospy.loginfo("Started run")
        self.run_active = True
        self.run_start = rospy.Time.now()
    
    def stop_run(self):
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
        img = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        cmd = Twist()
        now = rospy.Time.now()

        if self.run_active and self.run_start_time is not None:
            if rospy.Time.now() - self.run_start_time >= rospy.Duration(self.run_max_duration):
                self.stop_run()
                return
        
        # ==================== GRASS SIGN RECOVERY - HIGHEST PRIORITY ====================
        # This is checked FIRST, before anything else
        if self.grass_recovery_state != "inactive":
            rospy.loginfo(f"[GRASS RECOVERY] State: {self.grass_recovery_state}")
            
            if self.grass_recovery_state == "pausing":
                # Stop and wait for 1 second
                elapsed = (now - self.grass_recovery_start_time).to_sec()
                rospy.loginfo(f"[GRASS RECOVERY] PAUSING... elapsed={elapsed:.2f}s / {self.grass_recovery_pause_duration}s")
                
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.cmd_pub.publish(cmd)
                
                if elapsed >= self.grass_recovery_pause_duration:
                    rospy.loginfo("[GRASS RECOVERY] Pause complete, starting RIGHT TURN")
                    self.grass_recovery_state = "turning"
                    self.grass_recovery_turn_frames = 0
                return
            
            elif self.grass_recovery_state == "turning":
                # Execute right turn
                self.grass_recovery_turn_frames += 1
                rospy.loginfo(f"[GRASS RECOVERY] TURNING RIGHT... frame {self.grass_recovery_turn_frames}/{self.grass_recovery_turn_duration}")
                
                cmd.linear.x = 0.0
                cmd.angular.z = -1.8  # Turn RIGHT
                self.cmd_pub.publish(cmd)
                
                if self.grass_recovery_turn_frames >= self.grass_recovery_turn_duration:
                    rospy.loginfo("[GRASS RECOVERY] *** RIGHT TURN COMPLETE ***")
                    self.grass_recovery_state = "inactive"
                return
        
        # ==================== CROSSWALK / PEDESTRIAN DETECTION ====================
        red_stripe_result = self.stripe_detector.check_red_stripe(img)
        
        if len(red_stripe_result) == 3:
            on_red, crossed_crosswalk, red_y = red_stripe_result
        else:
            on_red, crossed_crosswalk = red_stripe_result
        
        # Stop at FIRST red line
        if on_red and not self.crosswalk_active and not self.crosswalk_cleared:
            rospy.loginfo("="*60)
            rospy.loginfo("[CROSSWALK] STOPPING AT RED LINE!")
            rospy.loginfo("[CROSSWALK] Waiting for pedestrian...")
            rospy.loginfo("="*60)
            
            self.crosswalk_active = True
            self.crosswalk_cleared = False
            self.motion_detector.reset()
            
            cmd.linear.x = 0
            cmd.angular.z = 0
            self.cmd_pub.publish(cmd)
            return
        
        # If stopped, check for pedestrian
        if self.crosswalk_active and not self.crosswalk_cleared:
            H, W = img.shape[:2]
            y1 = int(H * 2 / 8)
            y2 = int(H * 5 / 8)
            movement_region = img[y1:y2, :]
            
            # Debug: show monitoring region
            debug_img = img.copy()
            cv.rectangle(debug_img, (0, y1), (W, y2), (0, 255, 255), 2)
            if self.motion_detector.has_seen_motion:
                cv.putText(debug_img, "PEDESTRIAN DETECTED - waiting to pass", (10, y1 - 10), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv.putText(debug_img, "WAITING FOR PEDESTRIAN...", (10, y1 - 10), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            #cv.imshow("crosswalk_view", debug_img)
            cv.waitKey(1)
            
            should_wait = self.motion_detector.check_motion(movement_region)
            
            if should_wait:
                cmd.linear.x = 0
                cmd.angular.z = 0
                self.cmd_pub.publish(cmd)
                return
            else:
                # Clear to cross!
                rospy.loginfo("="*60)
                rospy.loginfo("[CROSSWALK] *** CLEAR TO GO! ***")
                rospy.loginfo("="*60)
                
                self.crosswalk_cleared = True
                self.crosswalk_active = False
                
                # Boost across
                boost = Twist()
                boost.linear.x = self.boost_speed
                boost.angular.z = 0.0
                self.cmd_pub.publish(boost)
                rospy.sleep(self.boost_duration)
                return
        
        # ==================== DETECT PINK MARKERS ====================
        _, crossed_pink_marker = self.stripe_detector.check_pink_stripe(img)
        
        current_state = self.state_machine.state

        # ==================== PRE-TUNNEL FORWARD DRIVE ====================
        if self.pre_tunnel_forward:
            cmd = Twist()
            cmd.linear.x = 1.0
            cmd.angular.z = 0.0
            
            self.pre_tunnel_frames += 1
            
            if self.pre_tunnel_frames >= self.pre_tunnel_duration:
                self.pre_tunnel_forward = False
                rospy.loginfo("[PRE-TUNNEL] Forward drive complete")
            
            self.cmd_pub.publish(cmd)
            return

        # ==================== TUNNEL NAVIGATION STATE ====================
        if current_state == StateMachine.START_GRASS_NO_ROAD:
            tunnel_phase = self.tunnel_navigator.phase
            
            if tunnel_phase == 3:
                tunnel_sign_captured = self.sign_captured[6]
                cmd, complete = self.tunnel_navigator.navigate(img, sign_captured=tunnel_sign_captured)
                
                if not self.sign_captured[6]:
                    sign_cmd = self.try_capture_sign(img, cmd, phase=6)
                    if sign_cmd.linear.x != cmd.linear.x or sign_cmd.angular.z != cmd.angular.z:
                        cmd = sign_cmd
            else:
                cmd, complete = self.tunnel_navigator.navigate(img, sign_captured=self.sign_captured[6])
            
            self.cmd_pub.publish(cmd)
            return

        # ==================== CLIMBING MOUNTAIN STATE ====================
        if current_state == StateMachine.CLIMBING_MOUNTAIN:
            cmd, complete = self.tunnel_navigator.navigate(img, sign_captured=True)

            if not self.sign_captured[7]:
                captured = self.try_capture_sign_mountain(img, phase=7)
                
                if captured:
                    rospy.loginfo("="*70)
                    rospy.loginfo("FINAL SIGN 7 CAPTURED! STOPPING RUN.")
                    rospy.loginfo("="*70)
                    
                    stop_cmd = Twist()
                    self.cmd_pub.publish(stop_cmd)
                    self.stop_run()
                    return

            self.cmd_pub.publish(cmd)
            return
        
        # ==================== UPDATE STATE MACHINE ====================
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
        
        if actions["start_timer"] and not self.run_active:
            self.start_run()
        
        # ==================== NORMAL DRIVING ====================
        if self.run_active:
            drive_mode = actions["drive_mode"]
            self.current_phase = actions["sign_phase"]

            if drive_mode == "grass_road":
                self.lane_follower.detection_mode = "grass"
            else:
                self.lane_follower.detection_mode = "asphalt"
            
            # Lane following
            if drive_mode == "normal":
                cmd = self.lane_follower.get_command(img, center_bias=0.0)
            elif drive_mode == "after_crosswalk":
                cmd = self.lane_follower.get_command(img, center_bias=0.0)
            elif drive_mode == "in_loop":
                # Track when we entered the loop
                if self.in_loop_start_time is None:
                    self.in_loop_start_time = rospy.Time.now()
                    rospy.loginfo("[LOOP] Entered loop - using CENTER driving with LEFT BIAS to enter clockwise")
                
                elapsed = (rospy.Time.now() - self.in_loop_start_time).to_sec()
                
                # First 4 seconds: drive with LEFT BIAS to enter loop clockwise (go left into loop)
                if elapsed < 4.0:
                    rospy.loginfo_throttle(1.0, f"[LOOP] Center driving with left bias... {elapsed:.1f}s / 4.0s")
                    cmd = self.lane_follower.get_command(img, center_bias=0.035)  # Positive = aim left
                else:
                    # After 4 seconds: switch to LEFT LINE following
                    if elapsed < 4.5:  # Log once when switching
                        rospy.loginfo("[LOOP] Switching to LEFT LINE following mode!")
                    cmd = self.lane_follower.get_command_loop(img)
            elif drive_mode == "exit_loop":
                cmd = self.lane_follower.get_command(img, center_bias=-0.15)
            elif drive_mode == "grass_road":
                cmd = self.lane_follower.get_command(img, center_bias=0.0)
                cmd.linear.x = 1.0

            # Try to capture sign (respects cooldown)
            if self.current_phase is not None and not self.sign_captured[self.current_phase]:
                # Check cooldown before even trying
                if now >= self.sign_cooldown_time:
                    cmd = self.try_capture_sign(img, cmd)
            
            # Track loop distance
            if drive_mode == "in_loop":
                self.update_loop_distance(cmd)
            else:
                self.loop_distance = 0.0
                self.last_time = None
        
        self.cmd_pub.publish(cmd)
    
    # ========== Loop Distance Tracking ==========
    def update_loop_distance(self, cmd):
        now = rospy.Time.now().to_sec()
        if self.last_time is None:
            self.last_time = now
        else:
            dt = now - self.last_time
            if dt > 0:
                self.loop_distance += abs(cmd.linear.x) * dt
            self.last_time = now
        if self.loop_distance >= 14.0:
            self.loop_complete = True
    
    # ========== Sign Detection ==========
    def classify_word_groups(self, groups):
        words = []
        for word in groups:
            chars = [self.classifier.predict_letter(L) for L in word]
            words.append("".join(chars))
        return words

    def try_capture_sign(self, img, cmd, phase=None):
        now = rospy.Time.now()
        
        # Double-check cooldown
        if now < self.sign_cooldown_time:
            return cmd
        
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
        
        # Approach if too small
        if size < 0.2:
            cmd.linear.x = 0.4
            cmd.angular.z = 0.0
            return cmd
        
        # ========== SIGN IS READY TO CAPTURE ==========
        rospy.loginfo("="*60)
        rospy.loginfo(f"[SIGN CAPTURE] About to capture sign phase {target_phase}")
        rospy.loginfo("="*60)
        
        warped = self.sign_detector.warp_sign(img, sign["quad"])
        if warped is not None:
            #cv.imshow(f"sign_phase_{target_phase}", warped)
            cv.waitKey(1)

            gray_board = self.sign_detector.warp_gray_square(warped)
            if gray_board is None:
                rospy.loginfo("gray warp failed")
                return cmd

            #cv.imshow("gray_board", gray_board)
            cv.waitKey(1)

            letter_mask = self.sign_detector.extract_letters_only(gray_board)
            #cv.imshow("letter_mask", letter_mask)
            cv.waitKey(1)

            top_words, bottom_words = self.sign_detector.segment_words(letter_mask)

            for wi, word in enumerate(top_words):
                for li, L in enumerate(word):
                    #cv.imshow(f"TOP_word{wi}_letter{li}", L)
                    cv.waitKey(1)
            for wi, word in enumerate(bottom_words):
                for li, L in enumerate(word):
                    #cv.imshow(f"BOT_word{wi}_letter{li}", L)
                    cv.waitKey(1)

            top_strings = self.classify_word_groups(top_words)
            bottom_strings = self.classify_word_groups(bottom_words)

            self.clue_board_number += 1
            msg = String()
            bottom_text = " ".join(bottom_strings) if bottom_strings else ""
            msg.data = f"TeamRed,multi21,{self.clue_board_number},{bottom_text}"
            self.score_pub.publish(msg)

            rospy.loginfo(f"{self.clue_board_number},{bottom_text}")
            rospy.loginfo(f"TOP WORDS: {top_strings}")
            rospy.loginfo(f"BOTTOM WORDS: {bottom_strings}")

        # ========== MARK SIGN AS CAPTURED ==========
        self.sign_captured[target_phase] = True
        rospy.loginfo(f"[SIGN CAPTURE] *** CAPTURED SIGN {target_phase} ***")
        
        # ========== SET COOLDOWN AND SPECIAL ACTIONS BASED ON WHICH SIGN ==========
        if target_phase == 3:
            # After exit loop sign (sign 3), wait 5 seconds before looking for sign 4
            self.sign_cooldown_time = now + rospy.Duration(5.0)
            rospy.loginfo(f"[SIGN 3] 5 second cooldown before looking for sign 4")
            
        elif target_phase == 4:
            # ========== GRASS SIGN - TRIGGER RECOVERY SEQUENCE ==========
            rospy.loginfo("="*60)
            rospy.loginfo("[SIGN 4 - GRASS] *** TRIGGERING GRASS RECOVERY ***")
            rospy.loginfo("[SIGN 4 - GRASS] Will pause for 1 second, then turn RIGHT")
            rospy.loginfo("="*60)
            
            self.sign_cooldown_time = now + rospy.Duration(3.0)
            self.grass_recovery_state = "pausing"
            self.grass_recovery_start_time = now
            self.grass_recovery_turn_frames = 0
            
        elif target_phase == 5:
            # Bridge sign - trigger pre-tunnel forward drive
            self.sign_cooldown_time = now + rospy.Duration(3.0)
            self.pre_tunnel_forward = True
            self.pre_tunnel_frames = 0
            rospy.loginfo(f"[SIGN 5] Starting pre-tunnel forward drive")
            
        else:
            # Default 3 second cooldown
            self.sign_cooldown_time = now + rospy.Duration(3.0)
        
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        return cmd

    def try_capture_sign_mountain(self, img, phase=7):
        now = rospy.Time.now()
        if now < self.sign_cooldown_time:
            return False
        
        use_light_blue = (self.sign_colors[phase] == "light_blue")
        sign = self.sign_detector.find_sign(img, use_light_blue=use_light_blue)
        
        if sign is None:
            return False
        
        width = sign["frame_width"]
        sign_x = sign["center_x"]
        sign_width = sign["bbox_width"]
        
        x_pos = sign_x / width
        size = sign_width / width
        
        rospy.loginfo_throttle(0.5, f"[MOUNTAIN SIGN 7] x_pos={x_pos:.2f}, size={size:.3f}")
        
        if size >= 0.12 and 0.2 < x_pos < 0.8:
            warped = self.sign_detector.warp_sign(img, sign["quad"])
            if warped is not None:
                #cv.imshow(f"sign_phase_{phase}", warped)
                cv.waitKey(1)
                rospy.loginfo(f"[MOUNTAIN] CAPTURED SIGN {phase}!")
            
            self.sign_captured[phase] = True
            self.sign_cooldown_time = now + rospy.Duration(3.0)
            return True
        
        return False


if __name__ == "__main__":
    node = ControllerNode()
    rospy.spin()