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
from letter_classifier import LetterClassifier



class ControllerNode:
    """Main robot controller - handles lane following, signs, and crosswalk detection"""
    
    def __init__(self):
        rospy.init_node("controller_node")
        
        self.bridge = CvBridge()
        
        # ROS topics
        image_topic = rospy.get_param("~image_topic", "/B1/pi_camera/image_raw")
        cmd_topic = rospy.get_param("~cmd_vel_topic", "/B1/cmd_vel")
        
        self.image_sub = rospy.Subscriber(image_topic, Image, self.image_callback, queue_size=1)
        self.cmd_pub = rospy.Publisher(cmd_topic, Twist, queue_size=10)
        self.score_pub = rospy.Publisher("/score_tracker", String, queue_size=1)
        
        # Main components
        self.state_machine = StateMachine()
        self.lane_follower = LaneFollower(detection_mode="asphalt")
        self.stripe_detector = StripeDetector()
        self.sign_detector = SignDetector()
        
        # Run status
        self.run_active = False
        self.stopped = False
        
        # Sign tracking - one sign per phase (0=start, 1=after crosswalk, 2=loop, 3=after loop)
        self.sign_colors = {0: "dark_blue", 1: "dark_blue", 2: "light_blue", 3: "dark_blue", 4: "dark_blue"}
        self.sign_captured = {0: False, 1: False, 2: False, 3: False, 4: False}
        self.current_phase = None
        self.sign_cooldown_time = rospy.Time(0)
        self.clue_board_number = 0  # To track clue board submissions
        
        # Loop distance tracking (for knowing when to exit)
        self.loop_distance = 0.0
        self.last_time = None
        self.loop_complete = False
        
        # Start keyboard listener
        self.key_thread = threading.Thread(target=self.listen_for_stop, daemon=True)
        self.key_thread.start()

        # Start CNN classifier
        self.classifier = LetterClassifier("/home/fizzer/ros_ws/src/controller_pkg/Util/conv_model(2).tflite")

    
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
        # Wait for score tracker to connect
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
        
        # Stop movement
        stop_cmd = Twist()
        self.cmd_pub.publish(stop_cmd)
        
        # Send stop message
        msg = String()
        msg.data = "TeamRed,multi21,-1,WHATEVER"
        self.score_pub.publish(msg)
        rospy.loginfo("Stopped run")
        
        rospy.sleep(3.0)
        
        # Try to close camera window
        try:
            rosnode.kill_nodes(["/B1_camera_view"])
        except:
            pass
        
        self.stopped = True
    
    # ========== Main Image Processing Loop ==========
    def image_callback(self, image_msg):
        """Called every frame - this is the main control loop"""
        img = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        
        # Detect crosswalk
        _, crossed_crosswalk = self.stripe_detector.check_red_stripe(img)

        # Detect pink markers
        _, crossed_pink_marker = self.stripe_detector.check_pink_stripe(img)
        
        # Update state machine
        events = {
            "crossed_crosswalk": crossed_crosswalk,
            "sign1_captured": self.sign_captured[1],
            "loop_complete": self.loop_complete,
            "crossed_first_pink": crossed_pink_marker
        }

        actions = self.state_machine.update(events)
        
        # Start timer on first frame
        if actions["start_timer"] and not self.run_active:
            self.start_run()
        
        # Get movement command
        cmd = Twist()
        
        if self.run_active:
            drive_mode = actions["drive_mode"]
            self.current_phase = actions["sign_phase"]

            # Switch lane follower detection mode based on drive mode
            if drive_mode == "grass_road":
                self.lane_follower.detection_mode = "grass"
            else:
                self.lane_follower.detection_mode = "asphalt"
            
            # Lane following with different biases for different sections
            if drive_mode == "normal":
                cmd = self.lane_follower.get_command(img, center_bias=0.0)
            
            elif drive_mode == "after_crosswalk":
                cmd = self.lane_follower.get_command(img, center_bias=0.0)
            
            elif drive_mode == "in_loop":
                # Bias left to stay in loop
                cmd = self.lane_follower.get_command(img, center_bias=0.035)
            
            elif drive_mode == "exit_loop":
                # Bias left to take exit
                cmd = self.lane_follower.get_command(img, center_bias=-0.15)

            # elif drive_mode == "grass_road":
            #     cmd = self.lane_follower.get_command(img, center_bias=0.0)
            
            # Try to capture sign if we haven't yet in this phase
            if self.current_phase is not None and not self.sign_captured[self.current_phase]:
                cmd = self.try_capture_sign(img, cmd)
            
            # Track distance while in loop
            if drive_mode == "in_loop":
                self.update_loop_distance(cmd)
            else:
                # Reset distance when not in loop
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
        if self.loop_distance >= 14.0:
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
        """Align with sign and capture it"""
        now = rospy.Time.now()
        
        # Cooldown prevents re-capturing same sign
        if now < self.sign_cooldown_time:
            return cmd
        
        # Get sign color for current phase
        use_light_blue = (self.sign_colors[self.current_phase] == "light_blue")
        
        # Look for sign
        sign = self.sign_detector.find_sign(img, use_light_blue=use_light_blue)
        
        if sign is None:
            return cmd  # No sign visible, keep lane following
        
        # Sign detection takes over control
        width = sign["frame_width"]
        sign_x = sign["center_x"]
        sign_width = sign["bbox_width"]
        
        # Where is the sign in the frame? (0.0 = left, 0.5 = center, 1.0 = right)
        x_pos = sign_x / width
        size = sign_width / width
        
        # Stop forward motion while aligning
        cmd.linear.x = 0.0
        
        # Step 1: Center the sign horizontally
        if x_pos < 0.35:
            # Sign is left, turn left
            cmd.angular.z = 0.8
            return cmd
        
        if x_pos > 0.65:
            # Sign is right, turn right
            cmd.angular.z = -0.8
            return cmd
        
        # Step 2: Sign is centered - check if it's big enough
        if size < 0.2:
            # Too small, move forward slowly
            cmd.linear.x = 0.4
            cmd.angular.z = 0.0
            return cmd
        
        # Step 3: Sign is centered and big enough - capture it!
        warped = self.sign_detector.warp_sign(img, sign["quad"])

        if warped is not None:
            phase_name = f"sign_phase_{self.current_phase}"
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
        return cmd


if __name__ == "__main__":
    node = ControllerNode()
    rospy.spin()