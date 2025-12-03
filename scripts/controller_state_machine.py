#!/usr/bin/env python3

import rospy

class StateMachine:
    """Manages high-level robot behavior through different track sections"""
    
    # State definitions
    STARTUP = 0
    BEFORE_CROSSWALK = 1
    AFTER_CROSSWALK = 2
    IN_LOOP = 3
    AFTER_LOOP = 4
    START_GRASS_ROAD = 5
    GRASS_BRIDGE = 6
    START_GRASS_NO_ROAD = 7  # Tunnel navigation
    CLIMBING_MOUNTAIN = 8    # Post-tunnel mountain climb
    
    def __init__(self):
        self.state = self.STARTUP
        self.timer_started = False
        self.frame_count = 0
    
    def update(self, events):
        """Update state based on events
        
        Input events:
            crossed_crosswalk: True when robot crosses red stripe
            sign1_captured: True when sign phase 1 is captured
            loop_complete: True when robot has driven one lap in loop
            crossed_first_pink: True when first pink marker crossed
            grass_sign_done: True when grass sign (phase 4) captured
            bridge_sign_done: True when bridge sign (phase 5) captured
            tunnel_exited: True when tunnel navigator reaches phase 6 (mountain)
        
        Returns actions dict:
            start_timer: True once at beginning
            drive_mode: "normal", "after_crosswalk", "in_loop", "exit_loop", "grass_road", "tunnel"
            sign_phase: 0-7 (which sign to look for)
        """
        self.frame_count += 1
        
        crossed_crosswalk = events.get("crossed_crosswalk", False)
        sign1_captured = events.get("sign1_captured", False)
        loop_complete = events.get("loop_complete", False)
        crossed_first_pink = events.get("crossed_first_pink", False)
        grass_sign_done = events.get("grass_sign_done", False)
        bridge_sign_done = events.get("bridge_sign_done", False)
        tunnel_exited = events.get("tunnel_exited", False)

        actions = {
            "start_timer": False,
            "drive_mode": "normal",
            "sign_phase": None
        }
        
        # ===== STARTUP =====
        if self.state == self.STARTUP:
            if not self.timer_started:
                actions["start_timer"] = True
                self.timer_started = True
            
            self._change_state(self.BEFORE_CROSSWALK)
        
        # ===== BEFORE CROSSWALK (initial straight) =====
        elif self.state == self.BEFORE_CROSSWALK:
            actions["drive_mode"] = "normal"
            actions["sign_phase"] = 0
            
            if crossed_crosswalk:
                self._change_state(self.AFTER_CROSSWALK)
        
        # ===== AFTER CROSSWALK (heading toward loop) =====
        elif self.state == self.AFTER_CROSSWALK:
            actions["drive_mode"] = "after_crosswalk"
            
            # Look for sign 1 first, then switch to sign 2 mode
            if not sign1_captured:
                actions["sign_phase"] = 1
            else:
                actions["sign_phase"] = 2
            
            # After enough frames, assume we've reached the loop
            if self.frame_count > 440:
                self._change_state(self.IN_LOOP)
        
        # ===== IN LOOP =====
        elif self.state == self.IN_LOOP:
            actions["drive_mode"] = "in_loop"
            actions["sign_phase"] = 2
            
            if loop_complete:
                self._change_state(self.AFTER_LOOP)
        
        # ===== AFTER LOOP =====
        elif self.state == self.AFTER_LOOP:
            actions["drive_mode"] = "exit_loop"
            actions["sign_phase"] = 3

            if crossed_first_pink:
                self._change_state(self.START_GRASS_ROAD)

        # ===== START OF THE GRASS ROAD =====
        elif self.state == self.START_GRASS_ROAD:
            actions["drive_mode"] = "grass_road"
            actions["sign_phase"] = 4

            # After capturing grass sign, continue to bridge
            if grass_sign_done:
                self._change_state(self.GRASS_BRIDGE)
        
        # ===== GRASS BRIDGE =====
        elif self.state == self.GRASS_BRIDGE:
            actions["drive_mode"] = "grass_road"
            actions["sign_phase"] = 5

            # After capturing bridge sign, START tunnel navigation
            if bridge_sign_done:
                self._change_state(self.START_GRASS_NO_ROAD)
        
        # ===== TUNNEL NAVIGATION =====
        elif self.state == self.START_GRASS_NO_ROAD:
            actions["drive_mode"] = "tunnel_navigation"
            actions["sign_phase"] = 6
            
            # When tunnel navigator indicates we are climbing mountain (phase 6+)
            if tunnel_exited:
                self._change_state(self.CLIMBING_MOUNTAIN)

        # ===== CLIMBING MOUNTAIN =====
        elif self.state == self.CLIMBING_MOUNTAIN:
            actions["drive_mode"] = "tunnel_navigation"
            actions["sign_phase"] = 7 # Looking for the FINAL sign
            
        return actions
    
    def _change_state(self, new_state):
        """Helper to change states and reset frame counter"""
        state_names = {
            0: "STARTUP",
            1: "BEFORE_CROSSWALK", 
            2: "AFTER_CROSSWALK",
            3: "IN_LOOP",
            4: "AFTER_LOOP",
            5: "START_GRASS_ROAD",
            6: "GRASS_BRIDGE",
            7: "START_GRASS_NO_ROAD (TUNNEL)",
            8: "CLIMBING_MOUNTAIN"
        }

        rospy.loginfo("="*70)
        rospy.loginfo(f"STATE CHANGE: {state_names.get(self.state, 'UNKNOWN')} -> {state_names.get(new_state, 'UNKNOWN')}")
        rospy.loginfo("="*70)
        self.state = new_state
        self.frame_count = 0