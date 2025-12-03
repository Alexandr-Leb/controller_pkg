#!/usr/bin/env python3


class StateMachine:
    """Manages high-level robot behavior through different track sections"""
    
    # State definitions
    STARTUP = 0
    BEFORE_CROSSWALK = 1
    AFTER_CROSSWALK = 2
    IN_LOOP = 3
    AFTER_LOOP = 4
    START_GRASS_ROAD = 5
    
    def __init__(self):
        self.state = self.STARTUP
        self.timer_started = False
        self.frame_count = 0  # Counts frames in current state
    
    def update(self, events):
        """Update state based on events
        
        Input events:
            crossed_crosswalk: True when robot crosses second red stripe
            sign1_captured: True when sign phase 1 is captured
            loop_complete: True when robot has driven one lap in loop
        
        Returns actions dict:
            start_timer: True once at beginning
            drive_mode: "normal", "after_crosswalk", "in_loop", "exit_loop"
            sign_phase: 0, 1, 2, or 3 (which sign to look for)
        """
        self.frame_count += 1
        
        crossed_crosswalk = events.get("crossed_crosswalk", False)
        sign1_captured = events.get("sign1_captured", False)
        loop_complete = events.get("loop_complete", False)
        crossed_first_pink = events.get("crossed_first_pink", False)
        
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
            
            # Look for sign 1 first, then switch to sign 2 mode early
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
        
        return actions
    
    def _change_state(self, new_state):
        """Helper to change states and reset frame counter"""
        state_names = {
            0: "STARTUP",
            1: "BEFORE_CROSSWALK", 
            2: "AFTER_CROSSWALK",
            3: "IN_LOOP",
            4: "AFTER_LOOP",
            5: "START_GRASS_ROAD"
        }
        print(f"STATE CHANGE: {state_names.get(self.state, 'UNKNOWN')} -> {state_names.get(new_state, 'UNKNOWN')}")
        self.state = new_state
        self.frame_count = 0