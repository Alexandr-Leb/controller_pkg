import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from lane_follower import LaneFollower



#list of states start timer, line follow 1, read clue, car part, dirt road, baby yoda...
class ControllerNode:
    def __init__(self):
        rospy.init_node('controller_node')

        self.bridge = CvBridge()
        self.lane_follower = LaneFollower()

        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        self.clock_sub = rospy.Subscriber('/clock', Clock, self.clock_callback)

        self.cmd_pub = rospy.Publisher('/b1cmd_vel', Twist, queue_size=10)
        self.score_pub = rospy.Publisher('/score', String, queue_size=10)

        self.state = 1  # Initial state

    def run(self):
        rospy.spin()

    def state_machine(self):
        if self.state == States.START_TIMER:
            self.send_start_timer()
            self.state = States.LINE_FOLLOW
            return
        elif self.state == States.LINE_FOLLOW:
            # State 2 behavior
            pass
        







    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        steering_cmd = self.lane_follower.process_image(cv_image)
        self.cmd_pub.publish(steering_cmd)

    def clock_callback(self, msg):
        sim_time = msg.clock.to_sec()

    def start_timer(self):
        msg = String()
        msg.data = "0"
        self.score_pub.publish(msg)

    def stop_timer(self):
        msg = String()
        msg.data = "1"
        self.score_pub.publish(msg)

    def send_clue(self, location, text):
        msg = String()
        msg.data = ""
        self.score_pub.publish(msg)

if __name__ == '__main__':
    controller_node = ControllerNode()
    rospy.spin()

class States:
    START_TIMER = 1
    LINE_FOLLOW = 2
    READ_CLUE = 3
    #tbd other states

    
