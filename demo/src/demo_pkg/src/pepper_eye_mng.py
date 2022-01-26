#!/usr/bin/env python
from pepper_utils import Pepper
from settings import demo_settings
from std_msgs.msg import String
import rospy
import time

class PepperEyeManagementNode:

    def __init__(self):
        # E(nabled)/D(isabled)/W(ait)/S(earching)
        self.state = 'W'
        self.dialoging = False 
    
    def callback(self,data):
        if data.data == 'VAD/Disabled' and self.dialoging:
            self.state = 'D'
        elif data.data == 'VAD/Enabled' and self.dialoging:
            self.state = 'E'
        elif data.data == 'FieraMain/NotDialoging':
            self.dialoging = False 
        elif data.data == 'FieraMain/Dialoging':
            self.dialoging = True 
        elif data.data == 'HandRightBackTouched':
            self.state = 'W'
        elif data.data == 'HandLeftBackTouched':
            self.state = 'D'
    
    def start(self):
        # Server Initialization
        rospy.init_node('pepper_eye_mng')

        pepper_robot = Pepper(demo_settings.pepper.ip,demo_settings.pepper.port)
        led_mng = pepper_robot.get_session().service("ALLeds")

        #led_mng.randomEyes(duration)
        rate = rospy.Rate(1)

        # Subscriber initialization
        rospy.Subscriber("events", String, self.callback)

        while not rospy.is_shutdown():

            # rospy.loginfo(("Eye state: (%s,%s)"%(self.state,self.dialoging)))

            if self.dialoging:
                color = 0x000000FF if self.state == 'E' else 0x0000FF00
            else:
                color = 0x00FFFF00 if self.state == 'W' else 0x00FFA500

            duration = 1
            rotation_duration = 1
            led_mng.rotateEyes(color, rotation_duration, duration)

            rate.sleep()

        rospy.spin()

if __name__ == "__main__":
    try:
        node = PepperEyeManagementNode()
        node.start()
    except rospy.ROSInterruptException:
        pass