#!/usr/bin/env python
from pepper_utils import Pepper
from settings import demo_settings
import rospy

if __name__ == "__main__":
    try:
        pepper_robot = Pepper(demo_settings.pepper.ip,demo_settings.pepper.port)
        pepper_robot.wakeup()
        pepper_robot.stand()
    except rospy.ROSInterruptException:
        pass