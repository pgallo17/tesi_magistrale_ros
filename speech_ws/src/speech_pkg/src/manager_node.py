#!/usr/bin/python3

import rospy
from speech_pkg.srv import *

def run(req):
    rospy.loginfo(req)
    cmd, probs = classify(req)
    res = speech(cmd, probs)
    return ManagerResponse(res)

if __name__ == "__main__":
    rospy.init_node('manager')
    rospy.wait_for_service('classifier_service')
    rospy.wait_for_service('speech_service')
    rospy.Service('manager_service', Manager, run)
    classify = rospy.ServiceProxy('classifier_service', Classification)
    speech = rospy.ServiceProxy('speech_service', Talker)
    rospy.spin()