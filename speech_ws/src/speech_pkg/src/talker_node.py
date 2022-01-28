#!/usr/bin/python

import rospy
# import qi
from commands import command_eng, command_ita
from speech_pkg.srv import *

def get_command_str(index):
    return commands_list[index]

def callback(req):
    cmd_str = get_command_str(req.cmd)
    with open("/home/files/res.txt", "a") as fil:
        fil.write(cmd_str)
        fil.write("\n")
    print(cmd_str)
    # tts.say(cmd_str)
    return TalkerResponse(True)

def init_dict():
    command_eng[len(command_eng)] = "I do not understand"
    command_ita[len(command_ita)] = "Non ho capito"

if __name__ == "__main__":
    init_dict()
    rospy.init_node('talker')
    lang = "ita"
    commands_list = command_eng if lang == "eng" else command_ita

    # Connect to the robot
    print("Connecting to robot...")
    # session = qi.Session()
    # session.connect('tcp://10.0.1.214:9559')  # Robot IP
    print("Robot connected")

    # TextToSpeech service
    # tts = session.service("ALTextToSpeech")
    # tts.setLanguage("Italian" if lang == "ita" else "English")

    rospy.Service('speech_service', Talker, callback)

    rospy.spin()