#!/usr/bin/python

import rospy
import qi
from commands import command_eng, command_ita
from settings import pepper
from speech_pkg.srv import *

def get_command_str(index):
    return commands_list[index]

def callback(req):
    cmd_str = get_command_str(req.cmd)
    tts.say(cmd_str)
    return TalkerResponse(True)

def init_dict():
    command_eng[len(command_eng)] = "I do not understand"
    command_ita[len(command_ita)] = "Non ho capito"

if __name__ == "__main__":
    init_dict()
    rospy.init_node('talker')
    lang = pepper.speech.language.lower()
    commands_list = command_eng if lang == "eng" else command_ita

    # Connect to the robot
    print("Connecting to robot...")
    session = qi.Session()
    session.connect('tcp://%s:%s' % (pepper.ip, str(pepper.port)))  # Robot IP
    print("Robot connected")

    # TextToSpeech service
    tts = session.service("ALTextToSpeech")

    rospy.Service('speech_service', Talker, callback)

    rospy.spin()