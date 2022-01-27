#!/usr/bin/python3

import rospy
# import qi
import commands
from settings import pepper
from speech_pkg.srv import *

def get_command_str(index):
    return commands_list[index]

def callback(req):
    cmd_str = get_command_str(req.cmd)
    print(cmd_str)
    return TalkerResponse(True)

if __name__ == "__main__":
    lang: str = pepper.speech.language
    commands_list = commands.command_eng if lang.lower() == "eng" else commands.command_ita

    # Connect to the robot
    # session = qi.Session()
    # session.connect(f"tcp://{pepper.ip}:{pepper.port}")  # Robot IP

    # TextToSpeech service
    # tts = session.service("ALTextToSpeech")

    rospy.Service('speech_service', Talker, callback)

    rospy.spin()