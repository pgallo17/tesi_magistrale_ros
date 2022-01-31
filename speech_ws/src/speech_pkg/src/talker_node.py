#!/usr/bin/python

import rospy
# import qi
from commands import command_eng, command_ita
from speech_pkg.srv import *
import argparse
from lang_settings import AVAILABLE_LANGS
import sys

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, dest="lang", type=str)
    args, unknown = parser.parse_known_args(args=rospy.myargv(argv=sys.argv)[1:])
    if args.lang not in AVAILABLE_LANGS:
        raise Exception("Selected lang not available.\nAvailable langs:", AVAILABLE_LANGS)
    init_dict()
    rospy.init_node('talker')
    commands_list = command_eng if args.lang == "eng" else command_ita

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