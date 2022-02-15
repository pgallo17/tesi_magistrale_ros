#!/usr/bin/python
# coding=utf-8
import rospy
# import qi
from commands import command_eng, command_ita
from speech_pkg.srv import *
import argparse
from lang_settings import AVAILABLE_LANGS
import sys

ENG = {
    0: "Command predicted:",
    1: "Probability of rejection:",
    2: "Top 3 classes predicted:"
}

ITA = {
    0: "Comando predetto:",
    1: "Probabilità di rigetto:",
    2: "Top 3 classi predette:"
}

class TextController:
    def __init__(self, lang):
        self.lang = lang
        self.db_lang = ENG if self.lang == "eng" else ITA

    def get_lang_string(self, index):
        return self.db_lang[index]

def get_command_str(index):
    return commands_list[index]

def get_bests(probs):
    assert len(command_eng) == len(command_ita)
    cmds = list(range(len(probs)))
    values_dict = dict(zip(cmds, probs))
    reject_key = len(command_eng)-1
    reject_prob = round(values_dict[reject_key], 3)
    # del values_dict[reject_key]
    values_list = list(values_dict.items())
    values_list.sort(key=lambda x: x[1], reverse=True)
    bests = values_list[:N_BEST_VALUES]
    bests = list(map(lambda x: (x[0], round(x[1], 3)), bests))
    return bests, reject_prob

def create_string(cmd, bests, reject_prob):
    out_str = ""
    out_str += text_controller.get_lang_string(0) + " " + get_command_str(cmd) + '\n'
    out_str += text_controller.get_lang_string(1) + " " + str(reject_prob) + '\n'
    out_str += text_controller.get_lang_string(2) + "\n"
    for cmd, prob in bests:
        out_str += "%s %s\n" % (str(prob), get_command_str(cmd))
    return out_str

def callback(req):
    bests, reject_prob = get_bests(req.probs)
    out_str = create_string(req.cmd, bests, reject_prob)
    with open("/home/files/res.txt", "a") as fil:
        fil.write("*"*30)
        fil.write(out_str)
        fil.write("*" * 30)
        fil.write("\n")
    print(out_str)
    # print(get_command_str(req.cmd))
    return TalkerResponse(True)

def init_dict():
    command_eng[len(command_eng)] = "I do not understand"
    command_ita[len(command_ita)] = "Non ho capito"

def connect_robot():
    # Connect to the robot
    print("Connecting to robot...")
    session = qi.Session()
    session.connect('tcp://10.0.1.230:9559')  # Robot IP
    print("Robot connected")

    #TextToSpeech service
    tts = session.service("ALTextToSpeech")
    tts.setLanguage("Italian" if args.lang == "ita" else "English")
    return tts

def say(out_str):
    tts.say(out_str)

if __name__ == "__main__":
    N_BEST_VALUES = 3
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, dest="lang", type=str)
    args, unknown = parser.parse_known_args(args=rospy.myargv(argv=sys.argv)[1:])
    if args.lang not in AVAILABLE_LANGS:
        raise Exception("Selected lang not available.\nAvailable langs:", AVAILABLE_LANGS)
    init_dict()
    text_controller = TextController(args.lang)
    rospy.init_node('talker')
    commands_list = command_eng if args.lang == "eng" else command_ita

    tts = connect_robot()

    rospy.Service('speech_service', Talker, callback)

    rospy.spin()