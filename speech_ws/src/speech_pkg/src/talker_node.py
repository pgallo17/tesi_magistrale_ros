#!/usr/bin/python
# coding=utf-8
import rospy
import qi
from commands import command_eng, command_ita
from speech_pkg.srv import *
import argparse
from lang_settings import AVAILABLE_LANGS
import sys
import time

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
    '''with open("/home/files/res.txt", "a") as fil:
        fil.write("*"*30)
        fil.write(out_str)
        fil.write("*" * 30)
        fil.write("\n")'''
    print(out_str)
    _,prob=bests[0]
    if prob > 0.8 :
        if req.cmd in range(20,26):
            say(get_command_str(req.cmd))
            move_wheels(req.cmd)    
        else :
            say(get_command_str(req.cmd))
    else:
        say(get_command_str(len(command_eng)-1))
    return TalkerResponse(True)

def init_dict():
    command_eng[len(command_eng)] = "I do not understand"
    command_ita[len(command_ita)] = "Comando non riconosciuto"

def connect_robot():
    # Connect to the robot
    print("Connecting to robot...")
    session = qi.Session()
    session.connect('tcp://%s:9559' % IP )  # Robot IP
    print("Robot connected")

    motion_service = session.service("ALMotion")
    motion_service.wakeUp()

    #TextToSpeech service
    tts = session.service("ALTextToSpeech")
    tts.setLanguage("Italian" if args.lang == "ita" else "English")
    tts.say("Ciao")
    return tts,motion_service

def move_head(cmd):
    names  = ["HeadPitch"]
    fractionMaxSpeed  = 0.2
    if cmd==23:
        angles=0.2
    elif cmd==22:
        angles=-0.1
    try:
        motion_service.setAngles("HeadPitch", angles, fractionMaxSpeed)
        time.sleep(2.0)
        motion_service.setStiffnesses("Head", 0.0)
    except Exception:
        session = qi.Session()
        session.connect('tcp://%s:9559' % IP )
        motion_service = session.service("ALMotion")
        motion_service.setStiffnesses("Head", 1.0)
        motion_service.setAngles("HeadPitch", angles, fractionMaxSpeed)
        time.sleep(2.0)
        motion_service.setStiffnesses("Head", 0.0)

def move_wheels(cmd):
    x  = 0
    y  = 0
    theta = 0
    if cmd==20 :
        y-=0.2
    elif cmd==21:
        y+=0.2
    elif cmd==24:
        x+=0.2
    elif cmd==25:
        x-=0.2
    try:
        motion_service.moveTo(x,y,theta)
    except Exception:
        session = qi.Session()
        session.connect('tcp://%s:9559' % IP )
        motion_service = session.service("ALMotion")
        motion_service.moveTo(x,y,theta)
    
def move_arm():
    names  = ["LShoulderPitch"]
    angles  = [0.3]
    fractionMaxSpeed  = 0.2
    try:
        motion_service.setStiffnesses("LArm", 1.0)
        motion_service.setAngles(names, angles, fractionMaxSpeed)
        time.sleep(3.0)
        print('change')
        motion_service.setAngles(names, 0.5, fractionMaxSpeed)
        time.sleep(2.0)
        motion_service.setAngles(names, 0.7, fractionMaxSpeed)
        time.sleep(2.0)
        motion_service.setStiffnesses("LArm", 0.0)     
    except Exception:
        session = qi.Session()
        session.connect('tcp://%s:9559' % IP )
        motion_service = session.service("ALMotion")
        motion_service.setStiffnesses("LArm", 1.0)
        motion_service.setAngles(names, angles, fractionMaxSpeed)
        time.sleep(3.0)
        print('change')
        motion_service.setAngles(names, 0.5, fractionMaxSpeed)
        time.sleep(2.0)
        motion_service.setAngles(names, 0.8, fractionMaxSpeed)
        motion_service.setStiffnesses("LArm", 0.0)
    

def say(out_str):
    try:
        tts.say(out_str)
    except Exception:
        session = qi.Session()
        session.connect('tcp://%s:9559' % IP )

        tts = session.service("ALTextToSpeech")
        tts.setLanguage("Italian" if args.lang == "ita" else "English")
        tts.say(out_str)
    # time.sleep(0.5)

if __name__ == "__main__":
    N_BEST_VALUES = 3
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, dest="lang", type=str)
    parser.add_argument("--ip", required=True, dest="ip", type=str)
    args, unknown = parser.parse_known_args(args=rospy.myargv(argv=sys.argv)[1:])
    IP = args.ip
    if args.lang not in AVAILABLE_LANGS:
        raise Exception("Selected lang not available.\nAvailable langs:", AVAILABLE_LANGS)
    init_dict()
    tts,motion_service = connect_robot()
    text_controller = TextController(args.lang)
    rospy.init_node('talker')
    commands_list = command_eng if args.lang == "eng" else command_ita


    rospy.Service('speech_service', Talker, callback)

    rospy.spin()