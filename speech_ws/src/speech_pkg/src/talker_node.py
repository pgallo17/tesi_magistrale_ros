#!/usr/bin/python
# coding=utf-8
import qi
import argparse
import sys
import time
import rospy

def connect_robot(IP):
    # Connect to the robot
    print("Connecting to robot...")
    session = qi.Session()
    try:
        session.connect('tcp://%s:9559' % IP )  # Robot IP
        print("Robot connected")
    except RuntimeError:
        print ("Can't connect to Naoqi at ip \"" + IP +".\n"
               "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)

    

    #TextToSpeech service
    tts = session.service("ALTextToSpeech")
    tts.setLanguage("Italian" if args.lang == "ita" else "English")
    tts.say("Ciao")
    return session,tts



def main(session):
    """
    This example uses the setAngles method.
    """
    # Get the service ALMotion.

    motion_service  = session.service("ALMotion")
    navigation_service = session.service("ALNavigation")
    posture_service = session.service("ALRobotPosture")

    # Wake up robot
    motion_service.wakeUp()

    # Send robot to Pose Init
    #posture_service.goToPosture("StandInit", 0.5)

    motion_service.setStiffnesses("LArm", 1.0)
    #motion_service.moveInit()

    x  = 0
    y  = -0.5
    theta  = 0
    navigation_service.navigateTo(1.5, 0.0)

    '''# Example showing how to set angles, using a fraction of max speed
    names  = ["LShoulderPitch"]
    angles  = [0]
    fractionMaxSpeed  = 0.2
    motion_service.setAngles(names, angles, fractionMaxSpeed)
    
    '''
    print('before sleep')
    time.sleep(5.0)
    print('STOP')
    navigation_service.navigateTo(1.5, 0.0)
    time.sleep(5.0)
    print('STOP')
    
    #motion_service.setStiffnesses("LArm", 0.0)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--ip", required=True, dest="ip", type=str)
    parser.add_argument("--lang", required=True, dest="lang", type=str)
    args, unknown = parser.parse_known_args(args=rospy.myargv(argv=sys.argv)[1:])
    IP = args.ip 
    session,tts = connect_robot(IP)
    

    main(session)
    rospy.init_node('motion')
    rospy.spin()