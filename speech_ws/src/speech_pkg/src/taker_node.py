import rospy
import qi
import commands
from settings import pepper

def get_command_str(index):
    tts.say(commands_list[index])

if __name__ == "__main__":
    lang: str = pepper.speech.language
    commands_list = commands.command_eng if lang.lower() == "eng" else commands.command_ita

    # Connect to the robot
    session = qi.Session()
    session.connect(f"tcp://{pepper.ip}:{pepper.port}")  # Robot IP

    # TextToSpeech service
    tts = session.service("ALTextToSpeech")

    rospy.spin()