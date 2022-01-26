import sys
sys.path.append('../demo_utils')
from demo_utils import Singleton
import qi

class Pepper:
    '''A singleton class that represent a session connected to the pepper robot.

    # Arguments
        pepper_ip: string
            The Pepper network ip - `default "10.0.1.230"`
        pepper_port: int
            The Pepper network port - `default 9559`
        
    # Attributes
        session: qi.Session()
            The qi session connected to the robot. Must be used to the other classes that need to
            use Pepper functionalities.
    For more details refer to the qi and Pepper SDK library docs.
    '''
    __metaclass__ = Singleton

    def __init__(self, pepper_ip="10.0.1.207", pepper_port=9559):
        self.pepper_ip = pepper_ip
        self.pepper_port = pepper_port
        self.session = qi.Session()
        self.session.connect("tcp://{}:{}".format(pepper_ip, pepper_port))

    def rest(self):
        done = False
        try:
            self.session.service("ALMotion").rest()
            done=True
        except RuntimeError:
            self.__init__(self.pepper_ip, self.pepper_port)
        finally:
            if not done:
                self.session.service("ALMotion").rest()
    
    def wakeup(self):
        done = False
        try:
            self.session.service("ALMotion").wakeUp()
            done = True
        except RuntimeError:
            self.__init__(self.pepper_ip, self.pepper_port)
        finally:
            if not done:
                self.session.service("ALMotion").wakeUp()
    
    def say(self, sentence, animated=True, rspd=100, language="English"):
        done = False
        
        if rspd != 100:
            sentence = '\\rspd=%s\\'%rspd+sentence

        try:
            self.session.service("ALTextToSpeech").setLanguage(language)

            if animated:
                self.session.service("ALAnimatedSpeech").say(sentence)
            else:    
                self.session.service("ALTextToSpeech").say(sentence)
            
            done = True
        except RuntimeError:
            self.__init__(self.pepper_ip, self.pepper_port)
        finally:
            if not done:
                self.session.service("ALTextToSpeech").setLanguage(language)

                if animated:
                    self.session.service("ALAnimatedSpeech").say(sentence)
                else:    
                    self.session.service("ALTextToSpeech").say(sentence)

    def stand(self):
        done = False 
        try:
            self.session.service("ALRobotPosture").goToPosture("StandInit", 1.0)
            done = True
        except RuntimeError:
            self.__init__(self.pepper_ip, self.pepper_port)
        finally:
            if not done:
                self.session.service("ALRobotPosture").goToPosture("StandInit", 1.0)

    def reset_head_position(self):
        done = False
        try:
            self.session.service("ALMotion").angleInterpolationBezier(["HeadYaw", "HeadPitch"], [[0.7],  [0.7]], [[0], [-0.1]])
            done = True
        except RuntimeError:
            self.__init__(self.pepper_ip, self.pepper_port)
        finally:
            if not done:
                self.session.service("ALMotion").angleInterpolationBezier(["HeadYaw", "HeadPitch"], [[0.7],  [0.7]], [[0], [-0.1]])


    def following(self,toggle):
        done = False
        try:
            if toggle == True:
                awareness = self.session.service("ALAutonomousLife")
                awareness.setAutonomousAbilityEnabled("BasicAwareness",True)
                awareness.setAutonomousAbilityEnabled("ListeningMovement",True)
                awareness.setAutonomousAbilityEnabled("SpeakingMovement",True)
            else:
                self.session.service("ALAutonomousLife").setAutonomousAbilityEnabled("SpeakingMovement",False)
            done = True
        except RuntimeError:
            self.__init__(self.pepper_ip, self.pepper_port)
        finally:
            if not done:
                if toggle == True:
                    awareness = self.session.service("ALAutonomousLife")
                    awareness.setAutonomousAbilityEnabled("BasicAwareness",True)
                    awareness.setAutonomousAbilityEnabled("ListeningMovement",True)
                    awareness.setAutonomousAbilityEnabled("SpeakingMovement",True)
                else:
                    self.session.service("ALAutonomousLife").setAutonomousAbilityEnabled("SpeakingMovement",False)
    
    def get_session(self):
        return self.session