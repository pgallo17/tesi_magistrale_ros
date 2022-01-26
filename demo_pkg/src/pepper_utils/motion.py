import sys
sys.path.append("../demo_utils")
from demo_utils.motion import BodyMotion

from settings import demo_settings

from . import Pepper
import math

class PepperBodyMotion(BodyMotion, object):
    '''PepperBodyMotion is an abstract class to move the Pepper robot.

    It is based on the qi python 2 module.

    # Attributes
        pepper: Pepper
            An instance of the singleton Pepper
        motion_service: Service
            A pointer to the ALMotion service
        
    For more details refer to the qi and Pepper SDK library docs.
    '''

    def __init__(self):
        self.pepper = Pepper(demo_settings.pepper.ip,demo_settings.pepper.port)
        self.pepper.wakeup()
        self.motion_service = self.pepper.session.service("ALMotion")

    def reset_position(self):
        self.pepper.reset_head_position()

    def get_head_angles(self):
        angles = self.motion_service.getAngles("Body", True)
        yaw = angles[0]
        pitch = angles[1]
        return pitch, yaw

class PepperPhysicalBodyMotion(PepperBodyMotion):
    '''PepperBodyPhysicalBodyMotion implements PepperBodyMotion for the physical movement.

    When the rotation is specified, the head rotates first. If the yaw exceed the threshold specified in the constructor,
    the whole body rotates to that angle and the head's yaw is reset.
    The rotation is relative to the current frame of Pepper's head.

    # Arguments
        yaw_threshold: float
            The maximum rotation angle in both directions.
        
    For more details refer to the qi and Pepper SDK library docs.
    '''

    def __init__(self, yaw_threshold):
        super(PepperPhysicalBodyMotion, self).__init__()
        self.yaw_threshold = yaw_threshold

    def rotate(self, roll=0, pitch=0, yaw=0, time_interval=1):  
        current_pitch, current_yaw = self.get_head_angles()
        new_pitch = current_pitch + math.radians(pitch)
        new_yaw = current_yaw + math.radians(yaw)
        if new_yaw < -self.yaw_threshold or new_yaw > self.yaw_threshold:
            rotation_speed = 0.05
            self.motion_service.setAngles(["HeadYaw", "HeadPitch"], [0, new_pitch], [rotation_speed, rotation_speed])
            self.motion_service.moveTo(0, 0, new_yaw, 1.5)
        else:
            self.motion_service.angleInterpolationBezier(["HeadYaw", "HeadPitch"], [[time_interval],  [time_interval]], [[new_yaw], [new_pitch]])

class PepperVisualBodyMotion(PepperBodyMotion):
    '''PepperVisualPhysicalBodyMotion implements PepperBodyMotion for the visual movement.

    When the rotation is specified, only the head rotates up to the maximum allowed by the joints.
        
    For more details refer to the qi and Pepper SDK library docs.
    '''

    def __init__(self):
        super(PepperVisualBodyMotion, self).__init__()

    def rotate(self, roll=0, pitch=0, yaw=0, time_interval=1):  
        current_pitch, current_yaw = self.get_head_angles()
        new_pitch = current_pitch + math.radians(pitch)
        new_yaw = current_yaw + math.radians(yaw)
        self.motion_service.setAngles(["HeadYaw", "HeadPitch"], [new_yaw, new_pitch], time_interval)
        # self.motion_service.angleInterpolationBezier(["HeadYaw", "HeadPitch"], [[time_interval],  [time_interval]], [[new_yaw], [new_pitch]])
        # self.motion_service.angleInterpolation(["HeadYaw", "HeadPitch"], [new_yaw, new_pitch], [time_interval,  time_interval], False)