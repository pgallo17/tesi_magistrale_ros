from .respeaker import Tuning
import usb.core
import usb.util
from usb import USBError
import time

class ReSpeakerMicArrayV2:
    '''ReSpeakerMicArrayV2 implements the API to access to the ReSpeaker Mic Array V2.0 IoT microphone.

    The class implements the followings methods:

    - **\_\_init\_\_(self)**: constructor
    - **get_doa(self)**: returns a int representing the direction of arrival of the audio signal 
    - **get_vad(self)**: returns a bool representing if the audio signal is speech or not
    '''

    def __init__(self):
        self.dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
        if self.dev:
            self.mic_tuning = Tuning(self.dev)
        else:
            raise USBError

    def get_doa(self):
        return self.mic_tuning.direction
    
    def get_vad(self):
        return self.mic_tuning.is_voice()
