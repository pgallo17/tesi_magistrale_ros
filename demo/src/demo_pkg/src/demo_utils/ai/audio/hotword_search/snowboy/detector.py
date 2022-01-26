from .import snowboydecoder
from .import snowboydetect 
from .. import HotwordSearch
import threading
import os

TOP_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_FILE = os.path.join(TOP_DIR, "resources/common.res")

class SnowBoyHotwordSearch(HotwordSearch):
    '''SnowBoyHotwordSearch implements HotwordSearch through the use of the SnowBoy library.

    # Arguments
        model: str
            Model path: path to the hotword search model
        sensitivity: float
            Float between 0.0 and 1.0 representing the sensitivity of the hotword detector
        device_index: int
            Microphone index: index in the list speech_recognition.Microphone.list_microphone_names()
        source: object
            Audio source: None to use pyaudio. 
        
    For more details refer to the SnowBoy library docs.
    '''

    def __init__(self, model, sensitivity=0.5, device_index = None, source=None):
        self.model = model
        self.sensitivity = sensitivity
        self.device_index = device_index
        self.interrupted = False
        self.x = None
        self.source = source

        self.detector = None

    def start(self, callback):
        if self.detector is None:
            self.detector = snowboydecoder.HotwordDetector(self.model, sensitivity=self.sensitivity, device_index = self.device_index)
        
        if self.source is None:
            self.x = threading.Thread(target=self.detector.start, kwargs={
                "detected_callback":callback,
                "interrupt_check":lambda: self.interrupted,
                "sleep_time":0.03
            })
        else:
            self.x = threading.Thread(target=self.detector.start_with_source, kwargs={
                'source':self.source,
                "detected_callback":callback,
                "interrupt_check":lambda: self.interrupted,
                "sleep_time":0.03
            })

        self.x.start()
        
    def stop(self):
        self.interrupted = True

        if self.x is not None:
            self.x.join()

        self.detector.terminate()
