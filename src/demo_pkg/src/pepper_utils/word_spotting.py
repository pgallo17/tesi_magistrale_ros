import sys
sys.path.append('../demo_utils')
from demo_utils.ai.audio.hotword_search import HotwordSearch

from settings import demo_settings

from . import Pepper

class PepperHotwordSearch():
    '''PepperHotwordSearch implements HotwordSearch through the use of the libqi library.

    # Arguments
        vocabulary: list[str]
            List of words to recognize
        sensitivity: float
            Float between 0.0 and 1.0 representing the sensitivity of the hotword detector
        language: str
            Language of the words within the vocabulary.
        service_name: str
            Name of the qi service. 
        
    For more details refer to the libqi library docs.
    '''

    def __init__(self, vocabulary=['pepper'], sensitivity=0.4, language='Italian', service_name="kws_service"):
        self.pepper = Pepper(demo_settings.pepper.ip,demo_settings.pepper.port)
        self.mem = self.pepper.session.service("ALMemory")

        self.asr_service = self.pepper.session.service("ALSpeechRecognition")
        self.asr_service.setVisualExpression(False)
        self.asr_service.pause(True)
        self.asr_service.setLanguage(language)
        self.asr_service.setParameter("Sensitivity", sensitivity)

        try:
            self.asr_service.setVocabulary(vocabulary, True)
        except RuntimeError as error:
            self.asr_service.removeAllContext()
            self.asr_service.setVocabulary(vocabulary, True)
        
        self.service_name=service_name
        self.asr_service.subscribe(service_name)

        self.asr_service.pause(False)


    def start(self,callback):
        self.sub = self.mem.subscriber('WordRecognized')
        self.link = self.sub.signal.connect(callback)
        
    def stop(self):
        self.sub.signal.disconnect(self.link)
        self.asr_service.unsubscribe(self.service_name)