import numpy as np
import speech_recognition as sr
from speech_recognition import AudioData

from . import SpeechRecognition

class GoogleWebSpeechRecognition(SpeechRecognition):
    '''GoogleWebSpeechRecognition implements SpeechRecognition through the use of the speech_recognition library.

    Transcribe an audio signal into text using Google Web Speech API.

    # Arguments
        sample_rate: float 
            Sampling rate
        sample_width: float 
            Size in bytes of each sample
        Language: str 
            Language ID string (speech_recognition)

    For more details refer to the speech_recognition library docs.
    '''

    def __init__(self, sample_rate, sample_width, lang='it-IT'):
        self.r = sr.Recognizer() 
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.lang = lang


    def transcribe(self, audio_data):
        # Convert in convinient object
        audio = AudioData(np.array(audio_data, dtype='int16').tobytes(), self.sample_rate, self.sample_width)

        # Transcribe
        try:
            response = self.r.recognize_google(audio, language = self.lang)
        except sr.UnknownValueError:
            # Speech was unintelligible
            response = ""

        return response
    
    def stop(self):
        raise NotImplementedError