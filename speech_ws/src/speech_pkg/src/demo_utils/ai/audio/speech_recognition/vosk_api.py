from vosk import Model, KaldiRecognizer
import numpy as np
import json

from . import SpeechRecognition

class VoskSpeechRecognition(SpeechRecognition):
    '''VoskSpeechRecognition implements SpeechRecognition through the use of the vosk API library (CPU).

    Transcribe an audio signal into text.

    # Arguments
        model: str
            Path to the vosk model.
        sample_rate: float 
            Sampling rate - `default 16000`

    For more details refer to the vosk library docs.
    '''

    def __init__(self, model, sample_rate=16000):
        self.sample_rate = sample_rate 
        self.vosk_model = Model(model)
        self.rec = KaldiRecognizer(self.vosk_model, self.sample_rate)

    def transcribe(self, audio_data):

        # Transcribe
        self.rec.AcceptWaveform(np.array(audio_data, dtype='int16'))
        finalRecognition = json.loads(self.rec.Result())

        return finalRecognition['text']
    
    def stop(self):
        pass