from vosk import Model, KaldiRecognizer, SpkModel
import numpy as np
import json

from . import SpeechAnalysis

class VoskSpeechAnalysis(SpeechAnalysis):
    '''VoskSpeechAnalysis implements SpeechAnalysis through the use of the vosk API library (CPU).

    This class extract a speaker embedding from the audio sample and the spoken text.

    # Arguments
        model: str
            Path to the vosk model.
        sample_rate: float 
            Sampling rate - `default 16000`

    For more details refer to the vosk library docs.
    '''

    def __init__(self, model, spk_model, sample_rate=16000):
        self.sample_rate = sample_rate 
        self.vosk_model = Model(model)
        self.spk_model = SpkModel(spk_model)
        self.rec = KaldiRecognizer(self.vosk_model, self.spk_model, self.sample_rate)

    def process(self, audio_data):

        #rec = KaldiRecognizer(self.vosk_model, self.sample_rate)
        #rec.SetSpkModel(self.spk_model)

        self.rec.AcceptWaveform(np.array(audio_data, dtype='int16'))
        finalRecognition = json.loads(self.rec.Result())

        data = {
            'text':finalRecognition['text']
        }

        emb = finalRecognition.get('spk', None)

        return emb, data
    
    def stop(self):
        pass