import numpy as np
from demo_utils.ai.audio.voice_activity_detector import VoiceActivityDetector
import torch
import soundfile as sf

class MySileroVad(VoiceActivityDetector):
    def __init__(self, threshold, sampling_rate):
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=True)
        # model = model.cuda()
        self.model = model

    def int2float(self, sound):
        abs_max = np.abs(sound).max()
        sound = sound.astype('float32')
        if abs_max > 0:
            sound *= 1 / abs_max
        sound = sound.squeeze()  # depends on the use case
        return sound

    def is_speech(self, buffer):
        audio_int16 = np.frombuffer(buffer, dtype=np.int16)

        print(audio_int16)
        speech_save = np.reshape(audio_int16.copy(), (-1, 1))
        print(speech_save)
        sf.write(f"/home/files/before.wav", data=speech_save, samplerate=16000, format="WAV")

        audio_float32 = self.int2float(audio_int16.copy())

        speech_save = np.reshape(audio_int16.copy(), (-1, 1))
        sf.write(f"/home/files/after.wav", data=speech_save, samplerate=16000, format="WAV")

        new_confidence = self.model(torch.from_numpy(audio_float32), self.sampling_rate).item()
        return True if new_confidence > self.threshold else False