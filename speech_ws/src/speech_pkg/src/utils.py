import numpy as np
from demo_utils.ai.audio.voice_activity_detector import VoiceActivityDetector
import torch
import soundfile as sf
import nemo
import nemo.collections.asr as nemo_asr

class MySileroVad(VoiceActivityDetector):
    def __init__(self, threshold, sampling_rate):
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        model = nemo_asr.models.EncDecClassificationModel.from_pretrained(model_name="MarbleNet-3x2x64")
        model = model.cuda()
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
        audio_float32 = self.int2float(audio_int16.copy())
        audio32_torch = torch.from_numpy(audio_float32).cuda()
        new_confidence = self.model(audio32_torch, self.sampling_rate).item()
        return True if new_confidence > self.threshold else False