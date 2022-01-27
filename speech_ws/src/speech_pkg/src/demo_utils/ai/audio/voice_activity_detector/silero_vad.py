from . import VoiceActivityDetector
import torch
from typing import List
import torch.nn.functional as F
import warnings
import numpy as np


class SileroVAD(VoiceActivityDetector):
    def __init__(self,
                    model_path,
                    threshold: float = 0.5,
                    sampling_rate: int = 16000,
                    device = 'cpu'
                ):
        torch.set_grad_enabled(False)
        self.model = torch.jit.load(model_path, map_location=torch.device(device))
        self.model.eval()

        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.device = device

    def is_speech(self, buffer):

        # Byte to numpy
        audio = np.frombuffer(buffer, dtype='int16')
        # int16 to foat32
        audio = audio.astype(np.float32, order='C') / 32768.0

        if not torch.is_tensor(audio):
            try:
                audio = torch.Tensor(audio)
            except:
                raise TypeError("Audio cannot be casted to tensor. Cast it manually")

        if len(audio.shape) > 1:
            for i in range(len(audio.shape)):  # trying to squeeze empty dimensions
                audio = audio.squeeze(0)
            if len(audio.shape) > 1:
                raise ValueError("More than one dimension in audio. Are you trying to process audio with 2 channels?")

        audio = audio.to(self.device)
        self.model.reset_states()
        speech_prob = self.model(audio, self.sampling_rate).item()

        if speech_prob >= self.threshold:
            return True
        else:
            return False