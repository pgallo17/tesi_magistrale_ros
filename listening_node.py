import pyaudio as pa
# import rospy
from matplotlib import pyplot as plt
import torch
import numpy as np

class Microphone:
    def __init__(self):
        self.stream = self.open_stream()
        self._flag = True
        self.model, self.utils = self.load_model()

    @property
    def flag(self):
        return self._flag
    @flag.setter
    def flag(self, value):
        assert value == True or value == False
        self._flag = value
    def _set_flag(self):
        self.flag = False

    def int2float(self, sound):
        abs_max = np.abs(sound).max()
        sound = sound.astype('float32')
        if abs_max > 0:
            sound *= 1 / abs_max
        sound = sound.squeeze()  # depends on the use case
        return sound

    def open_stream(self):
        p = pa.PyAudio()

        stream = p.open(
            rate=SR,
            format=pa.paInt16,
            channels=1,
            input=True,
            input_device_index=0, #TODO
            stream_callback=None,
        )

        return stream

    def loop(self):
        data = []
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = self.utils
        vad_iterator = VADIterator(self.model)
        print("Started Recording")
        while True:
            audio_chunk = self.stream.read(FRAMES_PER_BUFFER)
            data.append(audio_chunk)
            audio_int16 = np.frombuffer(audio_chunk, np.int16)
            audio_float32 = self.int2float(audio_int16)
            speech_dict = vad_iterator(audio_float32, return_seconds=False)
            if speech_dict:
                print(speech_dict)

        print("Stopped the recording")
        self.flag = True
        audio_int16 = np.frombuffer(audio_chunk, np.int16)
        audio_float32 = self.int2float(audio_int16)
        return audio_int16, audio_float32

    def load_model(self):
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=True)
        return model, utils

if __name__ == "__main__":
    #CONSTANT
    frames_to_record = 1
    FORMAT = pa.paInt32
    CHANNELS = 1
    FRAMES_PER_BUFFER = 1024
    SR = 16000
    mic = Microphone()
    mic.loop()