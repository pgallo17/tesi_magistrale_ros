#!/usr/bin/python3
from demo_utils.io.audio import SpeechRecognitionVAD
from settings import demo_settings
from demo_utils.ai.audio.voice_activity_detector.silero_vad import SileroVAD
from speech_recognition import AudioSource
import numpy as np
import soundfile as sf
import rospy
from std_msgs.msg import Int16MultiArray, String
from demo_utils.ai.audio.voice_activity_detector import VoiceActivityDetector
import torch

class MySileroVad(VoiceActivityDetector):
    def __init__(self, threshold, sampling_rate):
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=True)
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
        new_confidence = self.model(torch.from_numpy(audio_float32), self.sampling_rate).item()
        return True if new_confidence > self.threshold else False

class ROSMicrophoneSource(AudioSource):

    def __init__(self, device_index=None, sample_rate=None, chunk_size=1024):
        self.device_index = device_index
        self.format = 8  # 16-bit int sampling
        self.SAMPLE_WIDTH = 2  # size in bytes of each sample (2 = 16 bit -> int16)
        self.SAMPLE_RATE = sample_rate  # sampling rate in Hertz
        self.CHUNK = chunk_size  # number of frames stored in each buffer

        self.audio = None
        self.stream = None

    def __enter__(self):
        self.stream = self.ROSAudioStream()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    class ROSAudioStream:

        def read(self, chunk):
            audio = rospy.wait_for_message('mic_data', Int16MultiArray)
            return np.array(audio.data, dtype=np.int16).tobytes()


silero = MySileroVad(threshold=0.5, sampling_rate=16000)

speechRecognition = SpeechRecognitionVAD(
            device_index = demo_settings.io.speech.device_index,
            sample_rate = 16000,
            chunk_size = demo_settings.io.speech.chunk_size,
            timeout = 1,
            phrase_time_limit = demo_settings.io.speech.phrase_time_limit,
            calibration_duration = demo_settings.io.speech.calibration_duration,
            format = demo_settings.io.speech.format,
            source=ROSMicrophoneSource(
                demo_settings.io.speech.device_index,
                16000,
                demo_settings.io.speech.chunk_size
            ),
            vad=silero
        )
rospy.init_node("silero")
# speechRecognition.calibrate()

i = 0
while True:
    print("Start")
    speech, timestamps = speechRecognition.get_speech_frame()
    print("speech:", speech, timestamps)
    print("i:", i)
    if speech is None:
        continue
    i += 1
    speech_save = np.reshape(speech.copy(), (-1, 1))
    sf.write(f"/home/files/{i}.wav", data=speech_save, samplerate=16000, format="WAV")

