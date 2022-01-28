#!/usr/bin/python3
from demo_utils.io.audio import SpeechRecognitionVAD
from settings import demo_settings
from demo_utils.ai.audio.voice_activity_detector.silero_vad import SileroVAD
from speech_recognition import AudioSource
import numpy as np
import soundfile as sf
import rospy
from std_msgs.msg import Int16MultiArray, String

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
            return np.array(audio.data, dtype='int16').tobytes()

silero = SileroVAD(
            demo_settings.ai.audio.vad.model,
            demo_settings.ai.audio.vad.threshold,
            demo_settings.ai.audio.vad.sampling_rate,
            demo_settings.ai.audio.vad.device
        )

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
speechRecognition.calibrate()

i = 0
while True:
    speech, timestamps = speechRecognition.get_speech_frame()
    print("speech:", speech, timestamps)
    print("i:", i)
    if speech is None:
        continue
    i += 1
    speech_save = np.reshape(speech.copy(), (-1, 1))
    sf.write(f"/home/files/{i}.wav", data=speech_save, samplerate=16000, format="WAV")

