from utils import MySileroVad
import librosa
import numpy as np
import time
import random

if __name__ == "__main__":
    th = 0.5
    duration = 2
    chunk = [480, 960, 1600]
    silero = MySileroVad(threshold=0.5, sampling_rate=16000)

    latency = []
    for ch in chunk:
        for i in range(200):
            audio_data = librosa.tone(frequency=random.randint(100, 512), sr=16000, duration=duration)
            split = len(audio_data) // ch
            audio_data_split = np.array_split(audio_data, split)
            for e in audio_data_split:
                data = e.tobytes()
                start_time = time.time()
                silero.is_speech(data)
                end_time = time.time()
                latency.append(end_time-start_time)
        mean_value = np.array(latency).mean()
        print("Chunk:", ch, "\tlatency:", mean_value)

