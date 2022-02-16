import time

from utils import MySileroVad
import numpy as np

if __name__ == "__main__":
    th = 0.5
    chunk = [480, 960, 1600]
    silero = MySileroVad(threshold=0.5, sampling_rate=16000)

    for ch in chunk:
        latency= []
        audio_data = np.random.rand(ch)
        for i in range(200):
            data = audio_data.tobytes()
            start_time = time.time()
            silero.is_speech(data)
            end_time = time.time()
            latency.append(end_time-start_time)
        mean_value = np.array(latency).mean()
        print("Chunk:", ch, "\tlatency:", mean_value)

