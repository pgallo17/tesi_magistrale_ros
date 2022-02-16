from utils import MySileroVad
import librosa
import numpy as np
from datetime import datetime

if __name__ == "__main__":
    th = 0.5
    duration = 2
    chunk = [480, 960, 1600]
    silero = MySileroVad(threshold=0.5, sampling_rate=16000)
    audio_data = librosa.tone(frequency=512, sr=16000, duration=duration)
    latency = []
    for ch in chunk:
        split = len(audio_data) // ch
        audio_data_split = np.array_split(audio_data, split)
        for i in range(200):
            start_time = datetime.now()
            for e in audio_data_split:
                silero.is_speech(e.tobytes())
            end_time = datetime.now()
            latency.append(end_time-start_time)
        mean_value = np.array(latency).mean()
        print("Chunk:", ch, "lantency:", mean_value)

