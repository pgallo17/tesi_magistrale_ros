import pyaudio as pa
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
from nemo.core.classes import IterableDataset
import torch
from torch.utils.data import DataLoader
import numpy as np
from model import Model
import sys
import time
from commands import command_eng, command_ita
sys.path.append(r"/home/tesi_magistrale_ros/speech_ws/src/speech_pkg/src")

def infer_signal(model, signal):
    data_layer.set_signal(signal)
    batch = next(iter(data_loader))
    audio_signal, audio_signal_len = batch
    audio_signal, audio_signal_len = audio_signal.to(model.device), audio_signal_len.to(model.device)
    logits = model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
    return logits

class AudioDataLayer(IterableDataset):
    @property
    def output_types(self):
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(self, sample_rate):
        super().__init__()
        self._sample_rate = sample_rate
        self.output = True

    def __iter__(self):
        return self

    def __next__(self):
        if not self.output:
            raise StopIteration
        self.output = False
        return torch.as_tensor(self.signal, dtype=torch.float32), \
               torch.as_tensor(self.signal_shape, dtype=torch.int64)

    def set_signal(self, signal):
        self.signal = signal.astype(np.float32)
        self.signal_shape = self.signal.size
        self.output = True

    def __len__(self):
        return 1

class Microphone:
    def __init__(self):
        self.stream = self.open_stream()
        self._flag = True

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
            input_device_index=24, #TODO
            stream_callback=None,
        )

        return stream

    def loop(self):
        while True:
            audio_int16 = []
            print("Started Recording")
            for i in range(55):
                try:
                    audio_chunk = self.stream.read(FRAMES_PER_BUFFER)
                except OSError:
                    print("error stream")
                    time.sleep(2)
                    audio_chunk = self.stream.read(FRAMES_PER_BUFFER)
                audio_data = np.frombuffer(audio_chunk, np.int16)
                audio_int16.extend(audio_data)
            print("End recording")
            audio_int16 = np.array(audio_int16)
            audio_float32 = self.int2float(audio_int16)
            logits = infer_signal(model, audio_float32)
            soft = model.predict(logits)
            soft = soft.cpu().detach().numpy()
            cmd = np.argmax(soft, axis=1)
            cmd = cmd.tolist()[0]
            if cmd == 28:
                cmd_str = "I do not understand"
            else:
                cmd_str = commands_list[cmd]
            print("*"*30)
            print(cmd_str)
            print("*" * 30)
            self.stream.close()
            time.sleep(0.5)
            self.stream = self.open_stream()

if __name__ == "__main__":
    #CONSTANT
    lang = "ita"
    data_layer = AudioDataLayer(sample_rate=16000)
    data_loader = DataLoader(data_layer, batch_size=1, collate_fn=data_layer.collate_fn)
    commands_list = command_eng if lang == "eng" else command_ita
    frames_to_record = 1
    FORMAT = pa.paInt32
    CHANNELS = 1
    FRAMES_PER_BUFFER = 1024
    SR = 16000
    exp_dir = r"/home/tesi_magistrale_ros/speech_ws/src/speech_pkg/experiments/2022-01-21_17-18-42"
    ckpt = "matchcboxnet--val_loss=0.4191-epoch=249.model"
    model = Model.load_backup(ckpt, exp_dir)
    model = model.eval()
    model = model.cuda()
    mic = Microphone()
    mic.loop()