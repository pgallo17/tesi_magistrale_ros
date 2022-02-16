from torch.profiler import profile, record_function, ProfilerActivity
from model import Model
import numpy as np
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
from nemo.core.classes import IterableDataset
from torch.utils.data import DataLoader
from settings import global_utils
import torch
from pathlib import Path
import time

def infer_signal(model, signal):
    data_layer.set_signal(signal)
    batch = next(iter(data_loader))
    audio_signal, audio_signal_len = batch
    audio_signal, audio_signal_len = audio_signal.to(model.device), audio_signal_len.to(model.device)
    start_time = time.time()
    logits = model(input_signal=audio_signal, input_signal_length=audio_signal_len)
    end_time = time.time()
    latency.append(end_time-start_time)
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

class Classifier:
    def __init__(self, lang):
        self.model = self.load_model(lang)
        self.model = self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()

    def _pcm2float(self, sound: np.ndarray):
        abs_max = np.abs(sound).max()
        sound = sound.astype('float32')
        if abs_max > 0:
            sound *= 1 / abs_max
        sound = sound.squeeze()  # depends on the use case
        return sound

    def _numpy2tensor(self, signal: np.ndarray):
        signal_size = signal.size
        signal_torch = torch.as_tensor(signal, dtype=torch.float32)
        signal_size_torch = torch.as_tensor(signal_size, dtype=torch.int64)
        return signal_torch, signal_size_torch

    def convert(self, signal):
        signal = np.array(signal)
        signal_nw = self._pcm2float(signal)
        return signal_nw

    def predict_cmd(self, signal: np.ndarray):
        logits = infer_signal(self.model, signal)
        probs = self.model.predict(logits)
        probs = probs.cpu().detach().numpy()
        cmd = np.argmax(probs, axis=1)
        return cmd, probs


    def load_model(self, lang):
        base_path = Path(global_utils.get_curr_dir(__file__)).parent.joinpath("experiments")
        if lang == "eng":
            exp_dir = base_path.joinpath("2022-01-19_23-29-46")
            ckpt = r"matchcboxnet--val_loss=0.369-epoch=249.model"
        else:
            exp_dir = base_path.joinpath("2022-01-31_17-32-48")
            ckpt = r"matchcboxnet--val_loss=0.3033-epoch=220.model"
        model = Model.load_backup(exp_dir=exp_dir, ckpt_name=ckpt)
        print("loaded model lang:", lang)
        print("model loaded:", exp_dir)
        return model

if __name__ == "__main__":
    lang = "ita"
    classifier = Classifier(lang)
    sr = 16000
    duration = 2 #seconds
    num_samples = sr * duration
    audio = np.random.rand(num_samples)
    data_layer = AudioDataLayer(sample_rate=16000)
    data_loader = DataLoader(data_layer, batch_size=1, collate_fn=data_layer.collate_fn)
    latency = []
    for e in range(200):
        classifier.predict_cmd(audio)
    mean_value = np.array(latency).mean()
    print("Running  time:", mean_value)
