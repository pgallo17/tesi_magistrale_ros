#!/usr/bin/python3

from model import Model
import numpy as np
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
from nemo.core.classes import IterableDataset
from torch.utils.data import DataLoader
from speech_pkg.srv import Classification, ClassificationResponse
from settings import pepper
import torch
import rospy

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

class Classifier:
    def __init__(self):
        self.model = self.load_model()
        self.model = self.model.eval()
        self.model = self.model.cuda()
        self.init_node()

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

    def parse_req(self, req):
        signal = self.convert(req.data.data)
        cmd, probs = self.predict_cmd(signal)
        assert len(cmd) == 1
        cmd = int(cmd[0])
        probs = probs.tolist()[0]
        return ClassificationResponse(cmd, probs)

    def init_node(self):
        rospy.init_node('classifier')
        s = rospy.Service('classifier_service', Classification, self.parse_req)
        rospy.spin()

    def load_model(self):
        if lang == "eng":
            exp_dir = r"/home/tesi_magistrale_ros/speech_ws/src/speech_pkg/experiments/2022-01-19_23-29-46"
            ckpt = r"matchcboxnet--val_loss=0.369-epoch=249.model"
            model = Model.load_backup(exp_dir=exp_dir, ckpt_name=ckpt)
        else:
            exp_dir = r"/home/tesi_magistrale_ros/speech_ws/src/speech_pkg/experiments/2022-01-21_17-18-42"
            ckpt = r"matchcboxnet--val_loss=0.4191-epoch=249.model"
            model = Model.load_backup(exp_dir=exp_dir, ckpt_name=ckpt)
        print("loaded model lang:", lang)
        print("model loaded:", exp_dir)
        return model

if __name__ == "__main__":

    data_layer = AudioDataLayer(sample_rate=16000)
    data_loader = DataLoader(data_layer, batch_size=1, collate_fn=data_layer.collate_fn)
    lang =  "eng" if pepper.speech.language.lower() == "english" else "ita"
    classifier = Classifier()