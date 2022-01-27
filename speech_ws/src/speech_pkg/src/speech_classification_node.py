from model import Model
import numpy as np
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from speech_pkg.srv import Classification, ClassificationResponse
import torch
import rospy

class Classifier:
    def __init__(self, exp_dir, ckpt):
        self.model = Model.load_backup(ckpt, exp_dir)
        self.model = self.model.eval()
        self.model = self.model.cuda()
        self.init_node()

    def _pcm2float(self, sig: np.ndarray):
        return AudioSegment._convert_samples_to_float32(sig)

    def _numpy2tensor(self, signal: np.ndarray):
        signal_size = signal.size
        signal_torch = torch.as_tensor(signal, dtype=torch.float32)
        signal_size_torch = torch.as_tensor(signal_size, dtype=torch.int64)
        return signal_torch, signal_size_torch

    def convert(self, signal: np.ndarray):
        signal_nw = self._pcm2float(signal)
        signal_nw, signal_len = self._numpy2tensor(signal_nw)
        return signal_nw, signal_len

    def predict_cmd(self, signal: torch.Tensor, signal_len: torch.Tensor):
        logits = self.model(input_signal=signal, input_signal_len=signal_len)
        probs = self.model.predict(logits)
        probs = probs.cpu().detach().numpy()
        cmd = np.argmax(probs, axis=1)
        return cmd, probs

    def parse_req(self, req):
        signal, signal_len = self.convert(req.data)
        cmd, probs = self.predict_cmd(signal, signal_len)
        return ClassificationResponse(cmd, probs)

    def init_node(self):
        rospy.init_node('classifier')
        s = rospy.Service('add_two_ints', Classification, self.parse_req)
        rospy.spin()

if __name__ == "__main__":
    exp_dir = r"C:\MIE CARTELLE\PROGRAMMAZIONE\GITHUB\tesi_magistrale\nemo_experiments\MatchboxNet-3x2x64\2022-01-19_23-29-46"
    ckpt = r"matchcboxnet--val_loss=0.369-epoch=249.model"
    classifier = Classifier(exp_dir, ckpt)