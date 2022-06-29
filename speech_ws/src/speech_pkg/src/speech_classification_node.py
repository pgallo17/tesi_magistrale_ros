#!/usr/bin/python3

from model import ModelID
import numpy as np
from speech_pkg.srv import Classification, ClassificationResponse
from settings import pepper, global_utils
import torch
import rospy
import sys
from pathlib import Path
import argparse
import tensorflow as tf
from lang_settings import AVAILABLE_LANGS
import os


class Classifier:
    def __init__(self, lang):
        self.model = ModelID((None,1))
        
        base_path = Path(global_utils.get_curr_dir(__file__)).parent.joinpath("nosynt_cos_mean_75")
        #exp_dir = base_path.joinpath("distiller_ita_no_synt.h5")
        print((base_path+"distiller_ita_no_synt.h5")
        self.model.load_weights(base_path+"distiller_ita_no_synt.h5")
        #self.model = self.load_model(lang)
        #self.model = self.model.eval()
        '''if torch.cuda.is_available():
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()
        '''
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
        x=np.reshape(signal,(1,signal.shape[0],1))
        emb,y=self.model.predict(x)
        #print(y[0])
        l=[]
        for ele in y[0]:
            l.append("{:.13f}".format(float(ele)))
        yPredMax =  np.argmax(y)
        return yPredMax,l[yPredMax]
        

    '''def predict_cmd(self, signal: np.ndarray):
        logits = infer_signal(self.model, signal)
        probs = self.model.predict(logits)
        probs = probs.cpu().detach().numpy()
        REJECT_LABEL = probs.shape[1] - 1
        if probs[0, REJECT_LABEL] >= 0.004:
            cmd = np.array([REJECT_LABEL])
            print(cmd.shape)
        else:
            cmd = np.argmax(probs, axis=1)
        return cmd, probs
    '''

    def parse_req(self, req):
        signal = self.convert(req.data.data)
        cmd, probs = self.predict_cmd(signal)
        '''assert len(cmd) == 1
        cmd = int(cmd[0])
        probs = probs.tolist()[0]'''
        return ClassificationResponse(cmd, probs)

    def init_node(self):
        rospy.init_node('classifier')
        s = rospy.Service('classifier_service', Classification, self.parse_req)
        rospy.spin()

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, dest="lang", type=str)
    args, unknown = parser.parse_known_args(args=rospy.myargv(argv=sys.argv)[1:])
    if args.lang not in AVAILABLE_LANGS:
        raise Exception("Selected lang not available.\nAvailable langs:", AVAILABLE_LANGS)
    classifier = Classifier(args.lang)