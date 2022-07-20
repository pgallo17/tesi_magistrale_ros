#!/usr/bin/python3


import numpy as np
from speech_pkg.srv import Classification, ClassificationResponse
from settings import pepper, global_utils
from std_msgs.msg import String,Int8,Float32MultiArray 
import rospy
import sys
from pathlib import Path
import argparse
import tensorflow as tf
from lang_settings import AVAILABLE_LANGS
import os
import onnxruntime

PARAMS = {
    'sample_rate': 16000,
    'stft_window_seconds': 0.025,
    'stft_hop_seconds': 0.010,
    'mel_bands': 64,
    'mel_min_hz': 125.0,
    'mel_max_hz': 7500.0,
}



window_length_samples = int(
        round(PARAMS['sample_rate'] * PARAMS['stft_window_seconds']))
hop_length_samples = int(
        round(PARAMS['sample_rate'] * PARAMS['stft_hop_seconds']))
fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
num_spectrogram_bins = fft_length // 2 + 1
num_fft=512
window_length=window_length_samples
hop_length=hop_length_samples
sr=PARAMS['sample_rate']
num_mel_bins=PARAMS['mel_bands']
num_spec_bins=num_spectrogram_bins
f_min=PARAMS['mel_min_hz']
f_max=PARAMS['mel_max_hz']
log_offset=0.001
pad_end = True

axis=[1, 2]
eps=1e-07

assert num_fft // 2 + 1 == num_spec_bins

        



class Classifier:
    def __init__(self, lang):
        #self.model = ModelID((None,1))
        
        '''base_path = Path(global_utils.get_curr_dir(__file__)).parent.joinpath("nosynt_cos_mean_75")
        exp_dir = base_path.joinpath("distiller_ita_no_synt.h5")
        print(exp_dir)'''
        onnx_model = '/../../home/speech_ws/model_onnx/ita_model.onnx'
        self.session=onnxruntime.InferenceSession(onnx_model,None,providers=['CPUExecutionProvider'])
        self.input_name=self.session.get_inputs()[0].name
        self.output_name=self.session.get_outputs()[1].name
        
        self.init_node()

    def _pcm2float(self, sound: np.ndarray):
        abs_max = np.abs(sound).max()
        sound = sound.astype('float32')
        if abs_max > 0:
            sound *= 1 / abs_max
        sound = sound.squeeze()  # depends on the use case
        return sound

    '''def _numpy2tensor(self, signal: np.ndarray):
        signal_size = signal.size
        signal_torch = torch.as_tensor(signal, dtype=torch.float32)
        signal_size_torch = torch.as_tensor(signal_size, dtype=torch.int64)
        return signal_torch, signal_size_torch'''

    def convert(self, signal):
        signal = np.array(signal)
        signal_nw = self._pcm2float(signal)
        return signal_nw

    
    def tf_log10(self,x):
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator


    def predict_cmd(self, signal: np.ndarray):
        x=np.reshape(signal,(1,signal.shape[0],1))
        print('classification----------------------------')

        # tf.signal.stft seems to be applied along the last axis
        session=tf.Session()
        with session as sess:
            lin_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins=num_mel_bins,num_spectrogram_bins=num_spec_bins,
            sample_rate=sr,lower_edge_hertz=f_min,upper_edge_hertz=f_max,)            
            stfts = tf.signal.stft(
                x[:,:,0], frame_length=window_length, frame_step=hop_length, pad_end=pad_end
            )
            mag_stfts = tf.abs(stfts)
       
            melgrams = tf.tensordot(tf.square(mag_stfts), lin_to_mel_matrix, axes=[2, 0])
           
            log_melgrams = self.tf_log10(melgrams + log_offset)


            mean_values = tf.math.reduce_mean(
                        log_melgrams, axis=axis, keepdims=True)

            dev_std = tf.math.reduce_std(
                log_melgrams, axis=axis, keepdims=True) + tf.constant(eps)
        
            norm_tensor = (log_melgrams - mean_values)/dev_std

            norm_tensor = tf.reshape(norm_tensor,(-1, norm_tensor.shape[2], 1))

            norm_tensor = tf.reshape(norm_tensor,(1, norm_tensor.shape[0], norm_tensor.shape[1],norm_tensor.shape[2]))
            
            norm_tensor=sess.run(norm_tensor) 
        
        result=self.session.run([self.output_name],{self.input_name:norm_tensor})

        
        '''l=[]
        for ele in result[0]:
            l.append("{:.13f}".format(float(ele)))'''
        

        yPredMax =  np.argmax(result)
        
        l=result[0].flatten().tolist()
        
        print(yPredMax,type(yPredMax),type(result))

        
        return yPredMax.item(),l
        


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