import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from layers import ZScoreNormalization, LogMelgramLayer
from MobileNetV3 import MobileNetV3_large
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, Activation, Input, Add, Lambda
import onnx
import onnxruntime


PARAMS = {
    'sample_rate': 16000,
    'stft_window_seconds': 0.025,
    'stft_hop_seconds': 0.010,
    'mel_bands': 64,
    'mel_min_hz': 125.0,
    'mel_max_hz': 7500.0,
}



onnx_model_ita = '/../../home/speech_ws/model_onnx/ita_model.onnx'
onnx_model_eng = '/../../home/speech_ws/model_onnx/eng_model.onnx'
session_ita=onnxruntime.InferenceSession(onnx_model_ita,None,providers=['CPUExecutionProvider'])
session_eng=onnxruntime.InferenceSession(onnx_model_eng,None,providers=['CPUExecutionProvider'])




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

assert num_fft // 2 + 1 == num_spec_bins
lin_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=num_mel_bins,
    num_spectrogram_bins=num_spec_bins,
    sample_rate=sr,
    lower_edge_hertz=f_min,
    upper_edge_hertz=f_max,)
        


def tf_log10(x):
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator


speech,_=librosa.load("ita_0.wav",sr=sr)
x=np.reshape(speech,(1,speech.shape[0],1))
# tf.signal.stft seems to be applied along the last axis
stfts = tf.signal.stft(
    x[:,:,0], frame_length=window_length, frame_step=hop_length, pad_end=pad_end
)
mag_stfts = tf.abs(stfts)

melgrams = tf.tensordot(tf.square(mag_stfts), lin_to_mel_matrix, axes=[2, 0])
log_melgrams = tf_log10(melgrams + log_offset)



axis=[1, 2]
eps=1e-07
mean_values = tf.math.reduce_mean(
            log_melgrams, axis=axis, keepdims=True)

dev_std = tf.math.reduce_std(
    log_melgrams, axis=axis, keepdims=True) + tf.constant(eps)
norm_tensor = (log_melgrams - mean_values)/dev_std

norm_tensor = tf.reshape(norm_tensor,(-1, norm_tensor.shape[2], 1))

norm_tensor = tf.reshape(norm_tensor,(1, norm_tensor.shape[0], norm_tensor.shape[1],norm_tensor.shape[2]))

session=tf.Session()
# This works
with session as sess:
  b=sess.run(norm_tensor) # ok because `sess.graph == graph`



input_name=session.get_inputs()[0].name
output_name=session.get_outputs()[1].name
result=session.run([output_name],{input_name:b})

print(result)
