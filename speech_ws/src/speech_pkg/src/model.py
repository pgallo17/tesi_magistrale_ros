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
import datetime


tf.compat.v1.enable_eager_execution()

PARAMS = {
    'sample_rate': 16000,
    'stft_window_seconds': 0.025,
    'stft_hop_seconds': 0.010,
    'mel_bands': 64,
    'mel_min_hz': 125.0,
    'mel_max_hz': 7500.0,
}

'''#create dataset.txt
path='/../../home/speech_ws/Dataset_real_conv/'
txt=open('dataset.txt','a')
for elem in os.listdir(path):
    isDirectory = os.path.isdir(path+elem)
    if isDirectory:
        for e in os.listdir(path+elem):
            isDirectory = os.path.isdir(path+elem+'/'+e)
            if isDirectory:
                for sound in os.listdir(path+elem+'/'+e):
                    wav=sound.split('.')
                    if wav[1] == 'wav':
                        txt.write(path+elem+'/'+e+'/'+sound+'\n')
                    

'''


onnx_model_ita = '/../../home/speech_ws/model_onnx/ita_model.onnx'
onnx_model_eng = '/../../home/speech_ws/model_onnx/eng_model.onnx'
session_ita=onnxruntime.InferenceSession(onnx_model_ita,None,providers=['CPUExecutionProvider'])
session_eng=onnxruntime.InferenceSession(onnx_model_eng,None,providers=['CPUExecutionProvider'])
input_name=session_eng.get_inputs()[0].name
output_name=session_eng.get_outputs()[1].name

def tf_log10(x):
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator




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
lin_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=num_mel_bins,
    num_spectrogram_bins=num_spec_bins,
    sample_rate=sr,
    lower_edge_hertz=f_min,
    upper_edge_hertz=f_max,)
        
seconds=[]


path='/../../home/speech_ws/Dataset_real_conv/'
txt=open('dataset.txt','r')
i=0

for elem in txt:
    i+=1
    wav=elem[:-1]
    speech,_=librosa.load(wav,sr=sr)
    x=np.reshape(speech,(1,speech.shape[0],1))
    # tf.signal.stft seems to be applied along the last axis
    second_before=datetime.datetime.now()
    stfts = tf.signal.stft(
        x[:,:,0], frame_length=window_length, frame_step=hop_length, pad_end=pad_end
    )
    mag_stfts = tf.abs(stfts)

    melgrams = tf.tensordot(tf.square(mag_stfts), lin_to_mel_matrix, axes=[2, 0])
    log_melgrams = tf_log10(melgrams + log_offset)
    
    mean_values = tf.math.reduce_mean(
                log_melgrams, axis=axis, keepdims=True)

    dev_std = tf.math.reduce_std(
        log_melgrams, axis=axis, keepdims=True) + tf.constant(eps)
    norm_tensor = (log_melgrams - mean_values)/dev_std

    norm_tensor = tf.reshape(norm_tensor,(-1, norm_tensor.shape[2], 1))

    norm_tensor = tf.reshape(norm_tensor,(1, norm_tensor.shape[0], norm_tensor.shape[1],norm_tensor.shape[2]))
    
    '''     
    session=tf.Session()
    with session as sess:
        tensor=sess.run(norm_tensor) # ok because `sess.graph == graph`
    '''
    #pre_inference=datetime.datetime.now()
    result_ita=session_eng.run([output_name],{input_name:norm_tensor.numpy()})
    second_after=datetime.datetime.now()
    #delta=second_after-pre_inference
    #print(delta.total_seconds())
    delta=second_after-second_before
    if i!=1:
        seconds.append(delta.total_seconds())
    print('iterazione',str(i))
    if i==1001:
        break



def Average(lst):
    return sum(lst) / len(lst)

average = Average(seconds)

print("Average of the list =", average)

    
    









