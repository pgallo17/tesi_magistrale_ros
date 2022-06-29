import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from layers import ZScoreNormalization, LogMelgramLayer
from MobileNetV3 import MobileNetV3_large

PARAMS = {
    'sample_rate': 16000,
    'stft_window_seconds': 0.025,
    'stft_hop_seconds': 0.010,
    'mel_bands': 64,
    'mel_min_hz': 125.0,
    'mel_max_hz': 7500.0,
}

def ModelID(input_shape):

    # Input
    inputs = Input(shape=input_shape)

    window_length_samples = int(
        round(PARAMS['sample_rate'] * PARAMS['stft_window_seconds']))
    hop_length_samples = int(
        round(PARAMS['sample_rate'] * PARAMS['stft_hop_seconds']))
    fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
    num_spectrogram_bins = fft_length // 2 + 1

    
    x = tf.keras.layers.Lambda(lambda x: x / tf.math.reduce_max(x,-2,keepdims=True))(inputs)
   
    # Mel Spectrogram
    x = LogMelgramLayer(
        num_fft=512,
        window_length=window_length_samples,
        hop_length=hop_length_samples,
        sr=PARAMS['sample_rate'],
        mel_bins=PARAMS['mel_bands'],
        spec_bins=num_spectrogram_bins,
        fmin=PARAMS['mel_min_hz'],
        fmax=PARAMS['mel_max_hz']
    )(x)

    # Normalize along coeffients and time
    x = ZScoreNormalization(axis=[1, 2])(x)
    
    x = tf.keras.layers.Reshape((-1, x.shape[2], 1))(x)
   
    # Backbone
    input_shape = (None, PARAMS['mel_bands'],1)
    backbone = MobileNetV3_large(input_shape=input_shape, input_tensor=None, num_classes=29, include_top=True, pooling='avg', weights=None)    
    #y_emb,y_class = backbone(x)
    y_class = backbone(x)


    # Final Model
    model = Model(inputs=inputs, outputs=y_class)

    return model
