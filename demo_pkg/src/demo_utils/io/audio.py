from . import AudioSource, SpeechSource
from .speech_rec_mod import TimedRecognizer
import speech_recognition as sr
import pyaudio
import numpy as np

class PyAudioSource(AudioSource):
    '''PyAudioSource implements AudioSource through the use of the PyAudio library.

    Opens a continuous stream from the microphone through PyAudio.

    # Arguments
        device_index: int
            Microphone index: index in the list speech_recognition.Microphone.list_microphone_names()
        sample_rate: float 
            Sampling rate - `default 16000`
        frames_per_buffer: int
            Chunk size (in samples) - `default 1024`
        channels: int / list
            Number of channels - `default 1`
        format: str
            Audio array dtype: 'int16' or 'float32' - `default 'int16'`
        
    For more details refer to the PyAudio library docs.
    '''
    def __init__(self,device_index=None, sample_rate=16000, channels=1, frames_per_buffer=1024, format='int16'):
        
        self.device_index=device_index
        self.sample_rate=sample_rate
        self.channels=channels
        self.frames_per_buffer=frames_per_buffer
        self.format=format

        # Create an instance of PyAudio
        self.p = pyaudio.PyAudio()

        # Open the audio stream
        self.stream = self.p.open(      
            output=False,  
            rate=self.sample_rate,					   
            format=pyaudio.paInt16 if format == 'int16' else pyaudio.paFloat32,		
            channels=self.channels,				        
            input=True,				        
            input_device_index=self.device_index,	        
            frames_per_buffer=self.frames_per_buffer,	
            stream_callback=None		    
        )
    
    def get_audio_frame(self):
        audio_data = self.stream.read(self.frames_per_buffer, exception_on_overflow=False)
        audio_array = np.frombuffer(audio_data, dtype=np.int16 if self.format == 'int16' else np.float32)
        
        return audio_array


    def stop(self):
        self.stream.close()
        self.p.terminate()



class SpeechRecognitionVAD(SpeechSource):
    '''SpeechRecognitionVAD implements SpeechSource through the use of the SpeechRecognition and PyAudio libraries.

    Extracts audio chunks that contain speech.

    # Arguments
        device_index: int
            Microphone index: index in the list speech_recognition.Microphone.list_microphone_names()
        sample_rate: float 
            Sampling rate - `default None`
        chunk_size: int
            Chunk size (in samples) - `default 1024`
        timeout: float
            Timeout for an input phrase - `default None`
        phrase_time_limit: float
            Input phrase limit - `default None`
        calibration_duration: float
            Time needed to perform the vad calibration - `default 1`
        format: str
            Audio array dtype: 'int16' or 'float32' - `default 'int16'`
        source: speech_recognition.AudioSource
            Audio source: None to use speech_recognition.Microphone. 
        
    For more details refer to the SpeechRecognition and PyAudio libraries docs.
    '''
    def __init__(self, device_index=None, sample_rate=None, chunk_size=1024, timeout=None, phrase_time_limit=None, calibration_duration=1, format='int16', source=None, vad=None):
        self.sr = TimedRecognizer()
        if source is None:
            self.mic = sr.Microphone(device_index, sample_rate, chunk_size)
        else:
            self.mic = source 
            
        self.timeout = timeout
        self.phrase_time_limit = phrase_time_limit
        self.calibration_duration = calibration_duration
        self.format = format
        self.vad = vad

    def get_speech_frame(self):
        # wait for a phrase
        try :
            with self.mic as s:
                speech_data,timestamps = self.sr.listen_timestamp(source=s,timeout=self.timeout, phrase_time_limit=self.phrase_time_limit, vad=self.vad) 
        except sr.WaitTimeoutError:
            return None, (None,None)

        # bytes to samples
        speech_data = np.frombuffer(speech_data.get_raw_data(), dtype=np.int16)

        # conversion if needed
        if self.format == 'float32':
            speech_data = self.pcm2float(speech_data)
        
        return speech_data, timestamps

    def calibrate(self):
        with self.mic as source:
            self.sr.adjust_for_ambient_noise(source, self.calibration_duration)  

    def stop(self):
        pass
