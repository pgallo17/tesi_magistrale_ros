class SpeechRecognition:
    '''Class defining the interface of a SpeechRecognition object.

    The methods to implement are:
    
    - **\_\_init\_\_(self)**: constructor that takes as input the module settings ( model path, sample rate, ... )
    - **transcribe(self, audio_data)**: transcribe an audio signal into text 
    - **stop(self)**: this function close open models and sessions if needed
    '''

    def __init__(self):
        pass

    def transcribe(self, audio_data):
        raise NotImplementedError
    
    def stop(self):
        raise NotImplementedError
