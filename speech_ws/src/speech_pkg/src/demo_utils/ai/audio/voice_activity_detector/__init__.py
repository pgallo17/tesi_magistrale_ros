
class VoiceActivityDetector:
    '''Class defining the interface of a SpeechAnalysis object.

    The methods to implement are:
    
    - **\_\_init\_\_(self)**: constructor that takes as input the module settings ( model path, sample rate, ... )
    - **process(self, audio_data)**: process an audio signal to extract information (age, gender, embedding) 
    - **stop(self)**: this function close open models and sessions if needed

    Utils method (already implemented) are:

    - **serialize(self, data)**: serialize a dictionary (json)
    '''

    def __init__(self):
        pass
        
    def is_speech(self, buffer):

        raise NotImplementedError