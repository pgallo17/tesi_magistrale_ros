
class HotwordSearch():
    '''Class defining the interface of a HotwordSearch object.

    The methods to implement are:
    
    - **\_\_init\_\_(self)**: constructor that takes as input the source settings (audio_source, hotword model, ... )
    - **start(self,callback)**: calls the callback every time the hotwork is detected 
    - **stop(self)**: this function close open streams
    '''

    def __init__(self):
        pass

    def start(self,callback):
        raise NotImplementedError
    
    def stop(self):
        raise NotImplementedError
