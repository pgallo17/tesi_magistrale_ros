class BodyMotion:
    '''An interface for a device capable of moving.

    The methods to implement are:
    
    - **\_\_init\_\_(self)**: constructor that initialized the device
    - **rotate(roll, pitch, yaw, time_interval)**: rotate the device of the angle increments in the time interval specified
    '''

    def rotate(self, roll=0, pitch=0, yaw=0, time_interval=1):
        raise NotImplementedError