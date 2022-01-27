#!/usr/bin/python3
import rospy
from std_msgs.msg import Int16MultiArray

from demo_utils.io.audio import PyAudioSource
from settings import demo_settings

class MicrophoneNode:
    '''MicrophoneNode implements the ROS interface for the microphone acquisition.

    The node has not subscription to any topic.

    The node publishes on the following topics:

    - **mic_data** : Int16MultiArray

    The available methods are:
    
    - **\_\_init\_\_(self)**: constructor
    - **start(self)**: starts the ros node instance
    '''

    def start(self):
        # Node and publisher initialization
        pub = rospy.Publisher('mic_data', Int16MultiArray, queue_size=3)
        rospy.init_node('microphone_node')

        # Stream initialization
        audio_stream = PyAudioSource(
            device_index = demo_settings.io.mic.device_index,
            sample_rate = demo_settings.io.mic.sample_rate,
            channels = demo_settings.io.mic.channels,
            frames_per_buffer = demo_settings.io.mic.frames_per_buffer,
            format = demo_settings.io.mic.format
        )

        while not rospy.is_shutdown():
            # Get data
            audio_frame = audio_stream.get_audio_frame()
            
            # Message preparation
            msg = Int16MultiArray()
            msg.data = audio_frame

            # Message publishing
            pub.publish(msg)

        # Close the stream
        audio_stream.stop() 

if __name__ == '__main__':
    microphone = MicrophoneNode()
    microphone.start()
    