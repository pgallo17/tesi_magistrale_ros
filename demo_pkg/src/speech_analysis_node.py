#!/usr/bin/env python3
from demo_pkg.msg import SpeechData, SpeechInfo

from demo_utils.ai.audio.speech_analysis.vosk_api import VoskSpeechAnalysis
from settings import demo_settings

import rospy


class Callback:

    def __init__(self, publisher, analyzer):
        self.pub = publisher
        self.analyzer = analyzer
        
    def __call__(self, data):
        emb, meta = self.analyzer.process(data.data)

        if meta['text'] == '' or emb is None:
            return

        # Message preparing
        msg = SpeechInfo()
        msg.header.stamp = rospy.Time.now()
        msg.data = emb
        msg.meta = self.analyzer.serialize(meta)
        msg.start_time = data.start_time
        msg.end_time = data.end_time
        
        # Message publishing
        self.pub.publish(msg)

        rospy.logdebug('SpeechInfo published')

class SpeechAnalysisNode:
    '''SpeechAnalysisNode implements the ROS service interface for speech analysis.

    The node subscribes to the following topics:

    - **speech_detection** : SpeechData custom message.

    The node publishes on the following topics:

    - **speech_analysis** : SpeechInfo custom format.
    
    The available methods are:

    - **\_\_init\_\_(self)**: constructor
    - **start(self)**: starts the ros node instance
    '''

    def start(self):

        # Speech Analysis Initialization
        speechAnalysis = VoskSpeechAnalysis(
            demo_settings.ai.audio.speech_rec.model,
            demo_settings.ai.audio.speaker_identification.model,
            demo_settings.ai.audio.speech_rec.sample_rate
        )

        # Server Initialization
        rospy.init_node('speech_analysis_node', anonymous=True)
        pub = rospy.Publisher('speech_info', SpeechInfo, queue_size=3)
        
        # Callback Initialization
        callback = Callback(pub, speechAnalysis)

        # Subscriber initialization
        rospy.Subscriber("speech_detection", SpeechData, callback)

        rospy.spin()


if __name__ == "__main__":
    server = SpeechAnalysisNode()
    server.start()
