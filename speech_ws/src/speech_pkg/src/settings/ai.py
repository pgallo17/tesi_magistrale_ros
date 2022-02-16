from dotmap import DotMap
from .global_utils import get_curr_dir
from pathlib import Path

ai = DotMap()

###########################################
################## AUDIO ##################
###########################################


################# HOTWORD #################

ai.audio.hotword.model = '/home/nvidia/DemoFramework/demo_ws/src/demo_pkg/models/audio/hotword_search/pepper.pmdl'
ai.audio.hotword.sensitivity = 0.
ai.audio.hotword.language = 'Italian'

################ SPEECH REC ################

ai.audio.speech_rec.model = '/home/nvidia/DemoFramework/demo_ws/src/demo_pkg/models/audio/speech_recognition/vosk-model-small-it-0.4'
# ai.audio.speech_rec.model = '/home/nvidia/DemoFramework/demo_ws/src/demo_pkg/models/audio/speech_recognition/vosk-model-small-en-us-0.15'
ai.audio.speech_rec.sample_rate = 16000
ai.audio.speech_rec.sample_width = 2
ai.audio.speech_rec.lang = 'it-IT'
ai.audio.speech_rec.timeout = 5

############### SPEAKER REC ################

ai.audio.speaker_identification.model = '/home/nvidia/DemoFramework/demo_ws/src/demo_pkg/models/audio/speaker_identification/vosk-model-spk-0.4'

############### VOICE ACTIV ################
ai.audio.vad.model = Path(get_curr_dir(__file__)).parent.parent.joinpath("vad", "silero_vad.jit")
ai.audio.vad.threshold = 0.15
ai.audio.vad.sampling_rate = 16000
ai.audio.vad.device = 'cuda'

###########################################
################## VIDEO ##################
###########################################


############## FACE ANALYSIS ##############
ai.video.face_analysis.multitask_model_index = 1 # 0. MobileNet_A - 1. MobileNet_B - 2.  MobileNet_C - 3. ResNet_A - 4. ResNet_B - 5. ResNet_C - 6. SeNet_A - 7. SeNet_B - 8. SeNet_C
ai.video.face_analysis.multitask_model_allow_growth = None # Default False
ai.video.face_analysis.multitask_gpu_max_mem = None # Default 0.3
ai.video.face_analysis.reidentification_save_faces = False
ai.video.face_analysis.reidentification_allow_growth = None
ai.video.face_analysis.reidentification_gpu_max_mem = None
ai.video.face_analysis.target_fps = 3
ai.video.face_analysis.moving_average.length = 5


############## FACE DETECTION ##############
ai.video.face_detection.confidence_threshold = 0.65
ai.video.face_detection.size_threshold = None
ai.video.face_detection.number_faces = 3
ai.video.face_detection.batch_size = 2

############## REIDENTIFICATION ##############
ai.video.reidentification.db_folder = "../models/reidentification/face_embeddings.pkl"
ai.video.reidentification.emb_size = 1024
ai.video.reidentification.th = 0.75
ai.audio.reidentification.db_folder = "../models/reidentification/voice_embeddings.pkl"
ai.audio.reidentification.emb_size = 128
ai.audio.reidentification.th = 0.87

################## TRACKING ##################
ai.tracking.reset_interval = 15

################## LANGUAGE ##################
ai.language.spacy_model = "it_core_news_lg"