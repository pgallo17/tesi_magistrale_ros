from dotmap import DotMap
io = DotMap()

################## STORE DATA ##################

io.store.store = False
io.store.root = "/media/nvidia/SSD"
io.store.audio_folder = "audio"
io.store.txt_folder = "text"
io.store.video_folder = "video"
io.store.video_len_secs = 1800 # 30 min
io.store.video_fps = 3

################## MICROPHONE ##################

io.mic.device_index =24 # ReSpeaker 4 Mic Array (UAC1.0): USB Audio (hw:2,0)
io.mic.sample_rate = 16000
io.mic.channels = 1 # Mono - 2 for Stereo
io.mic.frames_per_buffer = 1600
io.mic.format = 'int16'

#################### SPEECH ####################

io.speech.device_index=24  # ReSpeaker 4 Mic Array (UAC1.0): USB Audio (hw:2,0)
io.speech.sample_rate = 16000
io.speech.chunk_size = 960
io.speech.timeout = 10 # seconds
io.speech.phrase_time_limit = 5 # seconds
io.speech.calibration_duration = 2
# io.speech.calibration_event = "FieraMain/TextSpokenDone" 
io.speech.calibration_event = 'LeftBumperPressed'
io.speech.format = 'int16' # 'float32'

#################### CAMERA ####################

io.camera.device = 0 # 0. Realsense - 1. Pepper
io.camera.fps = 6
io.camera.pepper.resolution = 0 #0. 160x120 - 1. 320x240 - 2. 640x480 - 3. 1280x960
io.camera.pepper.camera = 0 #0. Top - 1. Bottom
io.camera.realsense.width = 640
io.camera.realsense.height = 480

#################### WEBVIEW ####################
io.webview.ip = "10.0.1.210"
io.webview.port = 5000