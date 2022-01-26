#!/usr/bin/env python
import sys
sys.path.append('../demo_utils')
from demo_utils.io import ImageSource

# from io import ImageSource
import numpy as np

TOP_CAMERA = 0
BOTTOM_CAMERA = 1
DEPTH_CAMERA = 2

RES_120P = 0
RES_240P = 1
RES_480P = 2
RES_960P = 3

COLORSPACE_GRAYSCALE = 8
COLORSPACE_RGB = 13

MODE_RGB = 0
MODE_DEPTH = 1
MODE_RGBD = 2

class PepperImageSource(ImageSource):
    '''PepperImageSource implements ImageSource to get images from the robot Pepper.

    It is based on the qi python 2 module

    # Arguments
        pepper: Pepper
            An instance of the singleton Pepper
        resolution: int
            An int flag that defines the resolution of the camera. 
            0. 160x120
            1. 320x240
            2. 640x480
            3. 1280x960 - `default 2`
        rgb_camera: int
            An int flag to select which RGB camera to use.
            0. Top camera
            1. Bottom camera - `default 0`
        fps: int
            The number of frames per  second - `default 20`
        
    For more details refer to the qi and Pepper SDK library docs.
    '''

    def __init__(self, pepper, resolution=RES_480P, rgb_camera=TOP_CAMERA, fps=20):
        # if mode == MODE_RGBD or mode == MODE_DEPTH:
        #    assert resolution == RES_120P
        if resolution == RES_120P:
            self.width, self.height = 160, 120
        elif resolution == RES_240P:
            self.width, self.height = 320, 240
        elif resolution == RES_480P:
            self.width, self.height = 640, 480
        elif resolution == RES_960P:
            self.width, self.height = 1280, 960
        else:
            self.width, self.height = None, None
        self.camera = pepper.session.service("ALVideoDevice")
        self.rgb_sub = self.camera.subscribeCamera("RGB Stream", rgb_camera, resolution, COLORSPACE_RGB, fps)
        self.depth_sub = self.camera.subscribeCamera("Depth Stream", DEPTH_CAMERA, -1, COLORSPACE_GRAYSCALE, fps)
        if not self.rgb_sub or not self.depth_sub:
            raise Exception("Camera is not initialized properly")

    def get_color_frame(self):
        raw_rgb = self.camera.getImageRemote(self.rgb_sub)
        image = np.frombuffer(raw_rgb[6], np.uint8).reshape(raw_rgb[1], raw_rgb[0], 3)
        return image
    
    def get_depth_frame(self):
        raw_depth = self.camera.getImageRemote(self.depth_sub)
        image = np.frombuffer(raw_depth[6], np.uint8).reshape(raw_depth[1], raw_depth[0], 1)
        return image
    
    def get_rgbd_frame(self):
        rgb = self.get_color_frame()
        depth = self.get_depth_frame()
        image = np.dstack((rgb, depth))
        return image

    def get_fov(self, mode="RGB"):
        hfov, vfov = 0, 0
        if mode == "RGB":
            hfov = 57.2 * np.pi / 180
            vfov = 44.3 * np.pi / 180
        elif mode == "DEPTH":
            hfov = 58 * np.pi / 180
            vfov = 45 * np.pi / 180
        return hfov, vfov
    
    def stop(self):
        self.camera.unsubscribe(self.rgb_sub)
        self.camera.unsubscribe(self.depth_sub)