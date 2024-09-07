#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 13:14:48 2020

@author: iampraveenvemula
"""
# import the necessary packages
from webcamvideostream import WebcamVideoStream

class myVideoStream:
	def __init__(self, src=0, usePiCamera=False, resolution=(320, 240),
		framerate=32):
		# check to see if the picamera module should be used
		if usePiCamera:
			# only import the picamera packages unless we are
			# explicity told to do so -- this helps remove the
			# requirement of `picamera[array]` from desktops or
			# laptops that still want to use the `imutils` package
			from pivideostream import PiVideoStream
 
			# initialize the picamera stream and allow the camera
			# sensor to warmup
			self.stream = PiVideoStream(resolution=resolution,
				framerate=framerate)
 
		# otherwise, we are using OpenCV so initialize the webcam
		# stream
		else:
			self.stream = WebcamVideoStream(src=src)