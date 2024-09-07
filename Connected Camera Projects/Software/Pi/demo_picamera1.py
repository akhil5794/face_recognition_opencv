#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 00:08:13 2020

@author: iampraveenvemula
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


#import imutils
from imutils.video import FPS
from picamera.array import PiRGBArray
from picamera import PiCamera

import os
#import time
from datetime import datetime
import warnings
warnings.simplefilter("ignore")

import pickle
import numpy as np
import cv2
#from playsound import playsound
import face_recognition

def load_known_emeddings():
    with open('savedModels/tf.pickle', 'rb') as f:
        known_embeddings = pickle.load(f)
    return known_embeddings

def stop_video(stream):
    # Release handle to the webcam
    cv2.destroyAllWindows()
    stream.close()
    rawCapture.close()
    camera.close()

def say_hello(name):
    fileName = "audioFiles/"+name.replace(" ", "_")+".mp3"
    cmd = 'mpg321 {}'.format(fileName)
    os.system(cmd)

#### Main Function ####

if __name__ == '__main__':
    
    # STEP-1: LOAD THE KNOWN FACE EMBEDDINGS
    
    print("[INFO] LOADING THE KNOWN FACE EMBEDDINGS...")

    known_face_encodings, known_face_names, face_time = load_known_emeddings()
    
    # STEP-2: START THE VIDEO STREAM

    print("[INFO] STARTING THE VIDEO...")
    
    camera = PiCamera()
    #camera.resolution = (320, 240)
    camera.framerate = 10
    rawCapture = PiRGBArray(camera)
    
    fps = FPS().start()
    
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        
        frame = np.copy(frame.array)
        
        scaling = 0.5
        global reverse_scaling 
        reverse_scaling= int(1/scaling)
        print('before', frame.shape)
        small_frame = cv2.resize(frame, (0, 0), fx=scaling, fy=scaling)
        print('after', small_frame.shape)
        
        face_locations,face_encodings, face_names = [],[],[]
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            name = "Unknown"
            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            distance = face_distances[best_match_index]
            if distance < 0.4:
                name = known_face_names[best_match_index]
                current_time = datetime.now()
                time_str_detected = face_time[name]
                diff_time = (current_time - time_str_detected).seconds
                if diff_time > 10: 
                    face_time[name] =  current_time
                    say_hello(name)
            face_names.append(name+" "+str(round(distance,4)))
            
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= reverse_scaling
            right *= reverse_scaling
            bottom *= reverse_scaling
            left *= reverse_scaling
    
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        rawCapture.truncate(0)
        
        fps.update()
    
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # STEP-7: STOP THE VIDEO FOR GRACEFUL EXIT
    print("[INFO] USER DECIDED TO CLOSE THE PROGRAM, CLEANING BEFORE CLOSE...")
    camera.close()
    
    
    
