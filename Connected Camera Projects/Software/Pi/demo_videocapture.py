#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:23:33 2020

@author: iampraveenvemula
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


#import imutils
from imutils.video import FPS
#from imutils.video import VideoStream
#from picamera.array import PiRGBArray
#from picamera import PiCamera
#import time
from datetime import datetime
import warnings
warnings.simplefilter("ignore")

import os
import pickle
import numpy as np
import cv2
#from playsound import playsound
import face_recognition

from webcamvideostream import WebcamVideoStream

def load_known_emeddings():
    with open('savedModels/tf.pickle', 'rb') as f:
        known_embeddings = pickle.load(f)
    return known_embeddings

def say_hello_pi(name):
    fileName = "audioFiles/"+name.replace(" ", "_")+".mp3"
    cmd = 'mpg321 {}'.format(fileName)
    os.system(cmd)

print("[INFO] LOADING THE KNOWN FACE EMBEDDINGS...")
    
#global known_face_encodings, known_face_names, face_time

known_face_encodings, known_face_names, face_time = load_known_emeddings()
    
print("[INFO] STARTING THE VIDEO...")
    
vs = WebcamVideoStream(src=0).start()
fps = FPS().start()


import time

start_time = time.time()

while True:
    frame = vs.read()
    #cv2.imshow('Video', frame)
    scaling = 0.5
    reverse_scaling = int(1/scaling)
    print('length of the frame before scaling {}'.format(frame.shape))
    frame = cv2.resize(frame, (0, 0), fx=scaling, fy=scaling)
    print('after {}'.format(frame.shape))
    #cv2.imshow('Video', frame)
    end_time = time.time()
    time_diff = end_time - start_time
        
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        distance = face_distances[best_match_index]
        if distance < 0.4:
            name = known_face_names[best_match_index]
            print(name)
            current_time = datetime.now()
            time_str_detected = face_time[name]
            diff_time = (current_time - time_str_detected).seconds
            if diff_time > 10: 
                face_time[name] =  current_time
                say_hello_pi(name)

        face_names.append(name+" "+str(round(distance,4)))

        #reverse_scaling = 1
        #Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        #Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= reverse_scaling
        right *= reverse_scaling
        bottom *= reverse_scaling
        left *= reverse_scaling
        #print('left, top, right, bottom', left, top, right, bottom)
        #Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    fps.update()

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
