#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 14:56:09 2019

@author: sourjeetgupta
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import cv2

from playsound import playsound

import glob
import face_recognition

from datetime import datetime

import warnings
warnings.simplefilter("ignore")



def get_class(img_path):
    return str(img_path.split('/')[-2])


def create_input_image_embeddings():
    
    known_face_encodings = []
    known_face_names = []
    initial_timestamp = datetime.now()
    face_time = {}
    for file in glob.glob("../Faces/*/*"):
    
        person_name = get_class(file)
        image_file = face_recognition.load_image_file(file)
        known_face_encodings.append(face_recognition.face_encodings(image_file)[0])
        known_face_names.append(person_name)
    
        face_time[person_name] = initial_timestamp
    
        create_input_image_embeddings_out = [known_face_encodings,known_face_names,face_time]
        
         
    return create_input_image_embeddings_out

def recognize_faces_in_cam(known_face_names,known_face_encodings,face_time):
    

    cv2.namedWindow("Face Recognizer")
    video_capture = cv2.VideoCapture(0)
    
    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
   
    while True:
    # Grab a single frame of video
        ret, frame = video_capture.read()
        scaling = 0.5
        reverse_scaling = int(1/scaling)
        #print(reverse_scaling)

    # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=scaling, fy=scaling)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
        if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                #matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
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
                        fileName = "../audioFiles/"+name.replace(" ", "_")+".mp3"
                        playsound(fileName)

                face_names.append(name+" "+str(round(distance,4)))

        process_this_frame = not process_this_frame


        # Display the results
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

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    
    


create_input_image_embeddings_out = create_input_image_embeddings()
known_face_encodings = create_input_image_embeddings_out[0] 
known_face_names = create_input_image_embeddings_out[1] 
face_time = create_input_image_embeddings_out[2] 


recognize_faces_in_cam(known_face_names,known_face_encodings,face_time)