#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 19:52:21 2020

@author: iampraveenvemula
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import os
import time
import sys 
import glob
import face_recognition

from datetime import datetime

import warnings
warnings.simplefilter("ignore")

known_face_encodings = []
known_face_names = []
initial_timestamp = datetime.now()
face_time = {}
input_image_embeddings = []

print('='*85)

for file_path in glob.glob("trainedFaces/*/*"):
    person_name = str(file_path.split('/')[-2])
    print('[INFO] Generating Embeddings for: {}, Image Path: {}'.format(person_name, file_path))
    try:
        image_file = face_recognition.load_image_file(file_path)
        known_face_encodings.append(face_recognition.face_encodings(image_file)[0])
        known_face_names.append(person_name)
        face_time[person_name] = initial_timestamp
        input_image_embeddings = [known_face_encodings,known_face_names,face_time]
    except:
        print('Could not Save the Embedding for the file : {}'.format(file_path))

print('='*85)

# Save the embeddings to a pickle file

print('[INFO] Saving Embeddings as a Pickle File...')

timestr = time.strftime("%Y%m%d_%H%M%S")

pkl_file = 'savedModels/trained_face_embeddings_'+timestr+'.pickle'

try:
    with open(pkl_file, 'wb') as f:
        pickle.dump(input_image_embeddings, f)
    print('[SUCCESS] File saved as: {}'.format(os.path.abspath(pkl_file)))     
except:
    print('[FAIL] File couldnot be saved, Error below:')
    e = sys.exc_info()[0]
    print(e)

print('='*85)
