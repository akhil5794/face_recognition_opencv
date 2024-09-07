#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import functools

import tensorflow as tf
import numpy as np
import cv2
import PIL.Image
import os


# In[2]:


def load_pb(path_to_pb):
    with tf.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


# In[3]:


load_pb('MobileFaceNet_9925_9680.pb')
    
inputs_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")


# In[4]:


def image_to_embedding(image, embeddings):
   
    image = cv2.resize(image, (112, 112), interpolation=cv2.INTER_CUBIC) 
    #image = cv2.resize(image, (112, 112), interpolation=cv2.INTER_AREA) 
    
    ### Image Thresholing added by Sourjeet
    #_,image = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ###
    
    #image = cv2.resize(image, (112, 112)) 
    img = image[...,::-1]
    x_train = np.array([img])
    x_train = (x_train-127.5)*0.0078125 ## normalizing image from -1 to 1
    with tf.Session() as sess:
        embed = sess.run(embeddings, feed_dict={inputs_placeholder:x_train})

    return sklearn.preprocessing.normalize(embed)


# In[31]:


def recognize_face(face_image, input_embeddings, embeddings):

    embedding = image_to_embedding(face_image, embeddings)
    
    minimum_distance = 200
    minimum_distance2 = 200
    name = None
    name2 = "None"
    best_distance = {}
    
    # Loop over  names and encodings.
    for input_name, input_embedding in input_embeddings:
        
        
        euclidean_distance = np.sqrt(((embedding-input_embedding)**2).sum())
        
        if ((input_name not in best_distance.keys()) or best_distance[input_name] > euclidean_distance):
            best_distance[input_name] = euclidean_distance

        #print('Euclidean distance from %s is %s' %(input_name, euclidean_distance))

    for (input_name, euclidean_distance) in best_distance.items():
        
        print('Euclidean distance from %s is %s' %(input_name, euclidean_distance))
        
        if euclidean_distance < minimum_distance:
            minimum_distance2 = minimum_distance
            if name != None:
                name2 = name
            minimum_distance = euclidean_distance
            name = input_name
    
    if minimum_distance < 1.1:
        min_dist = round(minimum_distance,4)
        min_dist2 = round(minimum_distance2,4)
        return_str = str(name)+" "+str(min_dist)
        return return_str
    else:
        min_dist = round(minimum_distance,4)
        return "Unknown_person"


# In[6]:


import glob
import sklearn.preprocessing


# In[7]:


def get_class(img_path):
    return str(img_path.split('/')[-2])


# In[23]:


def create_input_image_embeddings():
    ##input_embeddings = {}
    
    ##change by Sourjeet
    input_embeddings_all = []
    
    
    for file in glob.glob("New_images/*/*"):
        #print (file)
        
        person_name = get_class(file)
        image_file = cv2.imread(file, 1)
        
        ##change by Sourjeet
        input_embeddings_all.append([person_name,image_to_embedding(image_file, embeddings)])

    return input_embeddings_all


# In[22]:


def recognize_faces_in_cam(input_embeddings):
    

    cv2.namedWindow("Face Recognizer")
    vc = cv2.VideoCapture(0)
   

    font = cv2.FONT_HERSHEY_SIMPLEX
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    
    while vc.isOpened():
        _, frame = vc.read()
        img = frame
        height, width, channels = frame.shape

        
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Loop through all the faces detected 
        identities = []
        for (x, y, w, h) in faces:
            x1 = x
            y1 = y
            x2 = x+w
            y2 = y+h

            face_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]    
            identity = recognize_face(face_image, input_embeddings, embeddings)
            
            

            if identity is not None:
                img = cv2.rectangle(frame,(x1, y1),(x2, y2),(255,255,255),2)
                cv2.putText(img, str(identity), (x1+5,y1-5), font, 1, (255,255,255), 2)
        
        key = cv2.waitKey(10)
        cv2.imshow("Face Recognizer", img)


        
        if key == 27: # exit on ESC
            break    
    vc.release()
    cv2.destroyAllWindows()


# In[32]:


input_embeddings = create_input_image_embeddings()
#print(len(input_embeddings))


# In[34]:


recognize_faces_in_cam(input_embeddings)


# In[ ]:




