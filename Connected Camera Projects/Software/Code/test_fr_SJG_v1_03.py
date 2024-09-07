#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from collections import namedtuple
#import functools

import tensorflow as tf
import numpy as np
import cv2
#import PIL.Image
#import os

from gtts import gTTS
#import pyttsx3;
#from io import BytesIO

from playsound import playsound



from datetime import datetime

from operator import itemgetter 

import warnings
warnings.simplefilter("ignore")


# In[2]:


from scipy.spatial import distance


# In[3]:


def load_pb(path_to_pb):
    with tf.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


# In[4]:


load_pb('MobileFaceNet_9925_9680.pb')
    
inputs_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")


# In[5]:


def image_to_embedding(image, embeddings):
   
    #image = cv2.resize(image, (112, 112), interpolation=cv2.INTER_CUBIC) 
    image = cv2.resize(image, (112, 112), interpolation=cv2.INTER_AREA) 
    
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


# In[18]:


def recognize_face(face_image, input_embeddings, embeddings, face_time):

    embedding = image_to_embedding(face_image, embeddings)
    #pd.DataFrame(embedding).to_csv("TestImageEmbedding.csv")
    
    ## USe This 
    ## sorted(student_tuples, key=itemgetter(2), reverse=True)
    index = 0
    # Loop over  names and encodings.
    for _, img_embedding, _ in input_embeddings:
        
        
        
        #input_embeddings[index][2] = np.sqrt(((embedding-img_embedding)**2).sum())
        #input_embeddings[index][3] = distance.cosine(embedding,img_embedding)
        input_embeddings[index][2] = distance.cosine(embedding,img_embedding)
        index = index + 1

    sorted_distance = sorted(input_embeddings, key=itemgetter(2), reverse=False)
    #pd.DataFrame(input_embeddings).to_csv("image_details.csv")
    
    return_str = "Unknown_person"
    
    #for i in list(range(5)):
    #    print('Cosine distance from %s is %s' %(sorted_distance[i][0], sorted_distance[i][2]))
    #    print('-----------------------------------------------')
    
    print('Cosine distance from %s is %s' %(sorted_distance[0][0], sorted_distance[0][2]))
        
    #print('###############################################')
    print('-----------------------------------------------')
    
    if sorted_distance[0][2] < 0.4:
        return_str = str(sorted_distance[0][0])   ##+"|"+str(sorted_distance[0][3])
        current_time = datetime.now()
        time_str_detected = face_time[return_str]
        diff_time = (current_time - time_str_detected).seconds
        if diff_time > 10: 
            face_time[return_str] =  current_time
            print("Speaker Out: " , return_str)
            tts = gTTS('hello '+return_str, 'en-in')
            tts.save('hello.mp3')
            playsound('hello.mp3')
            
            return return_str
    else:
        return "Unknown_person"


# In[7]:


import glob
import sklearn.preprocessing


# In[8]:


def get_class(img_path):
    return str(img_path.split('/')[-2])


# In[9]:


def create_input_image_embeddings():
    ##input_embeddings = {}
    
    ##change by Sourjeet
    input_embeddings_all = []
    initial_timestamp = datetime.now()
    face_time = {}
    
    for file in glob.glob("New_images/*/*"):
        #print (file)
        
        person_name = get_class(file)
        
        
        image_file = cv2.imread(file, 1)
        
        ##change by Sourjeet
        input_embeddings_all.append([person_name,image_to_embedding(image_file, embeddings),1])
        
        face_time[person_name] = initial_timestamp
        
        create_input_image_embeddings_out = [input_embeddings_all,face_time]
        
         
    return create_input_image_embeddings_out


# In[16]:


def recognize_faces_in_cam(input_embeddings,face_time):
    

    cv2.namedWindow("Face Recognizer")
    vc = cv2.VideoCapture(0)
   

    font = cv2.FONT_HERSHEY_SIMPLEX
    #face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    
    
    while vc.isOpened():
        _, frame = vc.read()
        img = frame
        height, width, channels = frame.shape

        
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Loop through all the faces detected 
        #identities = []
        for (x, y, w, h) in faces:
            x1 = x
            y1 = y
            x2 = x+w
            y2 = y+h

            face_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]    
            identity= recognize_face(face_image, input_embeddings, embeddings,face_time)
        
            if identity is not None:
                img = cv2.rectangle(frame,(x1, y1),(x2, y2),(255,255,255),2)
                cv2.putText(img, str(identity), (x1+5,y1-5), font, 1, (255,255,255), 2)
        
        key = cv2.waitKey(10)
        cv2.imshow("Face Recognizer", img)


        
        if key == 27: # exit on ESC
            break    
    vc.release()
    cv2.destroyAllWindows()


# In[11]:


create_input_image_embeddings_out = create_input_image_embeddings()
input_embeddings = create_input_image_embeddings_out[0] 
face_time = create_input_image_embeddings_out[1] 





# In[19]:


recognize_faces_in_cam(input_embeddings,face_time)


# In[ ]:




