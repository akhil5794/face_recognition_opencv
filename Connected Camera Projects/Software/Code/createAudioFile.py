#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gtts import gTTS
import glob
 


# In[9]:


for file in glob.glob("../Faces/*"):
    
    Name = str(file.split('/')[-1])
    fileName = "../audioFiles/"+Name.replace(" ", "_")+".mp3"
    gtts = gTTS('hello '+Name, 'en-in')
    gtts.save(fileName)


# In[ ]:




