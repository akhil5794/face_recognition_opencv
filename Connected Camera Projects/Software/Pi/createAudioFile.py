#!/usr/bin/env python
from gtts import gTTS
import glob

for file in glob.glob("trainedFaces/*"):
    Name = str(file.split('/')[-1])
    fileName = "audioFiles/"+Name.replace(" ", "_")+".mp3"
    print(fileName)
    tts = gTTS('hello '+Name, 'en-in')
    tts.save(fileName)
