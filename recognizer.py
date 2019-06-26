# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 21:19:16 2019

@author: acencek
"""

import speech_recognition as sr

r = sr.Recognizer()

with sr.Microphone() as source:
    print('Powiedz cos! ')
    audio = r.listen(source)
    
    try:  
        text = r.recognize_google(audio, language='pl')
        print(f'Powiedziales: {text}')
    except:
        print('Przepraszam, powtorz jeszcze raz')

