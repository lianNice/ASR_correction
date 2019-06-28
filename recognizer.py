# -*- coding: utf-8 -*-
import speech_recognition as sr

r = sr.Recognizer()

with sr.Microphone() as source:
    print('Powiedz coś! ')
    audio = r.listen(source)
    
    try:  
        text = r.recognize_google(audio, language='pl')
        print(f'Powiedziałeś: {text}')
    except:
        print('Przepraszam, powtórz jeszcze raz')
