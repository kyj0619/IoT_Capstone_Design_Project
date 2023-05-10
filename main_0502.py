#!/usr/bin/python3
#-*-coding:utf-8 -*-

import os
import tensorflow as tf
from tensorflow import keras
import numpy
import librosa
import pyaudio
import wave
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

model = keras.models.load_model("/home/iot/바탕화면/iot_cap/cap/model.h5")

# model.summary()

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Recording...")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Finished recording.")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()


def get_mfcc(wav_file_path):
    y, sr = librosa.load(wav_file_path, offset=0, duration=10)
    mfcc = numpy.array(librosa.feature.mfcc(y=y, sr=sr))
    return mfcc
    

sounds = ['fire_alram_sound', 'bicycle', 'car_drivesound', 'car_siren', 'car_horn', 'motorcycle_horn', 'motorcycle_drivesound']
file_path = "/home/iot/바탕화면/iot_cap/firebase/output.wav"

features = get_mfcc(file_path)
print("features: " , features.shape)

max_channels = 1292
padded_features = []

padding = numpy.zeros((features.shape[0], max_channels - features.shape[1]), dtype=numpy.float32)
padded_feature = numpy.hstack((features, padding))
print(padded_feature.shape)
padded_features.append(padded_feature)
    

padded_features = numpy.expand_dims(padded_feature, axis=0)
# padded_feautres = numpy.array(padded_features)
print("padded_features shape: ", padded_features.shape)

y1 = model.predict(padded_features)
print(y1)
ind1 = numpy.argmax(y1)
sounds[ind1]
print(sounds[ind1])

cred = credentials.Certificate("/home/iot/바탕화면/iot_cap/firebase/test0502.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
doc_ref = db.collection(u'Test_0502').document(u'RaspberryPi')
doc_ref.set({
    u'sound_type' : sounds[ind1]
})
