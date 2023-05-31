import math
import os
import tensorflow as tf
from tensorflow import keras
import numpy
import librosa

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

model = keras.models.load_model("C:/Users/yongju kim/PycharmProjects/test/model.h5")
model.summary()


# def get_mfcc(wav_file_path):
#     y, sr = librosa.load(wav_file_path, offset=0, duration=5)
#     mfcc = numpy.array(librosa.feature.mfcc(y=y, sr=sr))
#     print(mfcc)
#     return mfcc

def get_mfcc(wav_file_path, start=0, duration=5):
    y, sr = librosa.load(wav_file_path, offset=start, duration=duration)
    if len(y) < duration * sr:
        # y = numpy.concatenate([y] * math.ceil((duration * sr) / len(y)))[:duration * sr]
        y = numpy.concatenate([y] * math.ceil((duration * sr) / len(y)))
    y = y[:duration * sr]  # Ensure y is always the exact same length
    mfcc = numpy.array(librosa.feature.mfcc(y=y, sr=sr))
    return mfcc


sounds = ['fire_alarm_sound', 'bicycle', 'name', 'car_drivesound', 'car_siren', 'car_horn', 'motorcycle_horn',
          'motorcycle_drivesound']
file_path = "C:/Users/yongju kim/Desktop/testfolder/1.자동차_547_1.wav"
features = get_mfcc(file_path)
features = numpy.expand_dims(features, axis=0)
print(features.shape)

# max_channels = 216
# padded_features = []
# padding = numpy.zeros((features.shape[0], max_channels - features.shape[1]), dtype=numpy.float32)
# padded_feature = numpy.hstack((features, padding))
# padded_features.append(padded_feature)
#
# padded_features = numpy.expand_dims(padded_feature, axis=0)

print("feature shape: ", features.shape)
y1 = model.predict(features)
print(y1)
ind1 = numpy.argmax(y1)
sounds[ind1]
print(sounds[ind1])
