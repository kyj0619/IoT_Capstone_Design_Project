import os
import tensorflow as tf
from tensorflow import keras
import numpy
import librosa


os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

model = keras.models.load_model("C:/Users/yongju kim/PycharmProjects/test2/model.h5")
model.summary()

# file_path = ["C:/Users/yongju kim/Desktop/testfolder/1.자동차_10_1.wav",
#              "C:/Users/yongju kim/Desktop/testfolder/1.자동차_293_1.wav",
#              "C:/Users/yongju kim/Desktop/testfolder/1.자동차_552_1.wav"]

# get_feature = main.get_feature()
# get_mfcc = main.get_mfcc()
# feature1 = get_feature(file_path1)
#
def get_mfcc(wav_file_path):
    y, sr = librosa.load(wav_file_path, offset=0, duration=30)
    mfcc = numpy.array(librosa.feature.mfcc(y=y, sr=sr))
    return mfcc

def get_melspectrogram(wav_file_path):
    y, sr = librosa.load(wav_file_path, offset=0, duration=30)
    melspectrogram = numpy.array(librosa.feature.melspectrogram(y=y, sr=sr))
    return melspectrogram

def get_feature(file_path):
  # Extracting Mel Spectrogram feature
    melspectrogram = get_melspectrogram(file_path)
    melspectrogram_mean = melspectrogram.mean(axis=1)
    melspectrogram_min = melspectrogram.min(axis=1)
    melspectrogram_max = melspectrogram.max(axis=1)
    melspectrogram_feature = numpy.concatenate((melspectrogram_mean, melspectrogram_min, melspectrogram_max))


    mfcc = get_mfcc(file_path)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_min = mfcc.min(axis=1)
    mfcc_max = mfcc.max(axis=1)
    mfcc_feature = numpy.concatenate((mfcc_mean, mfcc_min, mfcc_max))

    feature = numpy.concatenate((melspectrogram_feature, mfcc_feature))

    return feature

sounds = ['fire_alram_sound', 'bicycle', 'car_drivesound', 'car_siren',  'car_horn', 'motorcycle_horn', 'motorcycle_drivesound']
file_path = "C:/Users/yongju kim/Desktop/testfolder/firebell.wav"
feature = get_feature(file_path)
y1 = model.predict(feature.reshape(-1, 444, 1))
print(y1)
ind1 = numpy.argmax(y1)
sounds[ind1]
print(sounds[ind1])


tf.config.list_physical_devices('GPU')