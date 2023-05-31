import math
import os
import numpy
import tensorflow as tf
from tensorflow import keras
import librosa
from keras import regularizers
import matplotlib.pyplot as plt

tf.keras.backend.clear_session()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)

        # 필요한 만큼 메모리를 런타임에 할당하기
        # logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

        # GPU에 할당되는 전체 메모리 크기를 제한하는 방법
        # tf.config.experimental.set_virtual_device_configuration(
        #     gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
        # logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    except RuntimeError as e:
        print(e)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if tf.test.gpu_device_name():
    print("GPU 사용 가능")
    print("GPU 정보: ", tf.config.list_physical_devices('GPU'))
else:
    print("GPU 사용 불가")


# 1 행 n_mfcc mfcc 계수(주파수 축 세로축) / 0 열 n_time_steps 시간 값 (시간축 가로축)
# def get_mfcc(wav_file_path):
#     # y, sr = librosa.load(wav_file_path, offset=10, duration=5)
#     y, sr = librosa.load(wav_file_path)
#     mfcc = numpy.array(librosa.feature.mfcc(y=y, sr=sr))
#     # pade_mode
#     # sr 동일하게 resample
#     # y_downsampled = librosa.resample(y, orig_sr=sr, target_sr=22050)
#     # sr_downsampled = 22050
#     # mfcc = numpy.array(librosa.feature.mfcc(y=y_downsampled, sr=sr_downsampled, n_mfcc=20))
#     return mfcc

# 노패딩시 tss 길이 55 패딩 or 5초로 늘려서 추출시 216
# 5초로 자르고 5초보다 짧은 경우 5초에 맞도록 반복 추출
# def get_mfcc(wav_file_path):
#     y, sr = librosa.load(wav_file_path)
#
#     # 5초로 자르기
#     duration = 5  # 목표 녹음 길이 (초)
#     if len(y) > sr * duration:
#         y = y[:sr * duration]
#     else:
#         repeats = int(numpy.ceil(sr * duration / len(y)))
#         y = numpy.tile(y, repeats)[:sr * duration]
#
#     mfcc = librosa.feature.mfcc(y=y, sr=sr)
#     return mfcc

def get_duration(wav_file_path):
    y, sr = librosa.load(wav_file_path, sr=None)
    return len(y) / sr


def get_mfcc(wav_file_path, start=0, duration=5):
    y, sr = librosa.load(wav_file_path, offset=start, duration=duration)
    if len(y) < duration * sr:
        # y = numpy.concatenate([y] * math.ceil((duration * sr) / len(y)))[:duration * sr]
        y = numpy.concatenate([y] * math.ceil((duration * sr) / len(y)))
    y = y[:duration * sr]  # Ensure y is always the exact same length
    mfcc = numpy.array(librosa.feature.mfcc(y=y, sr=sr))
    return mfcc

directory = 'C:/Users/yongju kim/Desktop/testfolder'
sounds = ['fire_alarm_sound', 'bicycle', 'name', 'car_drivesound', 'car_siren', 'car_horn', 'motorcycle_horn',
          'motorcycle_drivesound']
# sounds = ['car_drivesound', 'fire_alarm_sound', 'bicycle', 'name']
# sounds = ['name']
features = []
labels = []

# 폴더의 파일들 특징 추출 및 라벨링 반복문
# total_files = 0
# for sound in sounds:
#     print("Calculating features for sound : " + sound)
#     files = os.listdir(directory + "/" + sound)
#     total_files += len(files)
#     for file in files:
#         file_path = directory + "/" + sound + "/" + file
#         features.append(get_mfcc(file_path))
#         label = sounds.index(sound)
#         labels.append(label)

# bicycle일 경우에만 자르기
total_files = 0
for sound in sounds:
    print("Calculating features for sound : " + sound)
    files = os.listdir(directory + "/" + sound)
    total_files += len(files)
    for file in files:
        file_path = directory + "/" + sound + "/" + file
        # if sound == 'bicycle' or sound == 'fire_alarm_sound':
        if sound == 'bicycle':
            if get_duration(file_path) >= 15:
                features.append(get_mfcc(file_path, start=10, duration=5))
            else:
                print(f"Skipping file {file_path} as it is less than 15 seconds.")
        else:
            features.append(get_mfcc(file_path))
        label = sounds.index(sound)
        labels.append(label)

# for feature in features:
#     print(feature.shape)

# 길이가 맞이 않는 특징 zero padding
# max_channels = 0
# for feature in features:
#     max_channels = max(max_channels, feature.shape[1])
#
# padded_features = []
# for feature in features:
#     padding = numpy.zeros((feature.shape[0], max_channels - feature.shape[1]), dtype=numpy.float32)
#     padded_feature = numpy.hstack((feature, padding))
#     padded_features.append(padded_feature)

# permutations = numpy.random.permutation(total_files)
# padded_features = numpy.array(padded_features)[permutations]
# labels = numpy.array(labels)[permutations]
# print("after padding padded_features", padded_features.shape)


permutations = numpy.random.permutation(total_files)
features = numpy.array(features)[permutations]
# features = numpy.array(features)
labels = numpy.array(labels)[permutations]
# labels = numpy.array(labels)
print(features.shape)

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

train_index = int(total_files * train_ratio)
val_index = int(total_files * (train_ratio + val_ratio))

features_train = features[0:train_index]
labels_train = labels[0:train_index]
print("features_train shape", features_train.shape)

features_val = features[train_index:val_index]
labels_val = labels[train_index:val_index]
print("features_val shape", features_val.shape)

features_test = features[val_index:total_files]
labels_test = labels[val_index:total_files]
print("features_test shape", features_test.shape)
# for i in features_test:
#     print("모든 값 보기", i)
# print(features_test)


# 입력 데이터의 형태(shape)를 변경
input_shape = (20, 216)
# mfcc 20 melpectrogram 128 concatencate 148

inputs = keras.Input(shape=input_shape, name="input")

# 합성곱 계층(Convolutional Layer) 추가
# padding="valid",   padding="same",
x = keras.layers.Conv1D(32, kernel_size=7, activation="relu", padding="same", name="conv1")(inputs)
x = keras.layers.MaxPooling1D(pool_size=2, name="maxpool1")(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Conv1D(64, kernel_size=5, activation="relu", padding="same", name="conv2")(x)
x = keras.layers.MaxPooling1D(pool_size=2, name="maxpool2")(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Conv1D(128, kernel_size=3, activation="relu", padding="same", name="conv3")(x)
x = keras.layers.MaxPooling1D(pool_size=2, name="maxpool3")(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Conv1D(256, kernel_size=2, activation="relu", padding="same", name="conv4")(x)
x = keras.layers.MaxPooling1D(pool_size=2, name="maxpool4")(x)
x = keras.layers.Dropout(0.2)(x)

# 완전 연결 계층 (Fully Connected Layer)을 위해 데이터 평탄화
x = keras.layers.Flatten(name="flatten")(x)

x = keras.layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.001), name="dense_1")(x)
# x = keras.layers.Dense(256, activation="relu", name="dense_1")(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001), name="dense_2")(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(8, activation="softmax", name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=keras.optimizers.RMSprop(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=(keras.metrics.SparseCategoricalAccuracy(name="accuracy"))
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=10,  # 10 에포크 동안 개선되지 않으면 중단
    restore_best_weights=True
)

history = model.fit(x=features_train, y=labels_train, callbacks=early_stopping, verbose=1,
                    validation_data=(features_val, labels_val), epochs=200)

# loss 그래프로 보여주기
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.legend(['Train', 'val'], loc='upper right')
plt.show()

loss, accuracy = model.evaluate(features_test, labels_test)
print("Loss:", loss)
print('Accuracy : ' + str(accuracy * 100) + '%')
model.summary()

model.save('model.h5')
