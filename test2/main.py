import os
import numpy
import tensorflow as tf
from tensorflow import keras
import librosa
from keras import regularizers
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if tf.test.gpu_device_name():
    print("GPU 사용 가능")
    print("GPU 정보: ", tf.config.list_physical_devices('GPU'))
else:
    print("GPU 사용 불가")

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
    melspectrogram_feature = numpy.concatenate( (melspectrogram_mean, melspectrogram_min, melspectrogram_max) )
    mfcc = get_mfcc(file_path)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_min = mfcc.min(axis=1)
    mfcc_max = mfcc.max(axis=1)
    mfcc_feature = numpy.concatenate((mfcc_mean, mfcc_min, mfcc_max))
    feature = numpy.concatenate((melspectrogram_feature, mfcc_feature))
    return feature


directory = 'C:/Users/yongju kim/Desktop/testfolder'
sounds = ['fire_alram_sound', 'bicycle', 'car_drivesound', 'car_siren',  'car_horn', 'motorcycle_horn', 'motorcycle_drivesound']
features = []
labels = []
# 시그널 프로세싱 단계를 앞에 넣는다
# frequency domain ftt
total_files = 0
for sound in sounds:
    print("Calculating features for sound : " + sound)
    files = os.listdir(directory + "/" + sound)
    total_files += len(files)
    for file in files:
        file_path = directory + "/" + sound + "/" + file
        features.append(get_feature(file_path))
        label = sounds.index(sound)
        labels.append(label)

permutations = numpy.random.permutation(total_files)
features = numpy.array(features)[permutations]
labels = numpy.array(labels)[permutations]
print(features.shape)

train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

train_index = int(total_files * train_ratio)
val_index = int(total_files * (train_ratio + val_ratio))

features_train = features[0:train_index]
labels_train = labels[0:train_index]
print(features_train.shape)

features_val = features[train_index:val_index]
labels_val = labels[train_index:val_index]
print(features_val.shape)

features_test = features[val_index:total_files]
labels_test = labels[val_index:total_files]
print(features_test.shape)

# 입력 데이터의 형태(shape)를 변경
input_shape = (444, 1)

inputs = keras.Input(shape=input_shape, name="input")

# 합성곱 계층(Convolutional Layer) 추가
x = keras.layers.Conv1D(32, kernel_size=2, activation="relu", name="conv1")(inputs)
x = keras.layers.MaxPooling1D(pool_size=2, name="maxpool1")(x)
x = keras.layers.Conv1D(64, kernel_size=2, activation="relu", name="conv1")(inputs)
x = keras.layers.MaxPooling1D(pool_size=2, name="maxpool1")(x)
x = keras.layers.Conv1D(128, kernel_size=2, activation="relu", name="conv2")(x)
x = keras.layers.MaxPooling1D(pool_size=2, name="maxpool2")(x)
x = keras.layers.Conv1D(256, kernel_size=2, activation="relu", name="conv1")(inputs)
x = keras.layers.MaxPooling1D(pool_size=2, name="maxpool1")(x)

# 완전 연결 계층 (Fully Connected Layer)을 위해 데이터 평탄화
x = keras.layers.Flatten(name="flatten")(x)

x = keras.layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001), name="dense_1")(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.001), name="dense_2")(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(7, activation="softmax", name="predictions")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=keras.optimizers.RMSprop(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=20,  # 20 에포크 동안 개선되지 않으면 중단
    restore_best_weights=True
)

history = model.fit(x=features_train, y=labels_train, callbacks=early_stopping, verbose=1,
                    validation_data=(features_val, labels_val), epochs=500)

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