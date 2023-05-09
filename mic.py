import pyaudio
import wave
import numpy as np
import time
import sounddevice
from scipy.io.wavfile import write

def calculate_volume(data):
    # FFT를 수행하여 주파수 도메인으로 변환
    fft_data = np.fft.fft(data)
    # 주파수 영역에서의 소리의 크기 계산
    volume = np.abs(fft_data).mean()
    return volume

# 마이크 장치의 인덱스 설정
input_device_index = 1  # 원하는 마이크 장치의 인덱스로 설정

THRESHOLD = 20000
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050

# PyAudio 초기화
p = pyaudio.PyAudio()
is_recording = False

while True:
    # 마이크 스트림 열기
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=input_device_index)

    # 데이터 읽기
    data = stream.read(1024)
    # 바이너리 데이터를 넘파이 배열로 변환
    samples = np.frombuffer(data, dtype=np.int16)
    # 음량 계산
    volume = calculate_volume(samples)

    # 결과 출력
    print(f"소리 크기: {volume:.2f} dB") #데시벨 출력

    if volume > THRESHOLD and not is_recording: #일정 데시벨을 넘었을 때
        print("-------------------Recording started-------------------")
        is_recording = True
        start_time = time.time()

    if is_recording:
        recording = sounddevice.rec((5 * 44100), samplerate=44100, channels=1)
        sounddevice.wait()
        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time >= 5: #녹음을 시작한지 5초가 지났을 때
            print("-------------------5second end-------------------")
            write("output.wav", 44100, recording)
            is_recording = False
