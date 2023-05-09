import wave
import librosa
import matplotlib
from matplotlib import pyplot as plt
import numpy

file_path = 'C:/Users/yongju kim/Desktop/testfolder/1.자동차_311_1.wav'  # WAV 파일 경로를 지정하세요.

sample_rate = 22050
def get_mfcc(wav_file_path):
    y, sr = librosa.load(wav_file_path, offset=0, duration=30)
    y_downsampled = librosa.resample(y, orig_sr=sr, target_sr=22050) # 22050Hz로 다운 샘플링합니다.
    sr_downsampled = 22050
    mfcc = numpy.array(librosa.feature.mfcc(y=y_downsampled, sr=sr_downsampled))
    print(mfcc)
    return mfcc

plt.figure(figsize=(10, 4))
librosa.display.specshow((get_mfcc(file_path)))
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()

with wave.open(file_path, 'rb') as file_path:
    sample_rate = file_path.getframerate()
    print(f"샘플링 레이트: {sample_rate} Hz")