import librosa
import os
dataset_file_path = 'D:\\Projects\\Projects\\pytorch_Projects\\GAN\\Datasets\\concept cell dataset\\concept cell audio final\\'

sound_class = 0
for paths, dirs, files in os.walk(dataset_file_path):
    print(dirs)
    for dir in dirs:
        current_path = paths + dir
        sound_class+=1
        for file in os.listdir(current_path):
            current_sound_file_path = current_path + '\\' + file
            break
        break
    break

# print(sound_class)
# print(Current_File)

print(current_sound_file_path)

# load .wav file using librosa
sample_rate = 16000
audio_data, sr = librosa.load(current_sound_file_path, sr=None)
print(len(audio_data), sr)

# Plotting time-domain of sound
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

def Audio_Display(data):
    # Sonify detected beat events
    tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sr)
    audio_beats = librosa.clicks(frames=beats, sr=sr)

    # Or generate a signal of the same length as audio_data
    audio_beats = librosa.clicks(frames=beats, sr=sr, length=len(audio_data))

    # Or use timing instead of frame indices
    times = librosa.frames_to_time(beats, sr=sr)
    audio_beat_times = librosa.clicks(times=times, sr=sr)

    # Or with a click frequency of 880Hz and a 500ms sample
    audio_beat_times880 = librosa.clicks(times=times, sr=sr,
                                     click_freq=880, click_duration=0.5)

    # Display click waveform next to the spectrogram
    plt.figure()
    Spec = librosa.feature.melspectrogram(y=audio_data, sr=sr)
    ax = plt.subplot(3, 1, 2)
    librosa.display.specshow(librosa.power_to_db(Spec, ref=np.max), x_axis='time', y_axis='mel')

    plt.subplot(3, 1, 1)# , sharex=ax)
    librosa.display.waveplot(audio_data, sr=sr, label='Beat Clicks')  # Plot raw data to wave shape
    plt.legend()
    plt.xlim()
    plt.tight_layout()

    plt.subplot(3, 1, 3)
    D = librosa.amplitude_to_db(librosa.stft(audio_data), ref=np.max)
    librosa.display.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency power spectrogram')
    plt.show()

Audio_Display(audio_data)



