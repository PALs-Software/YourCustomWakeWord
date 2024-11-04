import sys
sys.path.insert(0, "./")
from modules.settings import get_settings
import tensorflow as tf
import tensorflow_io as tfio
import os
from tqdm import tqdm
from numpy.lib.format import open_memmap
from modules.audio_features import get_spectrogram, normalise_audio
import numpy as np
import scipy

stride_size = 0.5 # Percentage relative to the window size
dipco_path = os.path.join(os.path.abspath(os.getcwd()), "../Dipco/audio")

settings = get_settings()

if not os.path.exists(settings.dipco_dir):
    os.makedirs(settings.dipco_dir)

if not os.path.exists(os.path.join(settings.dipco_dir, "raw_wavs")):
    os.makedirs(os.path.join(settings.dipco_dir, "raw_wavs"))

files = tf.io.gfile.glob(os.path.join(dipco_path, "**/*U01.CH1.wav"))
files = [x for x in files if not os.path.basename(x).startswith('.')]

element_count = 0
for file in tqdm(files, desc="Count dipco frames"):
    audio_tensor = tfio.audio.AudioIOTensor(file)
    audio_data = tf.cast(audio_tensor[:], tf.float32)
    audio_data_len = len(audio_data)

    current_index = 0
    while True:
        frame_data = audio_data[current_index: current_index + settings.window_size_in_samples]
        element_count += 1
        current_index += int(settings.window_size_in_samples * stride_size)
        if (current_index + settings.window_size_in_samples) > audio_data_len:
            break

print(f'Features: {element_count}')
print(f'{round(((element_count * stride_size * settings.window_size_in_samples) / settings.sample_rate) / 60 / 60, 2)}h of data')

dummy_audio_tensor = tfio.audio.AudioIOTensor(files[0])
dummy_audio_data = tf.cast(audio_tensor[:], tf.float32)
dummy_audio_data = normalise_audio(dummy_audio_data)
dummy_audio_data_frame = dummy_audio_data[0: settings.window_size_in_samples]

maps = dict()
for pool_spectrogram_features_value in settings.pre_pooling_spectrogram_features_values_to_test:
    path = os.path.join(settings.dipco_dir, f'dipco_pre_pool_{pool_spectrogram_features_value}_x.npy')
    dummy_spectrogram = get_spectrogram(dummy_audio_data_frame, False if pool_spectrogram_features_value == 0 else True, pool_spectrogram_features_value)
    maps[pool_spectrogram_features_value] =  open_memmap(path, mode='w+', dtype=np.float32, shape=(element_count, dummy_spectrogram.shape[0], dummy_spectrogram.shape[1], dummy_spectrogram.shape[2]))

overall_index = 0
progress = tqdm(range(element_count), desc="Prepare spectrograms for all frames of the dipco dataset")
for file in files:
    audio_tensor = tfio.audio.AudioIOTensor(file)
    audio_data = tf.cast(audio_tensor[:], tf.float32)   
    audio_data_len = len(audio_data)

    current_index = 0
    while True:
        frame_data = audio_data[current_index: current_index + settings.window_size_in_samples]
        frame_data = normalise_audio(frame_data)
        raw_wav_data = frame_data * 32767
        if isinstance(raw_wav_data, np.ndarray):
            raw_wav_data = raw_wav_data.astype(np.int16)
        else:
            raw_wav_data = raw_wav_data.numpy().astype(np.int16)

        wav_path = os.path.join(settings.dipco_dir, "raw_wavs" , f'{overall_index:07}.wav')
        scipy.io.wavfile.write(wav_path, 16000, raw_wav_data)
        
        for pool_spectrogram_features_value in settings.pre_pooling_spectrogram_features_values_to_test:
            spectrogram = get_spectrogram(frame_data, False if pool_spectrogram_features_value == 0 else True, pool_spectrogram_features_value)
            maps[pool_spectrogram_features_value][overall_index] = spectrogram

        overall_index += 1
        progress.update(1)
        current_index += int(settings.window_size_in_samples * stride_size)
        if (current_index + settings.window_size_in_samples) > audio_data_len:
            break

for pool_spectrogram_features_value in settings.pre_pooling_spectrogram_features_values_to_test:       
    maps[pool_spectrogram_features_value].flush()