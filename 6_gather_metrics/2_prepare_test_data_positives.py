import sys
sys.path.insert(0, "./")
from modules.settings import get_settings
import tensorflow as tf
import os
from tqdm import tqdm
from numpy.lib.format import open_memmap
from modules.audio_features import convert_wav_to_spectrogram, generate_augment_audio_with_background_audio_file
import numpy as np
from pathlib import Path

settings = get_settings()

files = tf.io.gfile.glob(os.path.join(settings.extended_test_positives_dir, "*.wav"))
files.sort()

raw_dir = os.path.join(settings.extended_test_positives_dir, "raw_wavs")
if not os.path.exists(raw_dir):
    os.makedirs(raw_dir)

generate_augment_audio_with_background_audio_file(files[0], raw_dir, f'{0:07}')
file_path = f'{0:07}_{Path(files[0]).name}'

maps = dict()
for pool_spectrogram_features_value in settings.pre_pooling_spectrogram_features_values_to_test:
    path = os.path.join(settings.extended_test_positives_dir, f'test_positives_pre_pool_{pool_spectrogram_features_value}_x.npy')
    dummy_spectrogram = convert_wav_to_spectrogram(os.path.join(raw_dir, file_path), False if pool_spectrogram_features_value == 0 else True, pool_spectrogram_features_value)
    maps[pool_spectrogram_features_value] =  open_memmap(path, mode='w+', dtype=np.float32, shape=(len(files), dummy_spectrogram.shape[0], dummy_spectrogram.shape[1], dummy_spectrogram.shape[2]))

i = 0
for file in tqdm(files):
    generate_augment_audio_with_background_audio_file(file, raw_dir, f'{i:07}')
    file_path = f'{i:07}_{Path(file).name}'

    for pool_spectrogram_features_value in settings.pre_pooling_spectrogram_features_values_to_test:
        spectrogram = convert_wav_to_spectrogram(os.path.join(raw_dir, file_path), False if pool_spectrogram_features_value == 0 else True, pool_spectrogram_features_value)
        maps[pool_spectrogram_features_value][i] = spectrogram

    i += 1

for pool_spectrogram_features_value in settings.pre_pooling_spectrogram_features_values_to_test:       
    maps[pool_spectrogram_features_value].flush()