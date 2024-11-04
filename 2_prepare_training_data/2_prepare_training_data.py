import sys
sys.path.insert(0, "./")
from modules.settings import get_settings
from modules.audio_features import convert_wav_to_spectrogram
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from numpy.lib.format import open_memmap

settings = get_settings()

positive_dir = os.path.join(settings.training_data_dir, 'augmented_data', 'positive')
negative_dir = os.path.join(settings.training_data_dir, 'augmented_data', 'negative')
if not os.path.exists(settings.training_data_dir):
    os.makedirs(settings.training_data_dir)

positive_files = tf.io.gfile.glob(positive_dir + "/*.wav")
negative_files = tf.io.gfile.glob(negative_dir + "/*.wav")
np.random.shuffle(positive_files)
np.random.shuffle(negative_files)

positive_data_length = len(positive_files)
negative_data_length = len(negative_files)

pre_pool_settings = tqdm(settings.pre_pooling_spectrogram_features_values_to_test)
for pool_spectrogram_features_value in pre_pool_settings:
    pre_pool_settings.set_description(f'Generate maps for pre_pooling setting {pool_spectrogram_features_value}')

    dir = os.path.join(settings.training_data_dir, 'raw_maps', f'pre_pooling_{pool_spectrogram_features_value}')
    if not os.path.exists(dir):
        os.makedirs(dir)

    dummy = convert_wav_to_spectrogram(positive_files[0], False if pool_spectrogram_features_value == 0 else True, pool_spectrogram_features_value)
    map_positive = open_memmap(os.path.join(dir, 'positives.npy'), mode='w+', dtype=np.float32, shape=(positive_data_length, dummy.shape[0], dummy.shape[1], dummy.shape[2]))
    map_negative = open_memmap(os.path.join(dir, 'negatives.npy'), mode='w+', dtype=np.float32, shape=(negative_data_length, dummy.shape[0], dummy.shape[1], dummy.shape[2]))

    # Unfortunately, tfio.audio.AudioIOTensor is not threadsafe and therefore parallelisation makes no sense here, because if this command is protected by a lock, the whole thing no longer scales properly
    samples_generated = 0
    for file in tqdm(positive_files, desc= f'Generate map for positives'):
        map_positive[samples_generated] = convert_wav_to_spectrogram(file, False if pool_spectrogram_features_value == 0 else True, pool_spectrogram_features_value)
        if samples_generated % 1000 == 0:
            map_positive.flush()
        samples_generated += 1
    map_positive.flush()

    samples_generated = 0
    for file in tqdm(negative_files, desc= f'Generate map for negatives'):
        map_negative[samples_generated] = convert_wav_to_spectrogram(file, False if pool_spectrogram_features_value == 0 else True, pool_spectrogram_features_value)
        if samples_generated % 1000 == 0:
            map_negative.flush()
        samples_generated += 1
    map_negative.flush()