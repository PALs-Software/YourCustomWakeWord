# Optional: Generates some test audio data with the given wake word and background audio data to see if everything works fine

import sys
sys.path.insert(0, "./")
from modules.settings import get_settings
from modules.audio_features import generate_augment_audio_with_background_audio_file
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import shutil

no_of_samples_to_generate = 100
settings = get_settings()

wake_word_output_dir = os.path.join(settings.work_space, 'generated_test_samples', 'wake_word')
negative_words_output_dir = os.path.join(settings.work_space, 'generated_test_samples', 'negative_words')
negative_audio_output_dir = os.path.join(settings.work_space, 'generated_test_samples', 'negative_audio')

if os.path.exists(os.path.join(settings.work_space, 'generated_test_samples')):
    shutil.rmtree(os.path.join(settings.work_space, 'generated_test_samples'))
os.makedirs(wake_word_output_dir)
os.makedirs(negative_words_output_dir)
os.makedirs(negative_audio_output_dir)

wake_word_audio_files = tf.io.gfile.glob(settings.raw_wake_word_training_dir + "/**/*.wav")
negative_words_files = tf.io.gfile.glob(settings.raw_negative_words_training_dir + "/**/*.wav")
negative_audio_files = tf.io.gfile.glob(settings.raw_negative_audio_training_dir + "/**/*.wav")

for i in tqdm(range(no_of_samples_to_generate)):
    generate_augment_audio_with_background_audio_file(np.random.choice(wake_word_audio_files), wake_word_output_dir, i)
    generate_augment_audio_with_background_audio_file(np.random.choice(negative_words_files), negative_words_output_dir, i)
    generate_augment_audio_with_background_audio_file(np.random.choice(negative_audio_files), negative_audio_output_dir, i)

print('Done')