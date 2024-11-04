import sys
sys.path.insert(0, "./")
from modules.settings import get_settings
from modules.audio_features import generate_augment_audio_with_background_audio_file, generate_augment_audio
import os
import tensorflow as tf
from tqdm import tqdm
import itertools as it
import shutil

settings = get_settings()

positive_dir = os.path.join(settings.training_data_dir, 'augmented_data', 'positive')
negative_dir = os.path.join(settings.training_data_dir, 'augmented_data', 'negative')

if os.path.exists(positive_dir):
    shutil.rmtree(positive_dir)
os.makedirs(positive_dir)
if os.path.exists(negative_dir):
    shutil.rmtree(negative_dir)
os.makedirs(negative_dir)

wake_word_audio_files = tf.io.gfile.glob(settings.raw_wake_word_training_dir + "/**/*.wav")
negative_words_files = tf.io.gfile.glob(settings.raw_negative_words_training_dir + "/**/*.wav")
negative_audio_files = tf.io.gfile.glob(settings.raw_negative_audio_training_dir + "/**/*.wav")
negative_background_audio_and_impulses_audio_files = tf.io.gfile.glob(settings.raw_background_audio_training_dir + "/**/*.wav") + tf.io.gfile.glob(settings.raw_impulses_training_dir + "/**/*.wav")

positive_data_length = int(settings.percentage_of_positive_data * settings.no_of_training_samples_to_generate)
negative_data_length = int(settings.no_of_training_samples_to_generate - positive_data_length)

# Unfortunately, tfio.audio.AudioIOTensor is not threadsafe and therefore parallelisation makes no sense here, because if this command is protected by a lock, the whole thing no longer scales properly
# Positive Data
samples_generated = 0
for wake_word_audio_file in tqdm(it.cycle(wake_word_audio_files), total=positive_data_length, desc= f'Prepare positive wake word samples'):
    generate_augment_audio_with_background_audio_file(wake_word_audio_file, positive_dir, f'{samples_generated:07}')
    samples_generated += 1
    if samples_generated >= positive_data_length:
        break

# Negative Data
negative_words_cycle = it.cycle(negative_words_files)
negative_audio_cycle = it.cycle(negative_audio_files)
negative_background_audio_and_impulses_audio_cycle = it.cycle(negative_background_audio_and_impulses_audio_files)
negative_data = []
for i in range(int(settings.negative_data_percentage_of_negative_words * negative_data_length)):
    negative_data.append(next(negative_words_cycle))
for i in range(int(settings.negative_data_percentage_of_negative_audio * negative_data_length)):
    negative_data.append(next(negative_audio_cycle))

samples_generated = 0
for negative_data_file in tqdm(negative_data, desc= f'Prepare negative samples'):
    generate_augment_audio_with_background_audio_file(negative_data_file, negative_dir, f'{samples_generated:07}')
    samples_generated += 1

for i in tqdm(range(int(settings.negative_data_percentage_of_background_audio * negative_data_length)), desc= f'Prepare negative background samples'):
    generate_augment_audio(next(negative_background_audio_and_impulses_audio_cycle), negative_dir, f'{samples_generated:07}')
    samples_generated += 1 