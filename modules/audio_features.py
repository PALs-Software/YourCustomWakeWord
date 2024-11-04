import sys
sys.path.insert(0, "./")
from modules.settings import get_settings

import tensorflow as tf
import numpy as np
import tensorflow_io as tfio
from tensorflow.python.ops import gen_audio_ops as audio_ops
import scipy
from pathlib import Path
import os

settings = get_settings()
background_audio_files = tf.io.gfile.glob(settings.raw_background_audio_training_dir + "/**/*.wav")
impulses_audio_files = tf.io.gfile.glob(settings.raw_impulses_training_dir + "/**/*.wav")

def get_spectrogram(audio, pool_spectrogram_features = False, pool_spectrogram_features_value = None):
    audio = normalise_audio(audio)
    spectrogram = audio_ops.audio_spectrogram(audio,
                                              window_size=settings.spectrogram_window_size,
                                              stride=settings.spectrogram_stride,
                                              magnitude_squared=True).numpy()
    spectrogram = tf.expand_dims(spectrogram, -1)
    if pool_spectrogram_features:
        spectrogram = tf.nn.pool(
            input=spectrogram,
            window_shape=[1, pool_spectrogram_features_value],
            strides=[1, pool_spectrogram_features_value],
            pooling_type='AVG',
            padding='SAME')
    
    spectrogram = tf.squeeze(spectrogram, axis=0)
    spectrogram = np.log10(spectrogram + 1e-6) # Add minimum value, to avoid zero with log 10
    
    return spectrogram

def generate_augment_audio_with_background_audio_file(source_file_path, output_dir, id):    
    audio = augment_audio_with_background_audio(source_file_path)
    audio *= 32767
    if isinstance(audio, np.ndarray):
        audio = audio.astype(np.int16)
    else:
        audio = audio.numpy().astype(np.int16)

    name = f'{id}_{Path(source_file_path).name}'
    scipy.io.wavfile.write(os.path.join(output_dir, name), 16000, audio)

def generate_augment_audio(source_file_path, output_dir, id):    
    audio = load_and_adjust_audio(source_file_path)
    audio *= 32767
    if isinstance(audio, np.ndarray):
        audio = audio.astype(np.int16)
    else:
        audio = audio.numpy().astype(np.int16)

    name = f'{id}_{Path(source_file_path).name}'
    scipy.io.wavfile.write(os.path.join(output_dir, name), 16000, audio)

def load_and_adjust_audio(file_path):
    audio_tensor = tfio.audio.AudioIOTensor(file_path)
    audio = tf.cast(audio_tensor[:], tf.float32)
    audio = normalise_audio(audio)
    audio = adjust_size_of_audio_if_needed(audio)
    audio = reposition_voice_in_audio_randomly(audio)
    return audio

def augment_audio_with_background_audio(file_path):    
    audio = load_and_adjust_audio(file_path)
    
    if np.random.uniform(0, 1) <= settings.add_background_audio_possibility:
        audio = add_background_audio(audio, background_audio_files)

    if np.random.uniform(0, 1) <= settings.add_background_audio_impulses_possibility:
        audio = add_background_audio(audio, impulses_audio_files)

    return audio

def convert_wav_to_spectrogram(file_path, pool_spectrogram_features, pool_spectrogram_features_value):
    audio_tensor = tfio.audio.AudioIOTensor(file_path)
    audio = tf.cast(audio_tensor[:], tf.float32)
    return get_spectrogram(audio, pool_spectrogram_features, pool_spectrogram_features_value)

def convert_wav_to_spectrogram_with_random_background_audio(file_path):
    audio = augment_audio_with_background_audio(file_path)
    return get_spectrogram(audio)

def normalise_audio(audio):
    audio = audio - np.mean(audio)
    max = np.max(np.abs(audio))
    if max != 0:
        audio = audio / max
    return audio

def reposition_voice_in_audio_randomly(audio):
    voice_start, voice_end = tfio.audio.trim(audio, axis=0, epsilon=settings.voice_noise_floor)
    buffer_to_the_end = len(audio) - voice_end
    random_offset = int(np.random.uniform(0, voice_start + buffer_to_the_end))
    return np.roll(audio, -random_offset + buffer_to_the_end)

def adjust_size_of_audio_if_needed(audio):
    current_length = audio.shape[0]
    if current_length==settings.window_size_in_samples:
        return audio
    
    if current_length < settings.window_size_in_samples:
        zeros_to_add = settings.window_size_in_samples - current_length
        zeros = tf.zeros([zeros_to_add,1], dtype=audio.dtype)
        return tf.concat([audio, zeros], 0)
    else:
        random_offset = int(np.random.uniform(0, current_length - settings.window_size_in_samples))
        return audio[random_offset:random_offset+settings.window_size_in_samples]
    
def add_background_audio(audio, background_audio_files):
    audio_background_file = np.random.choice(background_audio_files)
    audio_background_tensor = tfio.audio.AudioIOTensor(audio_background_file)

    background_length = len(audio_background_tensor)
    if background_length < settings.window_size_in_samples:
        background_audio = tf.cast(audio_background_tensor[:], tf.float32)        
        zeros_to_add = settings.window_size_in_samples - background_length
        random_offset = int(np.random.uniform(0, zeros_to_add))
        start_zeros = tf.zeros([random_offset,1], dtype=audio.dtype)
        end_zeros = tf.zeros([zeros_to_add - random_offset,1], dtype=audio.dtype)
        background_audio = tf.concat([start_zeros, background_audio, end_zeros], 0)
    else:
        audio_background_start = np.random.randint(0, len(audio_background_tensor) - settings.window_size_in_samples)
        background_audio = tf.cast(audio_background_tensor[audio_background_start:audio_background_start + settings.window_size_in_samples], tf.float32)

    background_audio = normalise_audio(background_audio)
    background_audio_volume = np.random.uniform(0, settings.max_audio_background_volume)
    return audio + background_audio_volume * background_audio