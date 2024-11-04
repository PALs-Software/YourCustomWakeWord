import sys
sys.path.insert(0, "./")
from modules.settings import get_settings
from modules.audio_features import get_spectrogram, normalise_audio
import tensorflow as tf
import os
import keras
import tensorflow_io as tfio
import numpy as np

file_name = "6.wav"
pre_pool_spectrogram_features_value = 3

settings = get_settings()
model =  keras.models.load_model(os.path.join(settings.model_training_dir, 'final_model', 'final.keras'))

path = os.path.join(settings.dipco_dir, 'raw_wavs', file_name)
audio_tensor = tfio.audio.AudioIOTensor(path)
audio_data = tf.cast(audio_tensor[:], tf.float32)
audio_data = normalise_audio(audio_data)
spectrogram = get_spectrogram(audio_data, False if pre_pool_spectrogram_features_value == 0 else True, pre_pool_spectrogram_features_value)

prediction_data = np.reshape(spectrogram, (1, spectrogram.shape[0], spectrogram.shape[1], spectrogram.shape[2]))
print(model.predict(prediction_data))