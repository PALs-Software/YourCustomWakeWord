import sys
sys.path.insert(0, "./")
from modules.settings import get_settings
from modules.audio_features import get_spectrogram
import os
import time
import pyaudio
import numpy as np
import keras

pre_pool_spectrogram_features_value = 6

model_dir_name = "final_model"

settings = get_settings()
model_dir = os.path.join(settings.model_training_dir, model_dir_name)

model =  keras.models.load_model(os.path.join(model_dir, 'final.keras'))
audio = pyaudio.PyAudio()
recorded_samples = np.zeros((settings.window_size_in_samples))
           
def stream_callback(data, frame_count, time_info, status):
    global recorded_samples

    new_samples = np.frombuffer(data, np.float32)
    recorded_samples = np.concatenate((recorded_samples, new_samples))

    if len(recorded_samples) > settings.window_size_in_samples:
        start = time.perf_counter()
        samples_to_check = recorded_samples[:settings.window_size_in_samples]

        recorded_samples = recorded_samples[-(len(recorded_samples) - int(settings.window_size_in_samples / 4)):] # step window size / 4 forward
        spectrogram = get_spectrogram(np.reshape(samples_to_check, (settings.window_size_in_samples, 1)), pre_pool_spectrogram_features_value != 0, pre_pool_spectrogram_features_value)

        prediction = model.predict(np.reshape(spectrogram, (1, spectrogram.shape[0], spectrogram.shape[1], spectrogram.shape[2])))
        if prediction[0][0] > 0.9:
            print("WAKEWORD WAS DETECTED")
        end = time.perf_counter()

        print(f"Possibility: {prediction[0][0]}")
        print(f"Calculation time: {int((end - start ) * 1000)}ms")

    return (data, pyaudio.paContinue)


stream = audio.open(
    input_device_index = 0,
    format = pyaudio.paFloat32,
    channels = 1,
    rate = settings.sample_rate,
    input = True,
    stream_callback = stream_callback,
    frames_per_buffer = 2048
)

stream.start_stream()
while stream.is_active():
    time.sleep(0.1)