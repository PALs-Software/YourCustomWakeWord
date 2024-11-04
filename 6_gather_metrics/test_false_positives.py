import sys
sys.path.insert(0, "./")
from modules.settings import get_settings
import tensorflow as tf
import os
import keras
import numpy as np

model_path = "final_model"
pre_pool_spectrogram_features_value = 6
batch_size = 250
min_level_for_positive = 0.5

settings = get_settings()
model =  keras.models.load_model(os.path.join(settings.model_training_dir, model_path, 'final.keras'))

print('Load dipco dataset')
x = np.load(os.path.join(settings.dipco_dir, f'dipco_pre_pool_{pre_pool_spectrogram_features_value}_x.npy'))
y = [0 for i in range(len(x))]

predictions = []
for i in range(0, len(x), batch_size):
    batch = x[i : i + batch_size]
    predictions.extend(model.predict_on_batch(batch))

i = 0
for prediction in predictions:    
    if prediction > min_level_for_positive:
        print(f'Detected false positive at audio snippet {i}')
    i += 1

decision = [1 if p > min_level_for_positive else 0 for p in predictions]

stride_size = 0.5
total_hours_of_negative_data = ((len(x) * stride_size * settings.window_size_in_samples) / settings.sample_rate) / 60 / 60
print(f'Total hours of data: {total_hours_of_negative_data}')

cm = tf.math.confusion_matrix(y, decision)
print(cm)

fp_rate = 0
fp_per_hour = 0
if sum(decision) != 0:
    fp_rate = cm[0][1] / len(x)
    fp_per_hour = float(cm[0][1]) / total_hours_of_negative_data

print(f'fp rate: {fp_rate}')
print(f'fp per hour: {fp_per_hour}')