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

print('Load extended test dataset')
x = np.load(os.path.join(settings.extended_test_positives_dir, f'test_positives_pre_pool_{pre_pool_spectrogram_features_value}_x.npy'))
y = [1 for i in range(len(x))]

predictions = []
for i in range(0, len(x), batch_size):
    batch = x[i : i + batch_size]
    predictions.extend(model.predict_on_batch(batch))

i = 0
for prediction in predictions:    
    if prediction < min_level_for_positive:
        print(f'Detected false negative at audio snippet {i}')
    i += 1

decisions = [1 if p > min_level_for_positive else 0 for p in predictions]

cm = tf.math.confusion_matrix(y, decisions)
print(cm)

print(f'fn rate: {cm[1][0] / len(x)}')