import sys
sys.path.insert(0, "./")
from modules.settings import get_settings
from modules.train import load_k_fold
import tensorflow as tf
import os
import keras
from pathlib import Path

pre_pool_spectrogram_features_value = 3
test_fold_id = 7
test_batch_size = 250
dipco_path = os.path.join(os.path.abspath(os.getcwd()), "../Dipco/audio")

settings = get_settings()
model =  keras.models.load_model(os.path.join(settings.model_training_dir, 'final_model', 'final.keras'))

x, y = load_k_fold(os.path.join(settings.training_data_dir, 'folds', f'pre_pooling_{pre_pool_spectrogram_features_value}'), test_fold_id)

predictions = []
for i in range(0, len(x), test_batch_size):
    batch = x[i : i + test_batch_size]
    predictions.extend(model.predict_on_batch(batch))

decision = [1 if p > 0.5 else 0 for p in predictions]
print(tf.math.confusion_matrix(y, decision))

clips = [str(i) for i in Path(dipco_path).glob("**/*U01.CH1.wav")]


print('')