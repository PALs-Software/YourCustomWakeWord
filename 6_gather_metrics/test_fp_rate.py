import sys
sys.path.insert(0, "./")
from modules.settings import get_settings
import os
import tensorflow as tf
from tqdm import tqdm
import keras
import csv
import datetime
import numpy as np

settings = get_settings()
batch_size = settings.training_batch_size
min_level_for_positive = 0.5

log_file = open(os.path.join(settings.model_training_dir, f'test_fp_rate_{datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S")}_model_results.csv'), "w", newline='')
csv_writer = csv.writer(log_file, delimiter=';')
csv_writer.writerow(['model_name', 'fp_rate', 'fp_per_hour'])
log_file.flush()

print('Load dipco dataset')
x_3 = np.load(os.path.join(settings.dipco_dir, f'dipco_pre_pool_3_x.npy'))
x_6 = np.load(os.path.join(settings.dipco_dir, f'dipco_pre_pool_6_x.npy'))
y = [0 for i in range(len(x_3))]

stride_size = 0.5
total_hours_of_negative_data = ((len(x_3) * stride_size * settings.window_size_in_samples) / settings.sample_rate) / 60 / 60
print(f'Total hours of data: {total_hours_of_negative_data}')

def create_metrics(model_name, model_directory, x):
    model = keras.saving.load_model(os.path.join(model_directory, 'final.keras'))

    predictions = []
    for i in range(0, len(x), batch_size):
        batch = x[i : i + batch_size]
        predictions.extend(model.predict_on_batch(batch))

    decision = [1 if p > min_level_for_positive else 0 for p in predictions]
    cm = tf.math.confusion_matrix(y, decision)

    fp_rate = 0
    fp_per_hour = 0
    if sum(decision) != 0:
        fp_rate = cm[0][1] / len(x)
        fp_per_hour = float(cm[0][1]) / total_hours_of_negative_data
   
    print(f'fp rate: {fp_rate}, fp per hour: {fp_per_hour}')
    
    csv_writer.writerow([model_name, fp_rate, fp_per_hour])
    log_file.flush()

subfolders = [ f.path for f in os.scandir(settings.model_training_dir) if f.is_dir() ]
for model_directory in tqdm(subfolders):
    model_name = os.path.basename(model_directory)
    print(f'Evaluate {model_name}')

    if ("p.3" in model_name):
        create_metrics(model_name, model_directory, x_3)
    if ("p.6" in model_name):
        create_metrics(model_name, model_directory, x_6)




