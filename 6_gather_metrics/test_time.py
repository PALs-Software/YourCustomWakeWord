import sys
sys.path.insert(0, "./")
from modules.settings import get_settings
from modules.train import load_k_fold
import os
import tensorflow as tf
from tqdm import tqdm
import time
import keras
import csv
import datetime

settings = get_settings()

test_x_4_pool_3, test_y_4_pool_3 = load_k_fold(os.path.join(settings.training_data_dir, 'folds', f'pre_pooling_3'), 4)
test_x_8_pool_3, test_y_8_pool_3 = load_k_fold(os.path.join(settings.training_data_dir, 'folds', f'pre_pooling_3'), 8)
test_x_4_pool_6, test_y_4_pool_6 = load_k_fold(os.path.join(settings.training_data_dir, 'folds', f'pre_pooling_6'), 4)
test_x_8_pool_6, test_y_8_pool_6 = load_k_fold(os.path.join(settings.training_data_dir, 'folds', f'pre_pooling_6'), 8)

test_dataset_4_pool_3 = tf.data.Dataset.from_tensor_slices((test_x_4_pool_3, test_y_4_pool_3)).batch(settings.training_batch_size)
test_dataset_8_pool_3 = tf.data.Dataset.from_tensor_slices((test_x_8_pool_3, test_y_8_pool_3)).batch(settings.training_batch_size)
test_dataset_4_pool_6 = tf.data.Dataset.from_tensor_slices((test_x_4_pool_6, test_y_4_pool_6)).batch(settings.training_batch_size)
test_dataset_8_pool_6 = tf.data.Dataset.from_tensor_slices((test_x_8_pool_6, test_y_8_pool_6)).batch(settings.training_batch_size)

log_file = open(os.path.join(settings.model_training_dir, f'test_metrics_{datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S")}_model_results.csv'), "w", newline='')
csv_writer = csv.writer(log_file, delimiter=';')
csv_writer.writerow(['model_name', 'test_loss', 'test_accuracy', 'time'])
log_file.flush()

def create_metrics(model_name, model_directory, test_dataset):
    model = keras.saving.load_model(os.path.join(model_directory, 'final.keras'))

    start = time.perf_counter()
    for i in range(5):
        metrics = model.evaluate(test_dataset)
    time_metric = int(((time.perf_counter() - start) * 1000) / 5)
    
    print(f'Loss: {metrics[0]}, Accuracy: {metrics[1]}, Time: {time_metric}ms')
    csv_writer.writerow([model_name, metrics[0], metrics[1], time_metric])
    log_file.flush()

subfolders = [ f.path for f in os.scandir(settings.model_training_dir) if f.is_dir() ]
for model_directory in tqdm(subfolders):
    model_name = os.path.basename(model_directory)
    print(f'Evaluate {model_name}')

    if ("f.0.4" in model_name) and ("p.3" in model_name):
        create_metrics(model_name, model_directory, test_dataset_4_pool_3)
    if ("f.0.4" in model_name) and ("p.6" in model_name):
        create_metrics(model_name, model_directory, test_dataset_4_pool_6)
    if ("f.3.8" in model_name) and ("p.3" in model_name):
        create_metrics(model_name, model_directory, test_dataset_8_pool_3)
    if ("f.3.8" in model_name) and ("p.6" in model_name):
        create_metrics(model_name, model_directory, test_dataset_8_pool_6)





