import sys
sys.path.insert(0, "./")
from modules.settings import get_settings
import os
import numpy as np
import gc
from numpy.lib.format import open_memmap
from tqdm import tqdm

settings = get_settings()

def save_data(data, map_x, map_y, description):
    i = 0
    for data_entry in tqdm(data, desc= description):
        map_x[i] = data_entry[1]
        map_y[i] = data_entry[0]
        if i % 1000 == 0:
            map_x.flush()
            map_y.flush()
        i += 1
    map_x.flush()
    map_y.flush()

pre_pool_settings = tqdm(settings.pre_pooling_spectrogram_features_values_to_test)
for pool_spectrogram_features_value in pre_pool_settings:
    pre_pool_settings.set_description(f'Generate folds for pre_pooling setting {pool_spectrogram_features_value}')

    positives_file_path = os.path.join(settings.training_data_dir, 'raw_maps', f'pre_pooling_{pool_spectrogram_features_value}', 'positives.npy')
    negatives_file_path = os.path.join(settings.training_data_dir, 'raw_maps', f'pre_pooling_{pool_spectrogram_features_value}', 'negatives.npy')
    positive_data = np.load(positives_file_path)
    negative_data = np.load(negatives_file_path)

    if (np.isnan(positive_data).any() or np.isnan(negative_data).any()):
        raise Exception("Data contains nans")

    data_set_len = len(positive_data) + len(negative_data)
    complete_training_data = []
    for positive_data_entry in positive_data:
        complete_training_data.append((1, positive_data_entry))
    del positive_data
    gc.collect()

    for negative_data_entry in negative_data:
        complete_training_data.append((0, negative_data_entry))
    del negative_data
    gc.collect()

    np.random.shuffle(complete_training_data)

    size_per_fold = int(len(complete_training_data) / settings.k_folds)
    print(f'Elements per fold {size_per_fold}')

    for i in range(settings.k_folds):
        dir = os.path.join(settings.training_data_dir, 'folds', f'pre_pooling_{pool_spectrogram_features_value}')
        if not os.path.exists(dir):
            os.makedirs(dir)

        x_path = os.path.join(dir, f'fold_{i}_x.npy')
        y_path = os.path.join(dir, f'fold_{i}_y.npy')

        dummy = complete_training_data[0][1]
        map_x = open_memmap(x_path, mode='w+', dtype=np.float32, shape=(size_per_fold, dummy.shape[0], dummy.shape[1], dummy.shape[2]))
        map_y = open_memmap(y_path, mode='w+', dtype=np.int64, shape=(size_per_fold,))
        
        data = complete_training_data[size_per_fold*i:size_per_fold * (i + 1)]
        save_data(data, map_x, map_y, f'Save data fold {i}/{settings.k_folds}')