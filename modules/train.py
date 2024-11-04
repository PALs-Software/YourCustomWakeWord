import sys
sys.path.insert(0, "./")
from modules.settings import get_settings
import os
import keras
import datetime
import numpy as np
import tensorflow as tf
import gc
import time

settings = get_settings()

def load_k_folds(k_folds_dir: str, no_of_folds: int):
    k_folds_x = []
    k_folds_y = []

    for i in range(no_of_folds):
        k_folds_x.append(np.load(os.path.join(k_folds_dir, f'fold_{i}_x.npy')))
        k_folds_y.append(np.load(os.path.join(k_folds_dir, f'fold_{i}_y.npy')))

    return k_folds_x, k_folds_y

def load_k_fold(k_folds_dir: str, fold_no: int):
    x = np.load(os.path.join(k_folds_dir, f'fold_{fold_no}_x.npy'))
    y = np.load(os.path.join(k_folds_dir, f'fold_{fold_no}_y.npy'))
    return x, y

def create_datasets_from_folds(k_folds_dir: str, no_of_folds: int, validation_fold_no: int, test_fold_no: int, pre_pool_value: int):
    validation_dataset: tf.data.Dataset
    test_dataset: tf.data.Dataset
    train_x = np.array([])
    train_y = []

    max_elements = getattr(settings.max_train_elements_gpu_can_handle_by_pre_pool, f'{pre_pool_value}', None)

    folds = list(range(no_of_folds))
    np.random.shuffle(folds)
    for i in folds:
        x, y = load_k_fold(k_folds_dir, i)
        if (max_elements != None) and (len(train_x) + len(x) >= max_elements):
            break
    
        if i != validation_fold_no and i != test_fold_no:
            if train_x.any():
                train_x = np.vstack((train_x, x))
            else:
                train_x = x
            train_y.extend(y)

    validation_x, validation_y = load_k_fold(k_folds_dir, validation_fold_no)
    test_x, test_y = load_k_fold(k_folds_dir, validation_fold_no)
    test_x = test_x[0:settings.test_data_size]
    test_y = test_y[0:settings.test_data_size]

    if (not settings.use_validation_split_only_by_max_elements_set) or (max_elements != None):
        validation_data_size = int(settings.validation_data_split * len(train_x))
        validation_x = validation_x[0:validation_data_size]
        validation_y = validation_y[0:validation_data_size]

    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(len(train_x)).repeat().batch(settings.training_batch_size)
    validation_dataset = tf.data.Dataset.from_tensor_slices((validation_x, validation_y)).repeat().batch(settings.training_batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(settings.training_batch_size)
    return train_dataset, validation_dataset, test_dataset, len(train_x), train_x[0].shape
   

def train(model: keras.Sequential, training_dataset, validation_dataset, train_length, train_dir: str, class_weights = None) -> keras.Sequential:
    log_dir = os.path.join(train_dir, 'log', datetime.datetime.now().strftime('%Y.%m.%d-%H.%M.%S'))
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath = os.path.join(train_dir, 'checkpoint.keras'),
        monitor = 'val_accuracy',
        mode = 'max',
        save_best_only = True)

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=0,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=True,
        start_from_epoch=15
    )

    model.fit(
        training_dataset,
        steps_per_epoch = train_length // settings.training_batch_size,
        epochs = settings.training_epochs,
        validation_data  = validation_dataset,
        validation_steps = settings.validation_steps,
        callbacks = [tensorboard_callback, model_checkpoint_callback, early_stopping],
        class_weight = class_weights
    )

    model.save(os.path.join(train_dir, 'trained.keras'))

    return model


def evaluate_model_between_checkpoint_model(current_model: keras.Sequential, test_dataset, train_dir: str):
    checkpoint_model = keras.saving.load_model(os.path.join(train_dir, 'checkpoint.keras'))

    start = time.perf_counter()
    metrics = current_model.evaluate(test_dataset)
    checkpoint_metrics = checkpoint_model.evaluate(test_dataset)
    time_metric = int(((time.perf_counter() - start) * 1000) / 2)

    if metrics[1] >= checkpoint_metrics[1]:
        current_model.save(os.path.join(train_dir, 'final.keras'))
        return metrics[0], metrics[1], time_metric
    else:
        checkpoint_model.save(os.path.join(train_dir, 'final.keras'))
        return checkpoint_metrics[0], checkpoint_metrics[1], time_metric
