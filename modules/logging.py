import sys
sys.path.insert(0, "./")
from modules.settings import get_settings
import os
import csv
import datetime

settings = get_settings()

class ModelResultLogger:
    def __init__(self, dir_path: str):
        self._create_log_file(dir_path, ['model_name', 'test_loss', 'test_accuracy', 'time', 'type', 'fold_id', 'train_size', 'number_of_params', 'pre_pooling', 'kernel_regularizer',
                                         'dense_no_of_layers', 'dense_layer_units_start', 'dense_layer_units_min', 'dense_half_units_per_layer', 'dense_with_layer_norm', 'dense_dropout', 'dense_layer_activation_setting',
                                         'conv_no_of_layers', 'conv_filters_start', 'conv_filters_min', 'conv_half_units_per_layer', 'conv_kernel', 'conv_pool_size',
                                         'lstm_no_of_layers', 'lstm_units_start', 'lstm_units_min', 'lstm_units_half_per_layer', 'lstm_activation', 'lstm_recurrent_activation', 'lstm_dropout', 'lstm_recurrent_dropout'
        ])

    def _create_log_file(self, dir_path: str, field_names):
        self.file = open(os.path.join(dir_path, f'{datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S")}_model_results.csv'), "w", newline='')
        self.csv_writer = csv.writer(self.file, delimiter=';')
        self.csv_writer.writerow(field_names)
        self.file.flush()

    def close_log_file(self):
        self.file.close()

    def log(self, model_name, metrics, type, fold_id, train_size, number_of_params, pre_pooling, kernel_regularizer,
            dense_no_of_layers, dense_layer_units_start, dense_layer_units_min, dense_half_units_per_layer, dense_with_layer_norm, dense_dropout, dense_layer_activation_setting,
            conv_no_of_layers, conv_filters_start, conv_filters_min, conv_half_units_per_layer, conv_kernel, conv_pool_size,
            lstm_no_of_layers, lstm_units_start, lstm_units_min, lstm_units_half_per_layer, lstm_activation, lstm_recurrent_activation, lstm_dropout, lstm_recurrent_dropout):
        
        self.csv_writer.writerow([model_name, metrics[0], metrics[1], metrics[2], type, fold_id, train_size, number_of_params, pre_pooling, kernel_regularizer,
                                 dense_no_of_layers, dense_layer_units_start, dense_layer_units_min, dense_half_units_per_layer, dense_with_layer_norm, dense_dropout, dense_layer_activation_setting,
                                 conv_no_of_layers, conv_filters_start, conv_filters_min, conv_half_units_per_layer, conv_kernel, conv_pool_size,
                                 lstm_no_of_layers, lstm_units_start, lstm_units_min, lstm_units_half_per_layer, lstm_activation, lstm_recurrent_activation, lstm_dropout, lstm_recurrent_dropout])
        self.file.flush()

def log_current_setting_no(current_setting_no: int):
    path = os.path.join(settings.work_space, "current_setting_no.log")
    f = open(path, "w")
    f.write(f'{current_setting_no}')
    f.close()

def get_last_tested_setting_no_from_log() -> int:
    path = os.path.join(settings.work_space, "current_setting_no.log")
    if not os.path.exists(path):
        return 0
    
    f = open(path, "r")
    last_tested_setting_no = f.read()
    f.close()
    return int(last_tested_setting_no)