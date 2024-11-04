import sys
sys.path.insert(0, "./")
from modules.settings import get_settings
from modules.train import create_datasets_from_folds, train, evaluate_model_between_checkpoint_model
from modules.models import create_dnn_model, create_dnn_model_extended, create_cnn_model, create_rnn_model
import os
import keras

model_name = 'final_model'
settings = get_settings()

model_output_dir = os.path.join(settings.model_training_dir, model_name)
if not os.path.exists(model_output_dir):
    os.makedirs(model_output_dir)

model_type = "cnn"
pre_pooling = 6
class_weights = None # { 0: 1.5, 1: 1.0 } # None for default weights 1:1 or adjust it to increase weights on negatives or positives to increase/decrease fp or fn rate

kernel_regularizer = None
dense_no_of_layers = 2
dense_layer_units_start = 64
dense_layer_units_min = 32
dense_half_units_per_layer = True
dense_with_layer_norm_setting = False
dense_layer_activation = "relu"
dense_dropout = 0.2

conv_no_of_layers = 6
conv_filters_start = 32
conv_filters_min = 32
conv_half_units_per_layer = False 
conv_kernel = (3,3)
conv_pool_size = (2,2)
conv_max_pooling_layers_max = 3

lstm_no_of_layers = 4
lstm_units_start = 128
lstm_units_min = 64
lstm_units_half_per_layer = True
lstm_activation = "tanh"
lstm_recurrent_activation = "sigmoid"
lstm_dropout = 0
lstm_recurrent_dropout = 0

train_dataset, validation_dataset, test_dataset, train_data_length, input_shape = create_datasets_from_folds(os.path.join(settings.training_data_dir, 'folds', f'pre_pooling_{pre_pooling}'), settings.k_folds, 0, 1, pre_pooling)

if model_type == 'dnn':
    model = create_dnn_model(input_shape, kernel_regularizer, dense_no_of_layers, dense_layer_units_start, dense_layer_units_min, dense_half_units_per_layer, dense_with_layer_norm_setting, dense_dropout)
elif model_type == 'dnn_extended':
    model = create_dnn_model_extended(input_shape, kernel_regularizer, dense_no_of_layers, dense_layer_units_start, dense_layer_units_min, dense_half_units_per_layer, dense_with_layer_norm_setting)
elif model_type == 'cnn':
    model = create_cnn_model(input_shape, kernel_regularizer, conv_no_of_layers, conv_filters_start, conv_filters_min, conv_half_units_per_layer, conv_kernel, conv_pool_size, dense_dropout, dense_no_of_layers, dense_layer_units_start, dense_layer_units_min, dense_half_units_per_layer, conv_max_pooling_layers_max)         
elif model_type == 'rnn':
    model = create_rnn_model(input_shape, kernel_regularizer, lstm_no_of_layers, lstm_units_start, lstm_units_min, lstm_units_half_per_layer, lstm_activation, lstm_recurrent_activation, lstm_dropout, lstm_recurrent_dropout, dense_dropout, dense_no_of_layers, dense_layer_units_start, dense_layer_units_min, dense_half_units_per_layer, dense_layer_activation)

model.compile(optimizer = 'adam', loss = keras.losses.BinaryCrossentropy(), metrics = ['accuracy'])
model.summary()

model = train(model, train_dataset, validation_dataset, train_data_length, model_output_dir)
metrics = evaluate_model_between_checkpoint_model(model, test_dataset, model_output_dir)

print(f'Accuracy: {metrics[1]}')
print(f'Loss: {metrics[0]}')
print(f'Execution Test Time: {metrics[2]}ms')
print(f'Model is saved at "{model_output_dir}" directory')