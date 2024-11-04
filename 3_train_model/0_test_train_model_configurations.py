import sys
sys.path.insert(0, "./")
from modules.settings import get_settings
from modules.train import create_datasets_from_folds, train, evaluate_model_between_checkpoint_model
from modules.models import create_dnn_model, create_dnn_model_extended, create_cnn_model, create_rnn_model
from modules.logging import ModelResultLogger, log_current_setting_no
import os
from keras import regularizers
import itertools as it
import keras
from tqdm import tqdm
import gc

settings = get_settings()

force_reset_after = 10
start_by_setting_no = 0
if len(sys.argv) >= 1:
    if sys.argv[1].startswith('--StartBySettingNo'):
        start_by_setting_no = int(sys.argv[1].split(' ')[1])
        print(f'Start by setting no: "{start_by_setting_no}"')

if not os.path.exists(settings.model_training_dir):
    os.makedirs(settings.model_training_dir)

logger = ModelResultLogger(settings.model_training_dir)

k_folds_settings = [[0, 4], [3,8]]

kernel_regularizer_settings = [None, regularizers.l2(0.001)]
no_of_dense_layers_settings = [2, 4, 6]
layer_units_start_settings = [128, 64, 32]
layer_units_min_settings = [64, 32]
half_units_per_layer_settings = [True, False]
with_layer_norm_settings = [True, False]
dropout = [0, 0.2]

no_of_conv_layers_settings = [2, 4, 6]
conv_filters_start_settings = [32, 16]
conv_filters_min_settings = [32]
conv_kernel_settings = [(3,3), (5,5)]
conv_pool_size_settings = [(2,2)]
conv_no_of_dense_layers_settings = [2, 4]
conv_layer_units_start_settings = [64]
conv_layer_units_min_settings = [32]
max_conv_max_pooling_layers = 3

no_of_lstms_layers_settings = [2, 4]
lstms_units_start_settings = [128, 64]
lstms_units_min_settings = [32]
lstm_activation_settings = ["tanh"] # only valid activation function for gpu support
activation_recurrent_activation = ["sigmoid"] # only valid activation function for gpu support
lstm_dropout = [0.0] # only valid setting for gpu support
lstm_dense_dropout = [0.0, 0.2]
lstm_dense_layer_activation = ["relu"]
lstm_no_of_dense_layers_settings = [2, 4]
lstm_layer_units_start_settings = [128, 64]
lstm_layer_units_min_settings = [32]

dnn_settings = list(it.product(            
    kernel_regularizer_settings, # kernel_regularizer
    no_of_dense_layers_settings, # no_of_dense_layers
    layer_units_start_settings, # layer_units_start
    layer_units_min_settings, # layer_units_min
    half_units_per_layer_settings, # half_units_per_layer
    with_layer_norm_settings, # with_layer_norm
    dropout, # dropout
))

dnn_extended_settings = list(it.product(
    kernel_regularizer_settings, # kernel_regularizer
    no_of_dense_layers_settings, # no_of_dense_layers
    layer_units_start_settings, # layer_units_start
    layer_units_min_settings, # layer_units_min
    half_units_per_layer_settings, # half_units_per_layer
    with_layer_norm_settings, # with_layer_norm
))

cnn_settings = list(it.product(
    kernel_regularizer_settings, # kernel_regularizer
    no_of_conv_layers_settings, # no_of_conv_layers
    conv_filters_start_settings, # conv_filters_start
    conv_filters_min_settings, # conv_filters_min
    half_units_per_layer_settings, # half_conf_filters_per_layer
    conv_kernel_settings, # conv_kernel
    conv_pool_size_settings, # pool_size
    dropout, # dropout
    conv_no_of_dense_layers_settings, # no_of_dense_layers
    conv_layer_units_start_settings, # layer_units_start
    conv_layer_units_min_settings, # layer_units_min
    half_units_per_layer_settings, # half_units_per_layer
))

rnn_settings = list(it.product(
    kernel_regularizer_settings, # kernel_regularizer
    no_of_lstms_layers_settings, # no_of_lstms
    lstms_units_start_settings, # lstms_units_start
    lstms_units_min_settings, # lstms_units_min
    half_units_per_layer_settings, # lstms_units_half_per_layer
    lstm_activation_settings, # lstm_activation
    activation_recurrent_activation, # activation_recurrent_activation
    lstm_dropout, # dropout
    lstm_dropout, # recurrent_dropout
    lstm_dense_dropout, # dense_dropout
    lstm_no_of_dense_layers_settings, # no_of_dense_layers
    lstm_layer_units_start_settings, # layer_units_start
    lstm_layer_units_min_settings, # layer_units_min
    half_units_per_layer_settings, # half_units_per_layer
    lstm_dense_layer_activation, # dense_layer_activation
))

# filter out combinations that do not make much sense
filtered_dnn_settings = []
for setting in dnn_settings:
    kernel_regularizer_setting, no_of_dense_layers_setting, layer_units_start_setting, layer_units_min_setting, half_units_per_layer_setting, with_layer_norm_setting, dropout_setting = setting
    if half_units_per_layer_setting and (layer_units_min_setting >= layer_units_start_setting):
        continue
    if (not half_units_per_layer_setting) and (layer_units_min_setting != layer_units_min_settings[0]):
        continue
    filtered_dnn_settings.append(setting)

filtered_dnn_extended_settings = []
for setting in dnn_extended_settings:
    kernel_regularizer_setting, no_of_dense_layers_setting, layer_units_start_setting, layer_units_min_setting, half_units_per_layer_setting, with_layer_norm_setting = setting
    if half_units_per_layer_setting and (layer_units_min_setting >= layer_units_start_setting):
        continue
    if (not half_units_per_layer_setting) and (layer_units_min_setting != layer_units_min_settings[0]):
        continue
    filtered_dnn_extended_settings.append(setting)

filtered_cnn_settings = []
for setting in cnn_settings:
    kernel_regularizer_setting, no_of_conv_layers_setting, conv_filters_start_setting, conv_filters_min_setting, half_units_per_layer_setting, conv_kernel_setting, conv_pool_size_setting, dropout_setting, conv_no_of_dense_layers_setting, conv_layer_units_start_setting, conv_layer_units_min_setting, dense_half_units_per_layer_setting = setting
    if half_units_per_layer_setting and (conv_filters_min_setting >= conv_filters_start_setting):
        continue
    if (not half_units_per_layer_setting) and (conv_filters_min_setting != conv_filters_min_settings[0]):
        continue

    if dense_half_units_per_layer_setting and (conv_layer_units_min_setting >= conv_layer_units_start_setting):
        continue
    if (not dense_half_units_per_layer_setting) and (conv_layer_units_min_setting != conv_layer_units_min_settings[0]):
        continue

    filtered_cnn_settings.append(setting)

filtered_rnn_settings = []
for setting in rnn_settings:
    kernel_regularizer_setting, no_of_lstms_layers_setting, lstms_units_start_setting, lstms_units_min_setting, lstms_units_half_per_layer_setting, lstm_activation_setting, activation_recurrent_activation_setting, dropout_setting, recurrent_dropout_setting, dense_dropout_setting, no_of_dense_layers_setting, layer_units_start_setting, layer_units_min_setting, half_units_per_layer_setting, dense_layer_activation_setting = setting
    if lstms_units_half_per_layer_setting and (lstms_units_min_setting >= lstms_units_start_setting):
        continue
    if (not lstms_units_half_per_layer_setting) and (lstms_units_min_setting != lstms_units_min_settings[0]):
        continue

    if half_units_per_layer_setting and (layer_units_min_setting >= layer_units_start_setting):
        continue
    if (not half_units_per_layer_setting) and (layer_units_min_setting != lstm_layer_units_min_settings[0]):
        continue

    if layer_units_start_setting > lstms_units_start_setting:
        continue

    filtered_rnn_settings.append(setting)

pre_pooling_and_k_folds_settings_length = len(settings.pre_pooling_spectrogram_features_values_to_test) * len(k_folds_settings)
print(f'Test "{len(filtered_dnn_settings) * pre_pooling_and_k_folds_settings_length}" combination of dnn models')
print(f'Test "{len(filtered_dnn_extended_settings) * pre_pooling_and_k_folds_settings_length}" combination of extended dnn models')
print(f'Test "{len(filtered_cnn_settings) * pre_pooling_and_k_folds_settings_length}" combination of cnn models')
print(f'Test "{len(filtered_rnn_settings) * pre_pooling_and_k_folds_settings_length}" combination of rnn models')
total_settings = len(filtered_dnn_settings) * pre_pooling_and_k_folds_settings_length + len(filtered_dnn_extended_settings) * pre_pooling_and_k_folds_settings_length + len(filtered_cnn_settings) * pre_pooling_and_k_folds_settings_length + len(filtered_rnn_settings) * pre_pooling_and_k_folds_settings_length
total_tqdm = tqdm(range(total_settings), 'Total')

current_test_no = 0
pre_pooling_settings = tqdm(settings.pre_pooling_spectrogram_features_values_to_test)
for pre_pooling_setting in pre_pooling_settings:
    pre_pooling_settings.set_description(f'Train models for pre_pooling setting {pre_pooling_setting}')
        
    for k_fold in tqdm(k_folds_settings):
        fold_no = f'{k_fold[0]}.{k_fold[1]}'
        train_dataset, validation_dataset, test_dataset, train_data_length, input_shape = create_datasets_from_folds(os.path.join(settings.training_data_dir, 'folds', f'pre_pooling_{pre_pooling_setting}'), settings.k_folds, k_fold[0], k_fold[1], pre_pooling_setting)

        for setting in tqdm(filtered_dnn_settings):
            total_tqdm.update(1)
            current_test_no += 1
            if current_test_no < start_by_setting_no:
                continue

            kernel_regularizer_setting, no_of_dense_layers_setting, layer_units_start_setting, layer_units_min_setting, half_units_per_layer_setting, with_layer_norm_setting, dropout_setting = setting
            model = create_dnn_model(input_shape, kernel_regularizer_setting, no_of_dense_layers_setting, layer_units_start_setting, layer_units_min_setting, half_units_per_layer_setting, with_layer_norm_setting, dropout_setting)
            model.compile(optimizer = 'adam', loss = keras.losses.BinaryCrossentropy(), metrics = ['accuracy'])
            model.summary()

            kernel_setting_f = kernel_regularizer_setting if kernel_regularizer_setting == None else kernel_regularizer_setting.l2
            model_name = f't.dnn_tl.{train_data_length}_f.{fold_no}_p.{pre_pooling_setting}_kr.{kernel_setting_f}_l.{no_of_dense_layers_setting}_u.{layer_units_start_setting}_um.{layer_units_min_setting}_h.{half_units_per_layer_setting}_n.{with_layer_norm_setting}_d.{dropout_setting}'
            model_train_dir = os.path.join(settings.model_training_dir, model_name)
            model = train(model, train_dataset, validation_dataset, train_data_length, model_train_dir)
            
            metrics = evaluate_model_between_checkpoint_model(model, test_dataset, model_train_dir)
            logger.log(model_name, metrics, 'dnn', fold_no, train_data_length, model.count_params(), pre_pooling_setting, kernel_setting_f,
                       no_of_dense_layers_setting, layer_units_start_setting, layer_units_min_setting, half_units_per_layer_setting, with_layer_norm_setting, dropout_setting,  'relu',
                       '', '', '', '', '', '',
                       '', '', '', '', '', '', '', '')
            
            log_current_setting_no(current_test_no)
            del model
            keras.backend.clear_session(free_memory=True)
            gc.collect()
            if (force_reset_after != 0) and ((current_test_no - start_by_setting_no) > force_reset_after):
                exit()

        for setting in tqdm(filtered_dnn_extended_settings):
            total_tqdm.update(1)
            current_test_no += 1
            if current_test_no < start_by_setting_no:
                continue

            kernel_regularizer_setting, no_of_dense_layers_setting, layer_units_start_setting, layer_units_min_setting, half_units_per_layer_setting, with_layer_norm_setting = setting
            model = create_dnn_model_extended(input_shape, kernel_regularizer_setting, no_of_dense_layers_setting, layer_units_start_setting, layer_units_min_setting, half_units_per_layer_setting, with_layer_norm_setting)
            model.compile(optimizer = 'adam', loss = keras.losses.BinaryCrossentropy(), metrics = ['accuracy'])
            model.summary()

            kernel_setting_f = kernel_regularizer_setting if kernel_regularizer_setting == None else kernel_regularizer_setting.l2
            model_name = f't.dnnextended_tl.{train_data_length}_f.{fold_no}_p.{pre_pooling_setting}_kr.{kernel_setting_f}_l.{no_of_dense_layers_setting}_u.{layer_units_start_setting}_um.{layer_units_min_setting}_h.{half_units_per_layer_setting}_n.{with_layer_norm_setting}'
            model_train_dir = os.path.join(settings.model_training_dir, model_name)
            model = train(model, train_dataset, validation_dataset, train_data_length, model_train_dir)
            
            metrics = evaluate_model_between_checkpoint_model(model, test_dataset, model_train_dir)
            logger.log(model_name, metrics, 'dnn_extended', fold_no, train_data_length,  model.count_params(), pre_pooling_setting, kernel_setting_f,
                       no_of_dense_layers_setting, layer_units_start_setting, layer_units_min_setting, half_units_per_layer_setting, with_layer_norm_setting, '',  'linear/relu',
                       '', '', '', '', '', '',
                       '', '', '', '', '', '', '', '')
            
            log_current_setting_no(current_test_no)
            del model
            keras.backend.clear_session(free_memory=True)
            gc.collect()
            if (force_reset_after != 0) and ((current_test_no - start_by_setting_no) > force_reset_after):
                exit()
            
        for setting in tqdm(filtered_cnn_settings):
            total_tqdm.update(1)
            current_test_no += 1
            if current_test_no < start_by_setting_no:
                continue
        
            kernel_regularizer_setting, no_of_conv_layers_setting, conv_filters_start_setting, conv_filters_min_setting, half_units_per_layer_setting, conv_kernel_setting, conv_pool_size_setting, dropout_setting, conv_no_of_dense_layers_setting, conv_layer_units_start_setting, conv_layer_units_min_setting, dense_half_units_per_layer_setting = setting
            model = create_cnn_model(input_shape, kernel_regularizer_setting, no_of_conv_layers_setting, conv_filters_start_setting, conv_filters_min_setting, half_units_per_layer_setting, conv_kernel_setting, conv_pool_size_setting, dropout_setting, conv_no_of_dense_layers_setting, conv_layer_units_start_setting, conv_layer_units_min_setting, dense_half_units_per_layer_setting, max_conv_max_pooling_layers)
            model.compile(optimizer = 'adam', loss = keras.losses.BinaryCrossentropy(), metrics = ['accuracy'])
            model.summary()

            kernel_setting_f = kernel_regularizer_setting if kernel_regularizer_setting == None else kernel_regularizer_setting.l2
            model_name = f't.cnn_tl.{train_data_length}_f.{fold_no}_p.{pre_pooling_setting}_kr.{kernel_setting_f}_cl.{no_of_conv_layers_setting}_cu.{conv_filters_start_setting}_cum.{conv_filters_min_setting}_ch.{half_units_per_layer_setting}_ck.{conv_kernel_setting}_cp.{conv_pool_size_setting}_cd.{dropout_setting}_dl.{conv_no_of_dense_layers_setting}_du.{conv_layer_units_start_setting}_dum.{conv_layer_units_min_setting}_dh.{dense_half_units_per_layer_setting}'
            model_train_dir = os.path.join(settings.model_training_dir, model_name)
            model = train(model, train_dataset, validation_dataset, train_data_length, model_train_dir)            
            metrics = evaluate_model_between_checkpoint_model(model, test_dataset, model_train_dir)
            logger.log(model_name, metrics, 'cnn', fold_no, train_data_length, model.count_params(), pre_pooling_setting, kernel_setting_f,
                       conv_no_of_dense_layers_setting, conv_layer_units_start_setting, conv_layer_units_min_setting, dense_half_units_per_layer_setting, '', dropout_setting, 'relu',
                       no_of_conv_layers_setting, conv_filters_start_setting, conv_filters_min_setting, half_units_per_layer_setting, conv_kernel_setting, conv_pool_size_setting,
                       '', '', '', '', '', '', '', '')
            
            log_current_setting_no(current_test_no)
            del model
            keras.backend.clear_session(free_memory=True)
            gc.collect()
            if (force_reset_after != 0) and ((current_test_no - start_by_setting_no) > force_reset_after):
                exit()
            
        for setting in tqdm(filtered_rnn_settings):
            total_tqdm.update(1)
            current_test_no += 1
            if current_test_no < start_by_setting_no:
                continue
        
            kernel_regularizer_setting, no_of_lstms_layers_setting, lstms_units_start_setting, lstms_units_min_setting, lstms_units_half_per_layer_setting, lstm_activation_setting, activation_recurrent_activation_setting, dropout_setting, recurrent_dropout_setting, dense_dropout_setting, no_of_dense_layers_setting, layer_units_start_setting, layer_units_min_setting, half_units_per_layer_setting, dense_layer_activation_setting = setting
            model = create_rnn_model(input_shape, kernel_regularizer_setting, no_of_lstms_layers_setting, lstms_units_start_setting, lstms_units_min_setting, lstms_units_half_per_layer_setting, lstm_activation_setting, activation_recurrent_activation_setting, dropout_setting, recurrent_dropout_setting, dense_dropout_setting, no_of_dense_layers_setting, layer_units_start_setting, layer_units_min_setting, half_units_per_layer_setting, dense_layer_activation_setting)
            model.compile(optimizer = 'adam', loss = keras.losses.BinaryCrossentropy(), metrics = ['accuracy'])
            model.summary()

            kernel_setting_f = kernel_regularizer_setting if kernel_regularizer_setting == None else kernel_regularizer_setting.l2
            model_name = f't.rnn_tl.{train_data_length}_f.{fold_no}_p.{pre_pooling_setting}_kr.{kernel_setting_f}_ll.{no_of_lstms_layers_setting}_lu.{lstms_units_start_setting}_lum.{lstms_units_min_setting}_lh.{lstms_units_half_per_layer_setting}_la.{lstm_activation_setting}_lra.{activation_recurrent_activation_setting}_lad.{dropout_setting}_lrad.{recurrent_dropout_setting}_dd.{dense_dropout_setting}_dl.{no_of_dense_layers_setting}_du.{layer_units_start_setting}_dum.{layer_units_min_setting}_dh.{half_units_per_layer_setting}_da.{dense_layer_activation_setting}'
            model_train_dir = os.path.join(settings.model_training_dir, model_name)
            model = train(model, train_dataset, validation_dataset, train_data_length, model_train_dir)
            
            metrics = evaluate_model_between_checkpoint_model(model, test_dataset, model_train_dir)
            logger.log(model_name, metrics, 'rnn', fold_no, train_data_length, model.count_params(), pre_pooling_setting, kernel_setting_f,
                       no_of_dense_layers_setting, layer_units_start_setting, layer_units_min_setting, half_units_per_layer_setting, '', dense_dropout_setting, dense_layer_activation_setting,
                       '', '', '', '', '', '',
                       no_of_lstms_layers_setting, lstms_units_start_setting, lstms_units_min_setting, lstms_units_half_per_layer_setting, lstm_activation_setting, activation_recurrent_activation_setting, dropout_setting, recurrent_dropout_setting)
            
            log_current_setting_no(current_test_no)
            del model
            keras.backend.clear_session(free_memory=True)
            gc.collect()
            if (force_reset_after != 0) and ((current_test_no - start_by_setting_no) > force_reset_after):
                exit()

logger.close_log_file()