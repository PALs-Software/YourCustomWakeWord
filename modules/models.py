import keras
from keras import layers

def create_dnn_model(input_shape,
                     kernel_regularizer,
                     no_of_dense_layers,
                     layer_units_start,
                     layer_units_min,
                     half_units_per_layer,
                     with_layer_norm,
                     dropout):

    layer_units = get_layer_units(no_of_dense_layers, layer_units_start, layer_units_min, half_units_per_layer)
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Flatten())

    model.add(layers.Dense(layer_units[0], activation='relu', kernel_regularizer=kernel_regularizer))
    if with_layer_norm:
        model.add(layers.LayerNormalization())

    dropout_position = int(no_of_dense_layers / 2)
    for i in range(1, no_of_dense_layers):
        if (dropout != 0) and (dropout_position == i):
            model.add(layers.Dropout(dropout))
    
        model.add(layers.Dense(layer_units[i], activation='relu', kernel_regularizer=kernel_regularizer))
        if with_layer_norm:
            model.add(layers.LayerNormalization())
    
    model.add(layers.Dense(1, activation='sigmoid', kernel_regularizer=kernel_regularizer))    
    return model

def create_dnn_model_extended(input_shape,
                     kernel_regularizer,
                     no_of_dense_layers,
                     layer_units_start,
                     layer_units_min,
                     half_units_per_layer,
                     with_layer_norm):
    
    layer_units = get_layer_units(no_of_dense_layers, layer_units_start, layer_units_min, half_units_per_layer)
    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Flatten())
    model.add(layers.Dense(layer_units[0], activation='linear', kernel_regularizer=kernel_regularizer))
    if with_layer_norm:
        model.add(layers.LayerNormalization())
    model.add(layers.ReLU())
    
    for i in range(1, no_of_dense_layers - 1):
        model.add(layers.Dense(layer_units[i], activation='linear', kernel_regularizer=kernel_regularizer))
        if with_layer_norm:
            model.add(layers.LayerNormalization())
        model.add(layers.ReLU())
    
    model.add(layers.Dense(layer_units[no_of_dense_layers - 1], activation='linear', kernel_regularizer=kernel_regularizer))      
    model.add(layers.Dense(1, activation='sigmoid', kernel_regularizer=kernel_regularizer))
    return model

def create_cnn_model(input_shape,
                     kernel_regularizer,
                     no_of_conv_layers,
                     conv_filters_start,
                     conv_filters_min,
                     half_conf_filters_per_layer,
                     conv_kernel,
                     pool_size,
                     dropout,
                     no_of_dense_layers,
                     layer_units_start,
                     layer_units_min,
                     half_units_per_layer,
                     max_conv_max_pooling_layers):
    
    conf_filters = get_layer_units(no_of_conv_layers, conv_filters_start, conv_filters_min, half_conf_filters_per_layer)
    dense_layer_units = get_layer_units(no_of_dense_layers, layer_units_start, layer_units_min, half_units_per_layer)
    max_pooling_layers_added = 0

    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))

    model.add(layers.Conv2D(conf_filters[0], conv_kernel, padding='same', activation='relu', kernel_regularizer=kernel_regularizer))
    model.add(layers.MaxPooling2D(pool_size=pool_size))
    max_pooling_layers_added += 1
    
    for i in range(no_of_conv_layers - 1):
        model.add(layers.Conv2D(conf_filters[i + 1], conv_kernel, padding='same', activation='relu', kernel_regularizer=kernel_regularizer))

        if max_pooling_layers_added < max_conv_max_pooling_layers:
            max_pooling_layers_added += 1
            model.add(layers.MaxPooling2D(pool_size=pool_size))

    model.add(layers.Flatten())

    if dropout != 0:
        model.add(layers.Dropout(dropout))

    for i in range(no_of_dense_layers):
        model.add(layers.Dense(dense_layer_units[i], activation='relu', kernel_regularizer=kernel_regularizer))
    
    model.add(layers.Dense(1, activation='sigmoid', kernel_regularizer=kernel_regularizer))
    return model

def create_rnn_model(input_shape,
               kernel_regularizer,
               no_of_lstms,
               lstms_units_start,
               lstms_units_min,
               lstms_units_half_per_layer,
               lstm_activation,
               activation_recurrent_activation,
               dropout,
               recurrent_dropout,
               dense_dropout,
               no_of_dense_layers,
               layer_units_start,
               layer_units_min,
               half_units_per_layer,
               dense_layer_activation
               ):
    
    lstm_units = get_layer_units(no_of_lstms, lstms_units_start, lstms_units_min, lstms_units_half_per_layer)
    dense_layer_units = get_layer_units(no_of_dense_layers, layer_units_start, layer_units_min, half_units_per_layer)
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_shape[0], input_shape[1])))

    for i in range(no_of_lstms):
        model.add(layers.Bidirectional(layers.LSTM(units=lstm_units[i], activation=lstm_activation, recurrent_activation=activation_recurrent_activation,
                                                   dropout=dropout, recurrent_dropout=recurrent_dropout, kernel_regularizer=kernel_regularizer,
                                                   return_sequences = i != (no_of_lstms - 1))))
    
    if dense_dropout != 0:
        model.add(layers.Dropout(dense_dropout))

    for i in range(no_of_dense_layers):
        model.add(layers.Dense(dense_layer_units[i], activation=dense_layer_activation, kernel_regularizer=kernel_regularizer))
    
    model.add(layers.Dense(1, activation='sigmoid', kernel_regularizer=kernel_regularizer))
    return model

def get_layer_units(no_of_layers, units_start, units_min, half_units_per_layer):
    units = []
    for i in range(no_of_layers):
        if half_units_per_layer:
            units.append(max(int(units_start / pow(2, i)), units_min))
        else:
            units.append(units_start)

    return units