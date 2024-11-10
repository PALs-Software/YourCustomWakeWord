import sys
sys.path.insert(0, "./")
from modules.settings import get_settings
from modules.train import load_k_fold
import tensorflow as tf
import os
import keras

# Convert model to tflite -> only for the microcontroller model necessary!

model_dir_name = "final_model"
final_pre_pooling_setting = 6
conversion_type = 'Dynamic range quantization' # Dynamic range quantization is a recommended starting point because it provides reduced memory usage and faster computation without you having to provide a representative dataset for calibration. This type of quantization, statically quantizes only the weights from floating point to integer at conversion time, which provides 8-bits of precision
conversion_type = 'Full integer quantization' # You can get further latency improvements, reductions in peak memory usage, and compatibility with integer only hardware devices or accelerators by making sure all model math is integer quantized.

settings = get_settings()
model_path = os.path.join(settings.model_training_dir, model_dir_name, 'final.keras')
model =  keras.models.load_model(model_path)
converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]   

if conversion_type == 'Full integer quantization':
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    x, y = load_k_fold(os.path.join(settings.training_data_dir, 'folds', f'pre_pooling_{final_pre_pooling_setting}'), 0)
    
    def representative_dataset_generator():
        for i in range(0, len(x), 100):
            yield [x[i:i+100]]

    converter.representative_dataset = representative_dataset_generator

tflite_model = converter.convert()
destination_path = os.path.join(settings.model_training_dir, model_dir_name, 'final.tflite')
with open(destination_path, 'wb') as f:
    f.write(tflite_model)

print(f'Converted model is saved at "{destination_path}"')