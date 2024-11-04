import sys
sys.path.insert(0, "./")
from modules.settings import get_settings
from modules.train import load_k_fold
import tensorflow as tf
import tf2onnx
import onnx
import os
import keras

final_pre_pooling_setting = 6
onnx_opset_version = 18
model_dir_name = "final_model"

settings = get_settings()
model =  keras.models.load_model(os.path.join(settings.model_training_dir, model_dir_name, 'final.keras'))
model.output_names=['output'] # Needed as fix in some tensorflow versions
x, y = load_k_fold(os.path.join(settings.training_data_dir, 'folds', f'pre_pooling_{final_pre_pooling_setting}'), 0)
example_entry = x[0]
input_signature = [tf.TensorSpec([None, example_entry.shape[0], example_entry.shape[1], example_entry.shape[2]], tf.float32, name='input')]

onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=onnx_opset_version)

destination_path = os.path.join(settings.model_training_dir, model_dir_name, 'final.onnx')
onnx.save(onnx_model, destination_path)

print(f'Converted model is saved at "{destination_path}"')