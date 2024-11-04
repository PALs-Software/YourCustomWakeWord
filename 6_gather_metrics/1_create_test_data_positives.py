# Use with requirements of 1_gather_and_gernate_training_data chapter

import sys
sys.path.insert(0, "./")
from modules.settings import get_settings

import os
import sys
from urllib.request import urlretrieve
from modules.progress import show_progress
import glob

settings = get_settings()

skip_first_x_samples = 2000 # skip entries used for training
no_of_samples_per_word = 5000

work_space = settings.work_space
piper_sample_generator_dir = os.path.join(work_space, 'piper-sample-generator')
piper_sample_generator_models_dir = os.path.join(piper_sample_generator_dir, 'models')
max_file_size = no_of_samples_per_word + skip_first_x_samples

sys.path.insert(0, os.path.abspath(piper_sample_generator_dir))
import generate_samples

def prepare_piper_generator_model(voice):
    model_path = os.path.join(piper_sample_generator_models_dir, voice)
    if not os.path.exists(model_path):
        urlretrieve(f"https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/{voice}", model_path, show_progress)

    return model_path

word_no = 0
piper_generator_model_path = prepare_piper_generator_model(settings.piper_generator_model)
for wake_word in settings.wake_words:
    generate_samples.generate_samples(
        model=piper_generator_model_path,
        text=wake_word,
        max_samples=max_file_size,
        batch_size=50, noise_scales=settings.piper_noise_scales, noise_scale_ws=settings.piper_noise_scale_ws, length_scales=settings.piper_length_scales,
        output_dir=settings.extended_test_positives_dir,
        auto_reduce_batch_size=True,
        file_names=[f'{word_no}_{i:07}.wav' for i in range(max_file_size)]
    )

    i = 0
    audio_files = glob.glob(settings.extended_test_positives_dir + f'/{word_no}_*.wav', recursive=True)
    audio_files.sort()
    for file in audio_files:
        if i >= skip_first_x_samples:
            break

        os.remove(file)
        i += 1    

    word_no += 1