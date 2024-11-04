import os
import json
from types import SimpleNamespace

class settings:
    sample_rate = ''

    base_directory = ''
    work_space = ''

    raw_training_data_dir = ''
    raw_wake_word_training_dir = ''
    raw_negative_words_training_dir = ''

    raw_negative_audio_training_dir = ''
    raw_impulses_training_dir = ''
    raw_background_audio_training_dir = ''

    training_data_dir = ''
    model_training_dir = ''
    wake_word_training_dir = ''
    negative_words_training_dir = ''

    dipco_dir = ''
    extended_test_positives_dir = ''

    wake_words = ''
    negative_words = ''

    common_voice_dataset_languages = ''
    piper_language_code = ''
    piper_models_to_ignore = ''
    piper_models_to_wake_word_with_question = ''
    
    piper_generator_model = ''
    piper_generator_sample_size = ''
    piper_generator_sample_size_per_negative_word = ''
    piper_length_scales = ''
    piper_noise_scales = ''
    piper_noise_scale_ws = ''

    azure_key = ''
    azure_region = ''
    azure_tts_languages = ''
    azure_tts_languages_negative_words = ''

    window_size_in_ms = ''
    window_size_in_samples = ''
    voice_noise_floor = ''
    add_background_audio_possibility = ''
    add_background_audio_impulses_possibility = ''
    max_audio_background_volume = ''
    spectrogram_window_size = ''
    spectrogram_stride = ''
    pre_pooling_spectrogram_features_values_to_test = ''

    no_of_training_samples_to_generate = ''
    negative_data_percentage_of_negative_words = ''
    negative_data_percentage_of_negative_audio = ''
    negative_data_percentage_of_background_audio = ''

    k_folds = ''
    max_train_elements_gpu_can_handle_by_pre_pool = ''
    use_validation_split_only_by_max_elements_set = ''
    validation_data_split = ''
    test_data_size = ''

    training_batch_size = ''
    training_epochs = ''
    validation_steps = ''

def get_settings() -> settings:
    with open('settings.json', 'r') as file:
        settings = json.loads(file.read(), object_hook=lambda d: SimpleNamespace(**d))
       
        settings.base_directory = os.path.abspath(os.getcwd())
        settings.work_space =  os.path.join(settings.base_directory, 'workspace')
        settings.raw_training_data_dir = os.path.join(settings.base_directory, 'workspace', 'raw_training_data')
        settings.raw_wake_word_training_dir = os.path.join(settings.raw_training_data_dir, 'wake_word')
        settings.raw_negative_words_training_dir = os.path.join(settings.raw_training_data_dir, 'negative_words')

        settings.raw_negative_audio_training_dir = os.path.join(settings.raw_training_data_dir, "negative_audio")
        settings.raw_impulses_training_dir = os.path.join(settings.raw_training_data_dir, "impulses")
        settings.raw_background_audio_training_dir = os.path.join(settings.raw_training_data_dir, "background_audio")

        settings.training_data_dir = os.path.join(settings.base_directory, 'workspace', 'training_data')
        settings.model_training_dir = os.path.join(settings.base_directory, 'workspace', 'model_training_dir')        
        settings.wake_word_training_dir = os.path.join(settings.training_data_dir, 'wake_word')
        settings.negative_words_training_dir = os.path.join(settings.training_data_dir, 'negative_words')

        settings.dipco_dir = os.path.join(settings.work_space, 'dipco')
        settings.extended_test_positives_dir = os.path.join(settings.work_space, 'extended_test_positives')

        settings.sample_rate = 16000 
        settings.window_size_in_samples = int((settings.window_size_in_ms / 1000) * settings.sample_rate)
        return settings