import sys
sys.path.insert(0, "./")
from modules.settings import get_settings

import os
import sys
from git import Repo
import uuid
from tqdm import tqdm
import librosa
import soundfile
import logging
from piper import PiperVoice
from piper.download import ensure_voice_exists, get_voices
from pathlib import Path
from typing import Any, Dict, Tuple
import itertools as it
import wave
from urllib.request import urlretrieve
from modules.progress import show_progress
from itertools import chain

settings = get_settings()
work_space = settings.work_space

piper_models_dir = os.path.join(work_space, 'piper_models')
piper_sample_generator_dir = os.path.join(work_space, 'piper-sample-generator')
piper_sample_generator_models_dir = os.path.join(piper_sample_generator_dir, 'models')

wake_word_dir = os.path.join(settings.raw_wake_word_training_dir, 'piper')
wake_word_dir_generator = os.path.join(settings.raw_wake_word_training_dir, 'piper_generator')
negate_words_dir = os.path.join(settings.raw_negative_words_training_dir, 'piper')
negate_words_dir_generator = os.path.join(settings.raw_negative_words_training_dir, 'piper_generator')

if not os.path.exists(wake_word_dir):
    os.makedirs(wake_word_dir)
if not os.path.exists(wake_word_dir_generator):
    os.makedirs(wake_word_dir_generator)
if not os.path.exists(negate_words_dir):
    os.makedirs(negate_words_dir)
if not os.path.exists(negate_words_dir_generator):
    os.makedirs(negate_words_dir_generator)

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

if not os.path.exists(piper_sample_generator_dir):
    print('Clone piper-sample-generator repository')
    Repo.clone_from("https://github.com/rhasspy/piper-sample-generator", piper_sample_generator_dir) 

def prepare_piper_model(model: str):
    if not os.path.exists(piper_models_dir):
        os.mkdir(piper_models_dir)

    model_path = os.path.join(piper_models_dir, f"{model}.onnx")
    config_path = os.path.join(piper_models_dir, f"{model}.onnx.json")

    if not os.path.exists(model_path):        
        print(f'Start downloading voice model {model}')
        voices_info = get_voices(piper_models_dir)

        aliases_info: Dict[str, Any] = {}
        for voice_info in voices_info.values():
            for voice_alias in voice_info.get("aliases", []):
                aliases_info[voice_alias] = {"_is_alias": True, **voice_info}

        voices_info.update(aliases_info)
        ensure_voice_exists(model, piper_models_dir, piper_models_dir, voices_info)

    return Path(model_path), Path(config_path)

def generate_audio_samples_with_piper_voices(language_code: str,
                                             text: str,
                                             output_dir: str,
                                             voice_models_to_ignore: list):
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    voices = get_voices(piper_models_dir)
    voices_for_language = [voice['key'] for voice in voices.values() if voice['language']['code'] == language_code]
    voices_for_language = [voice for voice in voices_for_language if voice not in voice_models_to_ignore]

    progress_bar_iter = tqdm(voices_for_language, f'Generate samples')
    for voice in progress_bar_iter:

        if ("?" in text) and (voice in settings.piper_models_to_wake_word_with_question):
            continue

        progress_bar_iter.set_description(f'Generate samples for voice "{voice}"')
        model_path, config_path = prepare_piper_model(voice)
        piper_voice = PiperVoice.load(model_path, config_path=config_path)

        settings_iter = it.product(            
                settings.piper_length_scales,
                settings.piper_noise_scales,
                settings.piper_noise_scale_ws,
        )
        for length_scale, noise_scale, noise_scale_w in tqdm(settings_iter):
            file_name = f'{language_code}_{voice}_{length_scale}_{noise_scale}_{noise_scale_w}_{text.replace("?", "_q")}.wav'
            with wave.open(os.path.join(output_dir, file_name), "wb") as wav_file:
                piper_voice.synthesize(text, wav_file, length_scale=length_scale, noise_scale=noise_scale, noise_w=noise_scale_w)

for wake_word in settings.wake_words:
    generate_audio_samples_with_piper_voices(settings.piper_language_code, wake_word, wake_word_dir, voice_models_to_ignore=settings.piper_models_to_ignore)

for negative_word in settings.negative_words:
    generate_audio_samples_with_piper_voices(settings.piper_language_code, negative_word, negate_words_dir, voice_models_to_ignore=settings.piper_models_to_ignore)

def prepare_piper_generator_model(voice):
    model_path = os.path.join(piper_sample_generator_models_dir, voice)
    if not os.path.exists(model_path):
        urlretrieve(f"https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/{voice}", model_path, show_progress)

    return model_path

print(piper_sample_generator_dir)
sys.path.insert(0, os.path.abspath(piper_sample_generator_dir))
import generate_samples

piper_generator_model_path = prepare_piper_generator_model(settings.piper_generator_model)
for wake_word in settings.wake_words:
    generate_samples.generate_samples(
        model=piper_generator_model_path,
        text=wake_word,
        max_samples=settings.piper_generator_sample_size,
        batch_size=50, noise_scales=settings.piper_noise_scales, noise_scale_ws=settings.piper_noise_scale_ws, length_scales=settings.piper_length_scales,
        output_dir=wake_word_dir_generator,
        auto_reduce_batch_size=True,
        file_names=[uuid.uuid4().hex + ".wav" for i in range(settings.piper_generator_sample_size)]
    )

for negative_word in settings.negative_words:
    generate_samples.generate_samples(
        model=piper_generator_model_path,
        text=negative_word, max_samples=settings.piper_generator_sample_size_per_negative_word,
        batch_size=50, noise_scales=settings.piper_noise_scales, noise_scale_ws=settings.piper_noise_scale_ws, length_scales=settings.piper_length_scales,
        output_dir= negate_words_dir_generator, auto_reduce_batch_size=True,
        file_names=[uuid.uuid4().hex + ".wav" for i in range(settings.piper_generator_sample_size_per_negative_word)]
    )

files_to_check = chain(Path(wake_word_dir).glob("**/*.wav"),
                        Path(negate_words_dir).glob("**/*.wav"),
                        Path(wake_word_dir_generator).glob("**/*.wav"),
                        Path(negate_words_dir_generator).glob("**/*.wav"))
for file_path in tqdm(files_to_check, desc='Resample files if necessary'):
    data, sample_rate = librosa.load(file_path, sr=None)
    if sample_rate == settings.sample_rate:
        continue

    resampled_data = librosa.resample(data, orig_sr=sample_rate, target_sr=settings.sample_rate)
    soundfile.write(file_path, resampled_data, settings.sample_rate)
