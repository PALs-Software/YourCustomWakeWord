import sys
sys.path.insert(0, "./")
from modules.settings import get_settings
import itertools as it
from tqdm import tqdm
import os
from itertools import chain
from pathlib import Path
import librosa
import soundfile

settings = get_settings()
work_space = settings.work_space

wake_word_dir = os.path.join(settings.raw_wake_word_training_dir, 'azure')
negate_words_dir = os.path.join(settings.raw_negative_words_training_dir, 'azure')

files_to_check = chain(Path(wake_word_dir).glob("**/*.wav"),
                        Path(negate_words_dir).glob("**/*.wav"))

for file_path in tqdm(files_to_check, desc='Resample files if necessary'):
    data, sample_rate = librosa.load(file_path, sr=None)
    if sample_rate == settings.sample_rate:
        continue
    
    print(f'Resample file {file_path}')
    resampled_data = librosa.resample(data, orig_sr=sample_rate, target_sr=settings.sample_rate)
    soundfile.write(file_path, resampled_data, settings.sample_rate)