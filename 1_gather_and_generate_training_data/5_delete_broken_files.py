
import sys
sys.path.insert(0, "./")
from modules.settings import get_settings
from tqdm import tqdm
import os
import librosa
import glob

settings = get_settings()
work_space = settings.work_space

audio_files = glob.glob(settings.raw_training_data_dir + '/**/*.wav', recursive=True)
for audio_file in tqdm(audio_files):
    try:
        y, sr = librosa.load(audio_file)
    except:
        print(f'Delete broken audio file: {audio_file}')
        os.remove(audio_file)