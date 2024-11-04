import sys
sys.path.insert(0, "./")
from modules.settings import get_settings
import os
import datasets
from tqdm import tqdm
import scipy
import numpy as np
from urllib.request import urlretrieve
from modules.progress import show_progress
import tarfile
from pathlib import Path
from zipfile import ZipFile
import shutil
import librosa
from huggingface_hub import login

login() # login to hugging face, so you get access to the mozilla-foundation/common_voice_17_0 dataset (Note you must accept that first via the hugging face website (https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0))

settings = get_settings()

# Load MIT_environmental_impulse_responses
output_dir = os.path.join(settings.raw_impulses_training_dir, "MIT_environmental_impulse_responses")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

    print('Load MIT_environmental_impulse_responses dataset')
    dataset = datasets.load_dataset("davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True)
    for row in tqdm(dataset):
        name = row['audio']['path'].split('/')[-1]
        scipy.io.wavfile.write(os.path.join(output_dir, name), 16000, (row['audio']['array']*32767).astype(np.int16))

# Load AudioSet
output_dir = os.path.join(settings.raw_background_audio_training_dir, "audioset")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

    print('Load AudioSet dataset')
    for i in range(1, 10):
        print(f'Load AudioSet dataset part {i} of 9')
        tar_path = os.path.join(output_dir, f"bal_train0{i}.tar")
        urlretrieve(f"https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/bal_train0{i}.tar", os.path.join(output_dir, tar_path), show_progress)

    for i in range(1, 10):
        print(f'Unzip AudioSet dataset part {i} of 9')
        tar_path = os.path.join(output_dir, f"bal_train0{i}.tar")
        tar = tarfile.open(tar_path, "r:")
        tar.extractall(output_dir)
        tar.close()

    audio_files = Path(os.path.join(output_dir, "audio", "bal_train")).glob("**/*.flac")
    for audio_file in tqdm(audio_files):
        try:
            librosa.load(audio_file)
        except:
            print(f'Delete broken audio file: {audio_file}')
            os.remove(audio_file)

    print('Convert AudioSet to 16khz sample rate')
    audioset_dataset = datasets.Dataset.from_dict({"audio": [str(i) for i in Path(os.path.join(output_dir, "audio")).glob("**/*.flac")]})
    audioset_dataset = audioset_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
    for row in tqdm(audioset_dataset):
        name = Path(row['audio']['path']).name.replace(".flac", ".wav")
        scipy.io.wavfile.write(os.path.join(output_dir, name), 16000, (row['audio']['array']*32767).astype(np.int16))


    for i in range(1, 10):
        tar_path = os.path.join(output_dir, f"bal_train0{i}.tar")
        os.remove(tar_path)
    shutil.rmtree(os.path.join(output_dir, "audio"))

#Load Free Music Archive dataset
output_dir = os.path.join(settings.raw_background_audio_training_dir, "fma")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

    print('Load Free Music Archive dataset')
    zip_path = os.path.join(output_dir, "fma_small.zip")
    urlretrieve("https://os.unil.cloud.switch.ch/fma/fma_small.zip", zip_path, show_progress)

    print('Unzip Free Music Archive dataset')
    with ZipFile(zip_path, 'r') as zip:
        zip.extractall(output_dir)
        zip.close()

    audio_files = Path(os.path.join(output_dir, "fma_small")).glob("**/*.mp3")
    for audio_file in tqdm(audio_files):
        try:
            librosa.load(audio_file)
        except:
            print(f'Delete broken audio file: {audio_file}')
            os.remove(audio_file)

    audioset_dataset = datasets.Dataset.from_dict({"audio": [str(i) for i in Path(os.path.join(output_dir, "fma_small")).glob("**/*.mp3")]})
    audioset_dataset = audioset_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
    for row in tqdm(audioset_dataset):
        name = Path(row['audio']['path']).name.replace(".mp3", ".wav")
        scipy.io.wavfile.write(os.path.join(output_dir, name), 16000, (row['audio']['array']*32767).astype(np.int16))


    shutil.rmtree(os.path.join(output_dir, "fma_small"))
    os.remove(zip_path)

#Load Common Voice dataset
output_dir = os.path.join(settings.raw_negative_audio_training_dir, "common_voice")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

    print('Load Common Voice dataset')
    for language in tqdm(settings.common_voice_dataset_languages):
        dataset = datasets.load_dataset("mozilla-foundation/common_voice_17_0", language, split="train", streaming=True, trust_remote_code=True)
        dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16000, mono=True)) 
        dataset_iter = iter(dataset)

        for i in tqdm(range(50000)):
            row = next(dataset_iter)
            sentence = row['sentence']
            for wake_word in settings.wake_words:
                if wake_word in sentence:
                    print(f'Skip row with sentence "{sentence}" because it contains wake word "{wake_word}"')
                    continue
            name = Path(row['path']).name.replace(".mp3", ".wav")
            scipy.io.wavfile.write(os.path.join(output_dir, name), 16000, (row["audio"]["array"]*32767).astype(np.int16))

#Load People's Speech dataset
output_dir = os.path.join(settings.raw_negative_audio_training_dir, "peoples_speech")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

    print("Load People's Speech dataset")
    dataset = datasets.load_dataset("MLCommons/peoples_speech", split="train", streaming=True, trust_remote_code=True)
    dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16000, mono=True)) 
    dataset_iter = iter(dataset)

    for i in tqdm(range(25000)):
        row = next(dataset_iter)
        sentence = row['text']
        for wake_word in settings.wake_words:
            if wake_word in sentence:
                print(f'Skip row with sentence "{sentence}" because it contains wake word "{wake_word}"')
                continue
        name = Path(row['id']).name.replace(".flac", ".wav")
        scipy.io.wavfile.write(os.path.join(output_dir, name), 16000, (row["audio"]["array"]*32767).astype(np.int16))