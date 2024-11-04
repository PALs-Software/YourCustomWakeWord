# Readme

Go through all folders starting at 1_* (The 0_* are optionally to get some more infos or test out model configurations) and install the requirements first in each step (python -m pip install -r 1_*\requirements.txt). It is best to create a separate virtual Python environment for each step. Then execute the individual files of each step by starting at 1_*.

## Creating virtual environment
``` shell
python -m venv v1
source v1/bin/activate
```

## Troubleshooting

### Piper-TTS package can not be installed
Piper-TTS package needs the exact python version 3.10.12 in a linux distribution

### webrtcvad package con not be installed python.h missing
Install the following packages:
sudo apt-get install python3.10-dev libpython3.10-dev

### tensorflow-io.so library can not be loaded
Make sure you use python 3.12.1 with the specified versions in the requirements.txt on a linux distribution