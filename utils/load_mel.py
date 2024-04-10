import torch
import os
import numpy as np
import torchaudio
from tqdm import tqdm

def pad_short_audio(audio, min_samples=32000):
    if(audio.size(-1) < min_samples):
        audio = torch.nn.functional.pad(audio, (0, min_samples - audio.size(-1)), mode='constant', value=0.0)
    return audio


class WaveDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datadir,
        sr=16000,
        limit_num=None,
    ):
        self.datalist = [os.path.join(datadir, x) for x in os.listdir(datadir)]
        self.datalist = sorted(self.datalist)
        if limit_num is not None:
            self.datalist = self.datalist[:limit_num]
        self.sr = sr

    def __getitem__(self, index):
        while True:
            try:
                filename = self.datalist[index]
                waveform = self.read_from_file(filename)
                if waveform.size(-1) < 1:
                    raise ValueError("empty file %s" % filename)
                break
            except Exception as e:
                print(index, e)
                index = (index + 1) % len(self.datalist)
        
        return waveform, os.path.basename(filename)

    def __len__(self):
        return len(self.datalist)

    def read_from_file(self, audio_file):
        audio, file_sr = torchaudio.load(audio_file)
        # Only use the first channel
        audio = audio[0:1,...]
        audio = audio - audio.mean()

        if file_sr != self.sr:
            audio = torchaudio.functional.resample(
                audio, orig_freq=file_sr, new_freq=self.sr
            )

            
        audio = pad_short_audio(audio, min_samples=32000)
        return audio

