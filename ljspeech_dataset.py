import os

import torch
import torchaudio
import numpy as np

from torch.utils.data import Dataset


class LJSpeechDataset(Dataset):
    def __init__(
        self,
        root: str
    ):
        self.wav_paths = sorted([os.path.join(root + "/wavs", f) for f in os.listdir(root + "/wavs")])
        self.sample_rate = torchaudio.load(self.wav_paths[0])[1]

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(
        self,
        idx: int
    ) -> torch.Tensor:

        return torchaudio.load(self.wav_paths[idx])[0][0:1, :].squeeze()
