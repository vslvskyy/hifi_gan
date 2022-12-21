import os
import typing as tp

import torch
import torchaudio
import numpy as np

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class LJSpeechDataset(Dataset):
    def __init__(
        self,
        root: str
    ):
        self.wav_paths = sorted([os.path.join(root + "/wavs", f) for f in os.listdir(root + "/wavs")])
        self.mel_paths = sorted([os.path.join(root + "/mels", f) for f in os.listdir(root + "/mels")])

        # self.wavs = [torchaudio.load(wav_path)[0][0:1, :] for wav_path in wav_paths]
        # self.mels = [torch.tensor(np.load(mel_path)) for mel_path in mel_paths]

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(
        self,
        idx: int
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(np.load(self.mel_paths[idx])),
            torchaudio.load(self.wav_paths[idx])[0][0:1, :].squeeze()
        )


def collator(
    batch: tp.List[tp.Tuple[torch.Tensor, torch.Tensor]]
) -> torch.Tensor:
    mels = [batch[i][0] for i in range(len(batch))]
    wavs = [batch[i][1] for i in range(len(batch))]
    return (
        pad_sequence(mels, batch_first=True).transpose(-1, -2),
        pad_sequence(wavs, batch_first=True),
    )
