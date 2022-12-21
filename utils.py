import typing as tp

import torch
import torch.nn as nn
import torchaudio
import librosa

from torch.nn.utils.rnn import pad_sequence

from configs import MelSpectrogramConfig


def collator(
    wavs: tp.List[torch.Tensor]
) -> torch.Tensor:
    return pad_sequence(wavs, batch_first=True)


def init_conv(
    layer: nn.Module,
    mean: float = 0.0,
    std: float = 0.01
):
    if (
        isinstance(layer, nn.Conv2d) or
        isinstance(layer, nn.Conv1d) or
        isinstance(layer, nn.ConvTranspose1d)
    ):
        layer.weight.data.normal_(mean, std)


class MelSpectrogram(nn.Module):

    def __init__(
        self,
        config: MelSpectrogramConfig
    ):
        super(MelSpectrogram, self).__init__()

        self.config = config

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sr,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels
        )

        # The is no way to set power in constructor in 0.5.0 version.
        self.mel_spectrogram.spectrogram.power = config.power

        # Default `torchaudio` mel basis uses HTK formula. In order to be compatible with WaveGlow
        # we decided to use Slaney one instead (as well as `librosa` does by default).
        mel_basis = librosa.filters.mel(
            sr=config.sr,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.f_min,
            fmax=config.f_max
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        :param audio: Expected shape is [B, T]
        :return: Shape is [B, n_mels, T']
        """

        mel = self.mel_spectrogram(audio) \
            .clamp_(min=1e-5) \
            .log_()

        return mel
