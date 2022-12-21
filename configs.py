import typing as tp

import torch

from dataclasses import dataclass


@dataclass
class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0

    # value of melspectrograms if we fed a silence into `MelSpectrogram`
    pad_value: float = -11.5129251


@dataclass
class ModelConfig:
    h_u: int = 512
    k_u: tp.Tuple[int] = (16, 16, 4, 4)
    k_r: tp.Tuple[int] = (3, 7, 11)
    d_r: tp.Tuple[tp.Tuple[int]] = tuple(((1, 1), (3, 1), (5, 1)) for _ in range(3))

    mpd_kernel_size: int = 5
    mpd_stride: int = 3


@dataclass
class TrainConfig:
    mpd_periods: tp.Tuple[int] = (2, 3, 5, 7, 11)
    msd_factors: tp.Tuple[int] = (1, 2, 4)
    batch_size: int = 1
    initial_lr: float = 2e-4
    adamw_betas: tp.Tuple[float, float] = (0.8, 0.99)
    weight_decay: float = 0.01
    gamma: float = 0.999
    alpha_fm: int = 2
    alpha_mel: int = 45
    steps_per_epoch: int = 250
    n_epochs: int = 200
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_step: int = 25
    save_step: int = 10
    log: bool = True
    checkpoint_path: str = "."
