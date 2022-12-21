
import argparse

import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

from configs import TrainConfig, ModelConfig, MelSpectrogramConfig
from discriminator import SubDiscriminatorP, MSD
from generator import HiFiGenerator
from ljspeech_dataset import LJSpeechDataset
from trainer import train
from utils import collator, MelSpectrogram


def run(args):
    dataset = LJSpeechDataset(args.train_data_path)
    dataloader = DataLoader(dataset, TrainConfig.batch_size, collate_fn=collator)

    g_model = HiFiGenerator(ModelConfig, MelSpectrogramConfig)
    mpd_blocks = [SubDiscriminatorP(p,  ModelConfig) for p in TrainConfig.mpd_periods]
    msd_blocks = [MSD(factor) for factor in TrainConfig.msd_factors]
    d_model = nn.ModuleList(mpd_blocks + msd_blocks)

    g_optimizer = AdamW(
        g_model.parameters(), lr=TrainConfig.initial_lr,
        betas=TrainConfig.adamw_betas, weight_decay=TrainConfig.weight_decay
    )
    d_optimizer = AdamW(
        d_model.parameters(), lr=TrainConfig.initial_lr,
        betas=TrainConfig.adamw_betas, weight_decay=TrainConfig.weight_decay
    )

    g_scheduler = ExponentialLR(g_optimizer, TrainConfig.gamma)
    d_scheduler = ExponentialLR(d_optimizer, TrainConfig.gamma)

    train(
        g_model, d_model,
        dataloader,
        g_optimizer, d_optimizer,
        g_scheduler, d_scheduler,
        MelSpectrogram(MelSpectrogramConfig),
        TrainConfig
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train HiFi-GAN")

    parser.add_argument("--train_data_path", type=str, required=True,
                        help="path to directory with train wavs")

    parser.add_argument("--test_data_path", type=str, required=True,
                        help="path to directory with test wavs")

    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="path to directory to save checkpoints")

    run(parser.parse_args())
