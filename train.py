
import argparse
import wandb

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
    train_config = TrainConfig(log=args.wandb_log, checkpoint_path=args.checkpoint_path)
    train_dataset = LJSpeechDataset(args.train_data_path)
    train_dataloader = DataLoader(train_dataset, train_config.batch_size, shuffle=True, collate_fn=collator)

    test_dataset = LJSpeechDataset(args.test_data_path)

    g_model = HiFiGenerator(ModelConfig, MelSpectrogramConfig)
    mpd_blocks = [SubDiscriminatorP(p,  ModelConfig) for p in train_config.mpd_periods]
    msd_blocks = [MSD(factor) for factor in train_config.msd_factors]
    d_model = nn.ModuleList(mpd_blocks + msd_blocks)

    g_optimizer = AdamW(
        g_model.parameters(), lr=train_config.initial_lr,
        betas=train_config.adamw_betas, weight_decay=train_config.weight_decay
    )
    d_optimizer = AdamW(
        d_model.parameters(), lr=train_config.initial_lr,
        betas=train_config.adamw_betas, weight_decay=train_config.weight_decay
    )

    g_scheduler = ExponentialLR(g_optimizer, train_config.gamma)
    d_scheduler = ExponentialLR(d_optimizer, train_config.gamma)

    if train_config.log:
        wandb.init(project="hifi_gan")

    train(
        g_model, d_model,
        train_dataloader, test_dataset,
        g_optimizer, d_optimizer,
        g_scheduler, d_scheduler,
        MelSpectrogram(MelSpectrogramConfig),
        train_config
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train HiFi-GAN")

    parser.add_argument("--train_data_path", type=str, required=True,
                        help="path to directory with train wavs")

    parser.add_argument("--test_data_path", type=str, required=True,
                        help="path to directory with test wavs")

    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="path to directory to save checkpoints")

    parser.add_argument("--wandb_log", action="store_true", required=False,
                        help="wether to log or not")

    run(parser.parse_args())
