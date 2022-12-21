import wandb

import torch
import torch.nn as nn

from tqdm import tqdm

from torch.utils.data import DataLoader

from configs import TrainConfig
from loss import HifiGanLoss

def train_one_epoch(
    generator: nn.Module,
    discriminator: nn.ModuleList,
    train_dataloader: DataLoader,
    g_optimizer,
    d_optimizer,
    loss_obj,
    get_mel_spec,
    train_config: TrainConfig
):
    generator.train()
    for d_block in discriminator:
        d_block.train()

    for i, real_wav in tqdm(enumerate(train_dataloader), total=train_config.steps_per_epoch):
        real_mels = get_mel_spec(real_wav)

        real_wav = real_wav.to(train_config.device)
        real_mels = real_mels.to(train_config.device)

        fake_wav = generator(real_mels)

        real_wav = real_wav.unsqueeze(1)
        if real_wav.shape[-1] < fake_wav.shape[-1]:
            tmp = torch.zeros_like(fake_wav)
            tmp[:, :, :real_wav.shape[-1]] = real_wav
            real_wav = tmp
        elif fake_wav.shape[-1] < real_wav.shape[-1]:
            tmp = torch.zeros_like(real_wav)
            tmp[:, :, :fake_wav.shape[-1]] = fake_wav
            fake_wav = tmp

        assert fake_wav.shape == real_wav.shape

        total_g_loss, total_d_loss, adv_g_loss, mel_loss, ftmp_loss = loss_obj(discriminator, real_wav, fake_wav.detach())

        d_optimizer.zero_grad()
        total_d_loss.backward()
        d_optimizer.step()

        total_g_loss, total_d_loss, adv_g_loss, mel_loss, ftmp_loss = loss_obj(discriminator, real_wav, fake_wav)

        g_optimizer.zero_grad()
        total_g_loss.backward()
        g_optimizer.step()


        if train_config.log and i % train_config.log_step == 0:
            print(f"total_g_loss: {total_g_loss.item()}, d_loss: {total_d_loss.item()}, mel_loss: {mel_loss.item()}, ftmp_loss: {ftmp_loss.item()}")
            wandb.log({
                "total_g_loss": total_g_loss,
                "total_d_loss": total_d_loss,
                "adv_g_loss": adv_g_loss,
                "mel_loss": mel_loss,
                "ftmp_loss": ftmp_loss
            })


def train(
    generator: nn.Module,
    discriminator: nn.ModuleList,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    g_optimizer,
    d_optimizer,
    g_scheduler,
    d_scheduler,
    get_mel_spec,
    train_config: TrainConfig
):
    loss_obj = HifiGanLoss(
        get_mel=get_mel_spec,
        alpha_fm=train_config.alpha_fm,
        alpha_mel=train_config.alpha_mel
    )
    generator.to(TrainConfig.device)
    discriminator.to(TrainConfig.device)

    for epoch in range(train_config.n_epochs):
        train_one_epoch(
            generator,
            discriminator,
            train_dataloader,
            g_optimizer,
            d_optimizer,

            loss_obj,
            get_mel_spec,
            train_config
        )

        d_scheduler.step()
        g_scheduler.step()


    # validate(generator, get_mel_spec, test_dataloader)
