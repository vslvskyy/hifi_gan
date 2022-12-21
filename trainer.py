import torch.nn as nn

from torch.utils.data import DataLoader

from loss import HifiGanLoss

def train_one_epoch(
    generator: nn.Module,
    discriminator: nn.ModuleList,
    train_dataloader: DataLoader,
    g_optimizer,
    d_optimizer,
    g_scheduler,
    d_scheduler,
    loss_obj,
    device
):
    generator.train()
    for d_block in discriminator:
        d_block.train()

    for real_mels, real_wav in train_dataloader:
        real_wav = real_wav.to(device)
        real_mels = real_mels.to(device)

        fake_wav = generator(real_mels)

        g_loss, d_loss = loss_obj(discriminator, real_wav, fake_wav)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        g_scheduler.step()

        g_loss, d_loss = loss_obj(discriminator, real_wav, fake_wav)

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        d_scheduler.step()


def train(
    generator: nn.Module,
    discriminator: nn.ModuleList,
    train_dataloader: DataLoader,
    g_optimizer,
    d_optimizer,
    g_scheduler,
    d_scheduler,
    n_epochs,
    get_mel_spec,
    device,
    alpha_fm: int = 2,
    alpha_mel: int = 45
):
    loss_obj = HifiGanLoss(
        get_mel=get_mel_spec,
        alpha_fm=alpha_fm,
        alpha_mel=alpha_mel
    )

    for epoch in range(n_epochs):
        train_one_epoch(
            generator,
            discriminator,
            train_dataloader,
            g_optimizer,
            d_optimizer,
            g_scheduler,
            d_scheduler,
            loss_obj,
            get_mel_spec,
            device
        )
