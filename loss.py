import typing as tp

import torch
import torch.nn as nn


def d_loss_func(
    d_real: torch.Tensor,
    d_fake: torch.Tensor
) -> torch.Tensor:
    return (
            nn.MSELoss(reduction="none")(d_real, torch.ones_like(d_real)) +
            nn.MSELoss(reduction="none")(d_fake, torch.zeros_like(d_fake))
        ).mean()


def g_loss_func(
    d_fake: torch.Tensor
) -> torch.Tensor:
    return nn.MSELoss()(d_fake, torch.ones_like(d_fake))


def mel_loss_func(
    real_wav: torch.Tensor,
    fake_wav: torch.Tensor,
    get_mel_spec: nn.Module
) -> torch.Tensor:

    return nn.L1Loss()(
        get_mel_spec(fake_wav),
        get_mel_spec(real_wav)
    )


def feature_map_loss(
    real_features: tp.List[torch.Tensor],
    fake_features: tp.List[torch.Tensor]
) -> torch.Tensor:
    bs = real_features[0].shape[0]

    l1_losses = torch.zeros(bs, len(real_features))
    for i, (real_ftmp, fake_ftmp) in enumerate(zip(real_features, fake_features)):

        to_add = nn.L1Loss(reduction="none")(fake_ftmp, real_ftmp).view(bs, -1)
        l1_losses[:, i] = to_add.mean(dim=-1)

    return l1_losses.sum(dim=1).mean()


class HifiGanLoss(nn.Module):
    def __init__(
        self,
        get_mel: nn.Module,
        alpha_fm: int = 2,
        alpha_mel: int = 45
    ):
        super(HifiGanLoss, self).__init__()

        self.get_mel = get_mel
        self.alpha_fm = alpha_fm
        self.alpha_mel = alpha_mel

    def forward(
        self,
        discriminator: nn.ModuleList,
        real: torch.Tensor,
        fake: torch.Tensor
    ):
        d_loss = torch.zeros(1)
        g_loss = mel_loss_func(
            real_wav=real,
            fake_wav=fake,
            get_mel_spec=self.get_mel
        )

        for d_block in discriminator:
            d_real, d_real_ftmp = d_block(real)
            d_fake, d_fake_ftmp = d_block(fake)

            g_loss += g_loss_func(d_fake) + feature_map_loss(d_real_ftmp, d_fake_ftmp)
            d_loss += d_loss_func(d_real, d_fake)

        return g_loss, d_loss
