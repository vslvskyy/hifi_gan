import typing as tp

import torch
import torch.nn as nn

from torch.nn.utils import weight_norm

from configs import ModelConfig, MelSpectrogramConfig
from utils import init_conv


class ResBlock(nn.Module):
    def __init__(
        self,
        n_channels: int,
        kernel_size: int,
        dilations: tp.Tuple[tp.Tuple[int]]
    ):
        super(ResBlock, self).__init__()
        self.layers = []
        for i in range(len(dilations)):
            new_layer = []
            for j in range(len(dilations[i])):
                new_layer.extend([
                    nn.LeakyReLU(0.1),
                    weight_norm(nn.Conv1d(
                        n_channels,
                        n_channels,
                        kernel_size=kernel_size,
                        dilation=dilations[i][j],
                        padding="same"
                    )),
                ])
            self.layers.append(nn.Sequential(*new_layer))

        for layer in self.layers:
            layer.apply(init_conv)

        self.layers = nn.ModuleList(self.layers)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return x


class MRF(nn.Module):
    def __init__(
        self,
        n_channels: int,
        kernel_sizes: tp.Tuple[int],
        dilations: tp.Tuple[tp.Tuple[int]]
    ):
        super(MRF, self).__init__()

        self.res_blocks = []
        for i in range(len(kernel_sizes)):
            self.res_blocks.append(
                ResBlock(n_channels, kernel_size=kernel_sizes[i], dilations=dilations[i])
            )

        self.res_blocks = nn.ModuleList(self.res_blocks)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        outs = torch.zeros_like(x)
        for res_block in self.res_blocks:
            outs = outs + res_block(x)
        return outs / len(self.res_blocks)


class HiFiGenerator(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        mel_spec_config: MelSpectrogramConfig
    ):
        super(HiFiGenerator, self).__init__()

        self.input_conv = weight_norm(nn.Conv1d(
            mel_spec_config.n_mels,
            model_config.h_u,
            kernel_size=7,
            dilation=1,
            padding=3
        ))

        self.blocks = []
        for i in range(len(model_config.k_u)):
            self.blocks.append(
                nn.Sequential(
                    nn.LeakyReLU(0.1),
                    weight_norm(nn.ConvTranspose1d(
                        model_config.h_u // (2**i),
                        model_config.h_u // (2**(i + 1)),
                        kernel_size=model_config.k_u[i],
                        stride=model_config.k_u[i] // 2,
                        padding=(model_config.k_u[i] - model_config.k_u[i] // 2) // 2
                    )),
                    MRF(model_config.h_u // (2**(i + 1)), kernel_sizes=model_config.k_r, dilations=model_config.d_r),
                )
            )

        for block in self.blocks:
            block.apply(init_conv)

        self.blocks = nn.Sequential(*self.blocks)

        self.final_conv = nn.Sequential(
            nn.LeakyReLU(0.1),
            weight_norm(nn.Conv1d(
                model_config.h_u // (2**len(model_config.k_u)),
                1,
                kernel_size=7,
                padding=3
            )),
            nn.Tanh()
        )

        self.final_conv.apply(init_conv)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return self.final_conv(self.blocks(self.input_conv(x)))
