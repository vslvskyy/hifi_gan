import typing as tp

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(
        self,
        n_channels: int,
        kernel_size: int,
        dilations: tp.List[tp.List[int]]
    ):
        super(ResBlock, self).__init__()
        self.layers = []
        for i in range(len(dilations)):
            new_layer = []
            for j in range(len(dilations[i])):
                new_layer.extend([
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(n_channels, n_channels, kernel_size=kernel_size, dilation=dilations[i][j], padding="same"),
                ])
            self.layers.append(nn.Sequential(*new_layer))

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
        kernel_sizes: tp.List[int],
        dilations: tp.List[tp.List[int]]
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
        h_u: int,
        k_u: tp.List[int],
        k_r: tp.List[int],
        d_r: tp.List[tp.List[int]]
    ):
        super(HiFiGenerator, self).__init__()

        self.input_conv = nn.Conv1d(80, h_u, kernel_size=7, dilation=1, padding=3)  # n_mels = 80

        self.blocks = []
        for i in range(len(k_u)):
            self.blocks.append(
                nn.Sequential(
                    nn.LeakyReLU(0.1),
                    nn.ConvTranspose1d(
                        h_u // (2**i),
                        h_u // (2**(i + 1)),
                        kernel_size=k_u[i],
                        stride=k_u[i] // 2,
                        padding=(k_u[i] - k_u[i] // 2) // 2
                    ),
                    MRF(h_u // (2**(i + 1)), kernel_sizes=k_r, dilations=d_r),
                )
            )

        self.blocks = nn.Sequential(*self.blocks)

        self.final_conv = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv1d(h_u // (2**len(k_u)), 1, kernel_size=7, padding=3),
            nn.Tanh()
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return self.final_conv(self.blocks(self.input_conv(x)))


