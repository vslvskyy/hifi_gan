import torch
import torch.nn as nn

from torch.nn.utils import weight_norm, spectral_norm


class SubDiscriminatorP(nn.Module):
    def __init__(
        self,
        p: int,
        kernel_size: int,
        stride: int
    ):
        super(SubDiscriminatorP, self).__init__()

        self.p = p

        n_channels = [1] + [2**(5 + i) for i in range(1, 5)] + [1024]
        self.layers = []
        for i in range(1, len(n_channels)):
            self.layers.append(
                weight_norm(nn.Sequential(
                    nn.Conv2d(
                        n_channels[i - 1],
                        n_channels[i],
                        kernel_size=(kernel_size, 1),
                        stride=(stride, 1),
                        padding=(2, 0)
                    )),
                    nn.LeakyReLU(0.1)
                )
            )
        self.layers[-1][0].stride = (1, 1)
        self.layers = nn.ModuleList(self.layers)

        self.final_layer = weight_norm(nn.Conv2d(
            self.layers[-1][0].out_channels,
            1,
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0)
        ))

    def forward(self, x):
        fmap = []

        bs, n_ch, time = x.shape
        if time % self.p != 0:
            x = torch.nn.functional.pad(x, (0, self.p - time % self.p), "reflect")

        x = x.view(bs, n_ch, -1, self.p)

        for layer in self.layers:
            x = layer(x)
            fmap.append(x)

        x = self.final_layer(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MSD(nn.Module):
    def __init__(
        self,
        downsample_factor: int=1
    ):
        super(MSD, self).__init__()

        if downsample_factor == 1:
            self.norm = spectral_norm
        else:
            self.norm = weight_norm

        self.avg_pooling = nn.Sequential(*[
            nn.AvgPool1d(kernel_size=4, stride=2, padding=2)
            for _ in range(
                torch.log2(torch.tensor([downsample_factor])).int().item()
            )
        ])

        n_channels = [1] + [128] * 2 + [256, 512] + [1024] * 3
        kernel_sizes = [15] + [41] * 5 + [5]
        strides = [1] + [2] * 2 + [4] * 2 + [1] * 2
        groups = [1, 4] + [16] * 4 + [1]
        paddings = [7] + [20] * 5 + [2]

        self.layers = []
        for i in range(len(kernel_sizes)):
            self.layers.append(
                nn.Sequential(
                    self.norm(nn.Conv1d(
                        n_channels[i],
                        n_channels[i + 1],
                        kernel_size=kernel_sizes[i],
                        stride=strides[i],
                        padding=paddings[i],
                        groups=groups[i]
                    )),
                    nn.LeakyReLU(0.1)
                )
            )
        self.layers = nn.ModuleList(self.layers)

        self.final_layer = self.norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        x = self.avg_pooling(x)

        fmap = []
        for layer in self.layers:
            x = layer(x)
            fmap.append(x)

        x = self.final_layer(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap
