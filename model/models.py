from typing import List

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

from model.commons import init_weights
from model.modules import (LRELU_SLOPE, Flip, ResBlock1, ResBlock2,
                           ResidualCouplingLayer)


class ResidualCouplingBlock(nn.Module):
    def __init__(self,
                 *,
                 channels: int,
                 hidden_channels: int,
                 kernel_size: int,
                 dilation_rate: int,
                 n_layers: int,
                 n_flows: int = 4,
                 gin_channels: int = 0):
        super(ResidualCouplingBlock, self).__init__()
        self.flows = nn.ModuleList()

        for _ in range(n_flows):
            self.flows.append(ResidualCouplingLayer(
                channels=channels,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                dilation_rate=dilation_rate,
                n_layers=n_layers,
                gin_channels=gin_channels,
                mean_only=True
            ))

            self.flows.append(Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)

        return x


class Generator(nn.Module):
    def __init__(self,
                 *,
                 initial_channel: int,
                 resblock: str,
                 resblock_kernel_sizes: List[int],
                 resblock_dilation_sizes: List[List[int]],
                 upsample_rates: List[int],
                 upsample_initial_channel: int,
                 upsample_kernel_sizes: List[int],
                 gin_channels: int = 0):
        super(Generator, self).__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = ResBlock1 if resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(nn.ConvTranspose1d(
                in_channels=upsample_initial_channel // (2 ** i),
                out_channels=upsample_initial_channel // (2 ** (i + 1)),
                kernel_size=k,
                stride=u,
                padding=(k - u) // 2
            )))

        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))

            for (k, d) in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(resblock(channels=ch, kernel_size=k, dilation=d))

        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)

        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None

            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        for module in self.ups:
            remove_weight_norm(module)

        for module in self.resblocks:
            module.remove_weight_norm()
