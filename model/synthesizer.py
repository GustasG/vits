import math
from typing import List

import monotonic_align
import torch
from torch import nn

from model.commons import generate_path, rand_slice_segments, sequence_mask
from model.encoders import PosteriorEncoder, TextEncoder
from model.models import Generator, ResidualCouplingBlock
from model.predictors import DurationPredictor, StochasticDurationPredictor
from params import Params
from text.symbols import get_vocabulary


class SynthesizerTrn(nn.Module):
    def __init__(self,
                 *,
                 n_vocab: int,
                 spec_channels: int,
                 segment_size,
                 inter_channels: int,
                 hidden_channels: int,
                 filter_channels: int,
                 n_heads: int,
                 n_layers: int,
                 kernel_size: int,
                 p_dropout: float,
                 resblock: str,
                 resblock_kernel_sizes: List[int],
                 resblock_dilation_sizes: List[List[int]],
                 upsample_rates: List[int],
                 upsample_initial_channel: int,
                 upsample_kernel_sizes: List[int],
                 gin_channels: int = 0,
                 use_sdp: bool = True):
        super(SynthesizerTrn, self).__init__()

        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.use_sdp = use_sdp

        self.enc_p = TextEncoder(
            n_vocab=n_vocab,
            out_channels=inter_channels,
            hidden_channels=hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout
        )

        self.dec = Generator(
            initial_channel=inter_channels,
            resblock=resblock,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            upsample_rates=upsample_rates,
            upsample_initial_channel=upsample_initial_channel,
            upsample_kernel_sizes=upsample_kernel_sizes,
            gin_channels=gin_channels
        )

        self.enc_q = PosteriorEncoder(
            in_channels=spec_channels,
            out_channels=inter_channels,
            hidden_channels=hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=16,
            gin_channels=gin_channels
        )

        self.flow = ResidualCouplingBlock(
            channels=inter_channels,
            hidden_channels=hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=4,
            gin_channels=gin_channels
        )

        if use_sdp:
            self.dp = StochasticDurationPredictor(
                in_channels=hidden_channels,
                filter_channels=192,
                kernel_size=3,
                p_dropout=0.5,
                n_flows=4,
                gin_channels=gin_channels
            )
        else:
            self.dp = DurationPredictor(
                in_channels=hidden_channels,
                filter_channels=256,
                kernel_size=3,
                p_dropout=0.5,
                gin_channels=gin_channels
            )

    @classmethod
    def from_params(cls, params: Params):
        symbols, _, _ = get_vocabulary(params.data.language)

        return cls(
            n_vocab=len(symbols),
            spec_channels=params.data.filter_length // 2 + 1,
            segment_size=params.train.segment_size // params.data.hop_length,
            inter_channels=params.model.inter_channels,
            hidden_channels=params.model.hidden_channels,
            filter_channels=params.model.filter_channels,
            n_heads=params.model.n_heads,
            n_layers=params.model.n_layers,
            kernel_size=params.model.kernel_size,
            p_dropout=params.model.p_dropout,
            resblock=params.model.resblock,
            resblock_kernel_sizes=params.model.resblock_kernel_sizes,
            resblock_dilation_sizes=params.model.resblock_dilation_sizes,
            upsample_rates=params.model.upsample_rates,
            upsample_initial_channel=params.model.upsample_initial_channel,
            upsample_kernel_sizes=params.model.upsample_kernel_sizes,
            gin_channels=params.model.gin_channels,
            use_sdp=params.model.use_sdp
        )

    def forward(self, x, x_lengths, y, y_lengths):
        g = None
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        with torch.inference_mode():
            # negative cross-entropy
            s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)  # [b, 1, t_s]
            neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r)  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True)  # [b, 1, t_s]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

        w = attn.sum(2)

        if self.use_sdp:
            l_length = self.dp(x, x_mask, w, g=g)
            l_length = l_length / torch.sum(x_mask)
        else:
            logw_ = torch.log(w + 1e-6) * x_mask
            logw = self.dp(x, x_mask, g=g)
            l_length = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(x_mask)  # for averaging

        # expand prior
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)

        return o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(self, x, x_lengths, noise_scale=1, length_scale=1, noise_scale_w=1.0, max_len=None):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        g = None

        if self.use_sdp:
            logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        else:
            logw = self.dp(x, x_mask, g=g)

        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)

        return o, attn, y_mask, (z, z_p, m_p, logs_p)
