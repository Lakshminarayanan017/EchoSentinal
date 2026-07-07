"""CRNN frame-level SED model: conv blocks + BiGRU + framewise classifier.

Time resolution is reduced by TIME_POOL (4) -> 25 posterior frames/second at
the standard 10 ms feature hop. Frequency is pooled away progressively.
~1.5M parameters: trains on modest hardware, runs comfortably on CPU.
"""

from __future__ import annotations

import torch
from torch import nn

from echosentinel.constants import N_MELS, NUM_CLASSES
from echosentinel.features.melspec import LogMel

TIME_POOL = 4  # total time downsampling of the conv stack


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, pool: tuple[int, int]) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),  # (freq, time)
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CRNN(nn.Module):
    def __init__(
        self,
        n_mels: int = N_MELS,
        n_classes: int = NUM_CLASSES,
        rnn_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.frontend = LogMel()
        self.bn_input = nn.BatchNorm2d(1)
        self.conv = nn.Sequential(
            ConvBlock(1, 32, pool=(2, 2)),   # mels/2, T/2
            ConvBlock(32, 64, pool=(2, 2)),  # mels/4, T/4
            ConvBlock(64, 128, pool=(2, 1)),  # mels/8, T/4
            ConvBlock(128, 128, pool=(2, 1)),  # mels/16, T/4
        )
        feat_dim = 128 * (n_mels // 16)
        self.rnn = nn.GRU(
            feat_dim, rnn_hidden, num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.1,
        )
        self.head = nn.Linear(2 * rnn_hidden, n_classes)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """(B, samples) waveform -> (B, T/TIME_POOL, n_classes) logits."""
        x = self.frontend(waveform)              # (B, mels, T)
        x = self.bn_input(x.unsqueeze(1))        # (B, 1, mels, T)
        x = self.conv(x)                         # (B, 128, mels/16, T/4)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        x, _ = self.rnn(x)
        return self.head(x)

    @torch.no_grad()
    def posteriors(self, waveform: torch.Tensor) -> torch.Tensor:
        """(samples,) or (B, samples) -> (T_out, C) or (B, T_out, C) sigmoid probs."""
        squeeze = waveform.ndim == 1
        if squeeze:
            waveform = waveform.unsqueeze(0)
        probs = torch.sigmoid(self.forward(waveform))
        return probs.squeeze(0) if squeeze else probs
