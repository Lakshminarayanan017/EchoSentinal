"""Feature-domain augmentation: SpecAugment time/frequency masking.

Applied on the spectrogram (after the PCEN/log-mel front-end) during training
only. Waveform-domain augmentation (gain, stretch, low-pass to simulate
low-SR test files, 8-bit quantization) lives in the scene synthesizer, which
is the natural place to apply it while events are still separable.
"""

from __future__ import annotations

import torch
from torch import nn


class SpecAugment(nn.Module):
    def __init__(
        self,
        time_masks: int = 2,
        time_mask_max: int = 50,   # frames
        freq_masks: int = 2,
        freq_mask_max: int = 8,    # mel bins
    ) -> None:
        super().__init__()
        self.time_masks = time_masks
        self.time_mask_max = time_mask_max
        self.freq_masks = freq_masks
        self.freq_mask_max = freq_mask_max

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """(B, mels, T) -> masked copy. No-op in eval mode."""
        if not self.training:
            return spec
        spec = spec.clone()
        b, mels, t = spec.shape
        for _ in range(self.freq_masks):
            f = int(torch.randint(0, self.freq_mask_max + 1, (1,)).item())
            if f > 0 and f < mels:
                f0 = int(torch.randint(0, mels - f, (1,)).item())
                spec[:, f0 : f0 + f, :] = 0.0
        for _ in range(self.time_masks):
            w = int(torch.randint(0, self.time_mask_max + 1, (1,)).item())
            if w > 0 and w < t:
                t0 = int(torch.randint(0, t - w, (1,)).item())
                spec[:, :, t0 : t0 + w] = 0.0
        return spec
