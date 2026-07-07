"""Log-mel spectrogram front-end shared by all models.

Geometry comes from constants (10 ms hop at 32 kHz, 64 mel bins). The module
is a torch ``nn.Module`` so it runs on GPU during training and stays inside
the exported model for inference.
"""

from __future__ import annotations

import torch
import torchaudio

from echosentinel.constants import FMIN, HOP_LENGTH, N_FFT, N_MELS, TARGET_SR


class LogMel(torch.nn.Module):
    def __init__(
        self,
        sr: int = TARGET_SR,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        n_mels: int = N_MELS,
        fmin: float = FMIN,
        eps: float = 1e-10,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=sr / 2,
            power=2.0,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """(batch, samples) or (samples,) -> (batch, n_mels, frames)."""
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        mel = self.melspec(waveform)
        return torch.log(mel + self.eps)
