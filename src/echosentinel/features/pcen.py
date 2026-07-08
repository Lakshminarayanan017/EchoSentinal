"""Per-Channel Energy Normalization (PCEN) front-end.

PCEN replaces log compression with a trainable automatic-gain-control (AGC)
that divides each mel channel by a smoothed running energy estimate before a
root compression:

    M[t] = (1 - s) * M[t-1] + s * E[t]          # first-order IIR smoother
    PCEN = (E / (eps + M)^alpha + delta)^r - delta^r

The AGC term suppresses stationary background (the continuous engine drone in
the PS-12 test files) while preserving transients (faint whale/dolphin calls,
sonar pings) — directly targeting the failure mode that sank v1. The smoother
is a recursive filter, so the same module serves streaming inference by
carrying ``M[t-1]`` across blocks (see ``forward`` ``state`` argument).

Parameters can be fixed (paper defaults) or trained (log-domain, kept
positive) — trainable PCEN often wins on SED. Defaults follow Wang et al. 2017
and the librosa implementation.
"""

from __future__ import annotations

import torch
import torchaudio
from torch import nn

from echosentinel.constants import FMIN, HOP_LENGTH, N_FFT, N_MELS, TARGET_SR


class PCEN(nn.Module):
    def __init__(
        self,
        sr: int = TARGET_SR,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        n_mels: int = N_MELS,
        fmin: float = FMIN,
        trainable: bool = True,
        s: float = 0.025,      # IIR smoother coefficient (time constant)
        alpha: float = 0.8,    # AGC exponent
        delta: float = 2.0,    # bias before root compression
        r: float = 0.5,        # root compression exponent
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.s = s
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=sr / 2,
            power=2.0,
        )
        # Store AGC params in log domain so they stay positive when trained.
        log_alpha = torch.log(torch.tensor(alpha))
        log_delta = torch.log(torch.tensor(delta))
        log_r = torch.log(torch.tensor(r))
        if trainable:
            self.log_alpha = nn.Parameter(log_alpha)
            self.log_delta = nn.Parameter(log_delta)
            self.log_r = nn.Parameter(log_r)
        else:
            self.register_buffer("log_alpha", log_alpha)
            self.register_buffer("log_delta", log_delta)
            self.register_buffer("log_r", log_r)

    def _smooth(
        self, energy: torch.Tensor, init: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """First-order IIR filter along time. energy: (B, mels, T).

        Returns (smoothed, last_state) where last_state feeds the next block.
        """
        b, mels, t = energy.shape
        m_prev = energy[:, :, 0] if init is None else init
        out = torch.empty_like(energy)
        for i in range(t):  # recursive; T is modest (per-block)
            m_prev = (1 - self.s) * m_prev + self.s * energy[:, :, i]
            out[:, :, i] = m_prev
        return out, m_prev

    def forward(
        self, waveform: torch.Tensor, state: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """(B, samples) or (samples,) -> (B, mels, T) PCEN features.

        When ``state`` is provided (streaming), returns (features, new_state)
        so the smoother is continuous across blocks; otherwise returns just
        the features.
        """
        squeeze = waveform.ndim == 1
        if squeeze:
            waveform = waveform.unsqueeze(0)
        energy = self.melspec(waveform)  # (B, mels, T)

        smoother, last = self._smooth(energy, state)
        alpha = torch.exp(self.log_alpha)
        delta = torch.exp(self.log_delta)
        r = torch.exp(self.log_r)
        smooth_pow = torch.exp(-alpha * torch.log(self.eps + smoother))
        pcen = (energy * smooth_pow + delta) ** r - delta**r

        if squeeze:
            pcen = pcen.squeeze(0)
        if state is not None:
            return pcen, last
        return pcen
