"""PANNs CNN14 backbone with a frame-level SED head — the "powerful engine".

CNN14 (Kong et al., 2020) pretrained on AudioSet gives audio priors that our
~280-file dataset cannot. We keep the CNN14 convolutional backbone verbatim
(module names match the published checkpoint so its conv/bn weights load) and
replace the clip-level classifier with a framewise segmentation head.

Time resolution: the AudioSet CNN14 pools (2,2) after all six blocks, which is
too coarse for event detection. We instead pool frequency on every block but
time on only the first two, yielding a 4x time downsample (25 fps at our 100
fps feature rate — same as the CRNN, so the inference stack is unchanged). The
conv *weights* are independent of pool sizes, so the pretrained backbone still
loads. The frontend is selectable (PCEN default, log-mel optional).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from echosentinel.constants import N_MELS, NUM_CLASSES
from echosentinel.features.augment import SpecAugment
from echosentinel.features.melspec import LogMel
from echosentinel.features.pcen import PCEN

TIME_POOL = 4  # total time downsampling of the backbone (2 blocks x /2)


class ConvBlock(nn.Module):
    """PANNs ConvBlock: two 3x3 convs + BN, pooling chosen at call time.
    Names (conv1/conv2/bn1/bn2) match the published CNN14 checkpoint."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, pool_size: tuple[int, int]) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_size != (1, 1):
            x = F.avg_pool2d(x, kernel_size=pool_size)
        return x


class PANNsCNN14SED(nn.Module):
    def __init__(
        self,
        frontend: str = "pcen",
        n_mels: int = N_MELS,
        n_classes: int = NUM_CLASSES,
        spec_augment: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.frontend_name = frontend
        self.frontend = PCEN() if frontend == "pcen" else LogMel()
        self.spec_augment = SpecAugment() if spec_augment else nn.Identity()
        self.bn0 = nn.BatchNorm2d(n_mels)  # normalizes per mel bin (PANNs bn0)

        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)

        self.dropout = nn.Dropout(dropout)
        # Framewise SED head: per-frame class logits + a segmentation-friendly
        # nonlinearity. 2048 backbone channels -> hidden -> classes.
        self.fc_frame = nn.Linear(2048, 512)
        self.head = nn.Linear(512, n_classes)

    # Frequency pooled every block; time only on the first two -> /4 time.
    _POOLS = [(2, 2), (2, 2), (2, 1), (2, 1), (2, 1), (2, 1)]

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """(B, samples) -> (B, T/4, n_classes) logits."""
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        spec = self.frontend(waveform)            # (B, mels, T)
        spec = self.spec_augment(spec)

        # bn0 normalizes each mel bin: put mels on the channel axis.
        x = spec.unsqueeze(1)                     # (B, 1, mels, T)
        x = x.transpose(1, 2)                     # (B, mels, 1, T)
        x = self.bn0(x)
        x = x.transpose(1, 2)                     # (B, 1, mels, T)

        for block, pool in zip(
            [self.conv_block1, self.conv_block2, self.conv_block3,
             self.conv_block4, self.conv_block5, self.conv_block6],
            self._POOLS,
        ):
            x = block(x, pool)
            x = self.dropout(x)

        x = x.mean(dim=2)                         # collapse frequency -> (B, 2048, T/4)
        x = x.transpose(1, 2)                     # (B, T/4, 2048)
        x = F.relu_(self.fc_frame(x))
        x = self.dropout(x)
        return self.head(x)

    @torch.no_grad()
    def posteriors(self, waveform: torch.Tensor) -> torch.Tensor:
        squeeze = waveform.ndim == 1
        if squeeze:
            waveform = waveform.unsqueeze(0)
        probs = torch.sigmoid(self.forward(waveform))
        return probs.squeeze(0) if squeeze else probs

    def load_backbone(self, checkpoint_path: str | Path) -> dict:
        """Load CNN14 conv/bn0 weights from the published AudioSet checkpoint.

        Only backbone keys (bn0, conv_block*) are copied; the frontend and the
        new SED head are left as-initialized. Returns a small report.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        own = self.state_dict()
        loaded, skipped = [], []
        for k, v in state.items():
            if (k.startswith("conv_block") or k.startswith("bn0")) and k in own and own[k].shape == v.shape:
                own[k] = v
                loaded.append(k)
            else:
                skipped.append(k)
        self.load_state_dict(own)
        return {"loaded": len(loaded), "skipped_source_keys": len(skipped)}
