"""Model factory: name -> constructor. Keeps training/inference code generic
so a PANNs/BEATs backbone (Phase 2) is a config change, not a code change."""

from __future__ import annotations

from torch import nn

from echosentinel.models.crnn import CRNN


def build_model(name: str, **kwargs) -> nn.Module:
    if name == "crnn":
        return CRNN(**kwargs)
    raise ValueError(f"Unknown model {name!r}. Available: crnn")


def model_frames_per_second(name: str) -> float:
    from echosentinel.constants import FRAMES_PER_SECOND
    from echosentinel.models.crnn import TIME_POOL

    if name == "crnn":
        return FRAMES_PER_SECOND / TIME_POOL
    raise ValueError(f"Unknown model {name!r}")
