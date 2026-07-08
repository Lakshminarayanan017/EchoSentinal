"""Model factory: name -> constructor. Keeps training/inference code generic
so switching CRNN <-> PANNs (or a future BEATs) is a config change, not a
code change. Both current models downsample time by 4 (25 fps posteriors)."""

from __future__ import annotations

from torch import nn

from echosentinel.constants import FRAMES_PER_SECOND
from echosentinel.models.crnn import CRNN
from echosentinel.models.crnn import TIME_POOL as CRNN_TIME_POOL
from echosentinel.models.panns import PANNsCNN14SED
from echosentinel.models.panns import TIME_POOL as PANNS_TIME_POOL


def build_model(name: str, **kwargs) -> nn.Module:
    if name == "crnn":
        return CRNN(**kwargs)
    if name == "panns":
        return PANNsCNN14SED(**kwargs)
    raise ValueError(f"Unknown model {name!r}. Available: crnn, panns")


def model_frames_per_second(name: str) -> float:
    pools = {"crnn": CRNN_TIME_POOL, "panns": PANNS_TIME_POOL}
    if name not in pools:
        raise ValueError(f"Unknown model {name!r}")
    return FRAMES_PER_SECOND / pools[name]
