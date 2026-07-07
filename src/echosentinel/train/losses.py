"""Framewise multi-label BCE with optional per-class positive weighting."""

from __future__ import annotations

import torch
from torch import nn


def framewise_bce(pos_weight: list[float] | None = None) -> nn.Module:
    weight = None
    if pos_weight is not None:
        weight = torch.tensor(pos_weight, dtype=torch.float32)
    return nn.BCEWithLogitsLoss(pos_weight=weight)
