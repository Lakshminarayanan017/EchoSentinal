"""Fast frame-level metrics used during training (the real objective, IER,
is computed event-level by src/echosentinel/eval/ier.py)."""

from __future__ import annotations

import torch

from echosentinel.constants import CLASS_MAP


def frame_f1(
    probs: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5
) -> dict[str, float]:
    """Per-class and macro F1 over (N, T, C) posteriors vs binary labels."""
    preds = (probs >= threshold).float()
    out: dict[str, float] = {}
    f1s = []
    for i, name in sorted((k - 1, v) for k, v in CLASS_MAP.items()):
        tp = float((preds[..., i] * labels[..., i]).sum())
        fp = float((preds[..., i] * (1 - labels[..., i])).sum())
        fn = float(((1 - preds[..., i]) * labels[..., i]).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        out[f"f1_{name}"] = f1
        f1s.append(f1)
    out["f1_macro"] = sum(f1s) / len(f1s)
    return out
