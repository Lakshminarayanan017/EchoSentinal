"""Long-file inference: stream audio in blocks, stitch frame posteriors.

Official test files reach 250 MB, so audio is decoded in overlapping blocks
(windowed reads at native rate, standardized per block). Each block's
posteriors are trimmed by the overlap margins and concatenated; the overlap
gives the model context so stitched frames match whole-file inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch

from echosentinel.audio.io import load_audio, probe

# posterior_fn: (samples,) float32 tensor -> (frames, classes) probabilities
PosteriorFn = Callable[[torch.Tensor], torch.Tensor]


def file_posteriors(
    path: str | Path,
    posterior_fn: PosteriorFn,
    fps: float,
    block_seconds: float = 30.0,
    overlap_seconds: float = 2.0,
    progress: Callable[[float], None] | None = None,
) -> np.ndarray:
    """Frame posteriors (total_frames, classes) for an arbitrarily long file.

    ``progress`` (if given) is called with a fraction in [0, 1] after each
    processed block — used by the web UI's live queue.
    """
    info = probe(path)
    total_s = info.duration
    if total_s <= block_seconds + 2 * overlap_seconds:
        y, _ = load_audio(path)
        probs = posterior_fn(torch.from_numpy(y))
        if progress:
            progress(1.0)
        return probs.cpu().numpy()

    chunks: list[np.ndarray] = []
    t0 = 0.0
    while t0 < total_s:
        t1 = min(t0 + block_seconds, total_s)
        read_start = max(t0 - overlap_seconds, 0.0)
        read_end = min(t1 + overlap_seconds, total_s)
        y, _ = load_audio(path, offset=read_start, duration=read_end - read_start)
        probs = posterior_fn(torch.from_numpy(y)).cpu().numpy()

        head = int(round((t0 - read_start) * fps))
        keep = int(round((t1 - t0) * fps))
        chunks.append(probs[head : head + keep])
        t0 = t1
        if progress:
            progress(min(t0 / total_s, 1.0))

    out = np.concatenate(chunks, axis=0)
    return out[: int(round(total_s * fps))]
