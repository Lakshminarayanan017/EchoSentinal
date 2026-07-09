"""Visual assets computed from real audio for the web console.

- ``waveform_peaks``: a min/max envelope (bucketed to a fixed column count)
  the frontend draws on a canvas — the real waveform, not a decoration.
- ``spectrogram_png``: a log-mel spectrogram rendered with the console's
  cyan colormap, streamed block-wise so 250 MB files stay memory-safe.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from echosentinel.audio.io import load_audio, probe
from echosentinel.constants import TARGET_SR
from echosentinel.features.melspec import LogMel

BLOCK_S = 60.0


def waveform_peaks(path: str | Path, columns: int = 2400) -> dict:
    """Bucketed min/max envelope over the whole file."""
    info = probe(path)
    total_s = max(info.duration, 1e-6)
    sec_per_col = total_s / columns
    mins = np.zeros(columns, dtype=np.float32)
    maxs = np.zeros(columns, dtype=np.float32)

    col = 0
    t0 = 0.0
    while t0 < total_s and col < columns:
        y, _ = load_audio(path, offset=t0, duration=min(BLOCK_S, total_s - t0))
        if len(y) == 0:
            break
        samples_per_col = max(int(sec_per_col * TARGET_SR), 1)
        n_full = len(y) // samples_per_col
        if n_full == 0:
            break
        chunked = y[: n_full * samples_per_col].reshape(n_full, samples_per_col)
        hi = min(col + n_full, columns)
        take = hi - col
        mins[col:hi] = chunked.min(axis=1)[:take]
        maxs[col:hi] = chunked.max(axis=1)[:take]
        col = hi
        t0 += n_full * samples_per_col / TARGET_SR

    peak = float(max(np.abs(mins).max(), np.abs(maxs).max(), 1e-6))
    return {
        "duration": round(total_s, 3),
        "columns": columns,
        "min": np.round(mins / peak, 4).tolist(),
        "max": np.round(maxs / peak, 4).tolist(),
    }


def _cyan_colormap(v: np.ndarray) -> np.ndarray:
    """(H, W) in [0,1] -> (H, W, 3) uint8: abyss black -> deep blue -> cyan -> white."""
    stops = np.array(
        [
            [3, 11, 16],      # abyss
            [8, 47, 73],      # deep blue
            [0, 145, 178],    # mid teal
            [0, 240, 255],    # radiant cyan
            [219, 252, 255],  # near white
        ],
        dtype=np.float32,
    )
    pos = np.array([0.0, 0.35, 0.6, 0.85, 1.0], dtype=np.float32)
    v = np.clip(v, 0.0, 1.0)
    out = np.zeros(v.shape + (3,), dtype=np.float32)
    for i in range(len(stops) - 1):
        m = (v >= pos[i]) & (v <= pos[i + 1])
        t = (v[m] - pos[i]) / (pos[i + 1] - pos[i] + 1e-9)
        out[m] = stops[i] + t[:, None] * (stops[i + 1] - stops[i])
    return out.astype(np.uint8)


def spectrogram_png(path: str | Path, out_png: str | Path, max_width: int = 3200) -> None:
    """Render the file's log-mel spectrogram to a PNG (time x mel, cyan map)."""
    from PIL import Image

    info = probe(path)
    total_s = max(info.duration, 1e-6)
    frontend = LogMel().eval()

    cols: list[np.ndarray] = []
    t0 = 0.0
    while t0 < total_s:
        y, _ = load_audio(path, offset=t0, duration=min(BLOCK_S, total_s - t0))
        if len(y) < 1024:
            break
        with torch.no_grad():
            mel = frontend(torch.from_numpy(y)).squeeze(0).numpy()  # (mels, T)
        cols.append(mel)
        t0 += BLOCK_S
    if not cols:
        raise ValueError(f"no audio decoded from {path}")

    spec = np.concatenate(cols, axis=1)  # (mels, frames)
    # pool time down to <= max_width columns
    width = min(max_width, spec.shape[1])
    pool = max(spec.shape[1] // width, 1)
    n = (spec.shape[1] // pool) * pool
    spec = spec[:, :n].reshape(spec.shape[0], -1, pool).mean(axis=2)

    # robust normalize (5th..99th percentile) then colormap; low freqs at bottom
    lo, hi = np.percentile(spec, [5, 99])
    norm = (spec - lo) / (hi - lo + 1e-9)
    rgb = _cyan_colormap(norm[::-1, :])  # flip: high freq on top
    Image.fromarray(rgb, "RGB").save(out_png, optimize=True)
