"""Robust audio loading: any WAV subtype -> TARGET_SR mono float32 in [-1, 1].

The dataset (and especially the official PS-12 test sets) mixes 8/16/24/32/64-bit
PCM and IEEE-float WAVs at sample rates from 1 kHz to 82 kHz, mono and multi-
channel. The stdlib ``wave`` module chokes on float WAVs ("unknown format 3"),
which is one of the ways the old prototype failed. Here ``soundfile`` (libsndfile)
is the primary reader with ``librosa``/audioread as the fallback for anything
exotic.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from echosentinel.constants import TARGET_SR


@dataclass(frozen=True)
class AudioInfo:
    """Header-level metadata, readable without decoding the samples."""

    path: str
    sr: int
    channels: int
    frames: int
    duration: float
    subtype: str
    format: str


def probe(path: str | Path) -> AudioInfo:
    """Read header metadata only (fast, no sample decode)."""
    info = sf.info(str(path))
    return AudioInfo(
        path=str(path),
        sr=info.samplerate,
        channels=info.channels,
        frames=info.frames,
        duration=info.frames / info.samplerate if info.samplerate else 0.0,
        subtype=info.subtype,
        format=info.format,
    )


def to_mono(y: np.ndarray) -> np.ndarray:
    """Collapse (frames, channels) or (frames,) to mono (frames,)."""
    if y.ndim == 1:
        return y
    return y.mean(axis=1)


def resample(y: np.ndarray, sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """High-quality resample via soxr; no-op when rates already match."""
    if sr == target_sr:
        return y
    return soxr.resample(y, sr, target_sr, quality="HQ")


def load_audio(
    path: str | Path,
    target_sr: int = TARGET_SR,
    offset: float = 0.0,
    duration: float | None = None,
) -> tuple[np.ndarray, int]:
    """Load any supported audio file as mono float32 at ``target_sr``.

    ``offset``/``duration`` are in seconds and applied at the native rate
    before resampling, so only the requested window is decoded (important for
    the multi-hundred-MB test files).

    Returns (samples, target_sr).
    """
    path = str(path)
    try:
        with sf.SoundFile(path) as f:
            start = int(offset * f.samplerate)
            frames = -1 if duration is None else int(duration * f.samplerate)
            if path.lower().endswith(".mp3"):
                # mpg123 seeks are unreliable (bit-reservoir errors); MP3s in
                # this project are small, so decode fully and slice in memory.
                y = f.read(dtype="float32", always_2d=False)
                end = len(y) if frames < 0 else min(start + frames, len(y))
                y = y[start:end]
            else:
                f.seek(start)
                y = f.read(frames=frames, dtype="float32", always_2d=False)
            sr = f.samplerate
    except Exception as primary_err:  # exotic container/encoding: fall back
        try:
            import librosa

            y, sr = librosa.load(
                path, sr=None, mono=False, offset=offset, duration=duration
            )
            if y.ndim > 1:  # librosa returns (channels, frames)
                y = y.T
        except Exception as fallback_err:
            raise RuntimeError(
                f"Could not decode {path!r}: soundfile failed with "
                f"{primary_err!r}, librosa failed with {fallback_err!r}"
            ) from fallback_err
        warnings.warn(f"soundfile failed on {path!r} ({primary_err}); used librosa fallback")

    y = to_mono(np.asarray(y, dtype=np.float32))
    y = resample(y, sr, target_sr).astype(np.float32, copy=False)
    # 8-bit and some float files can exceed [-1, 1] slightly after resampling.
    np.clip(y, -1.0, 1.0, out=y)
    return y, target_sr


def rms_normalize(y: np.ndarray, target_dbfs: float = -25.0, eps: float = 1e-10) -> np.ndarray:
    """Scale to a target RMS level (dBFS). Silent input is returned unchanged."""
    rms = float(np.sqrt(np.mean(np.square(y), dtype=np.float64)))
    if rms < eps:
        return y
    gain = 10.0 ** (target_dbfs / 20.0) / rms
    out = y * gain
    peak = float(np.max(np.abs(out)))
    if peak > 1.0:  # avoid clipping loud transients
        out /= peak
    return out.astype(np.float32, copy=False)
