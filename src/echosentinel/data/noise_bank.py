"""Background noise beds for the scene synthesizer.

Two sources:
- synthetic ambient ocean noise (1/f "pink-ish" spectrum, band-limited), and
- engine-drone windows cut from the vessel training recordings, mixed in at
  LOW level to play the role of distant shipping.

Design decision: a faint continuous engine drone is *ambient noise* (distant
vessel), not a labeled vessel event — only prominent vessel sound placed by
the synthesizer at event-level SNR is labeled class 1. This teaches the model
the level-dependent distinction the PS-12 test data exhibits (continuous
machine noise under faint target events) instead of firing "vessel"
everywhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from echosentinel.audio.io import load_audio, rms_normalize


def pink_noise(n_samples: int, rng: np.random.Generator, alpha: float = 1.0) -> np.ndarray:
    """Band-shaped noise with a 1/f^alpha power spectrum (ocean-ambient-like)."""
    white = rng.standard_normal(n_samples).astype(np.float32)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = freqs[1]  # avoid div-by-zero at DC
    spectrum /= freqs ** (alpha / 2.0)
    out = np.fft.irfft(spectrum, n=n_samples).astype(np.float32)
    return out / (np.max(np.abs(out)) + 1e-10)


class NoiseBank:
    """Provides background beds of arbitrary length."""

    def __init__(
        self,
        vessel_files: list[tuple[Path, float]],
        sr: int,
        rng: np.random.Generator,
        engine_bed_prob: float = 0.6,
        ambient_dbfs: float = -35.0,
        engine_gain_db_range: tuple[float, float] = (-14.0, -4.0),
    ) -> None:
        self.vessel_files = [(p, d) for p, d in vessel_files if d > 5.0]
        self.sr = sr
        self.rng = rng
        self.engine_bed_prob = engine_bed_prob
        self.ambient_dbfs = ambient_dbfs
        self.engine_gain_db_range = engine_gain_db_range

    def _engine_window(self, n_samples: int) -> np.ndarray | None:
        if not self.vessel_files:
            return None
        path, dur = self.vessel_files[self.rng.integers(len(self.vessel_files))]
        want_s = n_samples / self.sr
        offset = float(self.rng.uniform(0, max(dur - want_s, 0)))
        y, _ = load_audio(path, target_sr=self.sr, offset=offset, duration=want_s)
        if len(y) < n_samples:  # short file: tile to length
            reps = int(np.ceil(n_samples / max(len(y), 1)))
            y = np.tile(y, reps)
        return y[:n_samples]

    def bed(self, n_samples: int) -> np.ndarray:
        """Ambient bed, optionally with a quiet engine drone mixed under it."""
        out = rms_normalize(pink_noise(n_samples, self.rng), self.ambient_dbfs)
        if self.rng.random() < self.engine_bed_prob:
            engine = self._engine_window(n_samples)
            if engine is not None:
                gain_db = float(self.rng.uniform(*self.engine_gain_db_range))
                engine = rms_normalize(engine, self.ambient_dbfs + gain_db)
                out = out + engine
        return out.astype(np.float32, copy=False)
