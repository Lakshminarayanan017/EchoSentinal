"""Background noise beds for the scene synthesizer.

Three components, mixed per scene:
- synthetic ambient ocean noise (1/f "pink-ish" spectrum) at a RANDOMIZED
  level per scene (level invariance: the real test set spans near-silent
  recordings to loud ones, so training must too),
- a synthetic machinery/engine drone (low-frequency harmonic stack + rumble
  with slow amplitude modulation) standing in for platform engine noise, and
- optional MINED beds: quiet stationary windows extracted from the official
  unlabeled test recordings (scripts/02_build_noise_bank.py), giving the
  model the *real* background texture. Mined beds are mixed on top of the
  synthetic ambient, never used alone, and never labeled.

Design decision: all bed components are unlabeled background, and the engine
drone is SYNTHETIC — not cut from the vessel training recordings — so it is
timbrally distinct from the vessel *events* (real ship recordings placed at
event-level SNR, labeled class 1). Using real vessel audio as the bed taught
an earlier model that "continuous vessel timbre = background", collapsing
vessel recall.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def pink_noise(n_samples: int, rng: np.random.Generator, alpha: float = 1.0) -> np.ndarray:
    """Band-shaped noise with a 1/f^alpha power spectrum (ocean-ambient-like)."""
    white = rng.standard_normal(n_samples).astype(np.float32)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = freqs[1]  # avoid div-by-zero at DC
    spectrum /= freqs ** (alpha / 2.0)
    out = np.fft.irfft(spectrum, n=n_samples).astype(np.float32)
    return out / (np.max(np.abs(out)) + 1e-10)


def engine_drone(n_samples: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Synthetic machinery drone: a low fundamental with a few harmonics plus
    low-pass rumble, under a slow amplitude modulation. Deliberately generic
    so it does not mimic any specific target-vessel recording."""
    t = np.arange(n_samples) / sr
    f0 = float(rng.uniform(40, 130))  # engine fundamental (Hz)
    drone = np.zeros(n_samples, dtype=np.float32)
    n_harm = int(rng.integers(3, 7))
    for h in range(1, n_harm + 1):
        amp = 1.0 / h
        phase = float(rng.uniform(0, 2 * np.pi))
        drone += (amp * np.sin(2 * np.pi * f0 * h * t + phase)).astype(np.float32)
    # low-frequency rumble: pink noise steered to the low band
    rumble = pink_noise(n_samples, rng, alpha=2.0)
    drone = drone / (np.max(np.abs(drone)) + 1e-10) + 0.5 * rumble
    # slow AM (machinery load variation), 0.1-0.5 Hz
    am = 1.0 + 0.3 * np.sin(2 * np.pi * float(rng.uniform(0.1, 0.5)) * t)
    drone *= am.astype(np.float32)
    return (drone / (np.max(np.abs(drone)) + 1e-10)).astype(np.float32)


class NoiseBank:
    """Provides background beds of arbitrary length.

    ``bed()`` also returns the drawn ambient level so the synthesizer can set
    event SNRs relative to the actual bed of this scene.
    """

    def __init__(
        self,
        sr: int,
        rng: np.random.Generator,
        engine_bed_prob: float = 0.6,
        ambient_dbfs_range: tuple[float, float] = (-55.0, -30.0),
        engine_gain_db_range: tuple[float, float] = (-10.0, 2.0),
        mined_noise_dir: str | Path | None = None,
        mined_bed_prob: float = 0.5,
        mined_gain_db_range: tuple[float, float] = (-3.0, 6.0),
    ) -> None:
        self.sr = sr
        self.rng = rng
        self.engine_bed_prob = engine_bed_prob
        self.ambient_dbfs_range = ambient_dbfs_range
        self.engine_gain_db_range = engine_gain_db_range
        self.mined_bed_prob = mined_bed_prob
        self.mined_gain_db_range = mined_gain_db_range
        self.mined_files: list[Path] = []
        if mined_noise_dir is not None:
            self.mined_files = sorted(Path(mined_noise_dir).glob("*.wav"))

    def _mined_window(self, n_samples: int) -> np.ndarray | None:
        if not self.mined_files:
            return None
        from echosentinel.audio.io import load_audio  # local import: avoids cycle

        path = self.mined_files[self.rng.integers(len(self.mined_files))]
        y, _ = load_audio(path, target_sr=self.sr)
        if len(y) == 0:
            return None
        if len(y) < n_samples:  # tile short snippets to scene length
            y = np.tile(y, int(np.ceil(n_samples / len(y))))
        start = int(self.rng.integers(0, max(len(y) - n_samples, 1)))
        return y[start : start + n_samples]

    def bed(self, n_samples: int) -> tuple[np.ndarray, float]:
        """Compose a background bed. Returns (bed, ambient_dbfs_drawn)."""
        ambient_dbfs = float(self.rng.uniform(*self.ambient_dbfs_range))
        out = rms_normalize_arr(pink_noise(n_samples, self.rng), ambient_dbfs)

        if self.rng.random() < self.engine_bed_prob:
            engine = engine_drone(n_samples, self.sr, self.rng)
            gain_db = float(self.rng.uniform(*self.engine_gain_db_range))
            out = out + rms_normalize_arr(engine, ambient_dbfs + gain_db)

        if self.mined_files and self.rng.random() < self.mined_bed_prob:
            mined = self._mined_window(n_samples)
            if mined is not None:
                gain_db = float(self.rng.uniform(*self.mined_gain_db_range))
                out = out + rms_normalize_arr(mined, ambient_dbfs + gain_db)

        return out.astype(np.float32, copy=False), ambient_dbfs


def rms_normalize_arr(y: np.ndarray, target_dbfs: float, eps: float = 1e-10) -> np.ndarray:
    """RMS-normalize a noise array to a target dBFS (no peak guard; beds are
    summed and the scene is peak-limited later)."""
    rms = float(np.sqrt(np.mean(np.square(y), dtype=np.float64)))
    if rms < eps:
        return y.astype(np.float32, copy=False)
    gain = 10.0 ** (target_dbfs / 20.0) / rms
    return (y * gain).astype(np.float32, copy=False)
