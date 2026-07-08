"""Background noise beds for the scene synthesizer.

Two components:
- synthetic ambient ocean noise (1/f "pink-ish" spectrum), and
- a synthetic machinery/engine drone (low-frequency harmonic stack + rumble
  with slow amplitude modulation), standing in for the platform's own engine
  noise that pervades the PS-12 test recordings.

Design decision: the engine drone is *unlabeled background* — the continuous
machine noise the model must see through. Crucially it is SYNTHETIC, not cut
from the vessel training recordings, so it is timbrally distinct from the
vessel *events* (which are the real ship recordings placed at event-level
SNR and labeled class 1). Using real vessel audio as the bed taught an earlier
model that "continuous vessel timbre = background", collapsing vessel recall;
a distinct synthetic drone removes that collision while still forcing
robustness to a loud stationary background.
"""

from __future__ import annotations

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
    """Provides background beds of arbitrary length."""

    def __init__(
        self,
        sr: int,
        rng: np.random.Generator,
        engine_bed_prob: float = 0.6,
        ambient_dbfs: float = -35.0,
        engine_gain_db_range: tuple[float, float] = (-10.0, 2.0),
    ) -> None:
        self.sr = sr
        self.rng = rng
        self.engine_bed_prob = engine_bed_prob
        self.ambient_dbfs = ambient_dbfs
        self.engine_gain_db_range = engine_gain_db_range

    def bed(self, n_samples: int) -> np.ndarray:
        """Ambient bed, optionally with a synthetic engine drone mixed in."""
        out = rms_normalize_arr(pink_noise(n_samples, self.rng), self.ambient_dbfs)
        if self.rng.random() < self.engine_bed_prob:
            engine = engine_drone(n_samples, self.sr, self.rng)
            gain_db = float(self.rng.uniform(*self.engine_gain_db_range))
            engine = rms_normalize_arr(engine, self.ambient_dbfs + gain_db)
            out = out + engine
        return out.astype(np.float32, copy=False)


def rms_normalize_arr(y: np.ndarray, target_dbfs: float, eps: float = 1e-10) -> np.ndarray:
    """RMS-normalize a noise array to a target dBFS (no peak guard; beds are
    summed and the scene is peak-limited later)."""
    rms = float(np.sqrt(np.mean(np.square(y), dtype=np.float64)))
    if rms < eps:
        return y.astype(np.float32, copy=False)
    gain = 10.0 ** (target_dbfs / 20.0) / rms
    return (y * gain).astype(np.float32, copy=False)
