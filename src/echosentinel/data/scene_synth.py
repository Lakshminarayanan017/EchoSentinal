"""Scene synthesizer: composes labeled training/validation scenes.

A scene = background bed (NoiseBank) + K events drawn from the clean pools,
placed at non-overlapping random times with gaps between them (the PS-12
"classes separated in time" structure), each mixed at a controlled SNR
relative to the bed. Low SNRs dominate the sampling because faint events
under engine noise are where the v1 model failed and where the IER is won.

Class 4 (other_anthropogenic) has very little real data, so a procedural
generator (sonar pings/sweeps, impulsive clanks, airgun-like pulses)
supplements the pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from echosentinel.audio.io import load_audio, rms_normalize
from echosentinel.constants import NAME_TO_ID, NUM_CLASSES, TARGET_SR
from echosentinel.data.noise_bank import NoiseBank


@dataclass(frozen=True)
class SceneEvent:
    class_id: int  # 1-based PS-12 label
    start_s: float
    end_s: float


def _sonar_ping(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Repeated tonal pings with exponential decay tails."""
    freq = float(rng.uniform(1_000, 9_000))
    ping_len = int(sr * rng.uniform(0.05, 0.25))
    t = np.arange(ping_len) / sr
    ping = np.sin(2 * np.pi * freq * t) * np.exp(-t / rng.uniform(0.02, 0.1))
    n_pings = int(rng.integers(2, 6))
    interval = int(sr * rng.uniform(0.5, 1.5))
    out = np.zeros(interval * (n_pings - 1) + ping_len, dtype=np.float32)
    for i in range(n_pings):
        out[i * interval : i * interval + ping_len] += ping.astype(np.float32)
    return out


def _sonar_sweep(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Linear frequency sweep (active sonar chirp)."""
    dur = float(rng.uniform(0.5, 2.0))
    n = int(sr * dur)
    t = np.arange(n) / sr
    f0 = float(rng.uniform(500, 3_000))
    f1 = f0 + float(rng.uniform(1_000, 6_000))
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * dur))
    env = np.minimum(t / 0.05, np.minimum(1.0, (dur - t) / 0.1)).clip(0, 1)
    return (np.sin(phase) * env).astype(np.float32)


def _impulsive_clank(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Metallic impact: damped noise burst plus a few resonant modes."""
    dur = float(rng.uniform(0.3, 1.5))
    n = int(sr * dur)
    t = np.arange(n) / sr
    out = rng.standard_normal(n).astype(np.float32) * np.exp(-t / 0.03)
    for _ in range(int(rng.integers(2, 5))):
        f = float(rng.uniform(300, 5_000))
        out += np.float32(rng.uniform(0.2, 0.6)) * np.sin(2 * np.pi * f * t).astype(
            np.float32
        ) * np.exp(-t / rng.uniform(0.05, 0.4))
    return out


def _airgun_pulse(rng: np.random.Generator, sr: int) -> np.ndarray:
    """Low-frequency damped oscillation with a bubble-pulse tail."""
    dur = float(rng.uniform(0.5, 1.5))
    n = int(sr * dur)
    t = np.arange(n) / sr
    f = float(rng.uniform(20, 150))
    out = np.sin(2 * np.pi * f * t) * np.exp(-t / 0.15)
    bubble_delay = rng.uniform(0.1, 0.3)
    out += 0.4 * np.sin(2 * np.pi * f * 0.8 * (t - bubble_delay)) * np.exp(
        -np.abs(t - bubble_delay) / 0.1
    )
    return out.astype(np.float32)


_PROCEDURAL_CLASS4 = (_sonar_ping, _sonar_sweep, _impulsive_clank, _airgun_pulse)


class SceneSynthesizer:
    def __init__(
        self,
        manifest: pd.DataFrame,
        dataset_root: Path,
        rng: np.random.Generator,
        sr: int = TARGET_SR,
        snr_db_range: tuple[float, float] = (-5.0, 20.0),
        snr_skew_low: bool = True,
        event_dur_range: tuple[float, float] = (2.0, 12.0),
        min_gap_s: float = 1.0,
        procedural_class4_prob: float = 0.5,
        engine_bed_prob: float = 0.6,
        mined_noise_dir: str | Path | None = None,
        master_gain_db_range: tuple[float, float] = (-25.0, 0.0),
    ) -> None:
        self.rng = rng
        self.sr = sr
        self.snr_db_range = snr_db_range
        self.snr_skew_low = snr_skew_low
        self.event_dur_range = event_dur_range
        self.min_gap_s = min_gap_s
        self.procedural_class4_prob = procedural_class4_prob
        self.dataset_root = dataset_root
        # Master gain scales the FINISHED scene (bed + events together),
        # teaching level invariance: the real test recordings range from
        # near-silent to loud, and a model trained at one absolute level
        # under-detects at others (measured: 77% misses on quiet scenes).
        self.master_gain_db_range = master_gain_db_range

        self.pools: dict[int, list[tuple[Path, float]]] = {}
        for class_id, group in manifest.groupby("class_id"):
            self.pools[int(class_id)] = [
                (dataset_root / p, float(d))
                for p, d in zip(group["path"], group["duration_s"])
            ]
        # Engine beds are synthetic (see noise_bank) and independent of the
        # vessel event pool, so vessel events stay timbrally distinct from the
        # background drone. Mined beds add the real test-recording texture.
        self.noise_bank = NoiseBank(
            sr=sr, rng=rng, engine_bed_prob=engine_bed_prob,
            mined_noise_dir=mined_noise_dir,
        )
        # Classes we can actually draw events for.
        self.event_classes = [c for c, pool in self.pools.items() if pool]
        if NAME_TO_ID["other_anthropogenic"] not in self.event_classes:
            self.event_classes.append(NAME_TO_ID["other_anthropogenic"])

    def _draw_snr_db(self) -> float:
        lo, hi = self.snr_db_range
        if self.snr_skew_low:
            # min of two uniforms biases toward the low end
            return float(min(self.rng.uniform(lo, hi), self.rng.uniform(lo, hi)))
        return float(self.rng.uniform(lo, hi))

    def _draw_event_audio(self, class_id: int) -> np.ndarray:
        if class_id == NAME_TO_ID["other_anthropogenic"] and (
            not self.pools.get(class_id)
            or self.rng.random() < self.procedural_class4_prob
        ):
            gen = _PROCEDURAL_CLASS4[self.rng.integers(len(_PROCEDURAL_CLASS4))]
            return gen(self.rng, self.sr)

        path, dur = self.pools[class_id][self.rng.integers(len(self.pools[class_id]))]
        want = float(self.rng.uniform(*self.event_dur_range))
        want = min(want, dur)
        offset = float(self.rng.uniform(0, max(dur - want, 0)))
        y, _ = load_audio(path, target_sr=self.sr, offset=offset, duration=want)
        return y

    def make_scene(
        self, duration_s: float, n_events_range: tuple[int, int] = (0, 3)
    ) -> tuple[np.ndarray, list[SceneEvent]]:
        n_samples = int(duration_s * self.sr)
        scene, _ = self.noise_bank.bed(n_samples)
        bed_rms_db = 20 * np.log10(float(np.sqrt(np.mean(scene**2))) + 1e-10)

        n_events = int(self.rng.integers(n_events_range[0], n_events_range[1] + 1))
        events: list[SceneEvent] = []
        occupied: list[tuple[float, float]] = []

        for _ in range(n_events):
            class_id = int(self.event_classes[self.rng.integers(len(self.event_classes))])
            audio = self._draw_event_audio(class_id)
            if len(audio) == 0:
                continue
            ev_dur = len(audio) / self.sr
            if ev_dur > duration_s - 2 * self.min_gap_s:
                audio = audio[: int((duration_s - 2 * self.min_gap_s) * self.sr)]
                ev_dur = len(audio) / self.sr
            if ev_dur < 0.3:
                continue

            placed = self._find_slot(ev_dur, duration_s, occupied)
            if placed is None:
                continue
            start_s = placed
            occupied.append((start_s - self.min_gap_s, start_s + ev_dur + self.min_gap_s))

            snr_db = self._draw_snr_db()
            audio = rms_normalize(audio, bed_rms_db + snr_db)
            i0 = int(start_s * self.sr)
            scene[i0 : i0 + len(audio)] += audio[: n_samples - i0]
            events.append(SceneEvent(class_id, start_s, start_s + ev_dur))

        peak = float(np.max(np.abs(scene)))
        if peak > 1.0:
            scene /= peak
        # Level invariance: scale bed + events together so the model sees the
        # same scene at many absolute levels (labels are unaffected).
        master_gain = 10.0 ** (float(self.rng.uniform(*self.master_gain_db_range)) / 20.0)
        scene *= master_gain
        events.sort(key=lambda e: e.start_s)
        return scene.astype(np.float32, copy=False), events

    def _find_slot(
        self, ev_dur: float, scene_dur: float, occupied: list[tuple[float, float]]
    ) -> float | None:
        for _ in range(20):  # rejection sampling for a free interval
            start = float(self.rng.uniform(0, scene_dur - ev_dur))
            if all(
                start + ev_dur <= lo or start >= hi for lo, hi in occupied
            ):
                return start
        return None


def frame_labels(
    events: list[SceneEvent], n_frames: int, frames_per_second: float
) -> np.ndarray:
    """Strong labels: (n_frames, NUM_CLASSES) multi-hot matrix (column i =
    class_id i+1)."""
    labels = np.zeros((n_frames, NUM_CLASSES), dtype=np.float32)
    for ev in events:
        f0 = int(round(ev.start_s * frames_per_second))
        f1 = min(int(round(ev.end_s * frames_per_second)), n_frames)
        labels[f0:f1, ev.class_id - 1] = 1.0
    return labels
