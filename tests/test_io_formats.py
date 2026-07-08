"""The loader must decode every WAV subtype present in the real dataset.

For each distinct (format subtype, channel count) combination found across the
training and official test folders, load a short window from one representative
file and check the standardization contract: mono float32 at TARGET_SR in [-1, 1].
Skips gracefully when the Dataset folder is not present (e.g. CI without data).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from echosentinel.audio.io import load_audio, probe, rms_normalize
from echosentinel.constants import TARGET_SR

DATASET_ROOT = Path(__file__).resolve().parents[1] / "Dataset"

SCAN_FOLDERS = [
    "Marine Animals",
    "Natural Sounds",
    "Human made Objects",
    "Cargo dataset",
    "Tanker ds",
    "Tug ds",
    "passengership ds",
    "other_anthropogenic",
    "20251103_PS12",
    "I4_20251013_PS12",
]

AUDIO_PATTERNS = ("*.wav", "*.mp3")


def _representative_files() -> list[Path]:
    """One file per distinct (subtype, channels, sr-band) combination."""
    if not DATASET_ROOT.is_dir():
        return []
    seen: dict[tuple, Path] = {}
    for folder in SCAN_FOLDERS:
        folder_path = DATASET_ROOT / folder
        if not folder_path.is_dir():
            continue
        for pattern in AUDIO_PATTERNS:
            for wav in sorted(folder_path.glob(pattern)):
                try:
                    info = probe(wav)
                except Exception:
                    seen[("UNREADABLE", wav.name)] = wav
                    continue
                sr_band = "low" if info.sr <= 16_000 else ("mid" if info.sr <= 44_100 else "high")
                key = (info.subtype, info.channels, sr_band)
                seen.setdefault(key, wav)
    return sorted(seen.values())


FILES = _representative_files()


@pytest.mark.skipif(not FILES, reason="Dataset folder not available")
@pytest.mark.parametrize("wav", FILES, ids=lambda p: p.name)
def test_load_standardizes(wav: Path) -> None:
    y, sr = load_audio(wav, duration=2.0)
    assert sr == TARGET_SR
    assert y.dtype == np.float32
    assert y.ndim == 1
    assert len(y) > 0
    assert float(np.max(np.abs(y))) <= 1.0


@pytest.mark.skipif(not FILES, reason="Dataset folder not available")
def test_offset_window_reads_partial_file() -> None:
    wav = FILES[0]
    info = probe(wav)
    if info.duration < 3.0:
        pytest.skip("representative file too short for offset test")
    y, _ = load_audio(wav, offset=1.0, duration=1.0)
    assert abs(len(y) - TARGET_SR) <= TARGET_SR // 100  # ~1 s within resampling tolerance


def test_rms_normalize_silent_input_unchanged() -> None:
    silent = np.zeros(1000, dtype=np.float32)
    out = rms_normalize(silent)
    assert np.array_equal(out, silent)


def test_rms_normalize_hits_target_level() -> None:
    rng = np.random.default_rng(0)
    y = (rng.standard_normal(32_000) * 0.01).astype(np.float32)
    out = rms_normalize(y, target_dbfs=-25.0)
    rms_db = 20 * np.log10(float(np.sqrt(np.mean(out**2))))
    assert abs(rms_db - (-25.0)) < 0.5
