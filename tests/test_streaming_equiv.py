"""Chunked long-file inference must reproduce whole-file posteriors.

Stitching correctness is tested with a deterministic, purely local posterior
function (framewise RMS energy), so any gap/duplication/off-by-one in the
block logic shows up as a hard mismatch. A separate smoke test runs the real
CRNN through the chunked path.
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from echosentinel.constants import TARGET_SR
from echosentinel.infer.posteriors import file_posteriors

FPS = 25.0
HOP = int(TARGET_SR / FPS)


def rms_posteriors(wave: torch.Tensor) -> torch.Tensor:
    """Deterministic local 'model': per-frame RMS replicated over 4 classes."""
    n = (wave.shape[0] // HOP) * HOP
    frames = wave[:n].reshape(-1, HOP)
    rms = torch.sqrt((frames**2).mean(dim=1, keepdim=True))
    return rms.repeat(1, 4)


@pytest.fixture()
def long_wav(tmp_path: Path) -> Path:
    rng = np.random.default_rng(7)
    y = (rng.standard_normal(int(95.3 * TARGET_SR)) * 0.05).astype(np.float32)
    y[int(40 * TARGET_SR) : int(45 * TARGET_SR)] += (
        0.3 * np.sin(2 * np.pi * 440 * np.arange(5 * TARGET_SR) / TARGET_SR)
    ).astype(np.float32)
    path = tmp_path / "long.wav"
    sf.write(path, y, TARGET_SR, subtype="FLOAT")
    return path


def test_chunked_equals_wholefile(long_wav: Path):
    whole = file_posteriors(long_wav, rms_posteriors, FPS, block_seconds=10_000)
    chunked = file_posteriors(long_wav, rms_posteriors, FPS, block_seconds=30, overlap_seconds=2)
    assert whole.shape == chunked.shape
    np.testing.assert_allclose(chunked, whole, atol=1e-4)


def test_crnn_chunked_smoke(long_wav: Path):
    from echosentinel.models.crnn import CRNN

    torch.manual_seed(0)
    model = CRNN().eval()
    probs = file_posteriors(
        long_wav, model.posteriors, FPS, block_seconds=40, overlap_seconds=2
    )
    assert probs.shape[0] == int(95.3 * FPS)
    assert probs.shape[1] == 4
    assert np.all((probs >= 0) & (probs <= 1))
