"""Phase 2: PCEN front-end, SpecAugment, and the PANNs SED model.

Covers output shapes/ranges, PCEN's streaming-state continuity (the property
that lets the same module serve live inference), SpecAugment train/eval
behavior, and that both models plug into the registry with matching frame
rates and produce valid posteriors through the chunked inference path.
"""

import numpy as np
import pytest
import torch

from echosentinel.constants import N_MELS, NUM_CLASSES, TARGET_SR
from echosentinel.features.augment import SpecAugment
from echosentinel.features.pcen import PCEN
from echosentinel.models.registry import build_model, model_frames_per_second


def _wave(seconds=4.0, seed=0):
    rng = np.random.default_rng(seed)
    return torch.from_numpy((rng.standard_normal(int(seconds * TARGET_SR)) * 0.1).astype(np.float32))


def test_pcen_output_shape_and_finiteness():
    pcen = PCEN()
    feats = pcen(_wave())
    assert feats.shape[0] == N_MELS
    assert torch.isfinite(feats).all()


def test_pcen_suppresses_stationary_more_than_transient():
    """A steady tone (stationary) should be attenuated by the AGC relative to
    an impulsive burst of equal energy — the core reason PCEN helps here."""
    pcen = PCEN(trainable=False)
    n = TARGET_SR * 3
    t = torch.arange(n) / TARGET_SR
    tone = 0.1 * torch.sin(2 * np.pi * 2000 * t).float()
    burst = torch.zeros(n)
    burst[n // 2 : n // 2 + TARGET_SR // 10] = 1.0  # 0.1 s impulse
    tone_resp = pcen(tone).mean()
    burst_resp = pcen(burst).max()
    assert burst_resp > tone_resp


def test_pcen_streaming_state_is_continuous():
    """Streaming in two blocks while carrying the smoother state reproduces
    the same total feature length and stays finite; the second block's frames
    must differ from cold-starting it (proving state actually carries over)."""
    pcen = PCEN(trainable=False).eval()
    wave = _wave(seconds=6.0)
    whole = pcen(wave)
    half = wave.shape[0] // 2
    with torch.no_grad():
        f_a, s_a = pcen(wave[:half], state=torch.zeros(1, N_MELS))
        f_b_warm, _ = pcen(wave[half:], state=s_a)
        f_b_cold, _ = pcen(wave[half:], state=torch.zeros(1, N_MELS))
    streamed = torch.cat([f_a, f_b_warm], dim=-1)
    # a split mid-STFT-window shifts the total frame count by at most one
    assert abs(streamed.shape[-1] - whole.shape[-1]) <= 1
    assert streamed.shape[0] == whole.shape[0]
    assert torch.isfinite(streamed).all()
    # carrying real state changes the first frames of the next block
    assert not torch.allclose(f_b_warm[:, :5], f_b_cold[:, :5])


def test_specaugment_only_masks_in_training():
    aug = SpecAugment(time_masks=2, time_mask_max=20, freq_masks=2, freq_mask_max=8)
    spec = torch.ones(2, N_MELS, 200)
    aug.eval()
    assert torch.equal(aug(spec), spec)
    aug.train()
    out = aug(spec)
    assert (out == 0).any()  # something got masked
    assert out.shape == spec.shape


@pytest.mark.parametrize("name,kwargs", [
    ("crnn", {"frontend": "logmel"}),
    ("crnn", {"frontend": "pcen"}),
    ("panns", {"frontend": "pcen"}),
])
def test_models_produce_valid_frame_logits(name, kwargs):
    torch.manual_seed(0)
    model = build_model(name, **kwargs).eval()
    wave = _wave(seconds=4.0).unsqueeze(0)
    with torch.no_grad():
        logits = model(wave)
    fps = model_frames_per_second(name)
    expected_frames = int(4.0 * fps)
    assert abs(logits.shape[1] - expected_frames) <= 2
    assert logits.shape[2] == NUM_CLASSES
    probs = model.posteriors(wave.squeeze(0))
    assert torch.all((probs >= 0) & (probs <= 1))


def test_panns_frame_rate_matches_crnn():
    # inference stack assumes a single fps; both models must agree
    assert model_frames_per_second("panns") == model_frames_per_second("crnn") == 25.0
