"""Scene synthesizer invariants: length, event bounds, temporal separation,
frame-label alignment. Uses a tiny generated dataset, no real audio needed."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from echosentinel.constants import FRAMES_PER_SECOND, TARGET_SR
from echosentinel.data.scene_synth import SceneSynthesizer, frame_labels


@pytest.fixture()
def tiny_dataset(tmp_path: Path) -> tuple[pd.DataFrame, Path]:
    rng = np.random.default_rng(0)
    rows = []
    specs = [
        (1, "vessel", 20.0),        # long enough to serve as engine bed too
        (2, "marine_animal", 6.0),
        (3, "natural_sound", 8.0),
    ]
    for cid, name, dur in specs:
        y = (rng.standard_normal(int(dur * TARGET_SR)) * 0.1).astype(np.float32)
        rel = f"{name}.wav"
        sf.write(tmp_path / rel, y, TARGET_SR)
        rows.append(
            {"path": rel, "class_id": cid, "class_name": name,
             "duration_s": dur, "source_group": name}
        )
    return pd.DataFrame(rows), tmp_path


def test_scene_invariants(tiny_dataset):
    manifest, root = tiny_dataset
    synth = SceneSynthesizer(manifest, root, np.random.default_rng(1))
    for _ in range(10):
        scene, events = synth.make_scene(20.0, n_events_range=(1, 3))
        assert scene.shape == (20 * TARGET_SR,)
        assert scene.dtype == np.float32
        assert float(np.max(np.abs(scene))) <= 1.0
        for ev in events:
            assert 0 <= ev.start_s < ev.end_s <= 20.0
            assert ev.class_id in (1, 2, 3, 4)
        # events must be separated in time (PS-12 structure)
        for a, b in zip(events, events[1:]):
            assert a.end_s <= b.start_s


def test_procedural_class4_without_real_data(tiny_dataset):
    manifest, root = tiny_dataset  # no class 4 in the pool at all
    synth = SceneSynthesizer(
        manifest, root, np.random.default_rng(2), procedural_class4_prob=1.0
    )
    audio = synth._draw_event_audio(4)
    assert len(audio) > 0
    assert audio.dtype == np.float32


def test_frame_labels_alignment(tiny_dataset):
    manifest, root = tiny_dataset
    synth = SceneSynthesizer(manifest, root, np.random.default_rng(3))
    scene, events = synth.make_scene(15.0, n_events_range=(2, 3))
    n_frames = int(15 * FRAMES_PER_SECOND)
    labels = frame_labels(events, n_frames, FRAMES_PER_SECOND)
    assert labels.shape == (n_frames, 4)
    # at most one active class per frame (events don't overlap)
    assert labels.sum(axis=1).max() <= 1.0
    for ev in events:
        mid = int((ev.start_s + ev.end_s) / 2 * FRAMES_PER_SECOND)
        assert labels[mid, ev.class_id - 1] == 1.0
