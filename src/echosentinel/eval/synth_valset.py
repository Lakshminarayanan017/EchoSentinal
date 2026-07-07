"""Render a fixed synthetic validation set that mirrors test conditions.

Long scenes (default 60 s) with engine-noise beds and events at low SNR,
written to disk as wav + a ground-truth JSON in the competition format, so
`predict.py` output can be scored against it with the exact IER metric.
Events are drawn from the *validation* source groups only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from echosentinel.constants import TARGET_SR
from echosentinel.data.scene_synth import SceneSynthesizer
from echosentinel.infer.json_writer import build_results_json, write_results_json
from echosentinel.infer.postprocess import Event


def build_valset(
    manifest: pd.DataFrame,
    dataset_root: Path,
    out_dir: Path,
    n_scenes: int = 24,
    scene_seconds: float = 60.0,
    seed: int = 20260707,
    **synth_kwargs,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    synth = SceneSynthesizer(
        manifest, dataset_root, np.random.default_rng(seed), **synth_kwargs
    )
    per_file = []
    for i in range(1, n_scenes + 1):
        scene, events = synth.make_scene(scene_seconds, n_events_range=(1, 4))
        name = f"synthval_{i:03d}.wav"
        sf.write(out_dir / name, scene, TARGET_SR, subtype="FLOAT")
        gt = [Event(ev.class_id, ev.start_s, ev.end_s, 1.0) for ev in events]
        per_file.append((name, scene_seconds, gt))

    gt_path = out_dir / "ground_truth.json"
    write_results_json(gt_path, build_results_json(per_file, contributor="GROUND_TRUTH"))
    return gt_path
