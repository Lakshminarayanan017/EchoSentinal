"""Torch datasets over the scene synthesizer.

Scenes are synthesized online in ``__getitem__`` — every epoch sees new
mixtures, which is the main defence against overfitting the tiny clean pool.
Labels are built at the feature frame rate and max-pooled down to the model's
output frame rate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from echosentinel.constants import FRAMES_PER_SECOND
from echosentinel.data.scene_synth import SceneSynthesizer, frame_labels


def pool_labels(labels: np.ndarray, factor: int) -> np.ndarray:
    """Max-pool (T, C) frame labels by ``factor`` along time."""
    n = (labels.shape[0] // factor) * factor
    return labels[:n].reshape(-1, factor, labels.shape[1]).max(axis=1)


class SynthSEDDataset(Dataset):
    """Online-synthesized scenes with strong frame labels."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        dataset_root: Path,
        epoch_size: int = 512,
        scene_seconds: float = 10.0,
        n_events_range: tuple[int, int] = (0, 3),
        label_pool: int = 4,
        seed: int = 0,
        **synth_kwargs,
    ) -> None:
        self.epoch_size = epoch_size
        self.scene_seconds = scene_seconds
        self.n_events_range = n_events_range
        self.label_pool = label_pool
        self.seed = seed
        self._manifest = manifest
        self._root = dataset_root
        self._synth_kwargs = synth_kwargs
        self._synth: SceneSynthesizer | None = None  # built lazily per worker

    def _synthesizer(self) -> SceneSynthesizer:
        if self._synth is None:
            info = torch.utils.data.get_worker_info()
            worker = info.id if info else 0
            rng = np.random.default_rng(self.seed + 1000 * worker)
            self._synth = SceneSynthesizer(
                self._manifest, self._root, rng, **self._synth_kwargs
            )
        return self._synth

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, _: int) -> tuple[torch.Tensor, torch.Tensor]:
        synth = self._synthesizer()
        scene, events = synth.make_scene(self.scene_seconds, self.n_events_range)
        n_frames = int(self.scene_seconds * FRAMES_PER_SECOND)
        labels = frame_labels(events, n_frames, FRAMES_PER_SECOND)
        labels = pool_labels(labels, self.label_pool)
        return torch.from_numpy(scene), torch.from_numpy(labels)
