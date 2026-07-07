"""Training loop for frame-level SED models on synthesized scenes.

Split discipline: the clean pool is divided by ``source_group`` (recording
origin), never by file, so the same recording cannot appear on both sides.
The validation set is a fixed bank of scenes synthesized once from val-group
sources only.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from echosentinel.data.dataset import SynthSEDDataset, pool_labels
from echosentinel.data.scene_synth import SceneSynthesizer, frame_labels
from echosentinel.constants import FRAMES_PER_SECOND
from echosentinel.models.registry import build_model
from echosentinel.train.losses import framewise_bce
from echosentinel.train.metrics import frame_f1


def grouped_split(
    manifest: pd.DataFrame, val_fraction: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic split by hashing source_group."""
    def bucket(group: str) -> float:
        h = hashlib.sha1(group.encode()).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    is_val = manifest["source_group"].map(bucket) < val_fraction
    return manifest[~is_val].reset_index(drop=True), manifest[is_val].reset_index(drop=True)


def build_val_bank(
    manifest: pd.DataFrame,
    dataset_root: Path,
    n_scenes: int,
    scene_seconds: float,
    label_pool: int,
    seed: int = 4242,
    **synth_kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed validation scenes rendered once (deterministic seed)."""
    synth = SceneSynthesizer(
        manifest, dataset_root, np.random.default_rng(seed), **synth_kwargs
    )
    waves, labels = [], []
    n_frames = int(scene_seconds * FRAMES_PER_SECOND)
    for _ in range(n_scenes):
        scene, events = synth.make_scene(scene_seconds, (0, 3))
        waves.append(torch.from_numpy(scene))
        lab = pool_labels(frame_labels(events, n_frames, FRAMES_PER_SECOND), label_pool)
        labels.append(torch.from_numpy(lab))
    return torch.stack(waves), torch.stack(labels)


def train(
    manifest: pd.DataFrame,
    dataset_root: Path,
    out_path: Path,
    model_name: str = "crnn",
    epochs: int = 30,
    batch_size: int = 16,
    epoch_size: int = 512,
    scene_seconds: float = 10.0,
    lr: float = 1e-3,
    label_pool: int = 4,
    val_scenes: int = 48,
    num_workers: int = 0,
    device: str | None = None,
    log=print,
    **synth_kwargs,
) -> Path:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Training {model_name} on {device}")

    train_df, val_df = grouped_split(manifest)
    log(f"Split: {len(train_df)} train files / {len(val_df)} val files "
        f"({train_df['source_group'].nunique()}/{val_df['source_group'].nunique()} groups)")

    ds = SynthSEDDataset(
        train_df, dataset_root, epoch_size=epoch_size,
        scene_seconds=scene_seconds, label_pool=label_pool, **synth_kwargs,
    )
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
    val_waves, val_labels = build_val_bank(
        val_df if len(val_df) >= 10 else manifest,
        dataset_root, val_scenes, scene_seconds, label_pool, **synth_kwargs,
    )

    model = build_model(model_name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = framewise_bce().to(device)

    best_f1 = -1.0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        total = 0.0
        for waves, labels in loader:
            waves, labels = waves.to(device), labels.to(device)
            logits = model(waves)
            t = min(logits.shape[1], labels.shape[1])  # trim off-by-one frames
            loss = criterion(logits[:, :t], labels[:, :t])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss) * len(waves)
        sched.step()

        model.eval()
        with torch.no_grad():
            probs = []
            for i in range(0, len(val_waves), batch_size):
                batch = val_waves[i : i + batch_size].to(device)
                probs.append(torch.sigmoid(model(batch)).cpu())
            probs = torch.cat(probs)
            t = min(probs.shape[1], val_labels.shape[1])
            metrics = frame_f1(probs[:, :t], val_labels[:, :t])

        log(
            f"epoch {epoch:3d}  loss {total / len(ds):.4f}  "
            f"val f1_macro {metrics['f1_macro']:.3f}  "
            f"({', '.join(f'{k[3:]} {v:.2f}' for k, v in metrics.items() if k != 'f1_macro')})  "
            f"{time.time() - t0:.0f}s"
        )
        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            torch.save(
                {"model_name": model_name, "state_dict": model.state_dict(),
                 "f1_macro": best_f1, "epoch": epoch},
                out_path,
            )
    log(f"Best val f1_macro {best_f1:.3f} -> {out_path}")
    return out_path
