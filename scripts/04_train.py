"""Train a frame-level SED model on synthesized scenes.

Usage (from echosentinel_v2/, venv active):
    python scripts/04_train.py [--config configs/model_crnn.yaml]
                               [--epochs N] [--epoch-size N]   # quick overrides

On Colab/Kaggle: clone/upload the repo + Dataset, `pip install -e .`, then run
this script the same way — it auto-selects CUDA when available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from echosentinel.train.loop import train

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "model_crnn.yaml"))
    parser.add_argument("--data-config", default=str(PROJECT_ROOT / "configs" / "data.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--epoch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    data_cfg = OmegaConf.load(args.data_config)
    dataset_root = (PROJECT_ROOT / str(data_cfg.dataset_root)).resolve()
    manifest = pd.read_csv(PROJECT_ROOT / str(data_cfg.train_manifest_csv))

    train(
        manifest=manifest,
        dataset_root=dataset_root,
        out_path=PROJECT_ROOT / str(cfg.weights_out),
        model_name=str(cfg.model),
        epochs=args.epochs or int(cfg.epochs),
        batch_size=int(cfg.batch_size),
        epoch_size=args.epoch_size or int(cfg.epoch_size),
        scene_seconds=float(cfg.scene_seconds),
        lr=float(cfg.lr),
        label_pool=int(cfg.label_pool),
        val_scenes=int(cfg.val_scenes),
        num_workers=int(cfg.num_workers),
        device=args.device,
        snr_db_range=tuple(cfg.synth.snr_db_range),
        snr_skew_low=bool(cfg.synth.snr_skew_low),
        engine_bed_prob=float(cfg.synth.engine_bed_prob),
        procedural_class4_prob=float(cfg.synth.procedural_class4_prob),
    )


if __name__ == "__main__":
    main()
