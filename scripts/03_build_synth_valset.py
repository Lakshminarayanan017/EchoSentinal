"""Build the fixed synthetic validation set (wavs + ground_truth.json).

Usage (from echosentinel_v2/):
    python scripts/03_build_synth_valset.py [--n-scenes 24] [--seconds 60]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from echosentinel.eval.synth_valset import build_valset
from echosentinel.train.loop import grouped_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default=str(PROJECT_ROOT / "configs" / "data.yaml"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "out" / "synth_valset"))
    parser.add_argument("--n-scenes", type=int, default=24)
    parser.add_argument("--seconds", type=float, default=60.0)
    args = parser.parse_args()

    data_cfg = OmegaConf.load(args.data_config)
    dataset_root = (PROJECT_ROOT / str(data_cfg.dataset_root)).resolve()
    manifest = pd.read_csv(PROJECT_ROOT / str(data_cfg.train_manifest_csv))

    _, val_df = grouped_split(manifest)
    source = val_df if len(val_df) >= 10 else manifest
    gt = build_valset(
        source, dataset_root, Path(args.out),
        n_scenes=args.n_scenes, scene_seconds=args.seconds,
    )
    print(f"Validation set written to {args.out} (ground truth: {gt})")


if __name__ == "__main__":
    main()
