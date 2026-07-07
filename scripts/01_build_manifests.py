"""Build manifests/train.csv from the human-reviewed relabel_review.csv.

Usage (from echosentinel_v2/):
    python scripts/01_build_manifests.py [--config configs/data.yaml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from echosentinel.data.manifest import build_train_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "data.yaml"))
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    review_csv = PROJECT_ROOT / str(cfg.relabel_review_csv)
    if not review_csv.exists():
        raise SystemExit(f"{review_csv} not found — run scripts/00_audit_dataset.py first")

    out_csv = PROJECT_ROOT / str(cfg.train_manifest_csv)
    manifest = build_train_manifest(review_csv, out_csv)

    print(f"Wrote {out_csv} ({len(manifest)} files)")
    print("\nFiles per class:")
    print(manifest["class_name"].value_counts().to_string())
    print("\nTotal audio per class (minutes):")
    per_class = (manifest.groupby("class_name")["duration_s"].sum() / 60).round(1)
    print(per_class.to_string())
    print(f"\nDistinct source groups: {manifest['source_group'].nunique()}")


if __name__ == "__main__":
    main()
