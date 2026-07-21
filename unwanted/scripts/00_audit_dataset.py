"""Scan the Dataset source folders and write manifests/relabel_review.csv.

Usage (from echosentinel_v2/):
    python scripts/00_audit_dataset.py [--config configs/data.yaml]

Review the CSV afterwards — especially rows with confidence medium/low — and
edit the ``final_class`` column where the proposal is wrong. Then run
scripts/01_build_manifests.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from echosentinel.data.audit import audit_folders

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "data.yaml"))
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    dataset_root = (PROJECT_ROOT / str(cfg.dataset_root)).resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root not found: {dataset_root}")

    df = audit_folders(dataset_root, dict(cfg.source_folders))
    out = PROJECT_ROOT / str(cfg.relabel_review_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Audited {len(df)} files from {dataset_root}")
    print(f"\nProposed class counts:")
    print(df["proposed_class"].value_counts().to_string())
    print(f"\nConfidence breakdown:")
    print(df["confidence"].value_counts().to_string())
    flagged = df[df["confidence"] != "high"]
    if not flagged.empty:
        print(f"\n{len(flagged)} file(s) need human review (confidence != high):")
        for _, r in flagged.iterrows():
            print(f"  [{r.confidence:6s}] {r.path}  ->  {r.proposed_class}  ({r.reason})")
    errors = df[df["error"] != ""]
    if not errors.empty:
        print(f"\nWARNING: {len(errors)} file(s) had unreadable headers:")
        for _, r in errors.iterrows():
            print(f"  {r.path}: {r.error}")
    print(f"\nReview file written: {out}")
    print("Edit the 'final_class' column where needed, then run scripts/01_build_manifests.py")


if __name__ == "__main__":
    main()
