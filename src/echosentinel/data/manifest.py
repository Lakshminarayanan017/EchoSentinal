"""Manifest building: the reviewed audit CSV becomes the training manifest.

Manifests are the single source of truth for training data — no other code
walks the Dataset folders. ``train.csv`` columns:

    path, class_id, class_name, sr, channels, duration_s, source_group

``source_group`` identifies the recording origin (freesound uploader or the
individual file) and is the unit for grouped train/val splits, so the same
underlying recording never leaks across the split boundary.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from echosentinel.constants import NAME_TO_ID
from echosentinel.data.audit import source_group


def build_train_manifest(review_csv: Path, out_csv: Path) -> pd.DataFrame:
    """Convert a human-reviewed relabel CSV into the train manifest."""
    review = pd.read_csv(review_csv)

    unreadable = review[review["error"].notna() & (review["error"] != "")]
    if not unreadable.empty:
        print(f"Skipping {len(unreadable)} unreadable file(s):")
        for p in unreadable["path"]:
            print(f"  - {p}")
        review = review.drop(unreadable.index)

    bad = set(review["final_class"]) - set(NAME_TO_ID)
    if bad:
        raise ValueError(
            f"relabel CSV contains final_class values outside the PS-12 map: {sorted(bad)}"
        )

    manifest = pd.DataFrame(
        {
            "path": review["path"],
            "class_id": review["final_class"].map(NAME_TO_ID),
            "class_name": review["final_class"],
            "sr": review["sr"],
            "channels": review["channels"],
            "duration_s": review["duration_s"],
            "source_group": [
                source_group(Path(p).name, Path(p).parent.as_posix())
                for p in review["path"]
            ],
        }
    ).sort_values(["class_id", "path"], ignore_index=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_csv, index=False)
    return manifest
