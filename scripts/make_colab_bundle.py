"""Build a lean zip for uploading to Google Drive / Colab.

Includes the code, configs, manifests, and only the TRAINING audio folders —
it excludes the multi-GB official test sets (not needed to train), the venv,
.git, cached outputs, and model binaries. Result is ~1-2 GB instead of ~7 GB.

    python scripts/make_colab_bundle.py            # -> echosentinel_v2_colab.zip
    python scripts/make_colab_bundle.py --out X.zip

Upload the resulting zip to Drive, then run notebooks/train_colab.ipynb.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Directories never worth uploading.
EXCLUDE_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache",
                "out", "runs", ".ipynb_checkpoints"}
# Extensions to skip (trained/pretrained binaries are re-created on Colab).
EXCLUDE_SUFFIXES = {".pt", ".pth", ".ckpt", ".pyc"}


def _training_folders() -> set[str]:
    cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "data.yaml")
    train = set(cfg.source_folders.keys())
    test = set(cfg.test_folders)
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "echosentinel_v2_colab.zip"))
    args = parser.parse_args()

    train_folders, test_folders = _training_folders()
    dataset_root = PROJECT_ROOT / str(OmegaConf.load(PROJECT_ROOT / "configs" / "data.yaml").dataset_root)
    out_zip = Path(args.out)

    def keep(path: Path) -> bool:
        parts = set(path.relative_to(PROJECT_ROOT).parts)
        if parts & EXCLUDE_DIRS:
            return False
        if path.suffix.lower() in EXCLUDE_SUFFIXES:
            return False
        # Inside Dataset/, keep only training folders (skip test sets + mock).
        try:
            rel = path.relative_to(dataset_root)
        except ValueError:
            return True
        top = rel.parts[0] if rel.parts else ""
        return top in train_folders

    files, total = [], 0
    for p in PROJECT_ROOT.rglob("*"):
        if p.is_file() and p != out_zip and keep(p):
            files.append(p)
            total += p.stat().st_size

    print(f"Bundling {len(files)} files ({total / 1e6:.0f} MB uncompressed) -> {out_zip}")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in files:
            # Store under an 'echosentinel_v2/' top folder so it unzips cleanly.
            zf.write(p, Path("echosentinel_v2") / p.relative_to(PROJECT_ROOT))
    print(f"Done: {out_zip} ({out_zip.stat().st_size / 1e6:.0f} MB compressed)")
    print("Upload this to Google Drive, then run notebooks/train_colab.ipynb")


if __name__ == "__main__":
    main()
