"""Download the PANNs CNN14 (AudioSet) checkpoint once and cache it in weights/.

Run this ONCE on a machine with internet (dev box, or the first Colab cell).
After that, training/inference are fully offline — the checkpoint is baked
into the repo/Docker image.

    python scripts/fetch_pretrained.py

Source: Kong et al. PANNs, Cnn14_mAP=0.431.pth (~310 MB), Zenodo record
3987831. If the primary URL fails, download manually from
https://zenodo.org/records/3987831 and place the file at
weights/Cnn14_mAP=0.431.pth
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET = PROJECT_ROOT / "weights" / "Cnn14_mAP=0.431.pth"
URL = "https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"


def main() -> None:
    if TARGET.exists() and TARGET.stat().st_size > 100_000_000:
        print(f"Already present: {TARGET} ({TARGET.stat().st_size / 1e6:.0f} MB)")
        return
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading CNN14 checkpoint -> {TARGET}")

    def _progress(block: int, block_size: int, total: int) -> None:
        done = block * block_size
        pct = 100 * done / total if total > 0 else 0
        sys.stdout.write(f"\r  {done / 1e6:6.0f} MB / {total / 1e6:.0f} MB ({pct:4.1f}%)")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(URL, TARGET, _progress)
        print(f"\nSaved {TARGET.stat().st_size / 1e6:.0f} MB")
    except Exception as e:
        print(f"\nDownload failed: {e!r}")
        print("Download manually from https://zenodo.org/records/3987831")
        print(f"and place it at {TARGET}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
