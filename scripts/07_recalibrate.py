"""One-command recalibration after swapping in new model weights.

    python scripts/07_recalibrate.py

Thresholds are model-specific, so every new weights/panns_pcen.pt needs this
once: rebuilds the synthetic validation set (with mined real-noise beds when
available) and re-tunes the per-class hysteresis thresholds against the exact
IER metric, writing the winners into configs/inference.yaml. Restart the web
console afterwards to pick up the new weights + thresholds.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(args: list[str]) -> None:
    print(f"\n=== {' '.join(args)} ===")
    result = subprocess.run([sys.executable, *args], cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(f"step failed: {' '.join(args)}")


def main() -> None:
    weights = PROJECT_ROOT / "weights" / "panns_pcen.pt"
    if not weights.exists():
        raise SystemExit(f"{weights} not found — download the trained weights first.")

    mined = PROJECT_ROOT / "Dataset" / "mined_noise"
    valset_args = ["scripts/03_build_synth_valset.py", "--n-scenes", "24", "--seconds", "60"]
    if mined.is_dir() and any(mined.glob("*.wav")):
        valset_args.append("--mined")

    run(valset_args)
    run(["scripts/06_tune_thresholds.py", "--write"])

    print(
        "\nRecalibration complete: configs/inference.yaml updated for the new model."
        "\nRestart the console to serve it:  python -m echosentinel.server"
    )


if __name__ == "__main__":
    main()
