"""Mine background-noise beds from the official (unlabeled) test recordings.

For each test wav, several candidate windows are read and the QUIETEST, most
STATIONARY ones are kept — bottom-of-file RMS with low frame-to-frame energy
variation, i.e. windows most likely to be pure background (engine hum /
ambient) rather than containing a faint event. Snippets are written to
Dataset/mined_noise/ at 32 kHz mono PCM_16 and used by the scene synthesizer
as unlabeled bed texture (never labeled, never used alone — see noise_bank).

Contamination risk (a faint real event inside a "quiet" window) is mitigated
by the stationarity filter, by mixing mined beds under synthetic ambient at
moderate gain, and by the use_mined_noise A/B flag in training config.

Usage (from echosentinel_v2/):
    python scripts/02_build_noise_bank.py [--per-file 2] [--window 10]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from omegaconf import OmegaConf
from tqdm import tqdm

from echosentinel.audio.io import load_audio, probe
from echosentinel.constants import TARGET_SR

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def window_stats(y: np.ndarray, sr: int) -> tuple[float, float]:
    """(rms, stationarity) for a window; stationarity = coefficient of
    variation of 1-second sub-frame RMS (lower = steadier = safer bed)."""
    n = (len(y) // sr) * sr
    if n == 0:
        return 0.0, np.inf
    frames = y[:n].reshape(-1, sr)
    frame_rms = np.sqrt((frames**2).mean(axis=1)) + 1e-12
    return float(frame_rms.mean()), float(frame_rms.std() / frame_rms.mean())


def mine_file(
    wav: Path, out_dir: Path, per_file: int, window_s: float, candidates: int = 12
) -> int:
    try:
        info = probe(wav)
    except Exception:
        return 0
    if info.duration < window_s * 2:
        return 0

    offsets = np.linspace(0, info.duration - window_s, num=candidates)
    scored = []
    for off in offsets:
        try:
            y, _ = load_audio(wav, offset=float(off), duration=window_s)
        except Exception:
            continue
        rms, cv = window_stats(y, TARGET_SR)
        if rms > 1e-6 and cv < 0.35:  # non-digital-silence, stationary
            scored.append((rms, float(off), y))

    scored.sort(key=lambda t: t[0])  # quietest first
    written = 0
    for rms, off, y in scored[:per_file]:
        name = f"{wav.stem}_t{int(off):05d}.wav"
        sf.write(out_dir / name, y, TARGET_SR, subtype="PCM_16")
        written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default=str(PROJECT_ROOT / "configs" / "data.yaml"))
    parser.add_argument("--per-file", type=int, default=2)
    parser.add_argument("--window", type=float, default=10.0)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.data_config)
    dataset_root = (PROJECT_ROOT / str(cfg.dataset_root)).resolve()
    out_dir = dataset_root / str(cfg.get("noise_bed_folder", "mined_noise"))
    out_dir.mkdir(parents=True, exist_ok=True)

    wavs: list[Path] = []
    for folder in cfg.test_folders:
        wavs += sorted((dataset_root / str(folder)).glob("*.wav"))
    if not wavs:
        raise SystemExit("No test wavs found — check test_folders in the data config.")

    total = 0
    for wav in tqdm(wavs, desc="mining"):
        total += mine_file(wav, out_dir, args.per_file, args.window)
    print(f"Wrote {total} bed snippets ({args.window:.0f}s each) to {out_dir}")


if __name__ == "__main__":
    main()
