"""Tune per-class hysteresis thresholds (and median filter) against IER.

Posteriors are computed ONCE per validation file, then post-processing
parameters are searched by coordinate descent, scoring each candidate with
the exact competition metric. Because a miss (weight 1.0) costs 4x a false
alarm (0.25), the optimum typically lands at low thresholds — this script
finds where, per class, instead of guessing.

Usage (from echosentinel_v2/):
    python scripts/06_tune_thresholds.py --weights weights/panns_pcen.pt
        [--valset out/synth_valset] [--write]   # --write updates configs/inference.yaml

Re-run whenever the model weights change (each model has its own calibration).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from echosentinel.audio.io import probe
from echosentinel.constants import CLASS_MAP
from echosentinel.eval.ier import IERScorer
from echosentinel.infer.json_writer import events_by_file, read_results_json
from echosentinel.infer.posteriors import file_posteriors
from echosentinel.infer.postprocess import probs_to_events
from echosentinel.models.registry import build_model, model_frames_per_second

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HIGH_GRID = [0.15, 0.2, 0.3, 0.4, 0.5, 0.6]
LOW_RATIO_GRID = [0.4, 0.6, 0.8]       # low = ratio * high
MEDIAN_GRID = [0.3, 0.7, 1.2]          # seconds


def score(
    cached: list[tuple[str, np.ndarray]],
    reference: dict[str, list[tuple[int, float, float]]],
    fps: float,
    thresholds: dict[str, dict[str, float]],
    median_seconds: float,
) -> float:
    scorer = IERScorer()
    for name, probs in cached:
        events = probs_to_events(probs, fps, thresholds, median_seconds=median_seconds)
        hyp = [(e.category_id, e.start, e.end) for e in events]
        scorer.add_file(reference.get(name, []), hyp, uri=name)
    return scorer.report()["ier"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(PROJECT_ROOT / "weights" / "panns_pcen.pt"))
    parser.add_argument("--valset", default=str(PROJECT_ROOT / "out" / "synth_valset"))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "inference.yaml"))
    parser.add_argument("--write", action="store_true", help="write tuned values into the config")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.weights, map_location=device, weights_only=True)
    model = build_model(ckpt["model_name"], **ckpt.get("model_kwargs", {}))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    fps = model_frames_per_second(ckpt["model_name"])

    valset = Path(args.valset)
    reference = events_by_file(read_results_json(valset / "ground_truth.json"))

    cached: list[tuple[str, np.ndarray]] = []
    for wav in tqdm(sorted(valset.glob("*.wav")), desc="posteriors"):
        probs = file_posteriors(wav, lambda w: model.posteriors(w.to(device)), fps)
        cached.append((wav.name, probs))

    cfg = OmegaConf.load(args.config)
    thresholds = {k: dict(v) for k, v in OmegaConf.to_container(cfg.thresholds).items()}
    median_seconds = float(cfg.posteriors.median_filter_seconds)

    best = score(cached, reference, fps, thresholds, median_seconds)
    print(f"Starting IER with current config: {best:.4f}")

    for round_i in range(2):  # two rounds of coordinate descent
        for cname in CLASS_MAP.values():
            for high in HIGH_GRID:
                for ratio in LOW_RATIO_GRID:
                    trial = {k: dict(v) for k, v in thresholds.items()}
                    trial[cname] = {"high": high, "low": round(high * ratio, 3)}
                    ier = score(cached, reference, fps, trial, median_seconds)
                    if ier < best:
                        best, thresholds = ier, trial
        for med in MEDIAN_GRID:
            ier = score(cached, reference, fps, thresholds, med)
            if ier < best:
                best, median_seconds = ier, med
        print(f"Round {round_i + 1}: IER {best:.4f}  median {median_seconds}s")
        for cname, th in thresholds.items():
            print(f"  {cname:20s} high {th['high']:.2f}  low {th['low']:.2f}")

    print(f"\nBest IER on valset: {best:.4f}")
    if args.write:
        cfg.posteriors.median_filter_seconds = median_seconds
        for cname, th in thresholds.items():
            cfg.thresholds[cname].high = float(th["high"])
            cfg.thresholds[cname].low = float(th["low"])
        OmegaConf.save(cfg, args.config)
        print(f"Wrote tuned values to {args.config}")
    else:
        print("(dry run — pass --write to update configs/inference.yaml)")


if __name__ == "__main__":
    main()
