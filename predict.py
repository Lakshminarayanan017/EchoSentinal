"""echoSentinel v2 — competition entrypoint (also the Docker entrypoint).

    python predict.py --input_dir <folder of .wav> --output_json <results.json>
                      [--weights weights/crnn_baseline.pt]
                      [--config configs/inference.yaml]

Fully offline: loads baked weights, streams each file in blocks, writes one
combined PS-12 JSON.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from echosentinel.audio.io import probe
from echosentinel.infer.json_writer import build_results_json, write_results_json
from echosentinel.infer.posteriors import file_posteriors
from echosentinel.infer.postprocess import probs_to_events
from echosentinel.models.registry import build_model, model_frames_per_second

PROJECT_ROOT = Path(__file__).resolve().parent


def load_model(weights_path: Path, device: str) -> tuple[torch.nn.Module, float]:
    ckpt = torch.load(weights_path, map_location=device, weights_only=True)
    model = build_model(ckpt["model_name"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, model_frames_per_second(ckpt["model_name"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--weights", default=str(PROJECT_ROOT / "weights" / "crnn_baseline.pt"))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "inference.yaml"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, fps = load_model(Path(args.weights), device)

    def posterior_fn(wave: torch.Tensor) -> torch.Tensor:
        return model.posteriors(wave.to(device))

    thresholds = OmegaConf.to_container(cfg.thresholds)
    wavs = sorted(Path(args.input_dir).glob("*.wav"))
    if not wavs:
        raise SystemExit(f"No .wav files in {args.input_dir}")

    per_file = []
    for wav in tqdm(wavs, desc="predict"):
        probs = file_posteriors(
            wav, posterior_fn, fps,
            block_seconds=float(cfg.block_seconds),
            overlap_seconds=float(cfg.block_overlap_seconds),
        )
        events = probs_to_events(
            probs, fps, thresholds,
            median_seconds=float(cfg.posteriors.median_filter_seconds),
            merge_gap_seconds=float(cfg.events.merge_gap_seconds),
            min_duration_seconds=float(cfg.events.min_duration_seconds),
            round_to_seconds=bool(cfg.events.round_to_seconds),
        )
        per_file.append((wav.name, probe(wav).duration, events))

    results = build_results_json(per_file, contributor=str(cfg.json.startup_name))
    write_results_json(args.output_json, results)
    n_events = len(results["annotations"])
    print(f"Wrote {args.output_json}: {len(per_file)} audio files, {n_events} events")


if __name__ == "__main__":
    main()
