"""Score a predictions JSON against a ground-truth JSON with the exact
competition metric (IER, pyannote-metrics).

Usage (from echosentinel_v2/):
    python scripts/05_evaluate_ier.py --predictions out/results.json \
                                      --ground-truth out/synth_valset/ground_truth.json
"""

from __future__ import annotations

import argparse

from echosentinel.eval.ier import IERScorer
from echosentinel.infer.json_writer import read_results_json


def events_by_file(results: dict) -> dict[str, list[tuple[int, float, float]]]:
    id_to_name = {a["id"]: a["file_name"] for a in results["audios"]}
    out: dict[str, list[tuple[int, float, float]]] = {n: [] for n in id_to_name.values()}
    for ann in results["annotations"]:
        out[id_to_name[ann["audio_id"]]].append(
            (ann["category_id"], float(ann["start_time"]), float(ann["end_time"]))
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground-truth", required=True)
    args = parser.parse_args()

    ref = events_by_file(read_results_json(args.ground_truth))
    hyp = events_by_file(read_results_json(args.predictions))

    scorer = IERScorer()
    for name, ref_events in sorted(ref.items()):
        file_ier = scorer.add_file(ref_events, hyp.get(name, []), uri=name)
        print(f"  {name}: IER {file_ier:.3f}  (ref {len(ref_events)} ev, hyp {len(hyp.get(name, []))} ev)")

    rep = scorer.report()
    print(
        f"\nAggregate IER: {rep['ier']:.4f}"
        f"\n  missed:    {rep['missed_detection_rate']:.4f} (weight 1.00)"
        f"\n  falsealarm:{rep['false_alarm_rate']:.4f} (weight 0.25)"
        f"\n  confusion: {rep['confusion_rate']:.4f} (weight 0.75)"
        f"\n  ref audio: {rep['total_ref_seconds']:.0f}s of labeled events"
    )


if __name__ == "__main__":
    main()
