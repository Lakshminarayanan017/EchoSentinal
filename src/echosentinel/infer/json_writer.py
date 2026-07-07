"""Competition JSON output (PS-12 Appendix-B format).

Top-level keys: info, audios, categories, annotations. Annotation timestamps
are whole seconds; ``duration`` is end - start.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from echosentinel.constants import CLASS_MAP
from echosentinel.infer.postprocess import Event


def build_results_json(
    per_file_events: list[tuple[str, float, list[Event]]],
    contributor: str = "ECHOSENTINEL",
) -> dict:
    """``per_file_events``: list of (file_name, duration_s, events)."""
    audios = []
    annotations = []
    ann_id = 1
    for audio_id, (file_name, duration_s, events) in enumerate(per_file_events, start=1):
        audios.append(
            {"id": audio_id, "file_name": file_name, "duration": round(duration_s, 2)}
        )
        for ev in events:
            annotations.append(
                {
                    "id": ann_id,
                    "audio_id": audio_id,
                    "category_id": ev.category_id,
                    "start_time": int(ev.start),
                    "end_time": int(ev.end),
                    "duration": int(ev.end - ev.start),
                    "score": round(float(ev.score), 4),
                }
            )
            ann_id += 1

    return {
        "info": {
            "description": "PS-12 underwater sound event detections",
            "contributor": contributor,
            "version": "2.0",
            "date_created": date.today().isoformat(),
        },
        "audios": audios,
        "categories": [{"id": cid, "name": name} for cid, name in CLASS_MAP.items()],
        "annotations": annotations,
    }


def write_results_json(path: str | Path, results: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def read_results_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
