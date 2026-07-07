"""Identification Error Rate — the exact competition metric.

IER = (0.25 * false_alarm + 1.0 * missed + 0.75 * confusion) / total_ref_duration
computed with pyannote-metrics, collar=0, skip_overlap=True. Note the
asymmetry: a missed event costs 4x a false alarm, which is why the inference
thresholds are recall-biased.
"""

from __future__ import annotations

from pyannote.core import Annotation, Segment
from pyannote.metrics.identification import IdentificationErrorRate

from echosentinel.constants import CLASS_MAP

WEIGHT_FALSE_ALARM = 0.25
WEIGHT_MISS = 1.0
WEIGHT_CONFUSION = 0.75


def to_annotation(events: list[tuple[int, float, float]], uri: str) -> Annotation:
    """events: (category_id, start_s, end_s) -> pyannote Annotation."""
    ann = Annotation(uri=uri)
    for cid, start, end in events:
        if end > start:
            ann[Segment(start, end)] = CLASS_MAP[cid]
    return ann


class IERScorer:
    """Accumulates IER over many files; report() gives the aggregate."""

    def __init__(self) -> None:
        self.metric = IdentificationErrorRate(
            confusion=WEIGHT_CONFUSION,
            miss=WEIGHT_MISS,
            false_alarm=WEIGHT_FALSE_ALARM,
            collar=0.0,
            skip_overlap=True,
        )

    def add_file(
        self,
        reference: list[tuple[int, float, float]],
        hypothesis: list[tuple[int, float, float]],
        uri: str,
    ) -> float:
        ref = to_annotation(reference, uri)
        hyp = to_annotation(hypothesis, uri)
        return float(self.metric(ref, hyp))

    def report(self) -> dict[str, float]:
        detail = self.metric[:]  # accumulated components
        total = detail["total"] or 1.0
        return {
            "ier": abs(self.metric),
            "missed_detection_rate": detail["missed detection"] / total,
            "false_alarm_rate": detail["false alarm"] / total,
            "confusion_rate": detail["confusion"] / total,
            "total_ref_seconds": detail["total"],
        }
