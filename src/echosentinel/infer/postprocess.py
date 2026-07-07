"""Posteriors -> competition-valid events.

Chain: per-class median filter -> hysteresis (onset/sustain) thresholds ->
merge short same-class gaps -> drop too-short events -> greedy non-overlap
resolution -> round timestamps to whole seconds.

Thresholds are recall-biased: in the IER a missed event costs 4x a false
alarm, so the sustain threshold in particular is kept low. The non-overlap
step is a hard requirement — PS-12 rejects submissions with overlapping
annotations — and is enforced last, on the rounded integer timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import median_filter as _median

from echosentinel.constants import CLASS_MAP


@dataclass(frozen=True)
class Event:
    category_id: int
    start: float
    end: float
    score: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def median_smooth(probs: np.ndarray, fps: float, seconds: float) -> np.ndarray:
    k = max(int(round(seconds * fps)) | 1, 1)  # odd kernel
    return _median(probs, size=(k, 1), mode="nearest")


def hysteresis_segments(
    p: np.ndarray, fps: float, high: float, low: float
) -> list[tuple[float, float, float]]:
    """(start_s, end_s, mean_prob) spans that rise above ``high`` and persist
    until falling below ``low``."""
    segments = []
    active = False
    start = 0
    for i, v in enumerate(p):
        if not active and v >= high:
            active = True
            start = i
            # extend backwards while above the sustain threshold
            while start > 0 and p[start - 1] >= low:
                start -= 1
        elif active and v < low:
            segments.append((start / fps, i / fps, float(p[start:i].mean())))
            active = False
    if active:
        segments.append((start / fps, len(p) / fps, float(p[start:].mean())))
    return segments


def merge_gaps(
    spans: list[tuple[float, float, float]], max_gap: float
) -> list[tuple[float, float, float]]:
    if not spans:
        return spans
    merged = [spans[0]]
    for s, e, sc in spans[1:]:
        ps, pe, psc = merged[-1]
        if s - pe <= max_gap:
            w1, w2 = pe - ps, e - s
            merged[-1] = (ps, e, (psc * w1 + sc * w2) / max(w1 + w2, 1e-9))
        else:
            merged.append((s, e, sc))
    return merged


def resolve_overlaps(events: list[Event], min_duration: float) -> list[Event]:
    """Greedy by score: keep the strongest events, trim weaker ones to the
    free time around them, drop leftovers shorter than ``min_duration``."""
    kept: list[Event] = []
    for ev in sorted(events, key=lambda e: e.score, reverse=True):
        free = [(ev.start, ev.end)]
        for k in kept:
            free = [
                piece
                for lo, hi in free
                for piece in ((lo, min(hi, k.start)), (max(lo, k.end), hi))
                if piece[1] - piece[0] > 0
            ]
        if not free:
            continue
        lo, hi = max(free, key=lambda x: x[1] - x[0])
        if hi - lo >= min_duration:
            kept.append(replace(ev, start=lo, end=hi))
    return sorted(kept, key=lambda e: e.start)


def probs_to_events(
    probs: np.ndarray,
    fps: float,
    thresholds: dict[str, dict[str, float]],
    median_seconds: float = 0.7,
    merge_gap_seconds: float = 1.0,
    min_duration_seconds: float = 1.0,
    round_to_seconds: bool = True,
) -> list[Event]:
    """(frames, classes) posteriors -> sorted, non-overlapping events."""
    smoothed = median_smooth(probs, fps, median_seconds)
    events: list[Event] = []
    for cid, cname in CLASS_MAP.items():
        th = thresholds[cname]
        spans = hysteresis_segments(smoothed[:, cid - 1], fps, th["high"], th["low"])
        spans = merge_gaps(spans, merge_gap_seconds)
        events += [
            Event(cid, s, e, sc)
            for s, e, sc in spans
            if e - s >= min_duration_seconds
        ]

    if round_to_seconds:
        events = [
            replace(ev, start=float(round(ev.start)), end=float(round(ev.end)))
            for ev in events
        ]
        events = [ev for ev in events if ev.end > ev.start]

    return resolve_overlaps(events, min_duration=1.0 if round_to_seconds else min_duration_seconds)
