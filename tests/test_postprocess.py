"""Post-processing invariants: hysteresis behavior and the hard guarantee
that emitted events never overlap (PS-12 rejects overlapping annotations)."""

import numpy as np
import pytest

from echosentinel.infer.postprocess import (
    Event,
    hysteresis_segments,
    merge_gaps,
    probs_to_events,
    resolve_overlaps,
)

FPS = 25.0
THRESHOLDS = {
    "vessel": {"high": 0.4, "low": 0.2},
    "marine_animal": {"high": 0.35, "low": 0.15},
    "natural_sound": {"high": 0.4, "low": 0.2},
    "other_anthropogenic": {"high": 0.3, "low": 0.15},
}


def test_hysteresis_extends_onset_backwards():
    p = np.array([0.0, 0.25, 0.3, 0.9, 0.9, 0.25, 0.1, 0.0])
    segs = hysteresis_segments(p, fps=1.0, high=0.5, low=0.2)
    assert len(segs) == 1
    start, end, _ = segs[0]
    assert start == 1.0  # walked back over frames above `low`
    assert end == 6.0    # sustained until dropping below `low`


def test_hysteresis_ignores_subthreshold_bump():
    p = np.array([0.0, 0.3, 0.3, 0.0])  # never reaches high
    assert hysteresis_segments(p, fps=1.0, high=0.5, low=0.2) == []


def test_merge_gaps_joins_close_spans():
    spans = [(0.0, 2.0, 0.8), (2.5, 4.0, 0.6), (10.0, 12.0, 0.9)]
    merged = merge_gaps(spans, max_gap=1.0)
    assert len(merged) == 2
    assert merged[0][:2] == (0.0, 4.0)


def test_resolve_overlaps_keeps_higher_score_and_trims():
    events = [
        Event(1, 0.0, 10.0, 0.9),
        Event(2, 5.0, 15.0, 0.6),  # overlaps -> trimmed to [10, 15]
    ]
    out = resolve_overlaps(events, min_duration=1.0)
    assert [(e.category_id, e.start, e.end) for e in out] == [
        (1, 0.0, 10.0),
        (2, 10.0, 15.0),
    ]


def _assert_no_overlap(events: list[Event]) -> None:
    for a, b in zip(events, events[1:]):
        assert a.end <= b.start, f"overlap: {a} vs {b}"


@pytest.mark.parametrize("seed", range(20))
def test_probs_to_events_never_overlaps(seed: int):
    """Property test: any posterior matrix yields non-overlapping integer events."""
    rng = np.random.default_rng(seed)
    frames = int(60 * FPS)
    # correlated random walk per class to create realistic event-ish blobs
    probs = np.clip(
        np.cumsum(rng.normal(0, 0.05, size=(frames, 4)), axis=0) + 0.3, 0, 1
    )
    events = probs_to_events(probs, FPS, THRESHOLDS)
    _assert_no_overlap(events)
    for ev in events:
        assert ev.start == int(ev.start) and ev.end == int(ev.end)
        assert ev.end - ev.start >= 1.0


def test_probs_to_events_finds_a_clear_event():
    frames = int(30 * FPS)
    probs = np.full((frames, 4), 0.05)
    probs[int(10 * FPS) : int(15 * FPS), 1] = 0.95  # marine event at 10-15 s
    events = probs_to_events(probs, FPS, THRESHOLDS)
    assert len(events) == 1
    ev = events[0]
    assert ev.category_id == 2
    assert abs(ev.start - 10) <= 1 and abs(ev.end - 15) <= 1
