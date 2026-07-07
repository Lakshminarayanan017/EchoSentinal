"""The output JSON must match the PS-12 Appendix-B structure exactly."""

from echosentinel.infer.json_writer import build_results_json
from echosentinel.infer.postprocess import Event


def _sample() -> dict:
    events = [Event(2, 3.0, 8.0, 0.87), Event(1, 12.0, 20.0, 0.71)]
    return build_results_json([("S2_Test_001.wav", 300.0, events)])


def test_top_level_keys():
    r = _sample()
    assert set(r) == {"info", "audios", "categories", "annotations"}


def test_categories_are_the_ps12_class_map():
    r = _sample()
    assert r["categories"] == [
        {"id": 1, "name": "vessel"},
        {"id": 2, "name": "marine_animal"},
        {"id": 3, "name": "natural_sound"},
        {"id": 4, "name": "other_anthropogenic"},
    ]


def test_annotation_schema_and_integer_seconds():
    r = _sample()
    ann = r["annotations"][0]
    assert set(ann) == {
        "id", "audio_id", "category_id", "start_time", "end_time", "duration", "score",
    }
    for a in r["annotations"]:
        assert isinstance(a["start_time"], int)
        assert isinstance(a["end_time"], int)
        assert a["duration"] == a["end_time"] - a["start_time"]
        assert 0.0 <= a["score"] <= 1.0
    assert [a["id"] for a in r["annotations"]] == [1, 2]


def test_audio_ids_link_annotations():
    r = _sample()
    audio_ids = {a["id"] for a in r["audios"]}
    assert all(a["audio_id"] in audio_ids for a in r["annotations"])
