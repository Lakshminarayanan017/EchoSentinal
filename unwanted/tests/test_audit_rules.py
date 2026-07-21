"""Unit tests for the relabel proposal rules and source grouping."""

from echosentinel.constants import CLASS_MAP, NAME_TO_ID
from echosentinel.data.audit import propose_class, source_group


def test_class_map_matches_ps12_spec():
    assert CLASS_MAP == {
        1: "vessel",
        2: "marine_animal",
        3: "natural_sound",
        4: "other_anthropogenic",
    }
    assert NAME_TO_ID["vessel"] == 1


def test_rain_in_human_made_folder_goes_to_natural():
    cls, conf, _ = propose_class("review", "217235__roofusj__rain-01.wav")
    assert cls == "natural_sound"
    assert conf == "high"


def test_earthquake_goes_to_natural():
    cls, _, _ = propose_class("review", "148002__loafdv__large-earthquake.wav")
    assert cls == "natural_sound"


def test_metal_impact_flagged_as_anthropogenic():
    cls, conf, _ = propose_class("review", "wood_metal_impact_03.wav")
    assert cls == "other_anthropogenic"
    assert conf in ("medium", "low")


def test_unknown_name_in_review_folder_is_low_confidence():
    _, conf, _ = propose_class("review", "103.wav")
    assert conf == "low"


def test_folder_label_wins_for_plain_folders():
    cls, conf, _ = propose_class("vessel", "40.wav")
    assert cls == "vessel"
    assert conf == "high"


def test_strays_in_downloaded_class4_folder_are_rerouted():
    cls, _, _ = propose_class(
        "other_anthropogenic", "20210205-sanctsound-ci01-03-dolphins-20190904t064203z.wav"
    )
    assert cls == "marine_animal"
    cls, _, _ = propose_class("other_anthropogenic", "Vess-05-large-vessel-clip.mp3")
    assert cls == "vessel"


def test_anthropogenic_keyword_beats_vessel_keyword():
    # pile driving recorded at a ferry terminal is class 4, not a vessel
    cls, _, _ = propose_class(
        "other_anthropogenic",
        "Pile-driving-high-level-MarthasVineyardFerryTerminal-NOAA-PAGroup-02.mp3",
    )
    assert cls == "other_anthropogenic"


def test_keyword_needs_token_start_boundary():
    # 'pile' must not fire inside 'Compiled', 'ping' not inside 'snapping'
    cls, _, _ = propose_class(
        "natural_sound", "Sdsc01-HourlySancSdCompiled-GRNMS-NOAA-soundscape-clip.mp3"
    )
    assert cls == "natural_sound"
    cls, _, _ = propose_class(
        "other_anthropogenic", "20210205-sanctsound-ci01-01-snappingshrimp-20181101.wav"
    )
    assert cls == "marine_animal"  # snapping shrimp is a marine animal


def test_uploader_name_does_not_trigger_keyword():
    # 'waveadventurer' is the Freesound uploader, not a wave sound; an
    # anchor-chain clip must stay other_anthropogenic
    cls, _, _ = propose_class("other_anthropogenic", "251555__waveadventurer__r05_0247.wav")
    assert cls == "other_anthropogenic"


def test_earthquake_misfiled_under_marine_is_relabeled():
    cls, conf, _ = propose_class(
        "marine_animal", "369485__mbari_mars__earthquake-audible-only-with-appropriate-speakers.wav"
    )
    assert cls == "natural_sound"
    assert conf == "medium"


def test_freesound_uploader_grouping():
    g = source_group("343682__mbari_mars__blue-whale-b-call-5x.wav", "Marine Animals")
    assert g == "freesound:mbari_mars"
    g2 = source_group("339343__mbari_mars__dolphins.wav", "Marine Animals")
    assert g2 == g  # same uploader -> same group -> never split across train/val


def test_numeric_names_group_per_file():
    assert source_group("40.wav", "Tanker ds") == "Tanker ds:40"
    assert source_group("41.wav", "Tanker ds") == "Tanker ds:41"
