"""Dataset audit: propose a PS-12 class for every clean training file.

The original folder labels are unreliable — most of "Human made Objects" is
rain/earthquake audio that belongs to ``natural_sound``. This module scans the
source folders, probes each file's header, and emits a review CSV with a
proposed class, a confidence, and the reason. A human confirms the CSV
(editing the ``final_class`` column where needed) before manifests are built;
the audit never relabels silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

from echosentinel.audio.io import probe

# Filename keywords that mark a recording's true class regardless of which
# folder it sits in (downloaded clips often land in a catch-all folder).
NATURAL_KEYWORDS = (
    "rain", "earthquake", "quake", "tremor", "volcano", "lava", "geophone",
    "rockslide", "landslide", "stonefall", "rockfall", "thunder", "storm",
    "wave", "wind", "ice", "iceberg", "hail",
)

# Keywords suggesting genuine non-vessel human-made sound (candidate class 4).
ANTHROPOGENIC_KEYWORDS = (
    "sonar", "ping", "echosounder", "airgun", "seismic", "seis-", "pile",
    "driving", "drill", "explosion", "blast", "chain", "anchor", "metal",
    "wood", "impact", "grind", "hammer", "machin", "scuba", "diver",
)

MARINE_KEYWORDS = (
    "whale", "hwsong", "dolphin", "porpoise", "shrimp", "seal", "sealion",
    "orca", "fish", "snapping", "clicks", "whistle",
)

VESSEL_KEYWORDS = (
    "vessel", "vess-", "ship", "boat", "tanker", "cargo", "tug", "propeller",
    "engine",
)

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg")


@dataclass
class AuditRow:
    path: str
    orig_folder: str
    proposed_class: str
    final_class: str  # starts equal to proposed_class; human edits this column
    confidence: str   # high | medium | low
    reason: str
    sr: int
    channels: int
    duration_s: float
    subtype: str
    error: str


def _keyword_hit(name: str, keywords: tuple[str, ...]) -> str | None:
    """Match a keyword at a token start only, so 'pile' does not fire inside
    'Compiled' nor 'ping' inside 'snapping'. No end boundary: filenames glue
    words together ('snappingshrimp' should still match 'snapping')."""
    lowered = name.lower()
    for kw in keywords:
        if re.search(r"(?<![a-z])" + re.escape(kw), lowered):
            return kw
    return None


def _keyword_class(filename: str) -> tuple[str, str] | None:
    """Best class suggested by filename keywords, or None.

    Priority order matters: anthropogenic terms win over vessel terms so that
    e.g. "Pile-driving-...-FerryTerminal" is not misread as a vessel, and
    natural wins over marine so a "thunderstorm" clip mentioning fish stays
    natural_sound.
    """
    for keywords, cls in (
        (ANTHROPOGENIC_KEYWORDS, "other_anthropogenic"),
        (NATURAL_KEYWORDS, "natural_sound"),
        (MARINE_KEYWORDS, "marine_animal"),
        (VESSEL_KEYWORDS, "vessel"),
    ):
        kw = _keyword_hit(filename, keywords)
        if kw:
            return cls, kw
    return None


def propose_class(folder_default: str, filename: str) -> tuple[str, str, str]:
    """Return (proposed_class, confidence, reason) for one file.

    ``folder_default`` is the class from configs/data.yaml, or "review" for
    folders whose contents need per-file keyword rules.
    """
    hit = _keyword_class(filename)

    if folder_default == "review":
        if hit:
            cls, kw = hit
            conf = "high" if cls == "natural_sound" else "medium"
            return cls, conf, f"filename keyword '{kw}'"
        return "other_anthropogenic", "low", "no keyword match — review by listening"

    if hit and hit[0] != folder_default:
        cls, kw = hit
        # Downloaded catch-all folders hold obvious strays (dolphins in the
        # anthropogenic folder); folder-curated sets get a verify flag instead.
        if folder_default == "other_anthropogenic":
            return cls, "medium", f"folder says {folder_default} but filename keyword '{kw}'"
        return cls, "medium", f"relabeled from {folder_default} folder: filename keyword '{kw}' — verify"

    return folder_default, "high", "folder label"


# Freesound-style names look like "343682__mbari_mars__blue-whale-....wav";
# the middle token is the uploader and makes a good leak-free grouping key.
_FREESOUND_RE = re.compile(r"^\d+__([^_].*?)__")


def source_group(filename: str, folder: str) -> str:
    m = _FREESOUND_RE.match(filename)
    if m:
        return f"freesound:{m.group(1)}"
    # Numeric/opaque names: treat each file as its own recording source but
    # keep the folder prefix so vessel types remain distinguishable.
    return f"{folder}:{Path(filename).stem}"


def audit_folders(dataset_root: Path, source_folders: dict[str, str]) -> pd.DataFrame:
    """Scan every configured source folder and build the review table."""
    rows: list[AuditRow] = []
    for folder, default_class in source_folders.items():
        folder_path = dataset_root / folder
        if not folder_path.is_dir():
            continue  # folder not downloaded/created yet
        files = sorted(
            p for p in folder_path.rglob("*")
            if p.suffix.lower() in AUDIO_EXTENSIONS
        )
        for wav in files:
            proposed, confidence, reason = propose_class(default_class, wav.name)
            sr = channels = 0
            duration = 0.0
            subtype = ""
            error = ""
            try:
                info = probe(wav)
                sr, channels = info.sr, info.channels
                duration, subtype = round(info.duration, 2), info.subtype
            except Exception as e:  # keep the row; a broken header is itself a finding
                error = repr(e)
                confidence = "low"
                reason += " | header unreadable"
            rows.append(
                AuditRow(
                    path=str(wav.relative_to(dataset_root)),
                    orig_folder=folder,
                    proposed_class=proposed,
                    final_class=proposed,
                    confidence=confidence,
                    reason=reason,
                    sr=sr,
                    channels=channels,
                    duration_s=duration,
                    subtype=subtype,
                    error=error,
                )
            )
    return pd.DataFrame([asdict(r) for r in rows])
