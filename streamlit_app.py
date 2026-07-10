"""echoSentinel — Streamlit console.

Single-file Streamlit front end for the PS-12 underwater sound-event detector.
It reuses the exact inference pipeline the FastAPI console used (audio IO ->
PCEN/CNN14 posteriors -> hysteresis post-processing -> PS-12 JSON); only the
serving/UI layer is different.

Run locally:      streamlit run streamlit_app.py
Deploy target:    Streamlit Community Cloud (main file = streamlit_app.py)

Model weights (~292 MB) are not in git; they are fetched from Google Drive on
first run and cached for the life of the container.
"""

from __future__ import annotations

import os

# Torch trips Streamlit's module watcher (torch.classes __path__); disable it
# before Streamlit is imported. .streamlit/config.toml sets this too — belt and
# suspenders so a bare `streamlit run` also works.
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import json
import sys
import tempfile
import time
import uuid
from pathlib import Path

import streamlit as st

# Make the src/ package importable without an install step (Community Cloud only
# installs requirements.txt; it does not `pip install .`).
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from echosentinel import __version__
from echosentinel.audio.io import probe
from echosentinel.constants import CLASS_MAP
from echosentinel.infer.json_writer import build_results_json
from echosentinel.infer.posteriors import file_posteriors
from echosentinel.infer.postprocess import probs_to_events
from echosentinel.models.registry import build_model, model_frames_per_second
from echosentinel.server.media import spectrogram_png, waveform_peaks

# ----------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------

WEIGHTS_PATH = ROOT / "weights" / "panns_pcen.pt"
INFERENCE_CFG = ROOT / "configs" / "inference.yaml"
# Same Drive file the Docker console pulled (README: weights not in git).
WEIGHTS_DRIVE_ID = "19IU8-RbiKg4C-yBqHY_wGhA-UVn_Pm7B"

ALLOWED_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg"}

CLASS_COLORS = {
    "vessel": "#00f0ff",
    "marine_animal": "#34f5c5",
    "natural_sound": "#38bdf8",
    "other_anthropogenic": "#ffb454",
}
CLASS_LABEL = {
    "vessel": "Vessel",
    "marine_animal": "Marine animal",
    "natural_sound": "Natural sound",
    "other_anthropogenic": "Other anthropogenic",
}

st.set_page_config(
    page_title="echoSentinel",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# cached heavy resources (weights download, model, config)
# ----------------------------------------------------------------------------


@st.cache_resource(show_spinner="Fetching model weights (~292 MB, first run only)…")
def ensure_weights() -> str:
    if not WEIGHTS_PATH.exists():
        WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        import gdown

        gdown.download(
            id=WEIGHTS_DRIVE_ID, output=str(WEIGHTS_PATH), quiet=False, fuzzy=True
        )
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            "Model weights could not be downloaded. Place panns_pcen.pt in weights/."
        )
    return str(WEIGHTS_PATH)


@st.cache_resource(show_spinner="Loading detection model…")
def load_model():
    ckpt = torch.load(ensure_weights(), map_location="cpu", weights_only=True)
    model = build_model(ckpt["model_name"], **ckpt.get("model_kwargs", {}))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    fps = model_frames_per_second(ckpt["model_name"])
    meta = {
        "model_name": ckpt["model_name"],
        "frontend": (ckpt.get("model_kwargs", {}) or {}).get("frontend", "pcen"),
        "epoch": int(ckpt.get("epoch", -1)),
        "val_f1_macro": round(float(ckpt.get("f1_macro", 0.0)), 4),
        "weights_file": WEIGHTS_PATH.name,
        "params_million": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
    }
    return model, fps, meta


@st.cache_resource
def load_cfg():
    return OmegaConf.load(INFERENCE_CFG)


# ----------------------------------------------------------------------------
# inference
# ----------------------------------------------------------------------------


def thresholds_for(cfg, sensitivity: dict[str, float]) -> dict[str, dict[str, float]]:
    """Scale calibrated thresholds by a per-class sensitivity multiplier.

    sensitivity > 1  -> more sensitive (lower thresholds, more detections)
    sensitivity < 1  -> stricter (higher thresholds, fewer detections)
    """
    base = OmegaConf.to_container(cfg.thresholds)
    out: dict[str, dict[str, float]] = {}
    for cname, th in base.items():
        m = 1.0 / max(float(sensitivity.get(cname, 1.0)), 1e-3)
        out[cname] = {
            "high": min(max(th["high"] * m, 0.01), 0.99),
            "low": min(max(th["low"] * m, 0.005), 0.95),
        }
    return out


def analyze(audio_path: Path, original_name: str, sensitivity: dict, cfg, model, fps,
            progress_cb) -> dict:
    info = probe(audio_path)

    probs = file_posteriors(
        audio_path,
        lambda w: model.posteriors(w),
        fps,
        block_seconds=float(cfg.block_seconds),
        overlap_seconds=float(cfg.block_overlap_seconds),
        progress=progress_cb,
    )
    events = probs_to_events(
        probs,
        fps,
        thresholds_for(cfg, sensitivity),
        median_seconds=float(cfg.posteriors.median_filter_seconds),
        merge_gap_seconds=float(cfg.events.merge_gap_seconds),
        min_duration_seconds=float(cfg.events.min_duration_seconds),
        round_to_seconds=bool(cfg.events.round_to_seconds),
    )
    results = build_results_json(
        [(original_name, info.duration, events)],
        contributor=str(cfg.json.startup_name),
    )

    counts: dict[str, int] = {}
    for ev in events:
        cname = CLASS_MAP[ev.category_id]
        counts[cname] = counts.get(cname, 0) + 1

    # visual assets (rendered from the real audio, same helpers as the old console)
    job_dir = Path(tempfile.mkdtemp(prefix="echosentinel_"))
    spec_path = job_dir / "spectrogram.png"
    try:
        spectrogram_png(audio_path, spec_path)
    except Exception:
        spec_path = None
    try:
        peaks = waveform_peaks(audio_path)
    except Exception:
        peaks = None

    return {
        "id": uuid.uuid4().hex[:8],
        "name": original_name,
        "created": time.time(),
        "duration": round(info.duration, 2),
        "sample_rate": info.sr,
        "n_events": len(events),
        "class_counts": counts,
        "events": [
            {
                "class": CLASS_MAP[ev.category_id],
                "category_id": ev.category_id,
                "start": ev.start,
                "end": ev.end,
                "score": round(float(ev.score), 4),
            }
            for ev in events
        ],
        "results": results,
        "audio_path": str(audio_path),
        "spectrogram_path": str(spec_path) if spec_path else None,
        "peaks": peaks,
    }


# ----------------------------------------------------------------------------
# formatting helpers
# ----------------------------------------------------------------------------


def fmt_time(s: float) -> str:
    s = max(0, int(round(s)))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def class_badges(counts: dict[str, int]) -> str:
    if not counts:
        return "<span style='opacity:.6'>no events detected</span>"
    chips = []
    for cname, n in counts.items():
        color = CLASS_COLORS.get(cname, "#888")
        chips.append(
            f"<span style='display:inline-block;margin:2px 6px 2px 0;padding:3px 10px;"
            f"border:1px solid {color};border-radius:20px;color:{color};font-size:12px;"
            f"font-family:monospace'>{CLASS_LABEL.get(cname, cname)} · {n}</span>"
        )
    return "".join(chips)


def timeline_chart(entry: dict):
    """Horizontal event timeline (Gantt-style) over the recording duration."""
    import altair as alt

    if not entry["events"]:
        return None
    df = pd.DataFrame(entry["events"])
    df["label"] = df["class"].map(CLASS_LABEL)
    order = [CLASS_LABEL[c] for c in CLASS_COLORS if CLASS_LABEL[c] in set(df["label"])]
    chart = (
        alt.Chart(df)
        .mark_bar(height=16, cornerRadius=3)
        .encode(
            x=alt.X("start:Q", title="time (s)",
                    scale=alt.Scale(domain=[0, entry["duration"]])),
            x2="end:Q",
            y=alt.Y("label:N", title=None, sort=order),
            color=alt.Color(
                "class:N",
                scale=alt.Scale(
                    domain=list(CLASS_COLORS.keys()),
                    range=list(CLASS_COLORS.values()),
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("label:N", title="class"),
                alt.Tooltip("start:Q", title="start (s)"),
                alt.Tooltip("end:Q", title="end (s)"),
                alt.Tooltip("score:Q", title="score", format=".3f"),
            ],
        )
        .properties(height=max(120, 34 * len(order)))
    )
    return chart


def waveform_chart(entry: dict):
    import altair as alt

    peaks = entry.get("peaks")
    if not peaks:
        return None
    cols = peaks["columns"]
    dur = peaks["duration"]
    t = np.linspace(0, dur, cols)
    df = pd.DataFrame({"t": t, "hi": peaks["max"], "lo": peaks["min"]})
    return (
        alt.Chart(df)
        .mark_area(opacity=0.85, color="#00f0ff")
        .encode(
            x=alt.X("t:Q", title="time (s)", scale=alt.Scale(domain=[0, dur])),
            y=alt.Y("lo:Q", title=None),
            y2="hi:Q",
        )
        .properties(height=90)
    )


# ----------------------------------------------------------------------------
# result rendering
# ----------------------------------------------------------------------------


def render_result(entry: dict) -> None:
    st.markdown(f"#### {entry['name']}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration", fmt_time(entry["duration"]))
    c2.metric("Sample rate", f"{entry['sample_rate'] / 1000:g} kHz" if entry["sample_rate"] else "—")
    c3.metric("Events", entry["n_events"])
    c4.metric("Classes present", len(entry["class_counts"]))

    st.markdown(class_badges(entry["class_counts"]), unsafe_allow_html=True)

    if entry.get("audio_path") and Path(entry["audio_path"]).exists():
        st.audio(entry["audio_path"])

    wf = waveform_chart(entry)
    if wf is not None:
        st.caption("Waveform")
        st.altair_chart(wf, use_container_width=True)

    if entry.get("spectrogram_path") and Path(entry["spectrogram_path"]).exists():
        st.caption("Log-mel spectrogram")
        st.image(entry["spectrogram_path"], use_container_width=True)

    tl = timeline_chart(entry)
    if tl is not None:
        st.caption("Detected-event timeline")
        st.altair_chart(tl, use_container_width=True)
    else:
        st.info("No events crossed the detection thresholds for this recording.")

    if entry["events"]:
        st.caption("Events")
        table = pd.DataFrame(entry["events"])[["class", "start", "end", "score"]].copy()
        table["class"] = table["class"].map(CLASS_LABEL)
        table = table.rename(columns={"start": "start (s)", "end": "end (s)"})
        st.dataframe(table, use_container_width=True, hide_index=True)

    stem = Path(entry["name"]).stem
    st.download_button(
        "⬇ Export PS-12 JSON",
        data=json.dumps(entry["results"], indent=2),
        file_name=f"{stem}_events.json",
        mime="application/json",
        key=f"dl_{entry['id']}",
    )


# ----------------------------------------------------------------------------
# app
# ----------------------------------------------------------------------------


def main() -> None:
    if "archive" not in st.session_state:
        st.session_state.archive = []  # most-recent-first

    cfg = load_cfg()

    st.title("🔊 echoSentinel")
    st.caption(
        "Underwater sound-event detection & classification — PS-12 · "
        "vessel / marine animal / natural / anthropogenic. Runs on CPU, fully offline."
    )

    # --- sidebar: model status + sensitivity ---
    with st.sidebar:
        st.subheader("Detection engine")
        try:
            model, fps, meta = load_model()
            st.success(f"{meta['model_name'].upper()} online")
            st.markdown(
                f"""
- **Architecture** PANNs CNN14 + SED head
- **Front-end** {meta['frontend'].upper()} (adaptive gain)
- **Checkpoint** {meta['weights_file']} · epoch {meta['epoch']}
- **Val F1 (macro)** {meta['val_f1_macro']}
- **Params** {meta['params_million']} M · CPU
- **Version** v{__version__}
"""
            )
        except Exception as e:  # weights missing / download failed
            model = fps = meta = None
            st.error(f"Model unavailable: {e}")

        st.divider()
        st.subheader("Class sensitivity")
        st.caption(
            "Thresholds are recall-biased (a missed contact costs 4× a false alarm). "
            "Raise a class to catch fainter events at the cost of more false alarms."
        )
        if st.button("Reset to calibrated", use_container_width=True):
            for cname in CLASS_COLORS:
                st.session_state[f"sens_{cname}"] = 1.0

        sensitivity: dict[str, float] = {}
        base = OmegaConf.to_container(cfg.thresholds)
        for cname in CLASS_COLORS:
            s = st.slider(
                CLASS_LABEL[cname],
                min_value=0.5, max_value=2.0, step=0.05,
                value=st.session_state.get(f"sens_{cname}", 1.0),
                key=f"sens_{cname}",
            )
            sensitivity[cname] = s
            eff = base[cname]["high"] / max(s, 1e-3)
            st.caption(f"onset ≈ {min(eff, 0.99):.2f}")

    # --- main: upload + analyze ---
    st.subheader("Analyze recordings")
    files = st.file_uploader(
        "Hydrophone recordings",
        type=["wav", "mp3", "flac", "ogg"],
        accept_multiple_files=True,
        help="WAV / MP3 / FLAC / OGG · any sample rate or bit depth.",
    )
    run = st.button(
        "▶ Run detection",
        type="primary",
        disabled=(model is None or not files),
    )

    if run and model is not None and files:
        for uf in files:
            suffix = Path(uf.name).suffix.lower()
            if suffix not in ALLOWED_SUFFIXES:
                st.warning(f"{uf.name}: unsupported file type — skipped.")
                continue

            tmp = Path(tempfile.mkdtemp(prefix="echo_in_")) / uf.name
            tmp.write_bytes(uf.getbuffer())

            prog = st.progress(0.0, text=f"{uf.name}: decoding…")

            def on_progress(frac: float, _name=uf.name, _bar=prog):
                _bar.progress(min(frac * 0.9, 0.9), text=f"{_name}: detecting…")

            try:
                entry = analyze(tmp, uf.name, sensitivity, cfg, model, fps, on_progress)
                prog.progress(1.0, text=f"{uf.name}: done")
                st.session_state.archive.insert(0, entry)
            except Exception as e:
                prog.empty()
                st.error(f"{uf.name}: {type(e).__name__}: {e}")

    # --- results ---
    archive = st.session_state.archive
    if archive:
        st.divider()
        st.subheader("Results")
        names = [f"{e['name']} · {fmt_time(e['duration'])} · {e['n_events']} events" for e in archive]
        idx = 0
        if len(archive) > 1:
            idx = names.index(st.selectbox("Recording", names, index=0))
        render_result(archive[idx])
    else:
        st.info("Upload one or more recordings and run detection to see results.")


if __name__ == "__main__":
    main()
