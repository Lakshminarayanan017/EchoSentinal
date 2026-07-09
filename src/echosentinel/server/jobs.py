"""Analysis job store and the single sequential worker.

Jobs persist to disk (out/webapp/jobs/<id>/ + registry.json) so the archive
survives restarts. One worker thread analyzes jobs FIFO — the model is
CPU-bound and parallel analyses would just thrash each other.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from omegaconf import OmegaConf

from echosentinel.audio.io import probe
from echosentinel.constants import CLASS_MAP
from echosentinel.infer.json_writer import build_results_json, write_results_json
from echosentinel.infer.posteriors import file_posteriors
from echosentinel.infer.postprocess import probs_to_events
from echosentinel.models.registry import build_model, model_frames_per_second
from echosentinel.server.media import spectrogram_png, waveform_peaks


@dataclass
class Job:
    id: str
    original_name: str
    status: str = "queued"  # queued | running | done | error
    stage: str = "queued"   # queued | decoding | detecting | rendering | done
    progress: float = 0.0
    created: float = field(default_factory=time.time)
    duration: float = 0.0
    sample_rate: int = 0
    error: str = ""
    n_events: int = 0
    class_counts: dict = field(default_factory=dict)
    sensitivity: dict = field(default_factory=dict)  # class -> multiplier (1.0 = calibrated)


class JobManager:
    def __init__(self, project_root: Path, weights: Path, inference_cfg: Path) -> None:
        self.root = project_root / "out" / "webapp"
        self.jobs_dir = self.root / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.root / "registry.json"
        self.weights_path = weights
        self.cfg = OmegaConf.load(inference_cfg)
        self.lock = threading.Lock()
        self.jobs: dict[str, Job] = self._load_registry()
        self.queue: "queue.Queue[str]" = queue.Queue()

        self.model = None
        self.model_meta: dict = {}
        self.fps = 25.0
        self._load_model()

        # re-queue jobs that were interrupted mid-run by a restart
        for job in self.jobs.values():
            if job.status in ("queued", "running"):
                job.status, job.stage, job.progress = "queued", "queued", 0.0
                self.queue.put(job.id)

        self.worker = threading.Thread(target=self._work, daemon=True)
        self.worker.start()

    # ---------- persistence ----------

    def _load_registry(self) -> dict[str, Job]:
        if not self.registry_path.exists():
            return {}
        data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        return {j["id"]: Job(**j) for j in data}

    def _save_registry(self) -> None:
        with self.lock:
            data = [asdict(j) for j in self.jobs.values()]
        self.registry_path.write_text(json.dumps(data, indent=1), encoding="utf-8")

    # ---------- model ----------

    def _load_model(self) -> None:
        ckpt = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        model = build_model(ckpt["model_name"], **ckpt.get("model_kwargs", {}))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        self.model = model
        self.fps = model_frames_per_second(ckpt["model_name"])
        self.model_meta = {
            "model_name": ckpt["model_name"],
            "model_kwargs": ckpt.get("model_kwargs", {}),
            "epoch": int(ckpt.get("epoch", -1)),
            "val_f1_macro": round(float(ckpt.get("f1_macro", 0.0)), 4),
            "weights_file": self.weights_path.name,
            "params_million": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        }

    # ---------- public API ----------

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def create(self, original_name: str, sensitivity: dict | None = None) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, original_name=original_name, sensitivity=sensitivity or {})
        self.job_dir(job_id).mkdir(parents=True, exist_ok=True)
        with self.lock:
            self.jobs[job_id] = job
        self._save_registry()
        return job

    def enqueue(self, job_id: str) -> None:
        self.queue.put(job_id)

    def delete(self, job_id: str) -> bool:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None or job.status == "running":
                return False
            del self.jobs[job_id]
        import shutil

        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)
        self._save_registry()
        return True

    def thresholds(self) -> dict:
        return OmegaConf.to_container(self.cfg.thresholds)

    def effective_thresholds(self, sensitivity: dict) -> dict:
        """Sensitivity multiplier per class scales the calibrated thresholds:
        >1 = stricter (fewer detections), <1 = more sensitive."""
        out = {}
        for cname, th in self.thresholds().items():
            k = float(sensitivity.get(cname, 1.0))
            out[cname] = {
                "high": min(max(th["high"] * k, 0.01), 0.99),
                "low": min(max(th["low"] * k, 0.005), 0.95),
            }
        return out

    # ---------- worker ----------

    def _work(self) -> None:
        while True:
            job_id = self.queue.get()
            with self.lock:
                job = self.jobs.get(job_id)
            if job is None:
                continue
            try:
                self._analyze(job)
                job.status, job.stage, job.progress = "done", "done", 1.0
            except Exception as e:  # job failures must not kill the worker
                job.status, job.stage = "error", "error"
                job.error = f"{type(e).__name__}: {e}"
            self._save_registry()

    def _analyze(self, job: Job) -> None:
        jd = self.job_dir(job.id)
        audio = jd / "audio.wav"
        job.status, job.stage = "running", "decoding"
        self._save_registry()

        info = probe(audio)
        job.duration = round(info.duration, 2)
        job.sample_rate = info.sr

        job.stage = "detecting"

        def on_progress(frac: float) -> None:
            job.progress = round(frac * 0.85, 4)  # detection = 85% of the bar

        probs = file_posteriors(
            audio,
            lambda w: self.model.posteriors(w),
            self.fps,
            block_seconds=float(self.cfg.block_seconds),
            overlap_seconds=float(self.cfg.block_overlap_seconds),
            progress=on_progress,
        )
        events = probs_to_events(
            probs,
            self.fps,
            self.effective_thresholds(job.sensitivity),
            median_seconds=float(self.cfg.posteriors.median_filter_seconds),
            merge_gap_seconds=float(self.cfg.events.merge_gap_seconds),
            min_duration_seconds=float(self.cfg.events.min_duration_seconds),
            round_to_seconds=bool(self.cfg.events.round_to_seconds),
        )

        results = build_results_json(
            [(job.original_name, info.duration, events)],
            contributor=str(self.cfg.json.startup_name),
        )
        write_results_json(jd / "results.json", results)

        job.n_events = len(events)
        counts: dict[str, int] = {}
        for ev in events:
            counts[CLASS_MAP[ev.category_id]] = counts.get(CLASS_MAP[ev.category_id], 0) + 1
        job.class_counts = counts

        job.stage, job.progress = "rendering", 0.88
        self._save_registry()
        (jd / "peaks.json").write_text(
            json.dumps(waveform_peaks(audio)), encoding="utf-8"
        )
        job.progress = 0.94
        spectrogram_png(audio, jd / "spectrogram.png")
