"""echoSentinel web console — FastAPI application.

    python -m echosentinel.server            # http://127.0.0.1:8710

Serves the static console (web/) and the analysis API. Fully offline: the
model, fonts-fallbacks, and all assets are local.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from echosentinel import __version__
from echosentinel.constants import CLASS_MAP
from echosentinel.server.jobs import JobManager

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WEB_DIR = PROJECT_ROOT / "web"

ALLOWED_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg"}


def create_app(
    weights: Path | None = None,
    inference_cfg: Path | None = None,
) -> FastAPI:
    app = FastAPI(title="echoSentinel", version=__version__, docs_url=None, redoc_url=None)
    manager = JobManager(
        PROJECT_ROOT,
        weights or PROJECT_ROOT / "weights" / "panns_pcen.pt",
        inference_cfg or PROJECT_ROOT / "configs" / "inference.yaml",
    )
    app.state.manager = manager

    # ---------- system ----------

    @app.get("/api/system")
    def system() -> dict:
        jobs = list(manager.jobs.values())
        done = [j for j in jobs if j.status == "done"]
        return {
            "version": __version__,
            "model": manager.model_meta,
            "classes": [
                {"id": cid, "name": name} for cid, name in CLASS_MAP.items()
            ],
            "thresholds": manager.thresholds(),
            "stats": {
                "files_analyzed": len(done),
                "events_detected": sum(j.n_events for j in done),
                "audio_minutes": round(sum(j.duration for j in done) / 60.0, 1),
                "queued_or_running": sum(
                    1 for j in jobs if j.status in ("queued", "running")
                ),
            },
        }

    # ---------- jobs ----------

    @app.get("/api/jobs")
    def list_jobs() -> list[dict]:
        jobs = sorted(manager.jobs.values(), key=lambda j: j.created, reverse=True)
        return [asdict(j) for j in jobs]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = manager.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")
        return asdict(job)

    @app.post("/api/jobs")
    async def create_job(
        file: UploadFile = File(...),
        sensitivity: str = Form("{}"),
    ) -> dict:
        name = Path(file.filename or "recording.wav").name
        if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
            raise HTTPException(400, f"unsupported file type: {name}")
        try:
            sens = {k: float(v) for k, v in json.loads(sensitivity).items()}
        except (json.JSONDecodeError, TypeError, ValueError):
            raise HTTPException(400, "sensitivity must be a JSON object of numbers")

        job = manager.create(name, sens)
        dest = manager.job_dir(job.id) / "audio.wav"
        with dest.open("wb") as out:
            while chunk := await file.read(1 << 20):
                out.write(chunk)
        manager.enqueue(job.id)
        return asdict(job)

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str) -> dict:
        if not manager.delete(job_id):
            raise HTTPException(409, "job is running or unknown")
        return {"deleted": job_id}

    # ---------- job artifacts ----------

    def _artifact(job_id: str, name: str) -> Path:
        path = manager.job_dir(job_id) / name
        if job_id not in manager.jobs or not path.exists():
            raise HTTPException(404, f"{name} not available")
        return path

    @app.get("/api/jobs/{job_id}/results")
    def results(job_id: str) -> JSONResponse:
        return JSONResponse(
            json.loads(_artifact(job_id, "results.json").read_text(encoding="utf-8"))
        )

    @app.get("/api/jobs/{job_id}/export")
    def export(job_id: str) -> FileResponse:
        job = manager.jobs[job_id]
        stem = Path(job.original_name).stem
        return FileResponse(
            _artifact(job_id, "results.json"),
            media_type="application/json",
            filename=f"{stem}_events.json",
        )

    @app.get("/api/jobs/{job_id}/peaks")
    def peaks(job_id: str) -> JSONResponse:
        return JSONResponse(
            json.loads(_artifact(job_id, "peaks.json").read_text(encoding="utf-8"))
        )

    @app.get("/api/jobs/{job_id}/spectrogram.png")
    def spectrogram(job_id: str) -> FileResponse:
        return FileResponse(_artifact(job_id, "spectrogram.png"), media_type="image/png")

    @app.get("/api/jobs/{job_id}/audio")
    def audio(job_id: str) -> FileResponse:
        return FileResponse(_artifact(job_id, "audio.wav"), media_type="audio/wav")

    # ---------- static console ----------

    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="console")

    return app


def main() -> None:
    import os

    import uvicorn

    # Local default binds loopback only; containers set HOST=0.0.0.0.
    host = os.environ.get("ECHOSENTINEL_HOST", "127.0.0.1")
    port = int(os.environ.get("ECHOSENTINEL_PORT", "8710"))
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
