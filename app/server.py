"""FastAPI app: REST API + static UI serving."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import app.config as cfg
from app import exporters, formatter, model_manager, transcribe
from app.models import JobStatus
from app.pipeline import delete_track_files, pipeline, store
from app.settings import settings

app = FastAPI(title="Refrain", version="0.1.0")

dl_manager = model_manager.DownloadManager()

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class JobIn(BaseModel):
    url: str


@app.post("/api/jobs")
def create_job(body: JobIn):
    if not body.url.strip():
        raise HTTPException(400, "url is required")
    job = pipeline.submit(body.url.strip())
    return {"id": job.id, "status": job.status.value, "message": job.message}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return {
        "id": job.id, "url": job.url, "status": job.status.value,
        "message": job.message, "trackId": job.track_id,
    }


@app.get("/api/tracks")
def list_tracks():
    return [
        {"id": t.id, "title": t.title, "artist": t.artist,
         "duration": t.duration, "hasTranscript":
             store.load_transcript(t.id) is not None}
        for t in store.list_tracks()
    ]


@app.get("/api/tracks/{track_id}")
def get_track(track_id: str):
    track = store.get_track(track_id)
    if not track:
        raise HTTPException(404, "track not found")
    lines = store.load_transcript(track_id) or []
    return {
        "id": track.id, "title": track.title, "artist": track.artist,
        "duration": track.duration,
        "audioUrl": f"/api/tracks/{track.id}/audio",
        "lines": [
            {"text": l.text, "start": l.start,
             "words": [{"text": w.text, "start": w.start, "end": w.end}
                       for w in l.words]}
            for l in lines
        ],
    }


@app.get("/api/tracks/{track_id}/audio")
def get_audio(track_id: str):
    track = store.get_track(track_id)
    if not track or not track.audio_file or not track.audio_file.exists():
        raise HTTPException(404, "audio not found")
    return FileResponse(track.audio_file, media_type="audio/mp4")


@app.get("/api/tracks/{track_id}/export")
def export_track(track_id: str, fmt: str = "lrc"):
    track = store.get_track(track_id)
    if not track:
        raise HTTPException(404, "track not found")
    lines = store.load_transcript(track_id) or []
    if not lines:
        raise HTTPException(404, "no transcript for this track")
    try:
        content = exporters.export(fmt, track.title, track.artist, lines)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    media = "text/plain" if fmt in ("txt", "lrc", "srt", "vtt") else "application/json"
    return PlainTextResponse(content, media_type=media,
                             headers={"Content-Disposition":
                                      f'attachment; filename="{track.title}.{fmt}"'})


@app.delete("/api/tracks/{track_id}")
def remove_track(track_id: str):
    if not store.get_track(track_id):
        raise HTTPException(404, "track not found")
    delete_track_files(track_id)
    return {"ok": True}


# --- retranscribe & transcript versions ---------------------------------------

class RetranscribeIn(BaseModel):
    backend: str | None = None   # auto | parakeet | groq | openai (None = settings)
    model: str | None = None     # local model id for this run only


@app.post("/api/tracks/{track_id}/retranscribe")
def retranscribe_track(track_id: str, body: RetranscribeIn):
    track = store.get_track(track_id)
    if not track:
        raise HTTPException(404, "track not found")
    if not track.audio_file or not track.audio_file.exists():
        raise HTTPException(409, "audio file missing — cannot retranscribe")
    if body.model and model_manager.get_spec(body.model) is None:
        raise HTTPException(400, f"unknown model '{body.model}'")
    job = store.create_job(
        track.url, kind="retranscribe", track_id=track_id,
        retranscribe_backend=body.backend, retranscribe_model=body.model)
    return {"id": job.id, "status": job.status.value, "message": job.message}


@app.get("/api/tracks/{track_id}/versions")
def list_versions(track_id: str):
    if not store.get_track(track_id):
        raise HTTPException(404, "track not found")
    index = store.list_versions(track_id)
    if index is None:
        return {"active": None, "versions": []}
    return index


@app.post("/api/tracks/{track_id}/versions/{version}/activate")
def activate_version(track_id: str, version: int):
    if not store.get_track(track_id):
        raise HTTPException(404, "track not found")
    if not store.activate_version(track_id, version):
        raise HTTPException(404, f"version {version} not found")
    return {"ok": True, "active": version}


# --- models & settings --------------------------------------------------------

@app.get("/api/settings")
def get_settings():
    data = settings.get()
    data["llmAvailable"] = formatter.llm_available()
    return data


@app.put("/api/settings")
def update_settings(body: dict):
    try:
        return settings.update(**{k: body[k] for k in ("backend", "model") if k in body})
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/models")
def list_models():
    downloads = {j["modelId"]: j for j in dl_manager.status()}
    models = model_manager.list_models()
    for m in models:
        m["download"] = downloads.get(m["id"], {"status": "idle", "error": None})
    return {"models": models, "settings": settings.get()}


@app.post("/api/models/{model_id}/download")
def download_model(model_id: str):
    if not model_manager.get_spec(model_id):
        raise HTTPException(404, "unknown model")
    return dl_manager.start(model_id)


@app.delete("/api/models/{model_id}")
def delete_model(model_id: str):
    if not model_manager.get_spec(model_id):
        raise HTTPException(404, "unknown model")
    job = dl_manager.status(model_id)
    if isinstance(job, dict) and job.get("status") == "downloading":
        raise HTTPException(409, "model is currently downloading")
    removed = model_manager.delete_model(model_id)
    transcribe.parakeet.drop_cache(model_id)
    if settings.get()["model"] == model_id:
        settings.update(model="parakeet-v3")
    return {"ok": removed}


@app.exception_handler(404)
def not_found(request, exc):  # SPA-friendly
    return JSONResponse({"detail": str(exc.detail)}, status_code=404)


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
