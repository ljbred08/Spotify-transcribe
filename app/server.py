"""FastAPI app: REST API + static UI serving."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import app.config as cfg
from app import exporters, formatter, transcribe
from app import cloud_models
from app.models import JobStatus
from app.pipeline import delete_track_files, pipeline, store
from app.settings import settings

app = FastAPI(title="Refrain", version="0.1.0")

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
    model: str | None = None     # cloud model id for this run only
    backend: str | None = None   # legacy alias, treated as model id


@app.post("/api/tracks/{track_id}/retranscribe")
def retranscribe_track(track_id: str, body: RetranscribeIn):
    track = store.get_track(track_id)
    if not track:
        raise HTTPException(404, "track not found")
    if not track.audio_file or not track.audio_file.exists():
        raise HTTPException(409, "audio file missing — cannot retranscribe")
    model_id = body.model or body.backend
    if model_id and cloud_models.get_model(model_id) is None:
        raise HTTPException(400, f"unknown model '{model_id}'")
    job = store.create_job(
        track.url, kind="retranscribe", track_id=track_id,
        retranscribe_backend=model_id, retranscribe_model=model_id)
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
    data["llm"] = formatter.llm_status_cached()
    return data


@app.put("/api/settings")
def update_settings(body: dict):
    keys = ("backend", "model", "cloud_model", "llm_format", "llm_url",
            "llm_model", "llm_model_ollama", "llm_provider",
            "transcribe_provider", "transcribe_model", "keys")
    try:
        out = settings.update(**{k: body[k] for k in keys if k in body})
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if any(k in body for k in ("llm_url", "llm_model", "llm_model_ollama",
                               "llm_provider", "keys", "check_llm")):
        formatter.invalidate_status_cache()  # invalidate availability cache
    out["llmAvailable"] = formatter.llm_available()
    out["llm"] = formatter.llm_status()
    return out


@app.get("/api/models")
def list_models():
    """Dynamic cloud catalog + current settings (incl. formatter block data)."""
    data = settings.get()
    data["llmAvailable"] = formatter.llm_available()
    data["llm"] = formatter.llm_status_cached()
    return {"models": cloud_models.list_models(), "settings": data,
            "formatter_models": cloud_models.formatter_catalog()}


@app.get("/api/providers")
def list_providers():
    """Provider dropdown data: key_env + whether a key is set (and from where)."""
    return {"providers": cloud_models.list_providers()}


@app.post("/api/llm/discover")
def discover_llm(body: dict | None = None):
    """Probe candidate ollama URLs (hint first, then localhost defaults)."""
    hint = ((body or {}).get("url") or "").strip() or None
    found = formatter.discover_server(hint=hint, current=settings.get().get("llm_url"))
    if found:
        settings.update(llm_url=found)
        formatter.invalidate_status_cache()
    return {"found": found, "llm": formatter.llm_status()}


@app.exception_handler(404)
def not_found(request, exc):  # SPA-friendly
    return JSONResponse({"detail": str(exc.detail)}, status_code=404)


@app.get("/", include_in_schema=False)
def index():
    resp = FileResponse(WEB_DIR / "index.html")
    # no heuristic caching: the UI is a single file, edits must show up on reload
    resp.headers["Cache-Control"] = "no-cache"
    return resp


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
