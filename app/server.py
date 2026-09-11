"""FastAPI app: REST API + static UI serving."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import app.config as cfg
from app import exporters
from app.models import JobStatus
from app.pipeline import delete_track_files, pipeline, store

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


@app.exception_handler(404)
def not_found(request, exc):  # SPA-friendly
    return JSONResponse({"detail": str(exc.detail)}, status_code=404)


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
