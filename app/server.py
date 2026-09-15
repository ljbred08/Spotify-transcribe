"""FastAPI app: REST API + static UI serving."""
from __future__ import annotations

import difflib
import re
import subprocess
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import app.config as cfg
from app import exporters, formatter, transcribe
from app import cloud_models
from app.models import JobStatus, Track, Word
from app.pipeline import delete_track_files, pipeline, store
from app.settings import settings

app = FastAPI(title="Refrain", version="0.1.0")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class JobIn(BaseModel):
    url: str


class TrackRenameIn(BaseModel):
    title: str | None = None
    artist: str | None = None


class LyricEditIn(BaseModel):
    lineIndex: int
    text: str


def _edit_lines(lines: list, line_index: int, text: str) -> list:
    """Apply a lyric text edit to a parsed transcript, preserving timings.

    Word tags only carry start times (a word's end is the next tag's start),
    so the word list is rebuilt by diffing old vs new tokens:
    - unchanged words keep their exact timings;
    - replaced words take over the old slot's timing (n new tokens spread
      across m old slots, extras sharing the last slot);
    - inserted words sit at the next surviving word's start (zero-width),
      or continue from the line's end when appending;
    - deleted words are simply dropped — the next survivor's start closes
      the gap automatically on the next parse.
    """
    new_tokens = text.split()
    if not new_tokens:
        raise ValueError("edited line cannot be empty")
    line = lines[line_index]
    old = line.words
    if old:
        sm = difflib.SequenceMatcher(
            a=[_norm_tok(w.text) for w in old],
            b=[_norm_tok(t) for t in new_tokens], autojunk=False)
        new_words: list[Word] = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag in ("equal", "replace"):
                slots = old[i1:i2]
                if slots:
                    for k, tok in enumerate(new_tokens[j1:j2]):
                        src = slots[min(k, len(slots) - 1)]
                        new_words.append(Word(text=tok, start=src.start,
                                              end=src.end))
                    continue
                # "replace" against nothing == insert
                i1 = min(i1, len(old))
            if tag == "insert" or not old[i1:i2]:
                if i1 < len(old):
                    nxt = old[i1]           # zero-width before the next word
                    new_words.extend(Word(text=t, start=nxt.start, end=nxt.start)
                                     for t in new_tokens[j1:j2])
                else:
                    tail = new_words[-1].end if new_words else line.start
                    new_words.extend(Word(text=t, start=tail, end=tail)
                                     for t in new_tokens[j1:j2])
            # "delete": drop the old words
        line.words = new_words
    else:
        # line without word timings: even split across a nominal 1s span
        span = line.start + 1.0
        step = (span - line.start) / len(new_tokens)
        line.words = [Word(text=t, start=line.start + k * step,
                           end=line.start + (k + 1) * step)
                      for k, t in enumerate(new_tokens)]
    line.text = " ".join(w.text for w in line.words)
    return lines


def _norm_tok(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


_AUDIO_EXTS = {".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus",
               ".webm", ".mp4"}
_MAX_UPLOAD_BYTES = 300 * 1024 * 1024


def _audio_duration(path: Path) -> float:
    """Best-effort duration via ffprobe; 0 when unavailable."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30, check=True)
        return float(out.stdout.strip())
    except Exception:  # noqa: BLE001 - duration is display metadata only
        return 0.0


@app.post("/api/tracks/upload")
async def upload_track(file: UploadFile = File(...)):
    """Create a track from a local audio file and queue its transcription.

    Downloads resolve metadata via yt-dlp; an upload already IS the audio,
    so the track row is created up front (title from the filename) and the
    job reuses the pipeline's transcribe-only path — no downloader involved.
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _AUDIO_EXTS:
        raise HTTPException(400, f"unsupported audio type '{ext or '?'}' — "
                                 f"expected one of {', '.join(sorted(_AUDIO_EXTS))}")
    audio_dir = cfg.DATA_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    dest = audio_dir / f"{uuid.uuid4().hex[:12]}{ext}"
    size = 0
    try:
        with dest.open("wb") as out:
            while chunk := await file.read(8 * 1024 * 1024):
                size += len(chunk)
                if size > _MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "file too large (max 300 MB)")
                out.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    except OSError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(500, f"could not save upload: {exc}") from exc
    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "empty file")

    stem = Path(file.filename or "audio").stem
    title = re.sub(r"^\d+\s*[-._)]\s*", "", stem.replace("_", " ")).strip()
    track = Track(id=uuid.uuid4().hex[:12], url="", title=title[:80] or "Uploaded audio",
                  artist="", duration=_audio_duration(dest), audio_file=dest)
    store.add_track(track)
    job = pipeline.submit("", kind="upload", track_id=track.id)
    return {"job": {"id": job.id, "status": job.status.value,
                    "message": job.message}, "trackId": track.id}


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


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    store.request_cancel(job_id)
    return {"id": job.id, "status": job.status.value, "message": job.message}


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


@app.patch("/api/tracks/{track_id}")
def rename_track(track_id: str, body: TrackRenameIn):
    """Edit track title/artist (double-click in the UI); exports use these."""
    if body.title is None and body.artist is None:
        raise HTTPException(400, "nothing to update")
    if body.title is not None and not body.title.strip():
        raise HTTPException(400, "title cannot be empty")
    track = store.update_track(track_id,
                               **{k: v.strip()
                                  for k, v in (("title", body.title),
                                               ("artist", body.artist))
                                  if v is not None})
    if track is None:
        raise HTTPException(404, "track not found")
    return {"id": track.id, "title": track.title, "artist": track.artist}


@app.post("/api/tracks/{track_id}/edit_line")
def edit_line(track_id: str, body: LyricEditIn):
    """Edit one lyric line; stored as a new transcript version (undoable)."""
    lines = store.load_transcript(track_id)
    if lines is None:
        raise HTTPException(404, "no transcript for this track")
    if not 0 <= body.lineIndex < len(lines):
        raise HTTPException(400, "lineIndex out of range")
    try:
        _edit_lines(lines, body.lineIndex, body.text)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    version = store.save_transcript(track_id, lines,
                                    backend=None, note="edited")
    return {"version": version, "lines": len(lines)}


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
        # [ti:]/[ar:] tags inside stored versions may be stale (title/artist
        # are editable); the track record is the source of truth
        content = exporters.export(fmt, track.title, track.artist, lines)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    media = ("text/plain" if fmt in ("txt", "lrc", "lrc-enhanced", "srt", "vtt")
             else "application/json")
    ext = "lrc" if fmt == "lrc-enhanced" else fmt
    return PlainTextResponse(content, media_type=media,
                             headers={"Content-Disposition":
                                      f'attachment; filename="{track.title}.{ext}"'})


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


@app.post("/api/tracks/{track_id}/improve_sync")
def improve_sync_track(track_id: str, body: RetranscribeIn):
    """Retime the active transcript from a transcription-only reference run."""
    track = store.get_track(track_id)
    if not track:
        raise HTTPException(404, "track not found")
    if not track.audio_file or not track.audio_file.exists():
        raise HTTPException(409, "audio file missing — cannot improve sync")
    if store.load_transcript(track_id) is None:
        raise HTTPException(409, "no transcript to improve")
    model_id = body.model or body.backend
    if not model_id:
        raise HTTPException(400, "a transcription model is required")
    spec = cloud_models.get_model(model_id)
    if spec is None:
        raise HTTPException(400, f"unknown model '{model_id}'")
    if spec.kind != "transcribe":
        raise HTTPException(400, "improve sync needs a transcription-only "
                                 "model (Whisper etc.), not an audio LLM")
    job = store.create_job(
        track.url, kind="improve_sync", track_id=track_id,
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
            "transcribe_provider", "transcribe_model", "keys",
            "temperature", "top_p", "transcribe_timeout", "show_timestamps",
            "highlight_style")
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
