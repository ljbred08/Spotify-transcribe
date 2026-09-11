"""Shared data model for tracks, jobs and word/line timings."""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import app.config as cfg


class JobStatus(str, Enum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    ALIGNING = "aligning"
    DONE = "done"
    ERROR = "error"


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Line:
    text: str
    start: float
    progress: float  # 0..1 karaoke sweep within the line (runtime only)
    words: list[Word]


@dataclass
class Track:
    id: str
    url: str
    title: str = "Untitled"
    artist: str = "Unknown artist"
    duration: float = 0.0
    audio_file: Path | None = None  # set by the downloader
    created_at: float = field(default_factory=time.time)


@dataclass
class Job:
    id: str
    url: str
    status: JobStatus = JobStatus.QUEUED
    message: str = "Queued"
    track_id: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class Store:
    """Thread-safe in-memory store with JSON persistence to disk."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._tracks: dict[str, Track] = {}
        self._queue: list[str] = []
        self._path = cfg.DATA_DIR / "library.json"
        (cfg.DATA_DIR / "transcripts").mkdir(parents=True, exist_ok=True)
        (cfg.DATA_DIR / "audio").mkdir(parents=True, exist_ok=True)
        self._load()

    # -- persistence --------------------------------------------------------
    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw: dict[str, Any] = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for t in raw.get("tracks", []):
            track = Track(
                id=t["id"], url=t["url"], title=t.get("title", "Untitled"),
                artist=t.get("artist", "Unknown artist"),
                duration=t.get("duration", 0.0),
                audio_file=Path(t["audio_file"]) if t.get("audio_file") else None,
                created_at=t.get("created_at", time.time()),
            )
            self._tracks[track.id] = track
        for j in raw.get("jobs", []):
            try:
                status = JobStatus(j.get("status", "error"))
            except ValueError:
                status = JobStatus.ERROR
            # jobs never survive a restart mid-flight
            if status in (JobStatus.QUEUED, JobStatus.DOWNLOADING,
                          JobStatus.TRANSCRIBING, JobStatus.ALIGNING):
                status = JobStatus.ERROR
                j["message"] = "Interrupted by restart"
            job = Job(
                id=j["id"], url=j["url"], status=status,
                message=j.get("message", ""), track_id=j.get("track_id"),
                created_at=j.get("created_at", time.time()),
                updated_at=j.get("updated_at", time.time()),
            )
            self._jobs[job.id] = job

    def _save(self) -> None:
        with self._lock:
            payload = {
                "tracks": [
                    {**asdict(t), "audio_file": str(t.audio_file) if t.audio_file else None}
                    for t in self._tracks.values()
                ],
                "jobs": [asdict(j) for j in self._jobs.values()],
            }
        self._path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    # -- jobs ---------------------------------------------------------------
    def create_job(self, url: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], url=url)
        with self._lock:
            self._jobs[job.id] = job
            self._queue.append(job.id)
            self._save()
        return job

    def get_job(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def update_job(self, job_id: str, **fields: Any) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            for key, value in fields.items():
                setattr(job, key, value)
            job.updated_at = time.time()
            self._save()
            return job

    def pop_next_job(self) -> Job | None:
        with self._lock:
            while self._queue:
                job_id = self._queue.pop(0)
                job = self._jobs.get(job_id)
                if job and job.status is JobStatus.QUEUED:
                    return job
        return None

    # -- tracks -------------------------------------------------------------
    def add_track(self, track: Track) -> None:
        with self._lock:
            self._tracks[track.id] = track
            self._save()

    def get_track(self, track_id: str) -> Track | None:
        with self._lock:
            return self._tracks.get(track_id)

    def list_tracks(self) -> list[Track]:
        with self._lock:
            return sorted(self._tracks.values(), key=lambda t: t.created_at, reverse=True)

    def delete_track(self, track_id: str) -> bool:
        with self._lock:
            existed = track_id in self._tracks
            self._tracks.pop(track_id, None)
            self._save()
        return existed

    # -- transcript persistence ----------------------------------------------
    def save_transcript(self, track_id: str, lines: list[Line]) -> None:
        payload = [
            {
                "text": line.text,
                "start": round(line.start, 3),
                "words": [
                    {"text": w.text, "start": round(w.start, 3), "end": round(w.end, 3)}
                    for w in line.words
                ],
            }
            for line in lines
        ]
        path = cfg.DATA_DIR / "transcripts" / f"{track_id}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                        encoding="utf-8")

    def load_transcript(self, track_id: str) -> list[Line] | None:
        path = cfg.DATA_DIR / "transcripts" / f"{track_id}.json"
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return [
            Line(
                text=l["text"],
                start=l["start"],
                progress=0.0,
                words=[Word(text=w["text"], start=w["start"], end=w["end"])
                       for w in l.get("words", [])],
            )
            for l in raw
        ]

    def audio_path(self, track_id: str) -> Path:
        return cfg.DATA_DIR / "audio" / f"{track_id}.m4a"
