"""Shared data model for tracks, jobs and word/line timings."""
from __future__ import annotations

import json
import shutil
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


TERMINAL_STATUSES = {JobStatus.DONE, JobStatus.ERROR}


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
    # "download" (default) or "retranscribe" (re-run ASR on existing audio)
    kind: str = "download"
    # per-job overrides for retranscribe jobs (None = use app settings)
    retranscribe_backend: str | None = None
    retranscribe_model: str | None = None
    # set when the user (or the timeout watchdog) asks to cancel; the
    # pipeline checks it at every stage boundary and raises Canceled
    cancel_requested: bool = False


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
                kind=j.get("kind", "download"),
                retranscribe_backend=j.get("retranscribe_backend"),
                retranscribe_model=j.get("retranscribe_model"),
                cancel_requested=j.get("cancel_requested", False),
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
    def create_job(self, url: str, *, kind: str = "download",
                   track_id: str | None = None,
                   retranscribe_backend: str | None = None,
                   retranscribe_model: str | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], url=url, kind=kind, track_id=track_id,
                  retranscribe_backend=retranscribe_backend,
                  retranscribe_model=retranscribe_model)
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
            # terminal jobs are frozen: a late worker-thread write must never
            # change a job's end state — including ERROR→ERROR, else the
            # worker's follow-up exception can overwrite the watchdog's
            # "Timed out after Ns" message with a less useful one
            if job.status in TERMINAL_STATUSES:
                fields.pop("status", None)
                fields.pop("message", None)
            for key, value in fields.items():
                setattr(job, key, value)
            job.updated_at = time.time()
            self._save()
            return job

    def request_cancel(self, job_id: str) -> Job | None:
        """Ask a running/queued job to stop; returns the refreshed job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status in TERMINAL_STATUSES:
                return job
            job.cancel_requested = True
            if job.status is JobStatus.QUEUED:
                job.status = JobStatus.ERROR
                job.message = "Canceled before start"
            else:
                job.message = "Canceling…"
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

    def update_track(self, track_id: str, *, title: str | None = None,
                     artist: str | None = None) -> Track | None:
        """Patch editable metadata (title/artist); None leaves a field as-is."""
        with self._lock:
            track = self._tracks.get(track_id)
            if track is None:
                return None
            if title is not None:
                track.title = title
            if artist is not None:
                track.artist = artist
            self._save()
            return track

    # -- transcript persistence (versioned) ----------------------------------
    def _transcript_dir(self, track_id: str) -> Path:
        return cfg.DATA_DIR / "transcripts" / track_id

    def _versions_index_path(self, track_id: str) -> Path:
        return self._transcript_dir(track_id) / "versions.json"

    def _migrate_legacy_transcript(self, track_id: str) -> None:
        """Move a pre-versioning transcripts/<id>.json into v1 of the new layout.

        Callers must hold self._lock (the index read-modify-write is not
        atomic on its own).
        """
        legacy = cfg.DATA_DIR / "transcripts" / f"{track_id}.json"
        if not legacy.exists():
            return
        d = self._transcript_dir(track_id)
        d.mkdir(parents=True, exist_ok=True)
        target = d / "v1.json"
        if not target.exists():
            shutil.move(str(legacy), str(target))
            self._write_versions_index(track_id, {
                "active": 1,
                "versions": [{"version": 1, "createdAt": time.time(),
                              "backend": None, "note": "imported from pre-versioning format"}],
            })
        else:
            legacy.unlink(missing_ok=True)

    def _read_versions_index(self, track_id: str) -> dict | None:
        p = self._versions_index_path(track_id)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _write_versions_index(self, track_id: str, index: dict) -> None:
        self._versions_index_path(track_id).write_text(
            json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")

    def save_transcript(self, track_id: str, lines: list[Line],
                        *, backend: str | None = None,
                        note: str | None = None,
                        match: int | None = None) -> int:
        """Persist lines as a new version; it becomes the active one.

        Returns the version number. Previous versions are kept so a bad
        re-run can always be rolled back. `match` is the improve-sync match
        accuracy (0-100); stored on the version so several resyncs can be
        compared at a glance in the UI.
        """
        # the index read-modify-write must not interleave with an activate()
        # from an endpoint thread or another job's save
        with self._lock:
            return self._save_transcript_locked(track_id, lines,
                                                backend=backend, note=note,
                                                match=match)

    def _save_transcript_locked(self, track_id: str, lines: list[Line],
                                *, backend: str | None,
                                note: str | None,
                                match: int | None = None) -> int:
        # Canonical storage is Enhanced LRC text (word-level tags) so the
        # format we get from audio LLMs is exactly what we keep; it is
        # re-parsed on every load, so parser improvements apply to every
        # stored version retroactively.
        from app.lrc import serialize
        track = self._tracks.get(track_id)
        title = track.title if track else ""
        artist = track.artist if track else ""
        self._migrate_legacy_transcript(track_id)
        d = self._transcript_dir(track_id)
        d.mkdir(parents=True, exist_ok=True)
        index = self._read_versions_index(track_id) or {"active": 0, "versions": []}
        version = (max((v["version"] for v in index["versions"]), default=0)) + 1
        (d / f"v{version}.lrc").write_text(
            serialize(title, artist, lines), encoding="utf-8")
        index["versions"].append({
            "version": version, "createdAt": time.time(), "backend": backend,
            "lines": len(lines), "note": note, "fmt": "lrc",
            **({"match": match} if match is not None else {}),
        })
        index["active"] = version
        self._write_versions_index(track_id, index)
        return version

    def load_transcript(self, track_id: str) -> list[Line] | None:
        with self._lock:
            return self._load_transcript_locked(track_id)

    def _load_transcript_locked(self, track_id: str) -> list[Line] | None:
        self._migrate_legacy_transcript(track_id)
        index = self._read_versions_index(track_id)
        if index is None:
            return None
        d = self._transcript_dir(track_id)
        active = index.get("active", 0)
        # Enhanced LRC is canonical; parse it fresh on every load.
        lrc_path = d / f"v{active}.lrc"
        if lrc_path.exists():
            from app.lrc import parse_enhanced
            try:
                parsed = parse_enhanced(lrc_path.read_text(encoding="utf-8"))
            except OSError:
                return None
            if not parsed:
                return None
            return [
                Line(text=ln["text"],
                     start=ln["words"][0]["s"],
                     progress=0.0,
                     words=[Word(text=w["t"], start=w["s"], end=w["e"])
                            for w in ln["words"]])
                for ln in parsed
            ]
        # legacy JSON versions (pre-LRC storage) still load as before
        path = d / f"v{active}.json"
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

    def list_versions(self, track_id: str) -> dict | None:
        """The versions index for a track (None when it has no transcripts)."""
        with self._lock:
            self._migrate_legacy_transcript(track_id)
            return self._read_versions_index(track_id)

    def activate_version(self, track_id: str, version: int) -> bool:
        with self._lock:
            self._migrate_legacy_transcript(track_id)
            index = self._read_versions_index(track_id)
            if not index or not any(v["version"] == version for v in index["versions"]):
                return False
            index["active"] = version
            self._write_versions_index(track_id, index)
            return True

    def audio_path(self, track_id: str) -> Path:
        return cfg.DATA_DIR / "audio" / f"{track_id}.m4a"
