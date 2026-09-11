"""Background job pipeline: download -> transcribe -> assemble -> persist."""
from __future__ import annotations

import threading
import traceback
import uuid
from pathlib import Path

import app.config as cfg
from app import downloader, exporters, transcribe
from app.models import Job, JobStatus, Line, Store, Track

store = Store()


class Pipeline:
    def __init__(self, store: Store) -> None:
        self.store = store
        self._threads: list[threading.Thread] = []
        self._semaphore = threading.Semaphore(cfg.JOB_CONCURRENCY)
        self._start_workers()

    def _start_workers(self) -> None:
        for i in range(cfg.JOB_CONCURRENCY):
            t = threading.Thread(target=self._worker_loop, name=f"worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def submit(self, url: str) -> Job:
        job = self.store.create_job(url)
        return job

    def _worker_loop(self) -> None:
        while True:
            job = self.store.pop_next_job()
            if job is None:
                threading.Event().wait(0.5)
                continue
            with self._semaphore:
                self._run_job(job)

    def _run_job(self, job: Job) -> None:
        try:
            self._progress(job, JobStatus.DOWNLOADING, "Resolving track…")
            dl = downloader.download_audio(job.url, cfg.DATA_DIR / "audio", job.id)
            track = Track(
                id=uuid.uuid4().hex[:12],
                url=job.url,
                title=dl.title,
                artist=dl.artist,
                duration=dl.duration,
                audio_file=dl.audio_path,
            )
            self.store.add_track(track)
            self.store.update_job(job.id, track_id=track.id)

            backend, note = transcribe.pick_backend()
            if backend is None:
                raise transcribe.TranscribeError(note)

            self._progress(job, JobStatus.TRANSCRIBING, note)
            result = backend.transcribe(dl.audio_path, lambda m: self._say(job, m))

            self._progress(job, JobStatus.ALIGNING, "Building synced lines…")
            assembly = exporters.assemble_lines(result)
            self.store.save_transcript(track.id, assembly.lines)
            self.store.update_job(job.id, status=JobStatus.DONE,
                                  message=f"Done via {result.backend} · "
                                          f"{len(assembly.lines)} lines")
        except Exception as exc:  # noqa: BLE001 - top-level job guard
            traceback.print_exc()
            self.store.update_job(job.id, status=JobStatus.ERROR, message=str(exc)[:300])

    def _progress(self, job: Job, status: JobStatus, message: str) -> None:
        self.store.update_job(job.id, status=status, message=message)

    def _say(self, job: Job, message: str) -> None:
        self.store.update_job(job.id, message=message)


pipeline = Pipeline(store)


def ensure_audio_extension(path: Path) -> Path:
    """No-op helper kept for future format handling."""
    return path


def delete_track_files(track_id: str) -> None:
    """Remove audio + transcript files when a track is deleted."""
    track = store.get_track(track_id)
    if not track:
        return
    if track.audio_file and track.audio_file.exists():
        track.audio_file.unlink(missing_ok=True)
    tp = cfg.DATA_DIR / "transcripts" / f"{track_id}.json"
    tp.unlink(missing_ok=True)
    store.delete_track(track_id)


def new_track_placeholder() -> Line:
    return Line(text="", start=0.0, progress=0.0, words=[])
