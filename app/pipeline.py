"""Background job pipeline: download -> transcribe -> assemble -> persist."""
from __future__ import annotations

import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

import app.config as cfg
from app import downloader, exporters, formatter, transcribe
from app.models import Job, JobStatus, Line, Store, Track, Word
from app.settings import settings

store = Store()


class JobCanceled(Exception):
    """Raised inside a job thread when cancel is requested or it times out."""


def _transcribe_timeout() -> float:
    """Watchdog limit for the whole transcribe stage (seconds)."""
    try:
        return max(10.0, float(settings.get().get("transcribe_timeout", 60)))
    except (TypeError, ValueError):
        return 60.0


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
            if job.kind == "retranscribe":
                self._run_retranscribe(job)
                return
            self._progress(job, JobStatus.DOWNLOADING, "Resolving track…")
            self._check_canceled(job)
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

            result, lines, llm_note = self._run_transcribe_with_timeout(
                job, dl.audio_path, title=track.title, artist=track.artist)

            version = self.store.save_transcript(track.id, lines,
                                                 backend=result.backend)
            self.store.update_job(job.id, status=JobStatus.DONE,
                                  message=f"Done via {result.backend} · "
                                          f"{len(lines)} lines{llm_note} · v{version}")
        except JobCanceled as exc:
            self.store.update_job(job.id, status=JobStatus.ERROR, message=str(exc))
        except Exception as exc:  # noqa: BLE001 - top-level job guard
            traceback.print_exc()
            self.store.update_job(job.id, status=JobStatus.ERROR, message=str(exc)[:300])

    def _check_canceled(self, job: Job) -> None:
        fresh = self.store.get_job(job.id)
        if fresh and fresh.cancel_requested:
            raise JobCanceled("Canceled")

    def _run_transcribe_with_timeout(self, job: Job, audio_path: Path, *,
                                     title: str, artist: str):
        """Run the transcribe+format stage with a cancel/timeout watchdog.

        The stage runs in a worker thread (can't be killed); the watchdog
        marks cancel_requested on expiry, then reports the timeout error
        itself. The worker either notices the flag at the next progress
        callback, or its HTTP call eventually returns and its write is
        discarded by the store's terminal-state guard. Either way the job
        ends exactly once, with the watchdog's message.
        """
        limit = _transcribe_timeout()
        deadline = time.monotonic() + limit
        fired = threading.Event()

        def watchdog() -> None:
            if not fired.wait(timeout=max(0.0, deadline - time.monotonic())):
                return  # worker finished first
            self.store.request_cancel(job.id)
            self.store.update_job(
                job.id, status=JobStatus.ERROR,
                message=f"Timed out after {int(limit)}s — the model or "
                        f"provider was too slow. Raise the limit under "
                        f"Advanced, or pick a faster model")

        wt = threading.Thread(target=watchdog, name=f"wd-{job.id}", daemon=True)
        wt.start()
        try:
            result, lines, llm_note = self._transcribe_and_format(
                job, audio_path, title=title, artist=artist)
            return result, lines, llm_note
        finally:
            fired.set()   # release the watchdog thread immediately

    def _transcribe_and_format(self, job: Job, audio_path: Path, *,
                               title: str = "", artist: str = ""):
        """Shared transcription+formatting; returns (result, lines, llm_note)."""
        from app import transcribe as tr
        from app.cloud_models import get_model
        job_now = self.store.get_job(job.id) or job
        backend_name = job_now.retranscribe_backend
        model_id = job_now.retranscribe_model

        backend, note = tr.pick_backend(backend=backend_name, model=model_id)
        if backend is None:
            raise tr.TranscribeError(note)

        self._progress(job, JobStatus.TRANSCRIBING, note)
        result = backend.transcribe(audio_path, lambda m: self._say(job, m),
                                    title=title, artist=artist)

        self._progress(job, JobStatus.ALIGNING, "Building synced lines…")
        llm_note = ""
        used_model = model_id
        if not used_model:
            from app.cloud_models import resolved_transcribe_model
            used_model = resolved_transcribe_model()
        spec = get_model(used_model)
        audio_llm = bool(spec and spec.kind == "audio_llm")
        if result.groups:
            # Audio LLMs return finished lyric lines — keep the model's own
            # grouping verbatim (no re-splitting); timings stay word-exact.
            lines = [
                Line(text=" ".join(w["text"] for w in g),
                     start=round(g[0]["start"], 3), progress=0.0,
                     words=[Word(text=w["text"], start=w["start"], end=w["end"])
                            for w in g])
                for g in result.groups
            ]
            if audio_llm:
                llm_note = " · formatted by audio model"
        elif audio_llm or not settings.get()["llm_format"]:
            # grouped results without preformed lines (e.g. validation dropped
            # a whole line) and formatting-disabled runs assemble from timings
            assembly = exporters.assemble_lines(result)
            lines = assembly.lines
            if audio_llm:
                llm_note = " · formatted by audio model"
        else:
            words = [Word(text=w["text"], start=w["start"], end=w["end"])
                     for w in result.words]
            self._progress(job, JobStatus.ALIGNING, "Formatting lyrics with LLM…")
            fr = formatter.format_words(words)
            lines = fr.lines
            llm_note = f" · {fr.llm_lines} LLM lines"
            if fr.fallback_lines:
                llm_note += f" ({fr.fallback_lines} auto)"
        return result, lines, llm_note

    def _run_retranscribe(self, job: Job) -> None:
        """Re-run transcription on a track's existing audio as a new version."""
        track = self.store.get_track(job.track_id or "")
        if not track or not track.audio_file or not track.audio_file.exists():
            raise transcribe.TranscribeError("Track audio is missing — cannot retranscribe")
        result, lines, llm_note = self._run_transcribe_with_timeout(
            job, track.audio_file, title=track.title, artist=track.artist)
        note = f"retranscribed via {result.backend}"
        version = self.store.save_transcript(track.id, lines,
                                             backend=result.backend, note=note)
        self.store.update_job(job.id, status=JobStatus.DONE,
                              message=f"Done via {result.backend} · "
                                      f"{len(lines)} lines{llm_note} · v{version} (kept v{version - 1})")

    def _progress(self, job: Job, status: JobStatus, message: str) -> None:
        self.store.update_job(job.id, status=status, message=message)

    def _say(self, job: Job, message: str) -> None:
        # called from inside backends between network calls: the natural
        # point to notice a cancel/timeout request mid-stage
        self._check_canceled(job)
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
    shutil.rmtree(cfg.DATA_DIR / "transcripts" / track_id, ignore_errors=True)
    (cfg.DATA_DIR / "transcripts" / f"{track_id}.json").unlink(missing_ok=True)
    store.delete_track(track_id)


def new_track_placeholder() -> Line:
    return Line(text="", start=0.0, progress=0.0, words=[])
