"""Background job pipeline: download -> transcribe -> assemble -> persist."""
from __future__ import annotations

import shutil
import threading
import traceback
import uuid
from pathlib import Path

import app.config as cfg
from app import downloader, exporters, formatter, transcribe
from app.models import Job, JobStatus, Line, Store, Track
from app.settings import settings

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
            if job.kind == "retranscribe":
                self._run_retranscribe(job)
                return
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

            result, lines, llm_note = self._transcribe_and_format(
                job, dl.audio_path)

            version = self.store.save_transcript(track.id, lines,
                                                 backend=result.backend)
            self.store.update_job(job.id, status=JobStatus.DONE,
                                  message=f"Done via {result.backend} · "
                                          f"{len(lines)} lines{llm_note} · v{version}")
        except Exception as exc:  # noqa: BLE001 - top-level job guard
            traceback.print_exc()
            self.store.update_job(job.id, status=JobStatus.ERROR, message=str(exc)[:300])

    def _transcribe_and_format(self, job: Job, audio_path: Path):
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
        result = backend.transcribe(audio_path, lambda m: self._say(job, m))

        self._progress(job, JobStatus.ALIGNING, "Building synced lines…")
        llm_note = ""
        used_model = model_id or settings.get().get("cloud_model") or ""
        spec = get_model(used_model)
        audio_llm = bool(spec and spec.kind == "audio_llm")
        if audio_llm or not settings.get()["llm_format"]:
            # audio LLMs already return formatted, synced lines; and when
            # formatting is disabled we assemble from timings directly
            assembly = exporters.assemble_lines(result)
            lines = assembly.lines
            if audio_llm:
                llm_note = " · formatted by audio model"
        else:
            from app.models import Word
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
        result, lines, llm_note = self._transcribe_and_format(job, track.audio_file)
        note = f"retranscribed via {result.backend}"
        version = self.store.save_transcript(track.id, lines,
                                             backend=result.backend, note=note)
        self.store.update_job(job.id, status=JobStatus.DONE,
                              message=f"Done via {result.backend} · "
                                      f"{len(lines)} lines{llm_note} · v{version} (kept v{version - 1})")

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
    shutil.rmtree(cfg.DATA_DIR / "transcripts" / track_id, ignore_errors=True)
    (cfg.DATA_DIR / "transcripts" / f"{track_id}.json").unlink(missing_ok=True)
    store.delete_track(track_id)


def new_track_placeholder() -> Line:
    return Line(text="", start=0.0, progress=0.0, words=[])
