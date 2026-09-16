"""Background job pipeline: download -> transcribe -> assemble -> persist."""
from __future__ import annotations

import logging
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

import app.config as cfg
from app import downloader, exporters, formatter, recovery, transcribe
from app.models import Job, JobStatus, Line, Store, Track, Word
from app.settings import settings

log = logging.getLogger(__name__)

store = Store()


class JobCanceled(Exception):
    """Raised inside a job thread when cancel is requested or it times out."""


def _groups_usable(groups) -> bool:
    """Do these groups look like lyric lines, or one whole-song blob?

    Sanity guard: some audio models return correct words/timings but no line
    breaks at all (one whole-song "line"). Trust grouping only when it looks
    like lyric lines; otherwise assemble from timings.
    """
    if not groups:
        return False
    total_words = sum(len(g) for g in groups)
    biggest = max(len(g) for g in groups)
    # a real lyric line is rarely >30 words; a blob is most of the song
    return not (len(groups) <= 2 and total_words > 40 and
                biggest > total_words * 0.6)


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
        # anything a crash interrupted *after* the model answered is finished
        # here, without paying for that transcription a second time
        self.recover_interrupted()

    def _start_workers(self) -> None:
        for i in range(cfg.JOB_CONCURRENCY):
            t = threading.Thread(target=self._worker_loop, name=f"worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def submit(self, url: str, *, kind: str = "download",
               track_id: str | None = None) -> Job:
        job = self.store.create_job(url, kind=kind, track_id=track_id)
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
            if job.kind == "improve_sync":
                self._run_improve_sync(job)
                return
            if job.kind == "upload":
                # audio already on disk (uploaded file); transcribe it directly
                self._run_upload(job)
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

            # Fire-and-forget metadata fetch in background: the fingerprint pass
            # needs no lyrics, and the transcription below gets one started early
            self._fetch_metadata_async(track)

            result, lines, llm_note = self._run_transcribe_with_timeout(
                job, dl.audio_path, title=track.title, artist=track.artist)

            version = self.store.save_transcript(track.id, lines,
                                                 backend=result.backend)
            recovery.clear_pending(job.id)     # safely saved: stop parking it
            # Second pass now that the lyrics exist — this is the one that can
            # search Genius by lyric line. Skipped when the first pass found art.
            self._fetch_metadata_async(track, lines=lines)
            self.store.update_job(job.id, status=JobStatus.DONE,
                                  message=f"Done via {result.backend} · "
                                          f"{len(lines)} lines{llm_note} · v{version}")
        except JobCanceled as exc:
            self.store.update_job(job.id, status=JobStatus.ERROR, message=str(exc))
            self._settle_parked(job)
        except Exception as exc:  # noqa: BLE001 - top-level job guard
            traceback.print_exc()
            self.store.update_job(job.id, status=JobStatus.ERROR, message=str(exc)[:300])
            self._settle_parked(job)

    def _fetch_metadata_async(self, track: Track,
                              *, lines: list[Line] | None = None) -> None:
        """Fetch metadata for a track in the background.

        Called twice per download: at submit time without lyrics (Shazam's audio
        fingerprint needs none) and again once the transcript exists, when the
        lyrics open up Genius's lyric search. The second call is skipped when the
        first already produced a cover, so the common path costs one lookup.

        Fire-and-forget either way: a miss (or a missing Genius key) leaves the
        track exactly as the downloader/upload named it. Only the artist is
        filled in when we don't have one — the title is the user's to edit.
        """
        def _run() -> None:
            try:
                import asyncio
                from app.metadata import art_for, fetch_metadata, lyrics_text
                if lines is not None and art_for(track.id):
                    return                      # the pre-transcription pass won
                loop = asyncio.new_event_loop()
                try:
                    result = loop.run_until_complete(fetch_metadata(
                        track.title, track.artist, track_id=track.id,
                        audio_path=track.audio_file,
                        lyrics=lyrics_text(lines) if lines else None))
                finally:
                    loop.close()
                unknown = not track.artist or track.artist == "Unknown artist"
                if result and unknown and result.get("artist"):
                    self.store.update_track(track.id, artist=result["artist"])
            except Exception as exc:  # noqa: BLE001 - never fail the job for this
                log.warning("Metadata fetch failed for %s: %s", track.title, exc)

        t = threading.Thread(target=_run, name=f"meta-{track.id}", daemon=True)
        t.start()

    def _check_canceled(self, job: Job) -> None:
        fresh = self.store.get_job(job.id)
        if fresh and fresh.cancel_requested:
            raise JobCanceled("Canceled")

    def _with_watchdog(self, job: Job, work):
        """Run work() with the cancel/timeout watchdog around it.

        The work runs in this worker thread (can't be killed); the watchdog
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
            # Event.wait returns True when the WORKER finished (event set),
            # False on genuine timeout — inverted from a naive reading
            if fired.wait(timeout=max(0.0, deadline - time.monotonic())):
                return  # worker finished first: nothing to do
            self.store.request_cancel(job.id)
            # marked separately from a user cancel: a timeout keeps whatever the
            # model already produced, because that answer was paid for
            self.store.update_job(job.id, timed_out=True)
            self.store.update_job(
                job.id, status=JobStatus.ERROR,
                message=f"Timed out after {int(limit)}s — the model or "
                        f"provider was too slow. Raise the limit under "
                        f"Advanced, or pick a faster model")

        wt = threading.Thread(target=watchdog, name=f"wd-{job.id}", daemon=True)
        wt.start()
        try:
            return work()
        finally:
            fired.set()   # release the watchdog thread immediately

    def _run_transcribe_with_timeout(self, job: Job, audio_path: Path, *,
                                     title: str, artist: str):
        """Run the transcribe+format stage with a cancel/timeout watchdog."""
        return self._with_watchdog(
            job, lambda: self._transcribe_and_format(
                job, audio_path, title=title, artist=artist))

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
        progress, state = self._parking_progress(job)
        result = backend.transcribe(audio_path, progress,
                                    title=title, artist=artist)
        # The one paid step is done: get its output on disk *before* anything
        # below can throw it away (a timeout, a cancel, a crash).
        self._park(job, "transcribed", backend=result.backend,
                   words=result.words, groups=result.groups)

        self._progress(job, JobStatus.ALIGNING, "Building synced lines…")
        lines, llm_note = self._lines_from_result(result, job)
        # …and the finished lines too, so a resume needs no second paid call
        # even when formatting lines took one.
        self._park(job, "lines", backend=result.backend,
                   lines=recovery.lines_to_payload(lines), note=llm_note)
        if state["canceled"] and not state["timed_out"]:
            # a timeout keeps going on purpose: the answer is already paid for
            raise JobCanceled("Canceled")
        return result, lines, llm_note

    def _lines_from_result(self, result, job: Job) -> tuple[list[Line], str]:
        """Group a transcription result into lyric lines (+ how that was done).

        Split out of the transcribe stage so a *resumed* job can apply exactly
        the same rules to a result that was parked on disk.
        """
        from app.cloud_models import get_model

        job_now = self.store.get_job(job.id) or job
        llm_note = ""
        used_model = job_now.retranscribe_model
        if not used_model:
            from app.cloud_models import resolved_transcribe_model
            used_model = resolved_transcribe_model()
        spec = get_model(used_model)
        audio_llm = bool(spec and spec.kind == "audio_llm")
        # a run the watchdog already gave up on does not get to buy another LLM
        # call for formatting: assemble from timings instead (free)
        timed_out = bool(job_now.timed_out)

        if result.groups and _groups_usable(result.groups):
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
        elif audio_llm or timed_out or not settings.get()["llm_format"]:
            # grouped results without preformed lines (e.g. validation dropped
            # a whole line) and formatting-disabled runs assemble from timings
            assembly = exporters.assemble_lines(result)
            lines = assembly.lines
            if audio_llm:
                llm_note = " · formatted by audio model"
            elif timed_out:
                llm_note = " · assembled from timings (the run had timed out)"
        else:
            words = [Word(text=w["text"], start=w["start"], end=w["end"])
                     for w in result.words]
            self._progress(job, JobStatus.ALIGNING, "Formatting lyrics with LLM…")
            fr = formatter.format_words(words)
            lines = fr.lines
            llm_note = f" · {fr.llm_lines} LLM lines"
            if fr.fallback_lines:
                llm_note += f" ({fr.fallback_lines} auto)"
        return lines, llm_note

    def _run_upload(self, job: Job) -> None:
        """Transcribe an uploaded file — audio exists, no download stage."""
        track = self.store.get_track(job.track_id or "")
        if not track or not track.audio_file or not track.audio_file.exists():
            raise transcribe.TranscribeError("Uploaded audio is missing")
        self._progress(job, JobStatus.TRANSCRIBING, "Reading uploaded audio…")
        self._check_canceled(job)
        result, lines, llm_note = self._run_transcribe_with_timeout(
            job, track.audio_file, title=track.title, artist=track.artist)
        version = self.store.save_transcript(track.id, lines,
                                             backend=result.backend)
        recovery.clear_pending(job.id)
        # Uploads have no download-time pass, so this IS their metadata fetch —
        # and by now the lyrics exist, so Genius's lyric search is available.
        self._fetch_metadata_async(track, lines=lines)
        self.store.update_job(job.id, status=JobStatus.DONE,
                              message=f"Done via {result.backend} · "
                                      f"{len(lines)} lines{llm_note} · v{version}")

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
        recovery.clear_pending(job.id)
        self.store.update_job(job.id, status=JobStatus.DONE,
                              message=f"Done via {result.backend} · "
                                      f"{len(lines)} lines{llm_note} · v{version} (kept v{version - 1})")

    def _run_improve_sync(self, job: Job) -> None:
        """Retime the ACTIVE transcript with a transcription-only model.

        Words and line breaks come from the current (good) transcript;
        timings come from the reference run — a transcribe-only backend's
        measured timestamps instead of an omni model's generated ones.
        """
        from app.sync import improve_sync
        track = self.store.get_track(job.track_id or "")
        if not track or not track.audio_file or not track.audio_file.exists():
            raise transcribe.TranscribeError("Track audio is missing — cannot improve sync")
        current = self.store.load_transcript(track.id)
        if not current:
            raise transcribe.TranscribeError("Track has no transcript to improve")

        # 1) reference run: transcription-only backend, no formatting stage
        job_now = self.store.get_job(job.id) or job
        backend, note = transcribe.pick_backend(
            backend=job_now.retranscribe_backend, model=job_now.retranscribe_model)
        if backend is None:
            raise transcribe.TranscribeError(note)
        progress, state = self._parking_progress(job)
        result = self._with_watchdog(
            job, lambda: backend.transcribe(
                track.audio_file, progress,
                title=track.title, artist=track.artist))
        # the reference pass is a paid call too — its words are all the aligning
        # below needs, so park them before anything can throw them away
        self._park(job, "sync_reference", backend=result.backend,
                   words=result.words)
        if state["canceled"] and not state["timed_out"]:
            raise JobCanceled("Canceled")

        # 2) retime the good transcript from the reference's word timings
        self._progress(job, JobStatus.ALIGNING,
                       "Aligning reference timings to the current lyrics…")
        new_lines, stats = improve_sync(current, result.words)
        # match accuracy: fraction of the transcript's words the reference
        # transcription found (fuzzy >= 0.5). Higher = the two models agreed,
        # so the retimed timings are trustworthy; low = reference misheard a
        # lot and timings may be worse than before.
        pct = round(stats["coverage"] * 100)
        note = (f"sync improved via {result.backend} "
                f"({pct}% match · {stats['matched']}/{stats['old_words']} matched, "
                f"{stats['forced']} forced, {stats['kept']} kept, "
                f"{stats['dropped']} dropped)")
        # the words/lines still come from the active transcript's model —
        # record THAT as the backend (keeps omni-lineage versions improvable
        # and visible under Post-processing); the reference goes in the note
        backend_label = self._sync_backend_label(track.id, result.backend)
        version = self.store.save_transcript(track.id, new_lines,
                                             backend=backend_label, note=note,
                                             match=pct)
        recovery.clear_pending(job.id)
        self.store.update_job(
            job.id, status=JobStatus.DONE,
            message=f"Sync improved · {stats['matched']}/{stats['old_words']} "
                    f"words retimed · {pct}% match "
                    f"({stats['kept']} kept, {stats['dropped']} dropped) · v{version}")

    def _sync_backend_label(self, track_id: str, reference_backend: str) -> str:
        """Whose words an improve-sync version holds — the active transcript's."""
        idx = self.store.list_versions(track_id) or {"versions": []}
        active_v = next((v for v in idx["versions"]
                         if v["version"] == idx.get("active")), None)
        return (active_v or {}).get("backend") or reference_backend

    # -- crash safety for paid calls -----------------------------------------
    def _parking_progress(self, job: Job):
        """A progress callback for a paid call that records but never raises.

        Raising from inside a backend's callback is how a cancel/timeout stops
        a run — but it can fire *after* the model answered, which would throw
        away a result we have already been billed for. So the callback only
        notes the request; the caller secures the result and then honours it.
        """
        state = {"canceled": False, "timed_out": False}

        def callback(message: str) -> None:
            fresh = self.store.get_job(job.id)
            if fresh:
                state["canceled"] = state["canceled"] or fresh.cancel_requested
                state["timed_out"] = state["timed_out"] or fresh.timed_out
            self.store.update_job(job.id, message=message)

        return callback, state

    def _park(self, job: Job, stage: str, **payload) -> None:
        """Put a paid result on disk before it can be lost."""
        fresh = self.store.get_job(job.id) or job
        recovery.save_pending(job.id, stage, trackId=fresh.track_id,
                              kind=fresh.kind, **payload)

    def _settle_parked(self, job: Job) -> None:
        """A job ended badly: keep a paid result if one arrived, else discard it.

        Only an explicit user cancel discards anything — a timeout or any other
        failure still saves whatever the model produced, because that answer is
        already paid for. Runs in an exception handler, so it must not raise:
        throwing here would take the worker thread down with it.
        """
        try:
            if not recovery.has_pending(job.id):
                return
            fresh = self.store.get_job(job.id) or job
            if fresh.cancel_requested and not fresh.timed_out:
                log.info("Job %s was canceled by the user — dropping the parked "
                         "result", job.id)
                recovery.clear_pending(job.id)
                return
            why = ("recovered after a timeout" if fresh.timed_out
                   else "recovered after the job failed")
            log.warning("Job %s ended badly but the model had answered — "
                        "keeping that result", job.id)
            self._recover_one(fresh, why=why)
        except Exception as exc:  # noqa: BLE001 - never kill the worker thread
            log.warning("Could not settle the parked result of job %s: %s",
                        job.id, exc)

    def recover_interrupted(self) -> int:
        """Finish jobs whose paid result outlived a crash. Called on startup."""
        ids = recovery.pending_job_ids()
        if not ids:
            return 0
        log.warning("Found %d job(s) with an unsaved transcription — resuming",
                    len(ids))
        t = threading.Thread(target=self._recover_all, args=(ids,),
                             name="recover", daemon=True)
        t.start()
        return len(ids)

    def _recover_all(self, ids: list[str]) -> None:
        for job_id in ids:
            try:
                job = self.store.get_job(job_id)
                if job is None:
                    recovery.clear_pending(job_id)   # no job left to attach it to
                    continue
                if not self._recover_one(job):
                    # nothing could be saved (its track or transcript is gone),
                    # but the job still must not sit mid-flight forever
                    self._end_unfinished(job_id)
            except Exception as exc:  # noqa: BLE001 - a resume must not kill boot
                log.warning("Could not resume job %s: %s", job_id, exc)
                recovery.clear_pending(job_id)
                self._end_unfinished(job_id, detail=str(exc))

    def _end_unfinished(self, job_id: str, *, detail: str = "") -> None:
        """Finish off a job a resume could not complete, so nothing hangs."""
        job = self.store.get_job(job_id)
        if job is None or job.status in (JobStatus.DONE, JobStatus.ERROR):
            return
        message = "Interrupted by restart — there was nothing left to resume"
        if detail:
            message += f" ({detail})"
        self.store.update_job(job_id, status=JobStatus.ERROR, message=message[:300])

    def _recover_one(self, job: Job, *, why: str = "recovered after restart") -> bool:
        """Persist the transcript behind a parked result. True if one was saved.

        Every stage can be finished from what is on disk: saved lines are used
        as they are, raw words go through the same grouping the live run would
        have used, and an improve-sync reference needs only local aligning.
        """
        payload = recovery.load_pending(job.id)
        if payload is None or job.status is JobStatus.DONE:
            recovery.clear_pending(job.id)   # already saved: nothing to redo
            return False
        track_id = payload.get("trackId") or job.track_id or ""
        track = self.store.get_track(track_id)
        if track is None:
            recovery.clear_pending(job.id)
            return False

        stage = payload.get("stage")
        backend = payload.get("backend")
        pct = None
        if stage == "sync_reference":
            from app.sync import improve_sync
            current = self.store.load_transcript(track_id)
            if not current:
                recovery.clear_pending(job.id)
                return False
            lines, stats = improve_sync(current, payload.get("words") or [])
            pct = round(stats["coverage"] * 100)
            backend = self._sync_backend_label(track_id, backend or "")
            llm_note = f" · {pct}% match"
        elif stage == "lines":
            lines = recovery.lines_from_payload(payload.get("lines"))
            llm_note = payload.get("note") or ""
        else:                                   # "transcribed": raw words only
            result = transcribe.TranscriptionResult(
                segments=[], words=payload.get("words") or [],
                backend=backend or "", groups=payload.get("groups"))
            lines, llm_note = self._lines_from_result(result, job)

        if not lines:
            recovery.clear_pending(job.id)
            return False
        version = self.store.save_transcript(track_id, lines, backend=backend,
                                             note=why, match=pct)
        recovery.clear_pending(job.id)
        self.store.update_job(
            job.id, status=JobStatus.DONE,
            message=f"{why.capitalize()} — the paid transcription was kept · "
                    f"{len(lines)} lines{llm_note} · v{version}")
        log.warning("Recovered %d line(s) for track %s from job %s (v%d)",
                    len(lines), track_id, job.id, version)
        return True

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
