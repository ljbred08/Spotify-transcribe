"""A paid API result must never be lost.

The rule under test: the moment a paid call returns, its output is on disk, and
whatever happens next — a crash, a restart, a watchdog timeout — a job can be
finished from that file without paying for the same transcription twice.
"""
import json
import time

import pytest

import app.config as cfg
from app import recovery, transcribe
from app.models import JobStatus, Line, Store, Track, Word
from app.pipeline import pipeline, store


@pytest.fixture
def pending(tmp_path, monkeypatch):
    """Park results in a temp dir instead of the real one."""
    monkeypatch.setattr(recovery, "PENDING_DIR", tmp_path / "pending")
    return tmp_path / "pending"


WORDS = [{"text": "Hello", "start": 0.0, "end": 0.4},
         {"text": "world", "start": 0.5, "end": 0.9}]


def _add_job(track_id: str, *, kind: str = "download") -> "object":
    job = store.create_job("u", kind=kind, track_id=track_id)
    store.update_job(job.id, status=JobStatus.TRANSCRIBING)
    return job


def _add_track(track_id: str, *, lines: list[Line] | None = None) -> Track:
    track = Track(id=track_id, url="u", title="T", artist="A")
    store.add_track(track)
    if lines is not None:
        store.save_transcript(track_id, lines, backend="seed")
    return track


def _lines(text: str = "Hello world") -> list[Line]:
    words = text.split()
    return [Line(text=text, start=0.0, progress=0.0,
                 words=[Word(t, i * 0.5, i * 0.5 + 0.4)
                        for i, t in enumerate(words)])]


def _versions(track_id: str) -> list[dict]:
    return (store.list_versions(track_id) or {"versions": []})["versions"]


# --- the parking mechanism ----------------------------------------------------

def test_a_parked_result_round_trips_and_leaves_no_temp_file(pending):
    recovery.save_pending("job-rt", "transcribed", words=WORDS, trackId="t1")

    assert recovery.has_pending("job-rt")
    payload = recovery.load_pending("job-rt")
    assert payload["stage"] == "transcribed"
    assert payload["jobId"] == "job-rt" and payload["trackId"] == "t1"
    assert payload["words"] == WORDS
    # atomic: the temp file was renamed into place, not left behind
    assert list(pending.glob("*.tmp")) == []

    recovery.clear_pending("job-rt")
    assert not recovery.has_pending("job-rt")
    assert recovery.load_pending("job-rt") is None


def test_pending_ids_are_listed_oldest_first(pending):
    recovery.save_pending("first", "transcribed", words=WORDS)
    recovery.save_pending("second", "lines", lines=[])

    assert recovery.pending_job_ids() == ["first", "second"]

    recovery.clear_pending("first")
    recovery.clear_pending("second")


def test_a_progress_callback_never_raises_on_cancel(pending):
    """A cancel must not be able to throw away a result mid-call."""
    _add_track("rec-cb")
    job = _add_job("rec-cb")
    try:
        callback, state = pipeline._parking_progress(job)
        store.request_cancel(job.id)

        callback("Transcribing…")          # must not raise JobCanceled

        assert state["canceled"] is True
        assert state["timed_out"] is False
        assert store.get_job(job.id).message == "Transcribing…"
    finally:
        store.delete_track("rec-cb")


def test_the_transcribe_stage_parks_both_stages(pending, monkeypatch, tmp_path):
    """The real stage leaves the paid result on disk, lines included."""
    words = [dict(w) for w in WORDS]
    seen: dict = {}

    class FakeBackend:
        name = "fake"

        def transcribe(self, audio_path, progress, *, title="", artist=""):
            progress("Transcribing…")
            return transcribe.TranscriptionResult(segments=[], words=words,
                                                  backend="fake-model",
                                                  groups=[words])

    def fake_pick(*, backend=None, model=None):
        return FakeBackend(), "fake model"

    monkeypatch.setattr(transcribe, "pick_backend", fake_pick)
    monkeypatch.setattr(pipeline.store, "get_job", store.get_job)
    track = _add_track("rec-park")
    job = _add_job("rec-park")
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"\x00" * 16)
    try:
        result, lines, note = pipeline._transcribe_and_format(
            job, audio, title="T", artist="A")

        payload = recovery.load_pending(job.id)
        assert result.backend == "fake-model"
        assert payload["stage"] == "lines"          # the finished lines too
        assert payload["backend"] == "fake-model"
        assert [d["text"] for d in payload["lines"]] == ["Hello world"]
        assert payload["lines"][0]["words"][0] == {"text": "Hello", "start": 0.0,
                                                   "end": 0.4}
        assert seen == {}
        assert [l.text for l in lines] == ["Hello world"]
    finally:
        recovery.clear_pending(job.id)
        store.delete_track("rec-park")


# --- resume -------------------------------------------------------------------

def test_recovery_finishes_a_job_from_parked_words(pending):
    _add_track("rec1")
    job = _add_job("rec1")
    recovery.save_pending(job.id, "transcribed", trackId="rec1", backend="paid-model",
                          words=WORDS, groups=[WORDS])
    try:
        assert pipeline._recover_one(job) is True

        lines = store.load_transcript("rec1")
        assert [l.text for l in lines] == ["Hello world"]
        version = _versions("rec1")[-1]
        assert version["backend"] == "paid-model"
        assert version["note"] == "recovered after restart"
        fresh = store.get_job(job.id)
        assert fresh.status is JobStatus.DONE
        assert "paid transcription was kept" in fresh.message
        assert not recovery.has_pending(job.id)      # consumed, not replayed
    finally:
        store.delete_track("rec1")


def test_recovery_reuses_parked_lines_without_formatting_again(pending, monkeypatch):
    """The second paid step (LLM line formatting) is not repeated either."""
    from app import formatter

    def explode(*a, **kw):
        raise AssertionError("recovery must not re-run the formatting LLM")

    monkeypatch.setattr(formatter, "format_words", explode)
    _add_track("rec2")
    job = _add_job("rec2")
    recovery.save_pending(job.id, "lines", trackId="rec2", backend="b",
                          lines=recovery.lines_to_payload(_lines("From the file")),
                          note=" · 2 LLM lines")
    try:
        assert pipeline._recover_one(job) is True
        assert [l.text for l in store.load_transcript("rec2")] == ["From the file"]
        assert "2 LLM lines" in store.get_job(job.id).message
    finally:
        store.delete_track("rec2")


def test_an_improve_sync_job_is_finished_from_its_reference_words(pending):
    """The reference pass is a paid call; the aligning it needs is local."""
    seed = _lines("Hello world")
    _add_track("rec3", lines=seed)
    job = _add_job("rec3", kind="improve_sync")
    ref = [{"text": "Hello", "start": 10.0, "end": 10.4},
           {"text": "world", "start": 10.5, "end": 10.9}]
    recovery.save_pending(job.id, "sync_reference", trackId="rec3",
                          backend="paid-whisper", words=ref)
    try:
        assert pipeline._recover_one(job) is True

        lines = store.load_transcript("rec3")
        assert [l.text for l in lines] == ["Hello world"]
        assert lines[0].words[0].start == pytest.approx(10.0)   # retimed
        assert store.get_job(job.id).status is JobStatus.DONE
        assert not recovery.has_pending(job.id)
    finally:
        store.delete_track("rec3")


def test_recovery_is_idempotent_and_skips_a_finished_job(pending):
    _add_track("rec4")
    job = _add_job("rec4")
    store.update_job(job.id, status=JobStatus.DONE, message="Done")
    recovery.save_pending(job.id, "transcribed", trackId="rec4", backend="b",
                          words=WORDS, groups=[WORDS])
    try:
        assert pipeline._recover_one(store.get_job(job.id)) is False
        assert _versions("rec4") == []               # no duplicate version
        assert not recovery.has_pending(job.id)      # and no stale file
    finally:
        store.delete_track("rec4")


def test_recovery_drops_a_result_whose_track_is_gone(pending):
    job = _add_job("no-such-track")
    recovery.save_pending(job.id, "transcribed", trackId="no-such-track",
                          backend="b", words=WORDS)
    try:
        assert pipeline._recover_one(job) is False
        assert not recovery.has_pending(job.id)
    finally:
        store.delete_track("no-such-track")


def test_recovery_drops_a_result_whose_job_is_gone(pending):
    recovery.save_pending("ghost-job", "transcribed", words=WORDS)

    pipeline._recover_all(["ghost-job"])

    assert not recovery.has_pending("ghost-job")


def test_a_resume_with_nothing_to_save_ends_the_job_instead_of_hanging(pending):
    """An improve-sync job for a track without a transcript cannot be resumed —
    but it must not sit at "transcribing" forever either."""
    _add_track("rec5")
    job = _add_job("rec5", kind="improve_sync")
    recovery.save_pending(job.id, "sync_reference", trackId="rec5", backend="b",
                          words=[{"text": "x", "start": 0.0, "end": 1.0}])
    try:
        pipeline._recover_all([job.id])
        fresh = store.get_job(job.id)
        assert fresh.status is JobStatus.ERROR
        assert "nothing left to resume" in fresh.message
        assert not recovery.has_pending(job.id)
    finally:
        store.delete_track("rec5")


def test_a_resume_that_explodes_ends_the_job_with_the_reason(pending, monkeypatch):
    _add_track("rec8")
    job = _add_job("rec8")
    recovery.save_pending(job.id, "lines", trackId="rec8", backend="b",
                          lines=[{"text": "hi", "start": 0, "words": []}])

    def boom(*a, **kw):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(store, "save_transcript", boom)
    try:
        pipeline._recover_all([job.id])
        fresh = store.get_job(job.id)
        assert fresh.status is JobStatus.ERROR
        assert "disk on fire" in fresh.message
        assert not recovery.has_pending(job.id)
    finally:
        monkeypatch.undo()
        store.delete_track("rec8")


def test_startup_recovery_reports_what_it_found(pending, monkeypatch):
    """Boot hands the whole backlog to the resume pass, off the boot thread."""
    assert pipeline.recover_interrupted() == 0

    seen: list[list[str]] = []
    monkeypatch.setattr(pipeline, "_recover_all", lambda ids: seen.append(sorted(ids)))
    recovery.save_pending("boot-1", "transcribed", words=WORDS)
    recovery.save_pending("boot-2", "lines", lines=[])

    assert pipeline.recover_interrupted() == 2
    deadline = time.time() + 5
    while not seen and time.time() < deadline:
        time.sleep(0.02)
    assert seen == [["boot-1", "boot-2"]]      # oldest first, both together

    recovery.clear_pending("boot-1")
    recovery.clear_pending("boot-2")


# --- timeouts vs. explicit cancels --------------------------------------------

def test_a_timeout_keeps_the_result_the_model_already_produced(pending):
    _add_track("rec6")
    job = _add_job("rec6")
    # the watchdog gave up while the model was still thinking…
    store.update_job(job.id, timed_out=True)
    store.update_job(job.id, status=JobStatus.ERROR, message="Timed out after 60s")
    # …and the answer arrived anyway
    recovery.save_pending(job.id, "transcribed", trackId="rec6", backend="slow-model",
                          words=WORDS, groups=[WORDS])
    try:
        pipeline._settle_parked(store.get_job(job.id))

        assert [l.text for l in store.load_transcript("rec6")] == ["Hello world"]
        assert _versions("rec6")[-1]["note"] == "recovered after a timeout"
        assert not recovery.has_pending(job.id)
    finally:
        store.delete_track("rec6")


def test_any_other_failure_also_keeps_the_result(pending):
    """Only a user cancel discards a paid result; a plain failure keeps it."""
    _add_track("rec9")
    job = _add_job("rec9")
    store.update_job(job.id, status=JobStatus.ERROR, message="boom")
    recovery.save_pending(job.id, "transcribed", trackId="rec9", backend="b",
                          words=WORDS, groups=[WORDS])
    try:
        pipeline._settle_parked(store.get_job(job.id))

        assert [l.text for l in store.load_transcript("rec9")] == ["Hello world"]
        assert _versions("rec9")[-1]["note"] == "recovered after the job failed"
        assert not recovery.has_pending(job.id)
    finally:
        store.delete_track("rec9")


def test_settling_never_raises_out_of_the_exception_handler(pending, monkeypatch):
    """Throwing here would take the worker thread down with it."""
    _add_track("rec10")
    job = _add_job("rec10")
    recovery.save_pending(job.id, "lines", trackId="rec10", backend="b",
                          lines=[{"text": "hi", "start": 0, "words": []}])

    def boom(*a, **kw):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(store, "save_transcript", boom)
    try:
        pipeline._settle_parked(store.get_job(job.id))   # must not raise
    finally:
        monkeypatch.undo()
        recovery.clear_pending(job.id)
        store.delete_track("rec10")


def test_an_explicit_cancel_drops_the_result(pending):
    """Stop means stop: a user cancel is not resurrected behind their back."""
    _add_track("rec7")
    job = _add_job("rec7")
    store.request_cancel(job.id)                 # user pressed Stop
    recovery.save_pending(job.id, "transcribed", trackId="rec7", backend="b",
                          words=WORDS, groups=[WORDS])
    try:
        pipeline._settle_parked(store.get_job(job.id))

        assert _versions("rec7") == []
        assert not recovery.has_pending(job.id)
    finally:
        store.delete_track("rec7")


# --- boot, with the real Store -------------------------------------------------

def test_store_resumes_an_interrupted_job_only_when_a_result_is_parked(
        pending, tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    library = tmp_path / "library.json"
    library.write_text(json.dumps({
        "tracks": [],
        "jobs": [{"id": "crash6", "url": "u", "status": "transcribing",
                  "message": "Transcribing…"}],
    }), encoding="utf-8")

    # nothing parked: written off, as before
    written_off = Store().get_job("crash6")
    assert written_off.status is JobStatus.ERROR
    assert written_off.message == "Interrupted by restart"

    recovery.save_pending("crash6", "transcribed", trackId="rec-x", words=WORDS)

    # a paid result is parked: the job stays alive for the pipeline to finish
    resumed = Store().get_job("crash6")
    assert resumed.status is JobStatus.TRANSCRIBING
    assert resumed.message == "Resuming after restart…"

    recovery.clear_pending("crash6")
