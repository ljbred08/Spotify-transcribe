"""Cancellation, terminal-state guard, and the timeout setting."""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models import Job, JobStatus, Store

import app.config as cfg
from app.settings import Settings


def fresh_store(tmp_path):
    import app.config as cfg
    old = cfg.DATA_DIR
    cfg.DATA_DIR = tmp_path
    try:
        s = Store()
        yield s
    finally:
        cfg.DATA_DIR = old


def test_cancel_queued_job_ends_it_immediately(tmp_path):
    store = next(fresh_store(tmp_path))
    job = store.create_job("https://x/song")
    store.request_cancel(job.id)
    got = store.get_job(job.id)
    assert got.cancel_requested is True
    assert got.status is JobStatus.ERROR
    assert "Canceled" in got.message
    # queue no longer hands it out
    assert store.pop_next_job() is None


def test_cancel_running_job_sets_flag_and_canceling_message(tmp_path):
    store = next(fresh_store(tmp_path))
    job = store.create_job("https://x/song")
    store.update_job(job.id, status=JobStatus.TRANSCRIBING)
    store.request_cancel(job.id)
    got = store.get_job(job.id)
    assert got.cancel_requested is True
    assert got.status is JobStatus.TRANSCRIBING
    assert got.message == "Canceling…"


def test_terminal_job_status_is_frozen(tmp_path):
    store = next(fresh_store(tmp_path))
    job = store.create_job("https://x/song")
    store.update_job(job.id, status=JobStatus.ERROR, message="boom")
    # a lagging worker thread must not resurrect the job
    store.update_job(job.id, status=JobStatus.DONE, message="late success")
    got = store.get_job(job.id)
    assert got.status is JobStatus.ERROR
    assert got.message == "boom"


def test_timeout_setting_validation_and_default():
    d = Path(".test-settings-data")
    d.mkdir(parents=True, exist_ok=True)
    old = cfg.DATA_DIR
    cfg.DATA_DIR = d
    try:
        st = Settings()
        assert st.get()["transcribe_timeout"] == 60
        st.update(transcribe_timeout=300)
        assert st.get()["transcribe_timeout"] == 300.0
        try:
            st.update(transcribe_timeout=5)
            raise AssertionError("5s should be rejected")
        except ValueError:
            pass
    finally:
        cfg.DATA_DIR = old
    st.update(transcribe_timeout=300)
    assert st.get()["transcribe_timeout"] == 300.0
    try:
        st.update(transcribe_timeout=5)
        raise AssertionError("5s should be rejected")
    except ValueError:
        pass


def test_pipeline_cancels_between_stages(tmp_path, monkeypatch):
    import app.pipeline as P
    import app.config as cfg

    old = cfg.DATA_DIR
    cfg.DATA_DIR = tmp_path
    try:
        store = Store()
        monkeypatch.setattr(P, "store", store)
        pipe = P.Pipeline.__new__(P.Pipeline)
        pipe.store = store
        job = store.create_job("https://x/song")
        store.update_job(job.id, status=JobStatus.DOWNLOADING)
        store.request_cancel(job.id)
        try:
            pipe._check_canceled(job)
            raise AssertionError("should have raised JobCanceled")
        except P.JobCanceled:
            pass
    finally:
        cfg.DATA_DIR = old
