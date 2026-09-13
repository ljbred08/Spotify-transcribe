"""Cancellation, terminal-state guard, and the timeout setting."""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models import Job, JobStatus, Store

import app.config as cfg
from app.settings import Settings


def _isolated_dir(name: str) -> Path:
    d = Path(name)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_cancel_queued_job_ends_it_immediately(tmp_path):
    old = cfg.DATA_DIR
    cfg.DATA_DIR = Path(tmp_path)
    try:
        store = Store()
        job = store.create_job("https://x/song")
        store.request_cancel(job.id)
        got = store.get_job(job.id)
        assert got.cancel_requested is True
        assert got.status is JobStatus.ERROR
        assert "Canceled" in got.message
        # queue no longer hands it out
        assert store.pop_next_job() is None
    finally:
        cfg.DATA_DIR = old


def test_cancel_running_job_sets_flag_and_canceling_message(tmp_path):
    old = cfg.DATA_DIR
    cfg.DATA_DIR = Path(tmp_path)
    try:
        store = Store()
        job = store.create_job("https://x/song")
        store.update_job(job.id, status=JobStatus.TRANSCRIBING)
        store.request_cancel(job.id)
        got = store.get_job(job.id)
        assert got.cancel_requested is True
        assert got.status is JobStatus.TRANSCRIBING
        assert got.message == "Canceling…"
    finally:
        cfg.DATA_DIR = old


def test_terminal_job_message_is_frozen():
    d = _isolated_dir(".test-cancel-data")
    old = cfg.DATA_DIR
    cfg.DATA_DIR = d
    try:
        store = Store()
        job = store.create_job("https://x/song")
        store.update_job(job.id, status=JobStatus.TRANSCRIBING)
        # watchdog wins the race: the real timeout message lands first
        store.update_job(job.id, status=JobStatus.ERROR,
                         message="Timed out after 120s")
        # a lagging worker's exception must NOT overwrite it (ERROR→ERROR too)
        store.update_job(job.id, status=JobStatus.ERROR,
                         message="OpenRouter error 400: context length")
        got = store.get_job(job.id)
        assert got.status is JobStatus.ERROR
        assert got.message == "Timed out after 120s"
    finally:
        cfg.DATA_DIR = old


def test_timeout_setting_validation_and_default():
    d = _isolated_dir(".test-cancel-settings")
    old = cfg.DATA_DIR
    cfg.DATA_DIR = d
    try:
        st = Settings()
        assert st.get()["transcribe_timeout"] == 180
        st.update(transcribe_timeout=300)
        assert st.get()["transcribe_timeout"] == 300.0
        try:
            st.update(transcribe_timeout=5)
            raise AssertionError("5s should be rejected")
        except ValueError:
            pass
    finally:
        cfg.DATA_DIR = old


def test_pipeline_cancels_between_stages(tmp_path, monkeypatch):
    import app.pipeline as P

    old = cfg.DATA_DIR
    cfg.DATA_DIR = Path(tmp_path)
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
