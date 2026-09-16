"""Never lose a paid API result.

A transcription is the one thing in this app that costs real money, and it is
also the slowest step — a minute or more of network time, which is exactly the
window a crash, a Ctrl-C, or a machine restart is most likely to land in. If
the model already answered and we lose that answer, the money is spent twice.

So the rule this module enforces is narrow and absolute: **the moment a paid
call returns, its output is on disk.** ``save_pending`` writes the raw result
(atomically — a crash mid-write leaves either the old file or the new one,
never a half-written one), and it stays there until the transcript has been
saved for real, at which point ``clear_pending`` removes it. On boot the
pipeline resumes anything still parked (see ``Pipeline.recover_interrupted``),
which finishes the job without paying again.

Stages, in the order the pipeline reaches them:

* ``transcribed`` — words + the backend's own line groups. The expensive one.
* ``lines`` — the finished lyric lines, once grouping/formatting is done.
* ``sync_reference`` — the retiming run's words for an *improve sync* job.

Nothing here imports the app's models at module scope: ``models.Store`` needs
``has_pending`` while loading, and a cycle would make the import order matter.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import app.config as cfg

log = logging.getLogger(__name__)

PENDING_DIR = cfg.DATA_DIR / "pending"


def _path(job_id: str) -> Path:
    safe = "".join(c for c in str(job_id) if c.isalnum() or c in "-_")
    return PENDING_DIR / f"{safe}.json"


def save_pending(job_id: str, stage: str, **payload: Any) -> None:
    """Park a paid result under ``job_id``, atomically.

    The write goes to a temp file in the same directory and is then renamed,
    so a reader (or a crash) never sees a truncated JSON document.
    """
    PENDING_DIR.mkdir(parents=True, exist_ok=True)
    body = {"jobId": job_id, "stage": stage, "at": time.time(), **payload}
    target = _path(job_id)
    tmp = target.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, target)
    except OSError as exc:            # a full disk must not fail the job
        log.warning("Could not park the pending result for job %s: %s", job_id, exc)
        tmp.unlink(missing_ok=True)


def load_pending(job_id: str) -> dict | None:
    p = _path(job_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def has_pending(job_id: str) -> bool:
    return _path(job_id).exists()


def clear_pending(job_id: str) -> None:
    """Drop the parked result — the transcript is saved, or it is being discarded."""
    _path(job_id).unlink(missing_ok=True)


def pending_job_ids() -> list[str]:
    """Job ids with a parked result, oldest first."""
    if not PENDING_DIR.exists():
        return []
    out = []
    for p in PENDING_DIR.glob("*.json"):
        if not p.name.endswith(".json.tmp"):
            out.append(p.stem)
    return sorted(out, key=lambda jid: _path(jid).stat().st_mtime)


# --- (de)serialising the app's line model -------------------------------------
# Kept here so the pipeline can park a result without a second conversion step,
# and so what lands on disk is plain JSON a human can read while debugging.

def lines_to_payload(lines) -> list[dict]:
    return [{"text": l.text, "start": l.start,
             "words": [{"text": w.text, "start": w.start, "end": w.end}
                       for w in (l.words or [])]}
            for l in lines or []]


def lines_from_payload(raw) -> list:
    from app.models import Line, Word

    return [Line(text=d.get("text", ""), start=float(d.get("start", 0.0)),
                 progress=0.0,
                 words=[Word(text=w.get("text", ""), start=float(w.get("start", 0.0)),
                             end=float(w.get("end", 0.0)))
                        for w in d.get("words") or []])
            for d in raw or []]
