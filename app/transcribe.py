"""Transcription backends (cloud) + the shared transcription data model.

All backends return a unified list of word dicts: {text, start, end}.
Backends themselves live in app.cloud_transcribe (transcribers and audio LLMs);
this module keeps the shared types, helpers and the pick_backend entry point.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import app.config as cfg

ProgressFn = Callable[[str], None]


@dataclass
class Segment:
    text: str
    start: float
    end: float


@dataclass
class TranscriptionResult:
    segments: list[Segment]
    words: list[dict]  # {text, start, end}
    backend: str


class TranscribeError(RuntimeError):
    pass


class Backend(Protocol):
    name: str

    def available(self) -> bool: ...

    def transcribe(self, audio_path: Path, progress: ProgressFn,
                   *, model_id: str | None = None) -> TranscriptionResult: ...


def _words_from_segments(segments: list[Segment]) -> list[dict]:
    """Evenly split segment text into pseudo-words when a backend gives none."""
    out: list[dict] = []
    for seg in segments:
        parts = seg.text.split()
        if not parts:
            continue
        span = max(seg.end - seg.start, 0.1)
        step = span / len(parts)
        for i, part in enumerate(parts):
            out.append({"text": part, "start": seg.start + i * step,
                        "end": seg.start + (i + 1) * step})
    return out


# --- registry -----------------------------------------------------------------


def get_backends() -> list[Backend]:
    """Backends in auto-preference order (cloud-only since the local purge)."""
    from app.cloud_transcribe import build
    from app.cloud_models import CATALOG
    out = []
    for m in CATALOG:
        b = build(m.id)
        if b is not None:
            out.append(b)
    return out


def pick_backend(*, backend: str | None = None,
                 model: str | None = None) -> tuple[Backend | None, str]:
    """Choose the cloud backend honoring per-run override or settings."""
    from app.cloud_transcribe import pick
    return pick(model_id=model or backend or None)
