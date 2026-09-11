"""Transcription backends: faster-whisper locally, Groq/OpenAI in the cloud.

All backends return a unified list of word dicts: {text, start, end}.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import httpx

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

    def transcribe(self, audio_path: Path, progress: ProgressFn) -> TranscriptionResult: ...


# --- local faster-whisper ----------------------------------------------------

class LocalWhisper:
    name = "local"

    def __init__(self) -> None:
        self._model = None
        self._model_name: str | None = None

    def available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def _load(self):
        if self._model is not None and self._model_name == cfg.WHISPER_MODEL:
            return self._model
        from faster_whisper import WhisperModel
        compute = "auto" if cfg.WHISPER_DEVICE == "auto" else cfg.WHISPER_DEVICE
        self._model = WhisperModel(cfg.WHISPER_MODEL, device=compute, compute_type="auto")
        self._model_name = cfg.WHISPER_MODEL
        return self._model

    def transcribe(self, audio_path: Path, progress: ProgressFn) -> TranscriptionResult:
        progress(f"Loading whisper model '{cfg.WHISPER_MODEL}'…")
        model = self._load()
        progress("Transcribing locally (this can take a while)…")
        segments_iter, info = model.transcribe(
            str(audio_path),
            word_timestamps=True,
            vad_filter=True,
        )
        segments: list[Segment] = []
        words: list[dict] = []
        for seg in segments_iter:
            segments.append(Segment(text=seg.text.strip(), start=seg.start, end=seg.end))
            for w in (seg.words or []):
                token = (w.word or "").strip()
                if token:
                    words.append({"text": token, "start": w.start or 0.0, "end": w.end or 0.0})
        return TranscriptionResult(segments=segments, words=words, backend=self.name)


# --- OpenAI-compatible cloud backends (Groq, OpenAI) --------------------------

class OpenAICompatible:
    def __init__(self, name: str, base_url: str, api_key_env: str, model: str) -> None:
        self.name = name
        self._base_url = base_url
        self._api_key = os.environ.get(api_key_env, "")
        self._model = model

    def available(self) -> bool:
        return bool(self._api_key)

    def transcribe(self, audio_path: Path, progress: ProgressFn) -> TranscriptionResult:
        progress(f"Uploading audio to {self.name} ({self._model})…")
        headers = {"Authorization": f"Bearer {self._api_key}"}
        with audio_path.open("rb") as fh:
            files = {"file": (audio_path.name, fh, "audio/mp4")}
            data = {
                "model": self._model,
                "response_format": "verbose_json",
                "timestamp_granularities[]": "word,segment",
            }
            with httpx.Client(timeout=300.0) as client:
                resp = client.post(f"{self._base_url}/audio/transcriptions",
                                   headers=headers, files=files, data=data)
        if resp.status_code != 200:
            raise TranscribeError(f"{self.name} error {resp.status_code}: {resp.text[:300]}")
        payload = resp.json()
        words = [
            {"text": w.get("word", "").strip(), "start": float(w.get("start", 0.0)),
             "end": float(w.get("end", 0.0))}
            for w in payload.get("words", [])
            if w.get("word", "").strip()
        ]
        segments = [
            Segment(text=s.get("text", "").strip(), start=float(s.get("start", 0.0)),
                    end=float(s.get("end", 0.0)))
            for s in payload.get("segments", [])
        ]
        if not words and segments:
            words = _words_from_segments(segments)
        if not words:
            raise TranscribeError(f"{self.name} returned no word timestamps")
        return TranscriptionResult(segments=segments, words=words, backend=self.name)


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
    return [
        OpenAICompatible("groq", "https://api.groq.com/openai/v1",
                         "GROQ_API_KEY", cfg.GROQ_MODEL),
        OpenAICompatible("openai", "https://api.openai.com/v1",
                         "OPENAI_API_KEY", cfg.OPENAI_MODEL),
        LocalWhisper(),
    ]


def pick_backend() -> tuple[Backend | None, str]:
    """Choose a backend honoring TRANSCRIBE_BACKEND; explain the choice."""
    pref = cfg.TRANSCRIBE_BACKEND
    backends = get_backends()
    by_name = {b.name: b for b in backends}
    if pref != "auto":
        b = by_name.get(pref)
        if not b:
            return None, f"Unknown TRANSCRIBE_BACKEND '{pref}'"
        if not b.available():
            return None, f"Backend '{pref}' is not available (missing key or package)"
        return b, f"Using {pref}"
    for b in backends:
        if b.available():
            return b, f"Using {b.name}"
    return None, "No transcription backend available. Install faster-whisper or set an API key."
