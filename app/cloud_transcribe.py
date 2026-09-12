"""Cloud transcription + audio-LLM backends.

Two families:
- TranscriptProvider (groq/openai whisper-style APIs): returns words with
  timestamps, exactly like the old local backends. Formatting happens later
  (cloud LLM or ollama).
- AudioLLM (Gemini, GPT-4o-audio): listens to the audio and returns fully
  formatted lyric lines WITH word-level timings in one call — no separate
  formatter, no resync. The response is strictly validated and any word that
  can't be mapped back to the audio is dropped, so karaoke sync stays honest.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from pathlib import Path

import httpx

import app.config as cfg
from app.cloud_models import CloudModel, get_model
from app.transcribe import Segment, TranscribeError, TranscriptionResult, _words_from_segments

ProgressFn = "ProgressFn"  # typing alias kept simple


# --- path 2: transcription-only providers --------------------------------------

class TranscriptProvider:
    """Whisper-style HTTP APIs (Groq, OpenAI). Word timestamps when available."""

    def __init__(self, model: CloudModel) -> None:
        self.m = model
        self.name = model.id

    def available(self) -> bool:
        return bool(os.environ.get(self.m.key_env, "").strip())

    def _key(self) -> str:
        return os.environ[self.m.key_env].strip()

    def _base_url(self) -> str:
        if self.m.provider == "groq":
            return "https://api.groq.com/openai/v1"
        return "https://api.openai.com/v1"

    def transcribe(self, audio_path: Path, progress) -> TranscriptionResult:
        progress(f"Uploading audio to {self.m.provider} ({self.m.name})…")
        headers = {"Authorization": f"Bearer {self._key()}"}
        with audio_path.open("rb") as fh:
            files = {"file": (audio_path.name, fh, "audio/mp4")}
            data = {
                "model": self.m.id.split(":", 1)[1],
                "response_format": "verbose_json",
                "timestamp_granularities[]": "word,segment",
            }
            with httpx.Client(timeout=600.0) as client:
                resp = client.post(f"{self._base_url()}/audio/transcriptions",
                                   headers=headers, files=files, data=data)
        if resp.status_code != 200:
            raise TranscribeError(
                f"{self.m.provider} error {resp.status_code}: {resp.text[:300]}")
        payload = resp.json()
        words = [
            {"text": w.get("word", "").strip(), "start": float(w.get("start", 0.0)),
             "end": float(w.get("end", 0.0))}
            for w in payload.get("words", []) if w.get("word", "").strip()
        ]
        segments = [
            Segment(text=s.get("text", "").strip(), start=float(s.get("start", 0.0)),
                    end=float(s.get("end", 0.0)))
            for s in payload.get("segments", [])
        ]
        if not words and segments:
            words = _words_from_segments(segments)
        if not words:
            raise TranscribeError(f"{self.m.name} returned no words")
        return TranscriptionResult(segments=segments, words=words,
                                   backend=self.m.id)


# --- path 1: audio LLMs ---------------------------------------------------------

_GEMINI_PROMPT = """Transcribe this song and return a karaoke-style lyric sheet.

Rules:
- Output ONLY a JSON object: {"lines": [{"text": "...", "words": [{"t": "<word>", "s": <start_sec>, "e": <end_sec>}, ...]}, ...]}
- Each line is one lyric line (a roughly grammatically complete thought), typically 3-12 words.
- Every word carries its start time "s" and end time "e" in seconds from the song's beginning. Times must be accurate to the actual singing — they drive a karaoke highlighter.
- Line text joins the words with single spaces.
- Standard lyric conventions: no periods at line ends; keep ? and ! and commas; capitalize line beginnings and "I".
- Transcribe exactly what is sung, including repeated choruses; do not censor or "fix" words.
- If a section is instrumental, skip it (no line for it).
"""


def _parse_audio_llm_json(raw: str) -> list[dict]:
    """Extract the lines array from the model's JSON reply."""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise TranscribeError("audio model returned no JSON object")
    try:
        js = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        raise TranscribeError(f"audio model JSON invalid: {exc}") from exc
    lines = js.get("lines")
    if not isinstance(lines, list) or not lines:
        raise TranscribeError("audio model returned no lines")
    return lines


def _lines_to_result(lines: list[dict], duration: float | None,
                     backend: str) -> TranscriptionResult:
    """Validate and flatten the audio-LLM lyric sheet into our word format.

    Every word must carry non-negative s/e and non-empty text; nonsense words
    are dropped. Words are re-sorted by start time (models occasionally swap
    two) and line text is rebuilt from the surviving words so the player's
    karaoke sweep always matches what it highlights.
    """
    words: list[dict] = []
    kept_lines: list[dict] = []
    for ln in lines:
        if not isinstance(ln, dict):
            continue
        ws = ln.get("words") or []
        clean = []
        for w in ws:
            if not isinstance(w, dict):
                continue
            text = str(w.get("t", "")).strip()
            try:
                s = float(w.get("s")); e = float(w.get("e"))
            except (TypeError, ValueError):
                continue
            if not text or s < 0 or e < s:
                continue
            clean.append({"text": text, "start": s, "end": e})
        if not clean:
            continue
        clean.sort(key=lambda w: (w["start"], w["end"]))
        kept_lines.append({"text": " ".join(w["text"] for w in clean),
                           "words": clean})
        words.extend(clean)
    if not words:
        raise TranscribeError("audio model returned no usable timed words")
    words.sort(key=lambda w: (w["start"], w["end"]))
    segments = [Segment(text=ln["text"],
                        start=ln["words"][0]["start"],
                        end=ln["words"][-1]["end"]) for ln in kept_lines]
    return TranscriptionResult(segments=segments, words=words, backend=backend)


class GeminiAudio:
    """Google Gemini via REST generateContent (audio inline, base64)."""

    def __init__(self, model: CloudModel) -> None:
        self.m = model
        self.name = model.id

    def available(self) -> bool:
        return bool(os.environ.get(self.m.key_env, "").strip())

    def transcribe(self, audio_path: Path, progress) -> TranscriptionResult:
        key = os.environ[self.m.key_env].strip()
        model_id = self.m.id.split(":", 1)[1]
        progress(f"Uploading audio to Gemini ({model_id})…")
        data = audio_path.read_bytes()
        mime = mimetypes.guess_type(audio_path.name)[0] or "audio/mp4"
        b64 = base64.b64encode(data).decode("ascii")
        body = {
            "contents": [{"parts": [
                {"text": _GEMINI_PROMPT},
                {"inline_data": {"mime_type": mime, "data": b64}},
            ]}],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
                "maxOutputTokens": 65536,
            },
        }
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model_id}:generateContent?key={key}")
        progress("Gemini is listening and formatting the lyrics…")
        with httpx.Client(timeout=900.0) as client:
            resp = client.post(url, json=body)
        if resp.status_code != 200:
            raise TranscribeError(f"Gemini error {resp.status_code}: {resp.text[:300]}")
        payload = resp.json()
        try:
            parts = payload["candidates"][0]["content"]["parts"]
            raw = "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError):
            raise TranscribeError(f"Gemini returned no text: {str(payload)[:300]}")
        lines = _parse_audio_llm_json(raw)
        return _lines_to_result(lines, None, self.m.id)


class OpenAIAudio:
    """GPT-4o-audio via the chat completions API (input_audio content part)."""

    def __init__(self, model: CloudModel) -> None:
        self.m = model
        self.name = model.id

    def available(self) -> bool:
        return bool(os.environ.get(self.m.key_env, "").strip())

    def transcribe(self, audio_path: Path, progress) -> TranscriptionResult:
        key = os.environ[self.m.key_env].strip()
        model_id = self.m.id.split(":", 1)[1]
        progress(f"Uploading audio to OpenAI ({model_id})…")
        data = audio_path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        body = {
            "model": model_id,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": _GEMINI_PROMPT},
                {"type": "input_audio", "input_audio": {"data": b64, "format": "mp4"}},
            ]}],
        }
        progress("GPT-4o is listening and formatting the lyrics…")
        with httpx.Client(timeout=900.0) as client:
            resp = client.post("https://api.openai.com/v1/chat/completions",
                               headers={"Authorization": f"Bearer {key}"}, json=body)
        if resp.status_code != 200:
            raise TranscribeError(f"OpenAI error {resp.status_code}: {resp.text[:300]}")
        try:
            raw = resp.json()["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError):
            raise TranscribeError("OpenAI returned no message content")
        lines = _parse_audio_llm_json(raw)
        return _lines_to_result(lines, None, self.m.id)


# --- dispatch -------------------------------------------------------------------

def build(model_id: str):
    """Construct the backend for a catalog model id (or None if unknown)."""
    m = get_model(model_id)
    if m is None:
        return None
    if m.kind == "transcribe":
        return TranscriptProvider(m)
    if m.provider == "gemini":
        return GeminiAudio(m)
    return OpenAIAudio(m)


def pick(model_id: str | None = None):
    """Choose a cloud backend from an explicit id or settings; explain choice."""
    from app.settings import settings
    mid = model_id or settings.get().get("cloud_model")
    if mid:
        m = get_model(mid)
        if m is None:
            return None, f"Unknown model '{mid}'"
        backend = build(mid)
        if not backend.available():
            return None, f"{m.name} needs {m.key_env} set"
        return backend, m.name
    # auto: first available in preference order
    from app.cloud_models import CATALOG
    for m in CATALOG:
        backend = build(m.id)
        if backend.available():
            return backend, m.name
    return None, ("No cloud transcription available — set GEMINI_API_KEY, "
                  "GROQ_API_KEY or OPENAI_API_KEY")
