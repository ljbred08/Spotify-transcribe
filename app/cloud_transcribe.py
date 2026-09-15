"""Cloud transcription + audio-LLM backends.

Two families:
- Transcription-only backends (whisper-style /audio/transcriptions APIs,
  OpenRouter's JSON STT endpoint, Gemini's Interactions API): return words
  with timestamps; formatting happens later (block 2 of the UI).
- AudioLLM (Gemini, OpenRouter omni models): listens to the audio and returns
  fully formatted lyric lines WITH word-level timings in one call — no
  separate formatter, no resync.

Response contract for audio LLMs: we ask for **Enhanced LRC** (A2 extended
LRC with per-word <mm:ss.xx> tags) — the format is abundantly represented in
training data, so big models emit it faithfully given just the title/artist.
The prompt is deliberately tiny; if the reply isn't parseable LRC we retry
once with the older JSON lyric-sheet prompt. Every parsed word is validated
(text non-empty, s/e finite with e >= s); anything that can't be mapped back
to the audio is dropped, so karaoke sync stays honest.
"""
from __future__ import annotations

import base64
import json
import re
import subprocess
import tempfile
from pathlib import Path

import httpx

from app.cloud_models import CloudModel, get_model, provider_key
from app.lrc import parse_enhanced as _parse_enhanced_lrc
from app.transcribe import Segment, TranscribeError, TranscriptionResult, _words_from_segments


# --- path 2: transcription-only providers --------------------------------------

def _provider_base_url(provider: str) -> str:
    if provider == "groq":
        return "https://api.groq.com/openai/v1"
    return "https://api.openai.com/v1"


def _stt_result_from_payload(model: CloudModel, payload: dict) -> TranscriptionResult:
    """Shape any OpenAI-compatible verbose_json STT reply into our result.

    Accepts both key styles — plain objects ({word,start,end}) and OpenRouter
    STT payloads that may nest words under extra keys — plus Deepgram-style
    {punctuated_word,start,end}. Words without valid finite timings are
    dropped so the karaoke sweep never stutters on garbage.
    """
    words: list[dict] = []
    for w in payload.get("words") or []:
        if not isinstance(w, dict):
            continue
        text = str(w.get("word") or w.get("text") or "").strip()
        try:
            s = float(w.get("start", 0.0))
            e = float(w.get("end", w.get("start", 0.0)))
        except (TypeError, ValueError):
            continue
        if not text or not (s >= 0) or e < s:
            continue
        words.append({"text": text, "start": s, "end": e})
    segments = [
        Segment(text=s.get("text", "").strip(), start=float(s.get("start", 0.0)),
                end=float(s.get("end", 0.0)))
        for s in payload.get("segments") or []
        if isinstance(s, dict)
    ]
    if not words:
        words = _words_from_segments(segments)
    if not words:
        raise TranscribeError(f"{model.name} returned no timed words")
    return TranscriptionResult(segments=segments, words=words, backend=model.id)


class TranscriptProvider:
    """Whisper-style HTTP APIs (Groq, OpenAI). Word timestamps when available."""

    def __init__(self, model: CloudModel) -> None:
        self.m = model
        self.name = model.id

    def available(self) -> bool:
        return bool(provider_key(self.m.provider))

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {provider_key(self.m.provider)}"}

    def _base_url(self) -> str:
        return _provider_base_url(self.m.provider)

    def transcribe(self, audio_path: Path, progress, *,
                   title: str = "", artist: str = "") -> TranscriptionResult:
        progress(f"Uploading audio to {self.m.provider} ({self.m.name})…")
        with audio_path.open("rb") as fh:
            files = {"file": (audio_path.name, fh, "audio/mp4")}
            data = {
                "model": self.m.api_model,
                "response_format": "verbose_json",
                # repeated form fields — one comma-joined string is a 400
                # (Groq validates each value against the enum)
                "timestamp_granularities[]": ["word", "segment"],
            }
            with httpx.Client(timeout=600.0) as client:
                resp = client.post(f"{self._base_url()}/audio/transcriptions",
                                   headers=self._headers(), files=files, data=data)
        if resp.status_code != 200:
            raise TranscribeError(
                f"{self.m.provider} error {resp.status_code}: {resp.text[:300]}")
        return _stt_result_from_payload(self.m, resp.json())

def _ffmpeg_convert(audio_path: Path, fmt: str) -> Path:
    """ffmpeg the audio into the container an upstream STT insists on.

    Used only for models whose upstream rejects the app's m4a outright
    (bare "Provider returned 400"). 16 kHz mono keeps the payload small;
    every affected model is a Whisper-class ASR that resamples internally
    anyway.
    """
    out = Path(tempfile.gettempdir()) / f"refrain_{audio_path.stem}_stt.{fmt}"
    proc = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(audio_path),
         "-ac", "1", "-ar", "16000", "-b:a", "64k", str(out)],
        capture_output=True, text=True)
    if proc.returncode != 0 or not out.exists():
        raise TranscribeError(f"ffmpeg failed to convert audio to {fmt}")
    return out


class OpenRouterSTT:
    """Transcription-only models via OpenRouter's JSON /audio/transcriptions.

    The endpoint takes base64 audio (raw bytes, not a data URI) and speaks
    OpenAI's verbose_json dialect: response_format "verbose_json" plus
    timestamp_granularities ["word"] yields a words array with per-word
    start/end. Providers without structured output reject verbose_json with
    a 400 — those models are kept out of the catalog on purpose.

    Singing quirk (probed 2026-09-14): deepgram/nova-3 and x-ai/grok-stt-1.0
    transcribe spoken audio flawlessly (59 s of TTS speech -> 168 words with
    exact timings) but return an EMPTY transcript (HTTP 200, no words, no
    text) for sung/music audio — even a 30 s song slice. Speech-only models,
    useless for this app; they are excluded from the catalog rather than
    worked around (chunking does not help:    a 30 s music chunk is empty too).
    """

    def __init__(self, model: CloudModel) -> None:
        self.m = model
        self.name = model.id

    def available(self) -> bool:
        return bool(provider_key(self.m.provider))

    def transcribe(self, audio_path: Path, progress, *,
                   title: str = "", artist: str = "") -> TranscriptionResult:
        key = provider_key(self.m.provider)
        if not key:
            raise TranscribeError("OpenRouter API key missing")
        fmt = self.m.stt_audio_format or "m4a"
        upload = audio_path
        if fmt != "m4a":
            # upstream rejects m4a outright (MAI-Transcribe 2 400s on it);
            # convert to the format its catalog entry demands
            progress(f"Converting audio to {fmt} for {self.m.name}…")
            upload = _ffmpeg_convert(audio_path, fmt)
        try:
            progress(f"Uploading audio to OpenRouter ({self.m.name})…")
            b64 = base64.b64encode(upload.read_bytes()).decode("ascii")
            body = {
                "model": self.m.api_model,
                "input_audio": {"data": b64, "format": fmt},
                "response_format": "verbose_json",
                "timestamp_granularities": ["segment", "word"],
            }
            progress(f"{self.m.name} is transcribing…")
            resp = _post_with_retries(
                lambda: httpx.post(
                    "https://openrouter.ai/api/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {key}"},
                    json=body, timeout=900.0),
                progress, "OpenRouter")
        finally:
            if upload is not audio_path:
                upload.unlink(missing_ok=True)
        if resp.status_code != 200:
            raise TranscribeError(
                f"OpenRouter error {resp.status_code}: {resp.text[:300]}")
        return _stt_result_from_payload(self.m, resp.json())


class GeminiTranscribe:
    """gemini-3.5-transcribe via Google's Interactions API.

    The dedicated ASR model returns measured word offsets as word_info
    annotations on the response content ("0.100s"-style strings). Audio is
    inlined base64: Interactions input items carry {type, data, mime_type}
    directly (NOT the generateContent-style inline_data nesting — that is
    rejected with "Unknown parameter 'inline_data' at 'input[0]'").
    """

    def __init__(self, model: CloudModel) -> None:
        self.m = model
        self.name = model.id

    def available(self) -> bool:
        return bool(provider_key(self.m.provider))

    def transcribe(self, audio_path: Path, progress, *,
                   title: str = "", artist: str = "") -> TranscriptionResult:
        key = provider_key(self.m.provider)
        if not key:
            raise TranscribeError("Gemini API key missing")
        progress(f"Uploading audio to Gemini ({self.m.api_model})…")
        import mimetypes
        mime = mimetypes.guess_type(audio_path.name)[0] or "audio/mp4"
        b64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")
        body = {
            "model": self.m.api_model,
            "input": [{"type": "audio", "data": b64, "mime_type": mime}],
            "generation_config": {"transcription_config": {
                "mode": {"type": "verbatim",
                         "timestamp_granularities": ["word"]},
            }},
        }
        progress(f"{self.m.name} is transcribing…")
        url = "https://generativelanguage.googleapis.com/v1beta/interactions"
        resp = _post_with_retries(
            lambda: httpx.post(url, headers={"x-goog-api-key": key},
                               json=body, timeout=900.0),
            progress, "Gemini")
        if resp.status_code != 200:
            raise TranscribeError(
                f"Gemini error {resp.status_code}: {resp.text[:300]}")
        try:
            steps = resp.json().get("steps") or []
        except ValueError as exc:
            raise TranscribeError("Gemini returned no JSON body") from exc

        def _offset(raw) -> float:
            # "12.3s" | "12.3" | "12s 300ms" -> seconds
            m = re.match(r"\s*(\d+(?:\.\d+)?)s?(?:\s*(\d+)ms)?", str(raw or ""))
            if not m:
                return -1.0
            sec = float(m.group(1))
            if m.group(2):
                sec += int(m.group(2)) / 1000.0
            return sec

        words: list[dict] = []
        for step in steps:
            for content in (step or {}).get("content") or []:
                for ann in (content or {}).get("annotations") or []:
                    if not isinstance(ann, dict) or ann.get("type") != "word_info":
                        continue
                    text = str(ann.get("text", "")).strip()
                    s, e = _offset(ann.get("start_offset")), _offset(ann.get("end_offset"))
                    if text and s >= 0 and e >= s:
                        words.append({"text": text, "start": s, "end": e})
        if not words:
            raise TranscribeError(
                f"{self.m.name} returned no word timestamps (got "
                f"{str(resp.json())[:200]})")
        words.sort(key=lambda w: (w["start"], w["end"]))
        return TranscriptionResult(segments=[], words=words, backend=self.m.id)


# --- path 1: audio LLMs ---------------------------------------------------------

_PROMPT_LRC = """Transcribe this song as Enhanced LRC (A2 extended LRC) with word-level timestamps.

Title: {title}
Artist: {artist}

Return only the raw contents of the .lrc file, nothing else."""

_PROMPT_JSON = """Transcribe this song and return a karaoke-style lyric sheet.

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
    """Extract the lines array from a JSON lyric-sheet reply (raises)."""
    m = re.search(r"\{.*\}", raw or "", re.S)
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


def _parse_json_safe(raw: str | None) -> list[dict] | None:
    try:
        return _parse_audio_llm_json(raw or "")
    except TranscribeError:
        return None


def _lines_to_result(lines: list[dict], backend: str) -> TranscriptionResult:
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
    return TranscriptionResult(
        segments=segments, words=words, backend=backend,
        groups=[ln["words"] for ln in kept_lines])


def _run_audio_llm(call_fn, progress, backend_id: str,
                   title: str, artist: str) -> TranscriptionResult:
    """Shared two-mode contract: Enhanced LRC first, JSON prompt as fallback."""
    prompt = _PROMPT_LRC.format(title=title or "Unknown title",
                                artist=artist or "Unknown artist")
    lines = None
    try:
        raw = call_fn(prompt, force_json=False)
    except TranscribeError:
        raw = None
    if raw:
        lines = _parse_enhanced_lrc(raw) or _parse_json_safe(raw)
    if not lines:
        progress("Falling back to the JSON lyric-sheet format…")
        raw = call_fn(_PROMPT_JSON, force_json=True)
        lines = _parse_json_safe(raw)
    if not lines:
        raise TranscribeError("audio model returned no usable lyrics")
    return _lines_to_result(lines, backend_id)


def _to_mp16k(audio_path: Path) -> Path:
    """ffmpeg the audio to a compact mono MP3 for inline audio parts.

    MP3, not WAV: a full-song 16 kHz mono WAV base64-overflows small
    context windows (e.g. gemini-3.5-flash-lite's 32k tokens) with a 400.
    A 32 kbps mono MP3 of the same song is ~10x smaller.
    """
    out = Path(tempfile.gettempdir()) / f"refrain_{audio_path.stem}_16k.mp3"
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_path), "-ac", "1", "-ar", "16000",
         "-b:a", "32k", str(out)],
        capture_output=True, text=True)
    if proc.returncode != 0 or not out.exists():
        raise TranscribeError("ffmpeg failed to convert audio for the audio model")
    return out


def _post_with_retries(post_fn, progress, label: str):
    """POST with up to 3 attempts on 5xx (aliases route to busy models)."""
    import time
    resp = None
    for attempt in range(3):
        resp = post_fn()
        if resp.status_code < 500:
            return resp
        if attempt < 2:
            progress(f"{label} busy ({resp.status_code}) — retrying…")
            time.sleep(8 * (attempt + 1))
    return resp


class OpenRouterAudio:
    """Audio LLMs via OpenRouter's OpenAI-compatible chat completions.

    Audio goes inline as base64 WAV in an input_audio part; text output is
    forced with modalities: ["text"] (required by some omni models).
    """

    def __init__(self, model: CloudModel) -> None:
        self.m = model
        self.name = model.id

    def available(self) -> bool:
        return bool(provider_key(self.m.provider))

    def transcribe(self, audio_path: Path, progress, *,
                   title: str = "", artist: str = "") -> TranscriptionResult:
        key = provider_key(self.m.provider)
        if not key:
            raise TranscribeError("OpenRouter API key missing")
        progress(f"Preparing audio for {self.m.name} (OpenRouter)…")
        wav = _to_mp16k(audio_path)
        b64 = base64.b64encode(wav.read_bytes()).decode("ascii")

        def call_fn(prompt: str, *, force_json: bool) -> str:
            from app.settings import settings
            s = settings.get()
            body = {
                "model": self.m.api_model,
                "temperature": s.get("temperature", 0.0),
                "top_p": s.get("top_p", 1.0),
                "max_tokens": 65536,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "input_audio",
                     "input_audio": {"data": b64, "format": "mp3"}},
                ]}],
            }
            if self.m.api_model.startswith("openai/"):
                # gpt-audio models emit speech by default; force text output
                body["modalities"] = ["text"]
            # soft latency bound: OpenRouter promotes providers whose rolling
            # p90 TTFT fits; if none qualify the best host still serves the
            # request (zero extra 4xx risk on models with a single provider)
            body["provider"] = {"preferred_max_latency": {"p90": 30.0}}
            progress(f"{self.m.name} is listening and formatting the lyrics…")
            try:
                resp = _post_with_retries(
                    lambda: httpx.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={"Authorization": f"Bearer {key}"},
                        json=body, timeout=900.0),
                    progress, "OpenRouter")
            finally:
                wav.unlink(missing_ok=True)
            if resp.status_code != 200:
                raise TranscribeError(
                    f"OpenRouter error {resp.status_code}: {resp.text[:300]}")
            try:
                return resp.json()["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError):
                raise TranscribeError("OpenRouter returned no message content")

        return _run_audio_llm(call_fn, progress, self.m.id, title, artist)


class GeminiAudio:
    """Google Gemini via REST generateContent (audio inline, base64)."""

    def __init__(self, model: CloudModel) -> None:
        self.m = model
        self.name = model.id

    def available(self) -> bool:
        return bool(provider_key(self.m.provider))

    def transcribe(self, audio_path: Path, progress, *,
                   title: str = "", artist: str = "") -> TranscriptionResult:
        key = provider_key(self.m.provider)
        if not key:
            raise TranscribeError("Gemini API key missing")
        model_id = self.m.api_model
        progress(f"Uploading audio to Gemini ({model_id})…")
        data = audio_path.read_bytes()
        import mimetypes
        mime = mimetypes.guess_type(audio_path.name)[0] or "audio/mp4"
        b64 = base64.b64encode(data).decode("ascii")
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model_id}:generateContent?key={key}")

        def call_fn(prompt: str, *, force_json: bool) -> str:
            from app.settings import settings
            s = settings.get()
            gen_cfg = {"temperature": s.get("temperature", 0.0),
                       "maxOutputTokens": 65536}
            if force_json:
                gen_cfg["responseMimeType"] = "application/json"
            body = {
                "contents": [{"parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": mime, "data": b64}},
                ]}],
                "generationConfig": gen_cfg,
            }
            progress("Gemini is listening and formatting the lyrics…")
            resp = _post_with_retries(
                lambda: httpx.post(url, json=body, timeout=900.0),
                progress, "Gemini")
            if resp.status_code != 200:
                raise TranscribeError(
                    f"Gemini error {resp.status_code}: {resp.text[:300]}")
            try:
                parts = resp.json()["candidates"][0]["content"]["parts"]
                return "".join(p.get("text", "") for p in parts)
            except (KeyError, IndexError):
                raise TranscribeError(
                    f"Gemini returned no text: {str(resp.json())[:300]}")

        return _run_audio_llm(call_fn, progress, self.m.id, title, artist)


class OpenAIAudio:
    """GPT-4o-audio via the chat completions API (input_audio content part)."""

    def __init__(self, model: CloudModel) -> None:
        self.m = model
        self.name = model.id

    def available(self) -> bool:
        return bool(provider_key(self.m.provider))

    def transcribe(self, audio_path: Path, progress, *,
                   title: str = "", artist: str = "") -> TranscriptionResult:
        key = provider_key(self.m.provider)
        if not key:
            raise TranscribeError("OpenAI API key missing")
        progress(f"Uploading audio to OpenAI ({self.m.api_model})…")
        data = audio_path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")

        def call_fn(prompt: str, *, force_json: bool) -> str:
            from app.settings import settings
            s = settings.get()
            body = {
                "model": self.m.api_model,
                "temperature": s.get("temperature", 0.0),
                "top_p": s.get("top_p", 1.0),
                "modalities": ["text"],
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "input_audio",
                     "input_audio": {"data": b64, "format": "mp4"}},
                ]}],
            }
            progress("The model is listening and formatting the lyrics…")
            resp = _post_with_retries(
                lambda: httpx.post("https://api.openai.com/v1/chat/completions",
                                   headers={"Authorization": f"Bearer {key}"},
                                   json=body, timeout=900.0),
                progress, "OpenAI")
            if resp.status_code != 200:
                raise TranscribeError(
                    f"OpenAI error {resp.status_code}: {resp.text[:300]}")
            try:
                return resp.json()["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError):
                raise TranscribeError("OpenAI returned no message content")

        return _run_audio_llm(call_fn, progress, self.m.id, title, artist)


# --- dispatch -------------------------------------------------------------------

def build(model_id: str):
    """Construct the backend for a catalog model id (or None if unknown)."""
    m = get_model(model_id)
    if m is None:
        return None
    if m.kind == "transcribe":
        if m.stt_flavor == "openrouter_stt":
            return OpenRouterSTT(m)
        if m.stt_flavor == "gemini_interactions":
            return GeminiTranscribe(m)
        return TranscriptProvider(m)
    if m.provider == "gemini":
        return GeminiAudio(m)
    if m.provider == "openrouter":
        return OpenRouterAudio(m)
    return OpenAIAudio(m)


def pick(model_id: str | None = None):
    """Choose a cloud backend from an explicit id or settings; explain choice."""
    from app.cloud_models import CATALOG, resolved_transcribe_model
    mid = model_id or resolved_transcribe_model()
    if mid:
        m = get_model(mid)
        if m is None:
            return None, f"Unknown model '{mid}'"
        backend = build(mid)
        if backend is None or not backend.available():
            info = get_model(mid)
            env = info.key_env if info else "the provider's API key"
            return None, f"{m.name} needs {env} set"
        return backend, m.name
    # auto: first available in preference order
    for m in CATALOG:
        backend = build(m.id)
        if backend is not None and backend.available():
            return backend, m.name
    return None, ("No cloud transcription available — set an API key in the "
                  "models panel (or GEMINI_API_KEY / OPENROUTER_API_KEY in "
                  "the environment)")
