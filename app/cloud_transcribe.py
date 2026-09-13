"""Cloud transcription + audio-LLM backends.

Two families:
- TranscriptProvider (whisper-style /audio/transcriptions APIs): returns words
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
import wave
from pathlib import Path

import httpx

from app.cloud_models import CloudModel, get_model, provider_key
from app.transcribe import Segment, TranscribeError, TranscriptionResult, _words_from_segments


# --- path 2: transcription-only providers --------------------------------------

def _provider_base_url(provider: str) -> str:
    if provider == "groq":
        return "https://api.groq.com/openai/v1"
    return "https://api.openai.com/v1"


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
                "timestamp_granularities[]": "word,segment",
            }
            with httpx.Client(timeout=600.0) as client:
                resp = client.post(f"{self._base_url()}/audio/transcriptions",
                                   headers=self._headers(), files=files, data=data)
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


# --- Enhanced LRC parsing --------------------------------------------------------

_LINE_TS = re.compile(r"^\[(\d{1,3}):(\d{1,2})(?:[.:](\d{1,3}))?\]")
_WORD_TS = re.compile(r"<(\d{1,3}):(\d{1,2})(?:[.:](\d{1,3}))?>")
_OFFSET_TAG = re.compile(r"^\[offset:\s*([+-]?\d+)\s*\]", re.I)


def _tag_seconds(mm: str, ss: str, frac: str | None) -> float:
    t = int(mm) * 60 + int(ss)
    if frac:
        t += int(frac) / (10 ** len(frac))
    return t


def _parse_enhanced_lrc(raw: str | None) -> list[dict] | None:
    """Parse Enhanced LRC (word-level <mm:ss.xx> tags) into our lines format.

    Word tags carry START times only, so a word's end is the next tag's time
    (next line's first tag for line-final words). Plain LRC lines without word
    tags get their text evenly interpolated across the line's time span.
    Returns None when nothing timed could be extracted.
    """
    if not raw or not raw.strip():
        return None
    content = "\n".join(ln for ln in raw.splitlines()
                        if not ln.strip().startswith("```"))
    offset_ms = 0.0
    entries: list[tuple[float, list[tuple[float, str]]]] = []
    for ln in content.splitlines():
        s = ln.strip()
        om = _OFFSET_TAG.match(s)
        if om:
            offset_ms = float(om.group(1))
            continue
        m = _LINE_TS.match(s)
        if not m:
            continue  # metadata tag ([ti:], [ar:], …) or noise
        t0 = _tag_seconds(*m.groups())
        rest = s[m.end():]
        tags = list(_WORD_TS.finditer(rest))
        # (start, token, explicit_end|None) per word
        wlist: list[tuple[float, str, float | None]] = []
        if tags:
            lead = rest[:tags[0].start()].strip()
            if lead:
                wlist.extend((t0, tok, None) for tok in lead.split())
            for i, wm in enumerate(tags):
                wt = _tag_seconds(*wm.groups())
                nxt = tags[i + 1].start() if i + 1 < len(tags) else len(rest)
                chunk = rest[wm.end():nxt].strip()
                if chunk:
                    wlist.extend((wt, tok, None) for tok in chunk.split())
                elif wlist:
                    # trailing/empty tag: explicit end for the last open word
                    # (Gemini ends lines with <mm:ss.xx> after the final word)
                    for j in range(len(wlist) - 1, -1, -1):
                        if wlist[j][2] is None:
                            wlist[j] = (wlist[j][0], wlist[j][1], wt)
                            break
        else:
            wlist.extend((t0, tok, None) for tok in rest.split())
        if wlist:
            entries.append((t0, wlist))
    if not entries:
        return None
    entries.sort(key=lambda x: x[0])
    adj = offset_ms / 1000.0
    lines: list[dict] = []
    for i, (_t0, wlist) in enumerate(entries):
        ws = []
        for j, (wt, tok, xend) in enumerate(wlist):
            start = wt - adj
            if xend is not None:
                end = xend - adj
            elif j + 1 < len(wlist):
                end = wlist[j + 1][0] - adj
            elif i + 1 < len(entries):
                end = entries[i + 1][1][0][0] - adj
            else:
                end = start + 1.0
            end = max(end, start)
            ws.append({"t": tok, "s": round(start, 3), "e": round(end, 3)})
        lines.append({"text": " ".join(w["t"] for w in ws), "words": ws})
    return lines


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
    """ffmpeg the audio to 16 kHz mono WAV (the format audio APIs accept)."""
    out = Path(tempfile.gettempdir()) / f"refrain_{audio_path.stem}_16k.wav"
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_path), "-ac", "1", "-ar", "16000",
         "-sample_fmt", "s16", str(out)],
        capture_output=True, text=True)
    if proc.returncode != 0 or not out.exists():
        raise TranscribeError("ffmpeg failed to convert audio for the audio model")
    return out


def _wav_bytes(path: Path) -> bytes:
    """File bytes via the wave module (validates our ffmpeg output)."""
    with wave.open(str(path), "rb") as w:
        return path.read_bytes()


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
        b64 = base64.b64encode(_wav_bytes(wav)).decode("ascii")

        def call_fn(prompt: str, *, force_json: bool) -> str:
            body = {
                "model": self.m.api_model,
                "temperature": 0.0,
                "modalities": ["text"],
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "input_audio",
                     "input_audio": {"data": b64, "format": "wav"}},
                ]}],
            }
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
            gen_cfg = {"temperature": 0.0, "maxOutputTokens": 65536}
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
            body = {
                "model": self.m.api_model,
                "temperature": 0.0,
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
        return TranscriptProvider(m)
    if m.provider == "gemini":
        return GeminiAudio(m)
    if m.provider == "openrouter":
        return OpenRouterAudio(m)
    return OpenAIAudio(m)


def pick(model_id: str | None = None):
    """Choose a cloud backend from an explicit id or settings; explain choice."""
    from app.cloud_models import CATALOG, resolved_transcribe_model
    from app.settings import settings
    _ = settings
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
    from app.cloud_models import CATALOG
    for m in CATALOG:
        backend = build(m.id)
        if backend is not None and backend.available():
            return backend, m.name
    return None, ("No cloud transcription available — set an API key in the "
                  "models panel (or GEMINI_API_KEY / OPENROUTER_API_KEY / "
                  "GROQ_API_KEY / OPENAI_API_KEY in the environment)")
