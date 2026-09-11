"""Transcription backends: Parakeet v3 (ONNX) locally, Groq/OpenAI in the cloud.

All backends return a unified list of word dicts: {text, start, end}.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
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


# --- local Parakeet v3 via onnx-asr ------------------------------------------

SR = 16000            # Parakeet's expected sample rate
SLICE_LEN = 28.0      # stay under the model's ~30s context window
SLICE_STEP = 27.5     # 0.5s overlap between slices; overlap words are deduped

def _ensure_wav(audio_path: Path) -> Path:
    """Convert any audio to 16 kHz mono WAV (Parakeet's expected input)."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="refrain_wav_", dir=cfg.DATA_DIR))
    out = tmp_dir / "input.wav"
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_path), "-ac", "1", "-ar", str(SR), str(out)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not out.exists():
        raise TranscribeError(f"ffmpeg WAV conversion failed: {proc.stderr[-300:]}")
    return out


def _read_wav(path: Path):
    """Read 16 kHz mono PCM-16 WAV into a normalized float32 numpy array."""
    import wave
    import numpy as np
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _merge_subwords(tokens: list[str], timestamps: list[float]) -> list[dict]:
    """Merge Parakeet subword tokens (' qu', 'ick') into words.

    A token beginning with a space starts a new word; token start times double
    as the previous word's end time. Returns [{text, start, end}].
    """
    words: list[dict] = []
    buf: str | None = None
    buf_start = 0.0
    for tok, ts in zip(tokens, timestamps):
        if tok.startswith(" "):
            if buf:
                words.append({"text": buf, "start": buf_start,
                              "end": max(ts, buf_start + 0.05)})
            buf = tok.strip()
            buf_start = float(ts)
        else:
            buf = (buf or "") + tok
    if buf:
        last_ts = float(timestamps[-1]) if timestamps else buf_start
        words.append({"text": buf, "start": buf_start,
                      "end": max(last_ts + 0.25, buf_start + 0.05)})
    return words


class ParakeetLocal:
    """NVIDIA Parakeet TDT 0.6B v3 (multilingual) via onnx-asr ONNX runtime.

    Whole-file recognition with word timestamps is the path — the TDT decoder
    handles full songs far better than windowing (no boundary garbles, hears
    quiet fade-out vocals). If a whole-file pass ever covers < 50% of a long
    track (the silent-truncation failure mode), we retry with overlapping
    windows as a fallback.
    """

    name = "parakeet"

    def __init__(self) -> None:
        self._model = None

    def available(self) -> bool:
        try:
            import onnx_asr  # noqa: F401
            return True
        except ImportError:
            return False

    def _load(self):
        if self._model is None:
            import onnx_asr
            self._model = onnx_asr.load_model(cfg.PARAKEET_MODEL)
        return self._model

    def transcribe(self, audio_path: Path, progress: ProgressFn) -> TranscriptionResult:
        wav = _ensure_wav(audio_path)
        try:
            progress("Loading Parakeet v3 (first run downloads the model)…")
            model = self._load()
            progress("Transcribing the whole file with Parakeet v3…")
            waveform = _read_wav(wav)
            duration = len(waveform) / SR

            words = self._recognize_slice(model, waveform, 0.0)

            # Safety net: if whole-file clearly under-covered a long track
            # (silent truncation), fall back to overlapping windows.
            if words and duration > 60.0:
                covered = words[-1]["end"] - words[0]["start"]
                if covered < 0.5 * duration:
                    progress("Whole-file pass looked short — retrying in windows…")
                    words = []
                    starts = _slice_starts(duration)
                    for i, cs in enumerate(starts):
                        progress(f"Transcribing {i + 1}/{len(starts)}…")
                        sl = waveform[int(cs * SR):int((cs + SLICE_LEN) * SR)]
                        words += self._recognize_slice(model, sl, cs)
                    words = _dedupe_overlap(words)

            if not words:
                raise TranscribeError(
                    "No vocals detected — this track appears to be instrumental")
            return TranscriptionResult(segments=[], words=words, backend=self.name)
        finally:
            wav.unlink(missing_ok=True)
            try:
                wav.parent.rmdir()
            except OSError:
                pass

    def _recognize_slice(self, model, slice_waveform, offset: float) -> list[dict]:
        try:
            res = model.with_timestamps().recognize(slice_waveform, sample_rate=SR)
        except Exception:
            return []
        toks = list(res.tokens)
        tss = [float(t) + offset for t in res.timestamps]
        return _merge_subwords(toks, tss)


def _slice_starts(duration: float) -> list[float]:
    """Window starts covering the whole duration (last tile may overlap)."""
    if duration <= SLICE_LEN:
        return [0.0]
    starts = [round(s, 2) for s in
              _frange(0.0, duration - SLICE_LEN + 0.01, SLICE_STEP)]
    if starts and starts[-1] + SLICE_LEN < duration - 1.0:
        starts.append(round(duration - SLICE_LEN, 2))
    return starts


def _frange(start: float, stop: float, step: float) -> list[float]:
    out = []
    v = start
    while v < stop:
        out.append(v)
        v += step
    return out


def _dedupe_overlap(words: list[dict], tol: float = 0.4) -> list[dict]:
    """Drop duplicate words produced by the 0.5s overlap between slices.

    A word appearing again within `tol` seconds of the previous instance
    (same text) is an overlap artifact. Words that merely start close but
    differ in text are kept — that's normal singing.
    """
    words.sort(key=lambda w: (w["start"], w["end"]))
    out: list[dict] = []
    for w in words:
        if out:
            p = out[-1]
            if (w["text"] == p["text"]
                    and abs(w["start"] - p["start"]) <= tol):
                continue
        out.append(w)
    return out


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
        ParakeetLocal(),
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
