"""Runtime configuration, loaded from environment / .env (stdlib only)."""
from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv() -> None:
    """Tiny .env loader so we don't need python-dotenv."""
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


_load_dotenv()


def _get(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


# --- App -------------------------------------------------------------------
DATA_DIR = Path(_get("APP_DATA_DIR", "data")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Download --------------------------------------------------------------
YTDLP_COOKIES_FILE = _get("YTDLP_COOKIES_FILE")
YTDLP_EXTRA = [f for f in _get("YTDLP_EXTRA").split(",") if f.strip()]

# --- Transcription ---------------------------------------------------------
TRANSCRIBE_BACKEND = _get("TRANSCRIBE_BACKEND", "auto").strip().lower()
WHISPER_MODEL = _get("WHISPER_MODEL", "base")
WHISPER_DEVICE = _get("WHISPER_DEVICE", "auto")
GROQ_API_KEY = _get("GROQ_API_KEY")
GROQ_MODEL = _get("GROQ_MODEL", "whisper-large-v3")
OPENAI_API_KEY = _get("OpenAI_API_KEY".lower()) or _get("OPENAI_API_KEY")
OPENAI_MODEL = _get("OPENAI_MODEL", "whisper-1")

# --- Jobs ------------------------------------------------------------------
JOB_CONCURRENCY = max(1, int(_get("JOB_CONCURRENCY", "1")))

__all__ = [
    "DATA_DIR", "YTDLP_COOKIES_FILE", "YTDLP_EXTRA", "TRANSCRIBE_BACKEND",
    "WHISPER_MODEL", "WHISPER_DEVICE", "GROQ_API_KEY", "GROQ_MODEL",
    "OPENAI_API_KEY", "OPENAI_MODEL", "JOB_CONCURRENCY",
]
