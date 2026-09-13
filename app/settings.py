"""Persisted app settings (transcription + formatter config, API keys)."""
from __future__ import annotations

import json
import os
import threading

import app.config as cfg

_VALID_TRANSCRIBE_PROVIDERS = {"gemini", "openrouter", "groq", "openai"}
_VALID_FORMATTER_PROVIDERS = {"gemini", "openrouter", "groq", "openai", "ollama"}

_DEFAULTS = {
    "backend": "cloud",
    "model": "",                       # legacy; unused by the cloud flow
    "cloud_model": "",                 # "<provider>:<api_model>" or "" = auto
    "transcribe_provider": "gemini",   # block 1 provider dropdown
    "transcribe_model": "",            # block 1 model (api_model string)
    "keys": {},                        # provider id -> API key (UI-saved)
    "llm_format": True,
    # --- advanced sampling params (mutually exclusive: one at 1.0) ----------
    "temperature": 0.0,                # greedy by default (top_p = 1)
    "top_p": 1.0,
    "transcribe_timeout": 180,        # seconds; watchdog for slow models
    # --- block 2 (formatter; only used with transcribe-only models) ----------
    "llm_provider": "ollama",
    "llm_model": "",                   # api model for the formatter provider
    # --- legacy ollama fields (still used when llm_provider == "ollama") -----
    "llm_url": os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"),
    "llm_model_ollama": os.environ.get("FORMATTER_MODEL", "qwen2.5:3b"),
}


_LEGACY_API = {  # old native model names -> new catalog api_model strings
    "gemini-2.5-flash": "gemini-flash-latest",
    "gemini-2.5-pro": "gemini-pro-latest",
    "google/gemini-flash-latest": "gemini-flash-latest",
    "google/gemini-pro-latest": "gemini-pro-latest",
    "gpt-4o-audio-preview": "openai/gpt-audio",
    "xiaomi/mimo-v2-omni": "xiaomi/mimo-v2.5",
    "thinking-machines/inkling": "thinkingmachines/inkling",
    "thinking-machines/inkling-small": "thinkingmachines/inkling-small",
    "nvidia/nemotron-3-nano-omni":
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
}


def _migrate(raw: dict) -> dict:
    """Map pre-provider settings onto the new fields."""
    if "llm_model" in raw and "llm_model_ollama" not in raw:
        # old llm_model was the ollama tag
        raw.setdefault("llm_model_ollama", raw["llm_model"])
    if raw.get("llm_provider") == "groq" and raw.get("cloud_formatter_model"):
        raw.setdefault("llm_model", raw["cloud_formatter_model"])
    cm = raw.get("cloud_model", "")
    if cm and ":" in cm:
        pid, api = cm.split(":", 1)
        if pid in ("gemini", "groq", "openai", "openrouter"):
            raw.setdefault("transcribe_provider", pid)
            raw.setdefault("transcribe_model", _LEGACY_API.get(api, api))
    tm = raw.get("transcribe_model", "")
    if tm:
        raw["transcribe_model"] = _LEGACY_API.get(tm, tm)
    if raw.get("llm_provider") == "openrouter" and raw.get("llm_model"):
        raw["llm_model"] = _LEGACY_API.get(raw["llm_model"], raw["llm_model"])
    return raw


class Settings:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._path = cfg.DATA_DIR / "settings.json"
        self._data = dict(_DEFAULTS)
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        raw = _migrate(raw)
        for k in _DEFAULTS:
            if k in raw:
                self._data[k] = raw[k]

    def _save(self) -> None:
        self._path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def get(self) -> dict:
        with self._lock:
            return dict(self._data)

    def update(self, **fields) -> dict:
        with self._lock:
            if "backend" in fields:
                self._data["backend"] = str(fields["backend"])
            if "model" in fields:
                self._data["model"] = str(fields["model"])
            if "cloud_model" in fields:
                self._data["cloud_model"] = str(fields["cloud_model"]).strip()
            if "transcribe_provider" in fields:
                val = str(fields["transcribe_provider"])
                if val not in _VALID_TRANSCRIBE_PROVIDERS:
                    raise ValueError("unknown transcription provider")
                self._data["transcribe_provider"] = val
            if "transcribe_model" in fields:
                self._data["transcribe_model"] = str(fields["transcribe_model"]).strip()
            if "llm_format" in fields:
                self._data["llm_format"] = bool(fields["llm_format"])
            if "temperature" in fields:
                try:
                    temp = float(fields["temperature"])
                except (TypeError, ValueError):
                    raise ValueError("temperature must be a number")
                if not 0.0 <= temp <= 2.0:
                    raise ValueError("temperature must be between 0 and 2")
                self._data["temperature"] = temp
                # sliders are locked: moving temperature off neutral (1)
                # snaps top_p back to its neutral 1, never both active
                if temp != 1.0 and self._data["top_p"] != 1.0:
                    self._data["top_p"] = 1.0
            if "top_p" in fields:
                try:
                    top_p = float(fields["top_p"])
                except (TypeError, ValueError):
                    raise ValueError("top_p must be a number")
                if not 0.05 <= top_p <= 1.0:
                    raise ValueError("top_p must be between 0.05 and 1")
                self._data["top_p"] = top_p
                if top_p != 1.0 and self._data["temperature"] != 1.0:
                    self._data["temperature"] = 1.0
            if "transcribe_timeout" in fields:
                try:
                    tmo = float(fields["transcribe_timeout"])
                except (TypeError, ValueError):
                    raise ValueError("transcribe_timeout must be a number")
                if tmo < 10:
                    raise ValueError("transcribe_timeout must be at least 10 seconds")
                self._data["transcribe_timeout"] = tmo
            if "llm_provider" in fields:
                val = str(fields["llm_provider"])
                if val not in _VALID_FORMATTER_PROVIDERS:
                    raise ValueError("unknown formatter provider")
                self._data["llm_provider"] = val
                # sensible default model when switching providers
                if not fields.get("llm_model"):
                    from app.cloud_models import formatter_default
                    self._data["llm_model"] = formatter_default(val)
            if "llm_model" in fields:
                self._data["llm_model"] = str(fields["llm_model"]).strip()
            if "llm_model_ollama" in fields:
                self._data["llm_model_ollama"] = str(fields["llm_model_ollama"]).strip()
            if "llm_url" in fields:
                val = str(fields["llm_url"]).strip()
                if not val:
                    raise ValueError("llm_url must be a non-empty string")
                self._data["llm_url"] = val
            if "keys" in fields:
                keys = fields["keys"]
                if not isinstance(keys, dict):
                    raise ValueError("keys must be an object")
                clean = {str(k): str(v).strip()
                         for k, v in keys.items() if str(v).strip()}
                self._data["keys"] = clean
            self._save()
            return dict(self._data)


settings = Settings()
