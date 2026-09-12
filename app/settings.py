"""Persisted app settings (backend + model selection)."""
from __future__ import annotations

import json
import os
import threading

import app.config as cfg

_DEFAULTS = {
    "backend": "cloud",
    "model": "parakeet-v3",          # legacy; unused by cloud flow
    "cloud_model": "",                # "<provider>:<model>" or "" = auto
    "llm_format": True,
    "llm_provider": "ollama",         # ollama | groq | openai
    "cloud_formatter_model": "",      # model name for groq/openai formatters
    # env vars seed the initial defaults; the UI can change them at runtime
    "llm_url": os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"),
    "llm_model": os.environ.get("FORMATTER_MODEL", "qwen2.5:3b"),
}
_VALID_BACKENDS = {"cloud", "auto", "parakeet", "groq", "openai"}


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
                if fields["backend"] not in _VALID_BACKENDS:
                    raise ValueError(f"backend must be one of {sorted(_VALID_BACKENDS)}")
                self._data["backend"] = fields["backend"]
            if "model" in fields:
                self._data["model"] = str(fields["model"])
            if "llm_format" in fields:
                self._data["llm_format"] = bool(fields["llm_format"])
            if "cloud_model" in fields:
                self._data["cloud_model"] = str(fields["cloud_model"]).strip()
            if "llm_provider" in fields:
                if fields["llm_provider"] not in ("ollama", "groq", "openai"):
                    raise ValueError("llm_provider must be ollama, groq or openai")
                self._data["llm_provider"] = fields["llm_provider"]
            if "cloud_formatter_model" in fields:
                self._data["cloud_formatter_model"] = str(fields["cloud_formatter_model"]).strip()
            for key in ("llm_url", "llm_model"):
                if key in fields:
                    val = str(fields[key]).strip()
                    if not val:
                        raise ValueError(f"{key} must be a non-empty string")
                    self._data[key] = val
            self._save()
            return dict(self._data)


settings = Settings()
