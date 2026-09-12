"""Dynamic cloud model catalog: transcription models and audio-capable LLMs.

Two transcription paths are supported:

1. audio_llm — a multimodal model (Gemini, GPT-4o-audio) listens to the audio
   and returns fully formatted, synced LRC-style lines. No formatter needed.
2. transcribe — a transcription-only model (Whisper via Groq/OpenAI) returns
   words with timestamps; a separate LLM (cloud or local ollama) formats them
   into lyric lines afterwards.

Availability is driven by which API keys are present in the environment.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class CloudModel:
    id: str            # "<provider>:<model>" — unique, used in settings
    name: str          # display name
    provider: str      # groq | openai | gemini
    kind: str          # transcribe | audio_llm
    key_env: str       # env var holding the API key
    notes: str


CATALOG: list[CloudModel] = [
    # --- transcription-only (word timestamps; needs a separate formatter) ----
    CloudModel("groq:whisper-large-v3", "Whisper large v3 (Groq)", "groq",
               "transcribe", "GROQ_API_KEY",
               "Fast word-level timestamps; best value"),
    CloudModel("groq:whisper-large-v3-turbo", "Whisper large v3 turbo (Groq)",
               "groq", "transcribe", "GROQ_API_KEY",
               "Cheaper/faster, slightly lower accuracy"),
    CloudModel("openai:whisper-1", "Whisper (OpenAI)", "openai",
               "transcribe", "OPENAI_API_KEY",
               "Word-level timestamps"),
    CloudModel("openai:gpt-4o-mini-transcribe", "GPT-4o-mini Transcribe (OpenAI)",
               "openai", "transcribe", "OPENAI_API_KEY",
               "Stronger transcription; segment timestamps only (words interpolated)"),
    CloudModel("openai:gpt-4o-transcribe", "GPT-4o Transcribe (OpenAI)", "openai",
               "transcribe", "OPENAI_API_KEY",
               "Best OpenAI transcription; segment timestamps only (words interpolated)"),
    # --- audio LLMs: listen + return formatted synced lines in one step -------
    CloudModel("gemini:gemini-2.5-flash", "Gemini 2.5 Flash", "gemini",
               "audio_llm", "GEMINI_API_KEY",
               "Listens and returns formatted synced lyrics in one step"),
    CloudModel("gemini:gemini-2.5-pro", "Gemini 2.5 Pro", "gemini",
               "audio_llm", "GEMINI_API_KEY",
               "Highest quality audio understanding; slower"),
    CloudModel("openai:gpt-4o-audio-preview", "GPT-4o Audio (OpenAI)", "openai",
               "audio_llm", "OPENAI_API_KEY",
               "Listens and returns formatted synced lyrics in one step"),
]

_BY_ID = {m.id: m for m in CATALOG}


def get_model(model_id: str) -> CloudModel | None:
    return _BY_ID.get(model_id)


def provider_available(provider: str, key_env: str) -> bool:
    import os
    return bool(os.environ.get(key_env, "").strip())


def list_models() -> list[dict]:
    """Catalog for the UI: availability + display fields."""
    out = []
    for m in CATALOG:
        d = asdict(m)
        d["available"] = provider_available(m.provider, m.key_env)
        out.append(d)
    return out


def any_available() -> bool:
    return any(m["available"] for m in list_models())


# --- formatter LLM providers (used on the "transcribe" path) ------------------

FORMATTER_PROVIDERS = {
    "ollama": {"label": "Ollama (local)", "key_env": None},
    "groq": {"label": "Groq", "key_env": "GROQ_API_KEY",
             "default_model": "llama-3.3-70b-versatile"},
    "openai": {"label": "OpenAI", "key_env": "OPENAI_API_KEY",
               "default_model": "gpt-4o-mini"},
}


def formatter_available(provider: str) -> bool:
    import os
    info = FORMATTER_PROVIDERS.get(provider)
    if not info:
        return False
    if provider == "ollama":
        from app import formatter
        return formatter.llm_available()
    key = os.environ.get(info["key_env"], "").strip()
    return bool(key)
