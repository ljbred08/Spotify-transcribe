"""Provider-driven cloud model catalog (audio LLMs + transcribers + formatters).

Structure mirrors the UI:
- PROVIDERS: the dropdown entries (Gemini, OpenRouter, Groq, OpenAI, Ollama),
  each with the env var that seeds its API key. Keys can also be set at
  runtime from the UI (settings.keys) which take precedence over the env.
- TRANSCRIBE_MODELS / FORMATTER_MODELS: the per-provider model lists that
  populate the second dropdown.

Two transcription paths:
1. audio_llm — an omni model (Gemini, OpenRouter audio models) listens to the
   audio and returns fully formatted, synced lyric lines. No formatter needed.
2. transcribe — a transcription-only model returns measured word timestamps;
   a separate LLM (block 2 of the UI) formats them afterwards. These carry
   an stt_flavor selecting the API shape: "" = OpenAI multipart
   /audio/transcriptions (Groq, OpenAI), "openrouter_stt" = OpenRouter's
   JSON /audio/transcriptions endpoint, "gemini_interactions" = Google's
   Interactions API (gemini-3.5-transcribe). stt_audio_format optionally
   overrides the upload format ("" = m4a for OpenRouter, native bytes
   otherwise) for upstreams that reject m4a outright (MAI-Transcribe 2).

Model ids are "<provider>:<api_model>"; api_model is the exact string the
provider's API expects (e.g. "google/gemini-flash-latest" on OpenRouter,
"gemini-flash-latest" on Google's native API).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Provider:
    id: str
    label: str
    key_env: str | None          # env var seeding the API key (None = local)
    purposes: tuple[str, ...]    # which UI blocks offer it: transcribe/formatter


PROVIDERS: dict[str, Provider] = {
    p.id: p for p in [
        Provider("gemini", "Google Gemini", "GEMINI_API_KEY",
                 ("transcribe", "formatter")),
        Provider("openrouter", "OpenRouter", "OPENROUTER_API_KEY",
                 ("transcribe", "formatter")),
        Provider("groq", "Groq", "GROQ_API_KEY",
                 ("transcribe", "formatter")),
        Provider("openai", "OpenAI", "OPENAI_API_KEY",
                 ("transcribe", "formatter")),
        Provider("ollama", "Ollama (local, free)", None, ("formatter",)),
    ]
}


@dataclass(frozen=True)
class CloudModel:
    id: str            # "<provider>:<api_model>" — unique, used in settings
    name: str          # display name
    provider: str      # PROVIDERS key
    kind: str          # transcribe | audio_llm
    key_env: str       # env var holding the API key ("" for local)
    notes: str
    api_model: str     # exact model string for the provider's API
    stt_flavor: str = ""  # transcribe-only API shape (see module docstring)
    stt_audio_format: str = ""  # transcribe-only upload format override


# (api_model, display name, kind, notes[, stt_flavor]) — transcription block.
# Only models that can return word-level timestamps are listed: the karaoke
# player is built on per-word sweep, so plain-text STT endpoints are useless
# here and deliberately excluded.
TRANSCRIBE_MODELS: dict[str, list[tuple]] = {
    "gemini": [
        ("gemini-flash-latest", "Gemini Flash (latest)", "audio_llm",
         "Routes to the newest production Gemini Flash; listens + formats in one step"),
        ("gemini-pro-latest", "Gemini Pro (latest)", "audio_llm",
         "Routes to the newest production Gemini Pro; highest quality, slower"),
        ("gemini-flash-lite-latest", "Gemini Flash Lite (latest)", "audio_llm",
         "Lightweight/fast Gemini Flash Lite; listens + formats in one step"),
        ("gemini-3.5-transcribe", "Gemini 3.5 Transcribe", "transcribe",
         "Google's dedicated ASR model; measured word timings; needs a "
         "formatter LLM (block 2)", "gemini_interactions"),
    ],
    "openrouter": [
        ("~google/gemini-flash-latest", "Gemini Flash (latest)", "audio_llm",
         "Newest production Gemini Flash via OpenRouter"),
        ("~google/gemini-pro-latest", "Gemini Pro (latest)", "audio_llm",
         "Newest production Gemini Pro via OpenRouter"),
        ("google/gemini-3.5-flash-lite", "Gemini 3.5 Flash Lite", "audio_llm",
         "Lightweight Gemini 3.5 Flash; fast and cheap"),
        ("openai/gpt-audio", "GPT Audio", "audio_llm",
         "OpenAI flagship native audio engine (text output)"),
        ("openai/gpt-audio-mini", "GPT Audio Mini", "audio_llm",
         "Fast, low-latency OpenAI audio model"),
        ("mistralai/voxtral-small-24b-2507", "Voxtral Small 24B", "audio_llm",
         "Mistral audio model; strong speech transcription (greedy sampling)"),
        ("meta/muse-spark-1.3", "Muse Spark 1.3", "audio_llm",
         "Meta multimodal audio model (privacy-locked tier)"),
        ("meta/muse-spark-1.3-contributor", "Muse Spark 1.3 (contributor)", "audio_llm",
         "Discounted tier; prompts may be used for Meta training"),
        ("thinkingmachines/inkling", "Inkling 41B", "audio_llm",
         "Open-weight 41B MoE with native audio comprehension"),
        ("thinkingmachines/inkling-small", "Inkling Small 12B", "audio_llm",
         "Efficient 12B MoE variant, fast audio reasoning"),
        ("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", "Nemotron 3 Nano Omni (free)", "audio_llm",
         "Lightweight NVIDIA omni model (free tier, reasoning)"),
        ("xiaomi/mimo-v2.5", "MiMo v2.5", "audio_llm",
         "Xiaomi open-weights omni model"),
        # --- transcription-only models via OpenRouter's STT endpoint ---
        # (every entry probe-verified on 2026-09-14: honors verbose_json AND
        # returns words for sung/music audio, which is the whole point here)
        ("microsoft/mai-transcribe-2", "MAI-Transcribe 2", "transcribe",
         "Microsoft's multilingual ASR (top of FLEURS); measured word "
         "timings; needs a formatter LLM (block 2)", "openrouter_stt",
         "mp3"),  # upstream 400s on m4a; everything else takes m4a

        ("qwen/qwen3-asr-0.6b", "Qwen3 ASR 0.6B", "transcribe",
         "Tiny open-weights ASR, near-free; measured word timings; needs a "
         "formatter LLM (block 2)", "openrouter_stt"),
        ("fish-audio/transcribe-1", "Fish Audio Transcribe 1", "transcribe",
         "Fish Audio ASR with language auto-detect; measured word timings; "
         "needs a formatter LLM (block 2)", "openrouter_stt"),
        ("openai/whisper-large-v3", "Whisper large v3 (via OpenRouter)", "transcribe",
         "Whisper billed through OpenRouter; measured word timings; needs a "
         "formatter LLM (block 2)", "openrouter_stt"),
        ("openai/whisper-large-v3-turbo", "Whisper large v3 turbo (via OpenRouter)", "transcribe",
         "Faster/cheaper Whisper billed through OpenRouter; measured word "
         "timings; needs a formatter LLM (block 2)", "openrouter_stt"),
        # deliberately NOT listed (no word timings for songs -> can't drive
        # the karaoke sweep; all probe-verified 2026-09-14):
        # - meta/muse-voice-transcribe-1.0: Meta-hosted only, plain text out
        # - mistralai/voxtral-small-24b-2507-stt, mistralai/voxtral-mini-transcribe,
        #   qwen/qwen3-asr-flash-2026-02-10: upstream rejects verbose_json
        #   ("does not support response_format verbose_json. Use json") -> text
        #   only via OpenRouter. Native Mistral/Alibaba APIs do have word
        #   timestamps — re-add if direct provider support is ever added.
        # - deepgram/nova-3, x-ai/grok-stt-1.0: SPEECH-ONLY upstreams — they
        #   transcribe spoken audio perfectly (59s TTS clip -> 168 timed words)
        #   but return an empty 200 (no words, no text) for sung/music audio,
        #   even a 30 s slice. Not length-related; chunking can't help.
        # - openai/gpt-4o-transcribe[-mini] via OpenRouter: same verbose_json
        #   rejection; use the native openai provider rows below instead.
    ],
    "groq": [
        ("whisper-large-v3", "Whisper large v3", "transcribe",
         "Fast word-level timestamps; needs a formatter LLM (block 2)"),
        ("whisper-large-v3-turbo", "Whisper large v3 turbo", "transcribe",
         "Cheaper/faster; needs a formatter LLM (block 2)"),
    ],
    "openai": [
        ("whisper-1", "Whisper", "transcribe",
         "Word-level timestamps; needs a formatter LLM (block 2)"),
        ("gpt-4o-transcribe", "GPT-4o Transcribe", "transcribe",
         "Strong transcription; segment timestamps only; needs a formatter"),
        ("gpt-4o-mini-transcribe", "GPT-4o mini Transcribe", "transcribe",
         "Cheaper transcription; segment timestamps only; needs a formatter"),
    ],
}

# (api_model, display name) per provider — formatter block (block 2).
FORMATTER_MODELS: dict[str, list[tuple[str, str]]] = {
    "ollama": [],   # populated dynamically from /api/tags
    "gemini": [
        ("gemini-flash-latest", "Gemini Flash (latest)"),
        ("gemini-pro-latest", "Gemini Pro (latest)"),
        ("gemini-flash-lite-latest", "Gemini Flash Lite (latest)"),
    ],
    # the omni list doubles as text formatters (they all output text);
    # transcription-only STT models are excluded (no chat surface)
    "openrouter": [(m[0], m[1]) for m in TRANSCRIBE_MODELS["openrouter"]
                   if m[2] == "audio_llm"],
    "groq": [
        ("llama-3.3-70b-versatile", "Llama 3.3 70B"),
        ("llama-3.1-8b-instant", "Llama 3.1 8B (fast)"),
    ],
    "openai": [
        ("gpt-4o-mini", "GPT-4o mini"),
        ("gpt-4o", "GPT-4o"),
    ],
}

_DEFAULT_FORMATTER_MODEL = {
    "ollama": "qwen2.5:3b",
    "gemini": "gemini-flash-latest",
    "openrouter": "google/gemini-flash-latest",
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
}


def _catalog() -> list[CloudModel]:
    out: list[CloudModel] = []
    for pid, models in TRANSCRIBE_MODELS.items():
        for entry in models:
            api_model, name, kind, notes = entry[:4]
            out.append(CloudModel(
                id=f"{pid}:{api_model}", name=name, provider=pid, kind=kind,
                key_env=PROVIDERS[pid].key_env or "", notes=notes,
                api_model=api_model,
                stt_flavor=entry[4] if len(entry) > 4 else "",
                stt_audio_format=entry[5] if len(entry) > 5 else ""))
    return out


CATALOG: list[CloudModel] = _catalog()
_BY_ID = {m.id: m for m in CATALOG}

# old native ids -> new catalog ids (settings migration)
_MIGRATE = {
    "gemini:gemini-2.5-flash": "gemini:gemini-flash-latest",
    "gemini:gemini-2.5-pro": "gemini:gemini-pro-latest",
    "gemini:google/gemini-flash-latest": "gemini:gemini-flash-latest",
    "gemini:google/gemini-pro-latest": "gemini:gemini-pro-latest",
    # upstream slugs that were renamed/retired
    "openrouter:xiaomi/mimo-v2-omni": "openrouter:xiaomi/mimo-v2.5",
    "openrouter:thinking-machines/inkling": "openrouter:thinkingmachines/inkling",
    "openrouter:thinking-machines/inkling-small": "openrouter:thinkingmachines/inkling-small",
    "openrouter:nvidia/nemotron-3-nano-omni":
        "openrouter:nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    # STT slugs retired from the catalog on 2026-09-14 (verified: no word
    # timestamps for songs — verbose_json rejection or speech-only upstreams);
    # remount onto a verified transcriber so saved settings keep working
    "openrouter:mistralai/voxtral-small-24b-2507-stt":
        "openrouter:openai/whisper-large-v3-turbo",
    "openrouter:mistralai/voxtral-mini-transcribe":
        "openrouter:openai/whisper-large-v3-turbo",
    "openrouter:qwen/qwen3-asr-flash-2026-02-10":
        "openrouter:openai/whisper-large-v3-turbo",
    "openrouter:deepgram/nova-3": "openrouter:openai/whisper-large-v3-turbo",
    "openrouter:x-ai/grok-stt-1.0": "openrouter:openai/whisper-large-v3-turbo",
}


def get_model(model_id: str) -> CloudModel | None:
    return _BY_ID.get(_MIGRATE.get(model_id, model_id))


def provider_key(provider: str) -> str:
    """API key for a provider: UI-saved key first, then the env seed."""
    info = PROVIDERS.get(provider)
    if info is None or not info.key_env:
        return ""
    try:
        from app.settings import settings
        saved = (settings.get().get("keys") or {}).get(provider, "")
    except Exception:
        saved = ""
    if saved and saved.strip():
        return saved.strip()
    return os.environ.get(info.key_env, "").strip()


def key_source(provider: str) -> str:
    """Where the active key comes from: "saved" | "env" | ""."""
    info = PROVIDERS.get(provider)
    if info is None or not info.key_env:
        return ""
    try:
        from app.settings import settings
        saved = (settings.get().get("keys") or {}).get(provider, "")
    except Exception:
        saved = ""
    if saved and saved.strip():
        return "saved"
    return "env" if os.environ.get(info.key_env, "").strip() else ""


def group_label(kind: str) -> str:
    """Dropdown section header for a model kind (see the web combo box)."""
    if kind == "audio_llm":
        return "Audio LLM · listens + formats in one step"
    return "Transcriber · measured word timings"


def list_models() -> list[dict]:
    """Catalog for the UI: availability + display fields + dropdown group."""
    out = []
    for m in CATALOG:
        d = asdict(m)
        d["available"] = bool(provider_key(m.provider))
        d["group"] = group_label(m.kind)
        out.append(d)
    return out


def any_available() -> bool:
    return any(m["available"] for m in list_models())


def list_providers() -> list[dict]:
    """Providers for the UI dropdowns + key-field prefill state."""
    out = []
    for p in PROVIDERS.values():
        out.append({
            "id": p.id,
            "label": p.label,
            "key_env": p.key_env or "",
            "purposes": list(p.purposes),
            "has_key": bool(provider_key(p.id)) or p.id == "ollama",
            "key_source": key_source(p.id),
        })
    return out


def formatter_default(provider: str) -> str:
    return _DEFAULT_FORMATTER_MODEL.get(provider, "")


def formatter_catalog() -> dict:
    """Per-provider formatter model lists for the UI (block 2 dropdowns)."""
    return {pid: [{"api_model": api, "name": name} for api, name in models]
            for pid, models in FORMATTER_MODELS.items()}


def resolved_transcribe_model() -> str:
    """The effective transcribe-model id from settings (block 1 of the UI)."""
    from app.settings import settings
    s = settings.get()
    tm = (s.get("transcribe_model") or "").strip()
    if tm:
        return f"{s.get('transcribe_provider') or 'gemini'}:{tm}"
    legacy = s.get("cloud_model") or ""
    return _MIGRATE.get(legacy, legacy)
