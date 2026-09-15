"""Cloud catalog integrity + settings migration for renamed model ids."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.cloud_models import (CATALOG, get_model, group_label, _MIGRATE,
                              formatter_catalog)


def test_openrouter_ids_are_current():
    ids = {m.id for m in CATALOG}
    assert "openrouter:xiaomi/mimo-v2.5" in ids
    assert "openrouter:thinkingmachines/inkling" in ids
    assert "openrouter:thinkingmachines/inkling-small" in ids
    assert "openrouter:nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free" in ids
    assert "openrouter:mistralai/voxtral-small-24b-2507" in ids
    # retired slugs are gone
    assert not any("mimo-v2-omni" in i for i in ids)
    assert not any("thinking-machines/" in i for i in ids)


def test_new_transcriber_models_in_catalog():
    ids = {m.id for m in CATALOG}
    # gemini's dedicated ASR model
    assert "gemini:gemini-3.5-transcribe" in ids
    # OpenRouter STT models, probe-verified to honor verbose_json AND return
    # words for sung/music audio
    for slug in ("microsoft/mai-transcribe-2", "qwen/qwen3-asr-0.6b",
                 "fish-audio/transcribe-1", "openai/whisper-large-v3",
                 "openai/whisper-large-v3-turbo"):
        assert f"openrouter:{slug}" in ids, slug
    # word-timestamp-less models are deliberately NOT in the catalog:
    # verbose_json rejection, no structured output, or speech-only upstreams
    assert "openrouter:meta/muse-voice-transcribe-1.0" not in ids
    assert "openrouter:mistralai/voxtral-small-24b-2507-stt" not in ids
    assert "openrouter:mistralai/voxtral-mini-transcribe" not in ids
    assert "openrouter:qwen/qwen3-asr-flash-2026-02-10" not in ids
    # speech-only upstreams: perfect on spoken audio, empty 200 on songs
    assert "openrouter:deepgram/nova-3" not in ids
    assert "openrouter:x-ai/grok-stt-1.0" not in ids
    assert "openrouter:openai/gpt-4o-transcribe" not in ids
    assert "openrouter:openai/gpt-4o-mini-transcribe" not in ids


def test_retired_stt_slugs_migrate_to_verified_transcriber():
    # saved settings pointing at removed STT slugs remount onto a working one
    for retired in ("openrouter:mistralai/voxtral-small-24b-2507-stt",
                    "openrouter:mistralai/voxtral-mini-transcribe",
                    "openrouter:qwen/qwen3-asr-flash-2026-02-10",
                    "openrouter:deepgram/nova-3",
                    "openrouter:x-ai/grok-stt-1.0"):
        assert get_model(retired).api_model == "openai/whisper-large-v3-turbo"


def test_stt_flavor_picked_per_provider():
    assert get_model("gemini:gemini-3.5-transcribe").stt_flavor \
        == "gemini_interactions"
    assert get_model("openrouter:openai/whisper-large-v3-turbo").stt_flavor \
        == "openrouter_stt"
    assert get_model("openrouter:qwen/qwen3-asr-0.6b").stt_flavor \
        == "openrouter_stt"
    # native multipart providers keep the default flavor
    assert get_model("groq:whisper-large-v3").stt_flavor == ""
    assert get_model("openai:whisper-1").stt_flavor == ""


def test_mai_transcribe_2_requires_mp3_upload():
    # probed 2026-09-14: MAI-2's upstream 400s on m4a outright ("Provider
    # returned 400") while every other OpenRouter STT model accepts it; it
    # returns 289 timed words for a full sung mp3
    m = get_model("openrouter:microsoft/mai-transcribe-2")
    assert m.stt_audio_format == "mp3"
    # every other OpenRouter STT model keeps the m4a default
    for mid in ("openrouter:qwen/qwen3-asr-0.6b", "openrouter:fish-audio/transcribe-1",
                "openrouter:openai/whisper-large-v3",
                "openrouter:openai/whisper-large-v3-turbo"):
        assert get_model(mid).stt_audio_format == "", mid


def test_group_labels_separate_kinds():
    assert "Audio LLM" in group_label("audio_llm")
    assert "Transcriber" in group_label("transcribe")
    audio = {m.id for m in CATALOG if group_label(m.kind).startswith("Audio LLM")}
    trans = {m.id for m in CATALOG if group_label(m.kind).startswith("Transcriber")}
    assert audio and trans
    assert not (audio & trans)


def test_renamed_ids_migrate():
    assert get_model("openrouter:xiaomi/mimo-v2-omni").api_model == "xiaomi/mimo-v2.5"
    assert get_model("openrouter:thinking-machines/inkling").api_model \
        == "thinkingmachines/inkling"
    assert get_model("openrouter:nvidia/nemotron-3-nano-omni").api_model \
        == "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"
    # every migration target exists in the catalog
    for target in _MIGRATE.values():
        assert get_model(target) is not None, target


def test_formatter_catalog_served_per_provider():
    cat = formatter_catalog()
    assert "ollama" in cat and cat["ollama"] == []
    or_models = [m["api_model"] for m in cat["openrouter"]]
    assert "xiaomi/mimo-v2.5" in or_models
    assert "mistralai/voxtral-small-24b-2507" in or_models
    assert all("name" in m for m in cat["gemini"])
    # transcription-only STT slugs must not leak into the formatter list
    assert "deepgram/nova-3" not in or_models
    assert "qwen/qwen3-asr-0.6b" not in or_models
