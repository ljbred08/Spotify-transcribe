"""Payload parsing for the OpenRouter STT and Gemini Interactions flavors."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.cloud_models import get_model
from app.cloud_transcribe import (_stt_result_from_payload, build,
                                  GeminiTranscribe, OpenRouterSTT,
                                  TranscriptProvider)


def test_stt_payload_parses_words_and_segments():
    m = get_model("openrouter:microsoft/mai-transcribe-2")
    result = _stt_result_from_payload(m, {
        "words": [
            {"word": "Hello", "start": 0.0, "end": 0.4},
            {"word": "  ", "start": 0.4, "end": 0.5},     # blank -> dropped
            {"word": "world", "start": "0.4", "end": "1.2"},  # str ok
            {"start": 2.0, "end": 2.5},                    # no text -> dropped
            {"word": "bad", "start": 5.0, "end": 4.0},     # e < s -> dropped
        ],
        "segments": [{"text": "Hello world", "start": 0.0, "end": 1.2}],
    })
    assert [w["text"] for w in result.words] == ["Hello", "world"]
    assert result.words[1]["start"] == 0.4 and result.words[1]["end"] == 1.2
    assert result.segments[0].text == "Hello world"
    assert result.backend == "openrouter:microsoft/mai-transcribe-2"


def test_stt_payload_falls_back_to_segment_interpolation():
    m = get_model("groq:whisper-large-v3")
    result = _stt_result_from_payload(m, {
        "text": "hello there",
        "segments": [{"text": "hello there", "start": 1.0, "end": 3.0}],
    })
    assert [w["text"] for w in result.words] == ["hello", "there"]
    assert result.words[0]["start"] == 1.0 and result.words[-1]["end"] == 3.0


def test_stt_payload_raises_without_any_timings():
    import pytest
    from app.transcribe import TranscribeError
    m = get_model("openrouter:qwen/qwen3-asr-0.6b")
    with pytest.raises(TranscribeError):
        _stt_result_from_payload(m, {"text": "words but no timings"})


def test_build_routes_stt_flavors():
    assert isinstance(build("openrouter:deepgram/nova-3"), OpenRouterSTT)
    assert isinstance(build("gemini:gemini-3.5-transcribe"), GeminiTranscribe)
    assert isinstance(build("groq:whisper-large-v3"), TranscriptProvider)
    assert isinstance(build("openai:gpt-4o-transcribe"), TranscriptProvider)


def _fake_audio(tmp_path):
    p = tmp_path / "x.m4a"
    p.write_bytes(b"fake-audio-bytes")
    return p


def test_openrouter_stt_request_body(tmp_path):
    m = get_model("openrouter:fish-audio/transcribe-1")
    backend = OpenRouterSTT(m)

    captured = {}
    class FakeResp:
        status_code = 200
        def json(self):
            return {"words": [{"word": "hi", "start": 0.0, "end": 0.3}]}
    def fake_post(url, **kw):
        captured["url"] = url
        captured["json"] = kw["json"]
        return FakeResp()

    import app.cloud_transcribe as ct
    orig = ct.httpx.post
    ct.httpx.post = fake_post
    try:
        result = backend.transcribe(_fake_audio(tmp_path), lambda msg: None)
    finally:
        ct.httpx.post = orig
    assert captured["url"].endswith("/audio/transcriptions")
    body = captured["json"]
    assert body["model"] == "fish-audio/transcribe-1"
    assert body["response_format"] == "verbose_json"
    assert body["timestamp_granularities"] == ["segment", "word"]
    assert body["input_audio"]["format"] == "m4a"
    assert result.words[0]["text"] == "hi"


def test_openrouter_stt_converts_to_model_format(tmp_path):
    # MAI-Transcribe 2 rejects m4a: the backend must transcode to mp3 and
    # label the payload accordingly (format string matches the actual bytes)
    m = get_model("openrouter:microsoft/mai-transcribe-2")
    backend = OpenRouterSTT(m)

    converted = tmp_path / "converted.mp3"
    converted.write_bytes(b"converted-mp3-bytes")
    captured = {}
    calls = []
    class FakeResp:
        status_code = 200
        def json(self):
            return {"words": [{"word": "hi", "start": 0.0, "end": 0.3}]}
    def fake_post(url, **kw):
        calls.append(kw["json"])
        return FakeResp()

    import app.cloud_transcribe as ct
    orig_post, orig_conv = ct.httpx.post, ct._ffmpeg_convert
    ct.httpx.post = fake_post
    ct._ffmpeg_convert = lambda p, fmt: converted
    try:
        backend.transcribe(_fake_audio(tmp_path), lambda msg: None)
    finally:
        ct.httpx.post = orig_post
        ct._ffmpeg_convert = orig_conv
    body = calls[0]
    assert body["model"] == "microsoft/mai-transcribe-2"
    assert body["input_audio"]["format"] == "mp3"
    assert body["input_audio"]["data"] == "Y29udmVydGVkLW1wMy1ieXRlcw=="
    # the temp transcode is cleaned up after the request
    assert not converted.exists()


def test_openrouter_stt_m4a_models_skip_conversion(tmp_path):
    # default-format models upload the original file untouched (no ffmpeg)
    m = get_model("openrouter:fish-audio/transcribe-1")
    backend = OpenRouterSTT(m)
    seen = []
    class FakeResp:
        status_code = 200
        def json(self):
            return {"words": [{"word": "hi", "start": 0.0, "end": 0.3}]}
    import app.cloud_transcribe as ct
    orig_post, orig_conv = ct.httpx.post, ct._ffmpeg_convert
    ct.httpx.post = lambda url, **kw: seen.append(kw) or FakeResp()
    ct._ffmpeg_convert = lambda p, fmt: (_ for _ in ()).throw(
        AssertionError("ffmpeg must not run for m4a models"))
    try:
        backend.transcribe(_fake_audio(tmp_path), lambda msg: None)
    finally:
        ct.httpx.post = orig_post
        ct._ffmpeg_convert = orig_conv
    assert seen[0]["json"]["input_audio"]["format"] == "m4a"


def test_gemini_transcribe_request_and_annotations(tmp_path):
    m = get_model("gemini:gemini-3.5-transcribe")
    backend = GeminiTranscribe(m)

    captured = {}
    class FakeResp:
        status_code = 200
        def json(self):
            return {"steps": [{"content": [{
                "type": "text",
                "text": "Hello world",
                "annotations": [
                    {"type": "word_info", "text": "Hello",
                     "start_offset": "0.100s", "end_offset": "0.450s"},
                    {"type": "word_info", "text": "world",
                     "start_offset": "0.500s", "end_offset": "0.850s"},
                    {"type": "word_info", "text": "junk",
                     "start_offset": "weird", "end_offset": "0.9s"},
                ],
            }]}]}
    def fake_post(url, **kw):
        captured["url"] = url
        captured["json"] = kw["json"]
        return FakeResp()

    import app.cloud_transcribe as ct
    orig = ct.httpx.post
    ct.httpx.post = fake_post
    try:
        result = backend.transcribe(_fake_audio(tmp_path), lambda msg: None)
    finally:
        ct.httpx.post = orig
    assert captured["url"] == \
        "https://generativelanguage.googleapis.com/v1beta/interactions"
    body = captured["json"]
    assert body["model"] == "gemini-3.5-transcribe"
    # Interactions input items carry data/mime_type directly (NOT the
    # generateContent-style inline_data nesting, which 400s)
    assert body["input"][0] == {"type": "audio", "data": "ZmFrZS1hdWRpby1ieXRlcw==",
                                "mime_type": "audio/mp4"}
    assert "inline_data" not in body["input"][0]
    assert body["generation_config"]["transcription_config"]["mode"] == {
        "type": "verbatim", "timestamp_granularities": ["word"]}
    assert [w["text"] for w in result.words] == ["Hello", "world"]
    assert result.words[0]["start"] == 0.1 and result.words[1]["end"] == 0.85
    assert result.words == sorted(result.words, key=lambda w: w["start"])
