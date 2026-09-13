"""Enhanced LRC parsing + audio-LLM two-mode contract tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.cloud_transcribe import (
    _lines_to_result,
    _parse_enhanced_lrc,
    _parse_json_safe,
    _run_audio_llm,
)
from app.transcribe import TranscribeError


LRC = """[ti:Never Gonna Give You Up]
[by:Test]
[00:12.00]<00:12.00>We're <00:12.30>no <00:12.50>strangers <00:12.90>to <00:13.10>love
[00:13.50]<00:13.50>You <00:13.70>know <00:13.90>the <00:14.10>rules <00:14.40>and <00:14.60>so <00:14.80>do <00:15.00>I"""

LRC_FENCED = "```\n" + LRC + "\n```"
LRC_PLAIN = "[00:10.00]hello world again\n[00:20.00]second line here"


def test_parse_enhanced_lrc_words_and_spans():
    lines = _parse_enhanced_lrc(LRC)
    assert lines is not None and len(lines) == 2
    w0 = lines[0]["words"][0]
    assert w0["t"] == "We're" and w0["s"] == 12.0
    # word end = next word's start
    assert w0["e"] == 12.3
    # line-final word ends at the next line's first word start
    last = lines[0]["words"][-1]
    assert last["t"] == "love" and last["e"] == 13.5
    assert lines[0]["text"] == "We're no strangers to love"
    # metadata tags ignored
    assert "Test" not in lines[0]["text"]


def test_parse_enhanced_lrc_fenced_and_plain_lines():
    fenced = _parse_enhanced_lrc(LRC_FENCED)
    assert fenced is not None and fenced[0]["text"] == "We're no strangers to love"

    plain = _parse_enhanced_lrc(LRC_PLAIN)
    assert plain is not None and len(plain) == 2
    toks = plain[0]["words"]
    assert [w["t"] for w in toks] == ["hello", "world", "again"]
    # even interpolation across the line span
    assert toks[0]["s"] == 10.0 and toks[-1]["e"] == 20.0


def test_parse_enhanced_lrc_offset_and_garbage():
    off = _parse_enhanced_lrc("[offset:+1000]\n[00:10.00]<00:10.00>shifted <00:10.50>words")
    assert off is not None
    assert off[0]["words"][0]["s"] == 9.0  # +1000ms shifts earlier

    assert _parse_enhanced_lrc("not lrc at all") is None
    assert _parse_enhanced_lrc("") is None
    assert _parse_enhanced_lrc(None) is None


def test_lines_to_result_drops_bad_words_rebuilds_text():
    lines = [
        {"text": "keep these", "words": [
            {"t": "keep", "s": 1.0, "e": 1.5},
            {"t": "", "s": 1.5, "e": 1.6},        # empty text -> dropped
            {"t": "bad", "s": 2.0, "e": 1.0},      # e < s -> dropped
            {"t": "these", "s": 1.5, "e": 2.0},
        ]},
        {"text": "all bad", "words": [
            {"t": "x", "s": -1.0, "e": 0.0},       # negative -> dropped
        ]},
        "not-a-dict",
    ]
    result = _lines_to_result(lines, "test:backend")
    assert [w["text"] for w in result.words] == ["keep", "these"]
    assert result.segments[0].text == "keep these"
    assert result.backend == "test:backend"


def test_run_audio_llm_prefers_lrc_then_json_fallback():
    calls = []

    def lrc_then_junk(prompt, *, force_json):
        calls.append((prompt, force_json))
        if len(calls) == 1:
            return LRC                     # good LRC on the first call
        raise AssertionError("should not need the JSON fallback")

    result = _run_audio_llm(lrc_then_junk, lambda m: None, "test:x",
                            "Never Gonna Give You Up", "Rick Astley")
    assert result.backend == "test:x"
    assert result.words[0]["text"] == "We're"
    # the LRC prompt (not JSON) was used and carried the metadata
    assert "Enhanced LRC" in calls[0][0]
    assert "Never Gonna Give You Up" in calls[0][0]
    assert calls[0][1] is False


def test_run_audio_llm_falls_back_to_json_prompt():
    def bad_lrc_good_json(prompt, *, force_json):
        if not force_json:
            return "totally not lrc"       # unparseable -> fallback
        return ('{"lines": [{"text": "hello there", "words": '
                '[{"t": "hello", "s": 0.0, "e": 0.5},'
                '{"t": "there", "s": 0.5, "e": 1.0}]}]}')

    result = _run_audio_llm(bad_lrc_good_json, lambda m: None, "test:x",
                            "Song", "Artist")
    assert [w["text"] for w in result.words] == ["hello", "there"]


def test_run_audio_llm_raises_when_both_fail():
    def always_junk(prompt, *, force_json):
        return "still not parseable"

    try:
        _run_audio_llm(always_junk, lambda m: None, "test:x", "S", "A")
        raised = False
    except TranscribeError:
        raised = True
    assert raised


def test_parse_json_safe_tolerates_junk():
    assert _parse_json_safe("no json here") is None
    assert _parse_json_safe('{"lines": []}') is None
