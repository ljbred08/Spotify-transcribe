"""Tests for line assembly and all export formats."""
from app.exporters import assemble_lines, export
from app.models import Line, Word
from app.transcribe import Segment, TranscriptionResult


def words(*pairs):
    return [{"text": t, "start": s, "end": e} for t, s, e in pairs]


def test_gap_splits_lines():
    result = TranscriptionResult(
        segments=[], backend="test",
        words=words(("hello", 0.0, 0.4), ("world", 0.5, 0.9),   # tight
                    ("never", 2.5, 3.0), ("more", 3.1, 3.6)),   # gap before
    )
    assembly = assemble_lines(result)
    assert len(assembly.lines) == 2
    assert assembly.lines[0].text == "hello world"
    assert assembly.lines[1].text == "never more"


def test_long_group_is_split():
    long_words = words(*[(f"w{i}", i * 0.1, i * 0.1 + 0.09) for i in range(30)])
    assembly = assemble_lines(TranscriptionResult(segments=[], backend="test",
                                                  words=long_words))
    assert all(len(l.text) <= 42 * 1.4 + 8 for l in assembly.lines)
    assert len(assembly.lines) > 1


def test_line_text_joins_words():
    assembly = assemble_lines(TranscriptionResult(
        segments=[], backend="test",
        words=words(("one", 0.0, 0.3), ("two", 0.35, 0.6)),
    ))
    assert assembly.text == "one two"
    assert assembly.lines[0].words[0].text == "one"


def test_lrc_format():
    lines = [Line(text="la la", start=61.234, progress=0.0,
                  words=[Word(text="la", start=61.234, end=61.5),
                         Word(text="la", start=61.6, end=61.9)])]
    out = export("lrc", "Song", "Artist", lines)
    assert out.startswith("[ti:Song]\n[ar:Artist]")
    assert "[01:01.23]la la" in out


def test_lrc_enhanced_word_tags():
    lines = [Line(text="la la", start=61.234, progress=0.0,
                  words=[Word(text="la", start=61.234, end=61.5),
                         Word(text="la", start=61.6, end=61.9)])]
    out = export("lrc-enhanced", "Song", "Artist", lines)
    assert out.startswith("[ti:Song]\n[ar:Artist]")
    assert "[01:01.23]<01:01.23>la <01:01.60>la" in out


def test_srt_cues():
    lines = [Line(text="hi", start=0.0, progress=0.0,
                  words=[Word(text="hi", start=0.0, end=1.5)])]
    out = export("srt", "t", "a", lines)
    assert "1\n00:00:00,000 --> 00:00:01,500\nhi" in out


def test_vtt_header():
    out = export("vtt", "t", "a", [])
    assert out.startswith("WEBVTT")


def test_txt_and_json():
    lines = [Line(text="hey", start=1.0, progress=0.0,
                  words=[Word(text="hey", start=1.0, end=2.0)])]
    assert export("txt", "t", "a", lines) == "hey\n"
    import json
    payload = json.loads(export("json", "Song", "Artist", lines))
    assert payload["title"] == "Song"
    assert payload["lines"][0]["end"] == 2.0


def test_unknown_format_raises():
    try:
        export("xml", "t", "a", [])
        assert False, "should have raised"
    except ValueError:
        pass
