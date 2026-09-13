"""Transcript storage: Enhanced LRC round-trip + legacy JSON loading."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models import Line, Store, Word


def make_lines():
    return [
        Line(text="We're no strangers to love", start=18.7, progress=0.0,
             words=[Word(text="We're", start=18.7, end=19.12),
                    Word(text="no", start=19.12, end=19.46),
                    Word(text="strangers", start=19.46, end=20.1),
                    Word(text="to", start=20.1, end=21.08),
                    Word(text="love", start=21.08, end=21.66)]),
        Line(text="You know the rules and so do I", start=23.1, progress=0.0,
             words=[Word(text="You", start=23.1, end=23.36),
                    Word(text="know", start=23.36, end=23.64),
                    Word(text="the", start=23.64, end=23.9),
                    Word(text="rules", start=23.9, end=24.78),
                    Word(text="and", start=24.78, end=25.6),
                    Word(text="so", start=25.6, end=25.86),
                    Word(text="do", start=25.86, end=26.14),
                    Word(text="I", start=26.14, end=26.54)]),
    ]


def test_lrc_roundtrip_preserves_words_and_grouping(tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.DATA_DIR", tmp_path)
    store = Store()
    track = __import__("app.models", fromlist=["Track"]).Track(
        id="t1", url="u", title="Never Gonna Give You Up", artist="Rick Astley")
    store.add_track(track)

    v = store.save_transcript("t1", make_lines(), backend="test:x")
    assert v == 1
    d = tmp_path / "transcripts" / "t1"
    assert (d / "v1.lrc").exists()
    text = (d / "v1.lrc").read_text(encoding="utf-8")
    assert "[ti:Never Gonna Give You Up]" in text
    assert "<00:21.08>love" in text                      # word start tags
    assert text.rstrip().endswith("<00:26.54>")          # trailing end tag

    loaded = store.load_transcript("t1")
    assert loaded is not None and len(loaded) == 2
    for orig, back in zip(make_lines(), loaded):
        assert back.text == orig.text
        assert back.start == orig.start
        assert [(w.text, w.start, w.end) for w in back.words] == \
               [(w.text, w.start, w.end) for w in orig.words]


def test_legacy_json_versions_still_load(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr("app.config.DATA_DIR", tmp_path)
    store = Store()
    track = __import__("app.models", fromlist=["Track"]).Track(
        id="t2", url="u", title="T", artist="A")
    store.add_track(track)

    d = tmp_path / "transcripts" / "t2"
    d.mkdir(parents=True)
    payload = [{"text": "hello world", "start": 1.0,
                "words": [{"text": "hello", "start": 1.0, "end": 1.5},
                          {"text": "world", "start": 1.5, "end": 2.0}]}]
    (d / "v1.json").write_text(json.dumps(payload), encoding="utf-8")
    (d / "versions.json").write_text(json.dumps(
        {"active": 1, "versions": [{"version": 1, "createdAt": 0.0}]}),
        encoding="utf-8")

    loaded = store.load_transcript("t2")
    assert loaded is not None and len(loaded) == 1
    assert loaded[0].text == "hello world"
    assert loaded[0].words[-1].end == 2.0


def test_versions_and_activation_on_lrc_storage(tmp_path, monkeypatch):
    monkeypatch.setattr("app.config.DATA_DIR", tmp_path)
    store = Store()
    track = __import__("app.models", fromlist=["Track"]).Track(
        id="t3", url="u", title="T", artist="A")
    store.add_track(track)

    v1 = store.save_transcript("t3", make_lines(), backend="a")
    one_word = [Line(text="only one", start=0.0, progress=0.0,
                     words=[Word(text="only", start=0.0, end=0.5),
                            Word(text="one", start=0.5, end=1.0)])]
    v2 = store.save_transcript("t3", one_word, backend="b", note="rerun")
    assert (v1, v2) == (1, 2)

    idx = store.list_versions("t3")
    assert idx["active"] == 2 and len(idx["versions"]) == 2
    assert idx["versions"][1]["fmt"] == "lrc"

    assert store.load_transcript("t3")[0].text == "only one"
    assert store.activate_version("t3", 1)
    back = store.load_transcript("t3")
    assert back[0].text == "We're no strangers to love"
    assert back[0].words[-1].end == 21.66
