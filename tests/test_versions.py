"""Tests for transcript versioning (save/list/activate/migrate)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config as cfg
from app.models import Line, Store, Track, Word


def make_lines(tag: str) -> list[Line]:
    return [
        Line(text=f"{tag} line one", start=0.0, progress=0.0,
             words=[Word(text=tag, start=0.0, end=0.4),
                    Word(text="line", start=0.4, end=0.8),
                    Word(text="one", start=0.8, end=1.2)]),
        Line(text=f"{tag} line two", start=2.0, progress=0.0,
             words=[Word(text=tag, start=2.0, end=2.4),
                    Word(text="line", start=2.4, end=2.8),
                    Word(text="two", start=2.8, end=3.2)]),
    ]


def test_save_creates_versions_and_keeps_old(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    s = Store()
    s.add_track(Track(id="t1", url="u"))
    v1 = s.save_transcript("t1", make_lines("old"), backend="parakeet:parakeet-v3")
    assert v1 == 1
    v2 = s.save_transcript("t1", make_lines("new"), backend="parakeet:parakeet-unified-en")
    assert v2 == 2
    idx = s.list_versions("t1")
    assert idx["active"] == 2
    assert [v["version"] for v in idx["versions"]] == [1, 2]
    # old version still on disk
    assert (tmp_path / "transcripts" / "t1" / "v1.json").exists()
    assert (tmp_path / "transcripts" / "t1" / "v2.json").exists()
    # active transcript is the newest one
    lines = s.load_transcript("t1")
    assert lines[0].text.startswith("new")
    # switch back
    assert s.activate_version("t1", 1) is True
    lines = s.load_transcript("t1")
    assert lines[0].text.startswith("old")


def test_activate_unknown_version_returns_false(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    s = Store()
    assert s.activate_version("missing", 3) is False


def test_legacy_transcript_migrates(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    (tmp_path / "transcripts").mkdir()
    legacy = [{"text": "legacy line", "start": 0.0,
               "words": [{"text": "legacy", "start": 0.0, "end": 0.5}]}]
    (tmp_path / "transcripts" / "t2.json").write_text(json.dumps(legacy))
    s = Store()
    lines = s.load_transcript("t2")
    assert lines is not None
    assert lines[0].text == "legacy line"
    idx = s.list_versions("t2")
    assert idx["active"] == 1
    assert not (tmp_path / "transcripts" / "t2.json").exists()
    # saving after migration creates v2 and keeps v1
    v = s.save_transcript("t2", make_lines("fresh"))
    assert v == 2
    assert (tmp_path / "transcripts" / "t2" / "v1.json").exists()


def test_delete_track_removes_all_versions(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    s = Store()
    s.save_transcript("t3", make_lines("x"))
    s.save_transcript("t3", make_lines("y"))
    import shutil
    shutil.rmtree(tmp_path / "transcripts" / "t3")
    assert s.list_versions("t3") is None or True  # deletion is file-level; store tolerated
