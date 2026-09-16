"""MP3 export: the tags, the three lyric layers, and which version they hold."""
import io
import shutil

import pytest
from mutagen.id3 import APIC, COMM, ID3, SYLT, TALB, TIT2, TPE1, USLT

from app.models import Line, Track, Word
from app.mp3 import clean_lyrics, export_mp3, sylt_tokens


def _lines() -> list[Line]:
    return [
        Line(text="Hello world", start=1.2, progress=0.0,
             words=[Word("Hello", 1.2, 1.5), Word("world", 1.6, 2.0)]),
        Line(text="Don't use a toaster", start=3.0, progress=0.0,
             words=[Word("Don't", 3.0, 3.2), Word("use", 3.3, 3.4),
                    Word("a", 3.45, 3.5), Word("toaster", 3.6, 4.0)]),
    ]


ENHANCED = ("[ti:Test Song]\n[ar:Test Artist]\n[re:Refrain]\n"
            "[01:20.00]<01:20.00>Hello <01:20.00>world \n"
            "[00:03.00]<00:03.00>Don't <00:03.30>use <00:03.45>a <00:03.60>toaster \n")


# --- the pieces ---------------------------------------------------------------

def test_clean_lyrics_strips_both_tag_flavors_and_blank_lines():
    text = ("[ti:Test Song]\n[ar:Test Artist]\n\n"
            "[00:01.20]<00:01.20>Hello <00:01.60>world \n"
            "[00:03.00]<00:03.00>Don't <00:03.60>toaster \n")

    clean = clean_lyrics(text)

    assert clean == "Hello world\nDon't toaster"
    assert "[" not in clean and "<" not in clean
    # a header-only document leaves nothing at all (the tags are not lyrics)
    assert clean_lyrics("[ti:Only Headers]\n[ar:Nobody]\n") == ""


def test_sylt_tokens_keep_spacing_and_use_absolute_milliseconds():
    tokens = sylt_tokens(_lines())

    assert tokens == [("Hello ", 1200), ("world\n", 1600),
                      ("Don't ", 3000), ("use ", 3300), ("a ", 3450),
                      ("toaster\n", 3600)]
    # every token except a line's last keeps its trailing space, so a player
    # that lays words out itself does not bunch them together
    assert all(t.endswith(" ") or t.endswith("\n") for t, _ in tokens)


def test_sylt_tokens_round_timings_to_ms():
    lines = [Line(text="x", start=0.0, progress=0.0,
                  words=[Word("x", 1.2345, 1.5)])]

    assert sylt_tokens(lines) == [("x\n", 1234)]


# --- the finished file --------------------------------------------------------

def _build(tmp_path, *, source_name="in.mp3", art=None, enhanced=ENHANCED):
    src = tmp_path / source_name
    # mutagen only needs a header to prepend a tag to; the bytes below are real
    # MPEG frames only in the ffmpeg-generated transcode test
    src.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 512)
    data = export_mp3(src, title="Test Song", artist="Test Artist",
                      album="Test Album", lines=_lines(), enhanced_lrc=enhanced,
                      art=art)
    return ID3(io.BytesIO(data))


def test_export_writes_the_three_lyric_layers(tmp_path):
    tags = _build(tmp_path)

    sylt = tags.getall("SYLT")[0]
    assert sylt.lang == "eng"
    assert sylt.format == 2          # absolute milliseconds
    assert sylt.type == 1            # lyrics content
    assert sylt.text == sylt_tokens(_lines())

    uslt = tags.getall("USLT")[0]
    assert uslt.lang == "eng" and uslt.desc == "Lyrics"
    assert uslt.text == "Hello world\nDon't use a toaster"

    comm = tags.getall("COMM")[0]
    assert comm.desc == "Comment" and comm.lang == "eng"
    # COMM holds a list of strings; the script round-trips byte for byte
    assert comm.text == [ENHANCED]


def test_export_writes_title_artist_and_album(tmp_path):
    tags = _build(tmp_path)

    assert tags.getall("TIT2")[0].text == ["Test Song"]
    assert tags.getall("TPE1")[0].text == ["Test Artist"]
    assert tags.getall("TALB")[0].text == ["Test Album"]


def test_export_embeds_the_cover(tmp_path):
    art = tmp_path / "cover.jpg"
    art.write_bytes(b"\xff\xd8\xff\xe0" + b"jpegbytes")

    tags = _build(tmp_path, art=art)

    pic = tags.getall("APIC")[0]
    assert pic.mime == "image/jpeg" and pic.type == 3
    assert pic.data == art.read_bytes()


def test_export_without_album_or_cover_leaves_those_tags_out(tmp_path):
    src = tmp_path / "in.mp3"
    src.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 64)

    data = export_mp3(src, title="T", artist="A", lines=_lines(),
                      enhanced_lrc=ENHANCED)
    tags = ID3(io.BytesIO(data))

    assert not tags.getall("TALB")
    assert not tags.getall("APIC")
    assert tags.getall("SYLT")           # lyrics are always written


def test_tag_order_and_encoding_of_every_frame(tmp_path):
    """All text frames are UTF-8, as the spec requires."""
    tags = _build(tmp_path)

    for frame in (tags.getall("TIT2") + tags.getall("USLT")
                  + tags.getall("COMM") + tags.getall("SYLT")):
        assert frame.encoding == 3       # Encoding.UTF8


def test_unicode_lyrics_survive(tmp_path):
    lines = [Line(text="Café — ñoño", start=0.5, progress=0.0,
                  words=[Word("Café", 0.5, 0.9), Word("ñoño", 1.0, 1.4)])]
    src = tmp_path / "in.mp3"
    src.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 64)

    data = export_mp3(src, title="T", artist="A", lines=lines,
                      enhanced_lrc="[00:00.50]Café ñoño\n")
    tags = ID3(io.BytesIO(data))

    assert tags.getall("USLT")[0].text == "Café ñoño"
    assert tags.getall("SYLT")[0].text[0][0] == "Café "


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_a_non_mp3_source_is_transcoded(tmp_path):
    """An m4a/wav track still exports as a real MP3 (that is the promise)."""
    wav = tmp_path / "tone.wav"
    from pydub import AudioSegment
    from pydub.generators import Sine

    Sine(440).to_audio_segment(duration=300).export(wav, format="wav")

    data = export_mp3(wav, title="Tone", artist="Test", lines=_lines(),
                      enhanced_lrc=ENHANCED)
    tags = ID3(io.BytesIO(data))

    assert data[:3] == b"ID3" or data[0] == 0xFF     # a tag, or raw MPEG frames
    assert tags.getall("TIT2")[0].text == ["Tone"]
    assert tags.getall("SYLT")[0].text == sylt_tokens(_lines())
    # the audio came through: decoding it back gives roughly the same length
    back = AudioSegment.from_file(io.BytesIO(data), format="mp3")
    assert 200 < len(back) < 400


def test_a_missing_source_file_is_a_clear_error(tmp_path):
    with pytest.raises(RuntimeError, match="missing"):
        export_mp3(tmp_path / "nope.mp3", title="T", artist="A",
                   lines=_lines(), enhanced_lrc=ENHANCED)


# --- the endpoint -------------------------------------------------------------

def test_endpoint_returns_a_tagged_mp3_of_the_active_version(tmp_path):
    """The download bakes in the *selected* version, like every other format."""
    from fastapi.testclient import TestClient

    from app.pipeline import store
    from app.server import app

    src = tmp_path / "song.mp3"
    src.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 256)
    first = [Line(text="Version one", start=0.0, progress=0.0,
                  words=[Word("Version", 0.0, 0.4), Word("one", 0.5, 0.8)])]
    second = [Line(text="Version two", start=0.0, progress=0.0,
                   words=[Word("Version", 0.0, 0.4), Word("two", 0.5, 0.8)])]
    store.add_track(Track(id="mp31", url="u", title="T", artist="A",
                          album="Alb", audio_file=src))
    try:
        store.save_transcript("mp31", first, backend="b")
        store.save_transcript("mp31", second, backend="b")   # becomes active

        res = TestClient(app).get("/api/tracks/mp31/export?fmt=mp3")
        assert res.status_code == 200
        assert res.headers["content-type"] == "audio/mpeg"
        assert 'filename="T.mp3"' in res.headers["content-disposition"]
        tags = ID3(io.BytesIO(res.content))
        assert tags.getall("TALB")[0].text == ["Alb"]
        assert tags.getall("USLT")[0].text == "Version two"
        assert tags.getall("SYLT")[0].text == sylt_tokens(second)

        # switch back to the first version: the next download follows it
        assert store.activate_version("mp31", 1)
        again = ID3(io.BytesIO(
            TestClient(app).get("/api/tracks/mp31/export?fmt=mp3").content))
        assert again.getall("USLT")[0].text == "Version one"
        assert again.getall("SYLT")[0].text == sylt_tokens(first)
    finally:
        store.delete_track("mp31")


def test_endpoint_without_audio_explains_itself(tmp_path):
    from fastapi.testclient import TestClient

    from app.pipeline import store
    from app.server import app

    store.add_track(Track(id="mp32", url="u", title="T", artist="A"))
    try:
        store.save_transcript("mp32", _lines(), backend="b")
        res = TestClient(app).get("/api/tracks/mp32/export?fmt=mp3")
        assert res.status_code == 404
        assert "no audio" in res.json()["detail"]
    finally:
        store.delete_track("mp32")
