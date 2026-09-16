"""Metadata pipeline tests: Shazam (audio fingerprint) first, then keyless Deezer,
then Shazam's text search, and Genius only when a token is configured — with
never a hard failure for the caller.

No network and no Shazam/Deezer/Genius credentials: every hop is stubbed.
"""
from __future__ import annotations

import asyncio
import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import app.config as cfg
from app import metadata
from app.metadata import (
    _album_from_sections,
    _ext_for,
    _norm_title,
    _titles_match,
    art_for,
    fetch_metadata,
)


# --- helpers ---------------------------------------------------------------

def shazam_payload(
    title: str = "Never Gonna Give You Up",
    artist: str = "Rick Astley",
    album: str = "Whenever You Need Somebody",
    cover: str = "https://img.test/cover.jpg",
) -> dict:
    """The shape shazamio's ``recognize`` returns for a hit."""
    return {"matches": [{"id": "1"}], "track": {
        "title": title, "subtitle": artist, "images": {"coverart": cover},
        "sections": [{"type": "SONG", "metadata": [
            {"title": "Album", "text": album}]}],
    }}


def run(coro):
    return asyncio.run(coro)


PNG = b"\x89PNG\r\n\x1a\n" + b"fake image body"
JPEG = b"\xff\xd8\xff\xe0" + b"fake image body"


class _Resp:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self) -> None:
        pass


class _Client:
    """Minimal async httpx client stand-in; counts downloads."""
    calls: list[str] = []
    body: bytes = JPEG

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, **kw):
        type(self).calls.append(url)
        return _Resp(type(self).body)


@pytest.fixture
def art_dir(tmp_path, monkeypatch):
    """Point the app's data dir at a temp copy so covers land in the sandbox."""
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    _Client.calls = []
    _Client.body = JPEG
    monkeypatch.setattr("httpx.AsyncClient", _Client)
    return tmp_path


def stub_hops(monkeypatch, recognize):
    """Patch every Shazam/Deezer/Genius hop so only the given one can succeed."""
    async def miss(*a, **kw):
        return None

    monkeypatch.setattr(metadata, "_shazam_recognize", recognize)
    monkeypatch.setattr(metadata, "_deezer_search", miss)
    monkeypatch.setattr(metadata, "_shazam_search", miss)
    monkeypatch.setattr(metadata, "_genius_lyrics_search", miss)


# The last hop searches Genius by lyric line, so its stub takes the lyrics first.
LYRICS = "Everything I ever learned that gets me through the worst"


def stub_all_hops(monkeypatch, **overrides):
    """Patch all four hops; any hop not overridden is a miss."""
    async def miss(*a, **kw):
        return None

    for name in ("_shazam_recognize", "_deezer_search", "_shazam_search",
                 "_genius_lyrics_search"):
        monkeypatch.setattr(metadata, name, overrides.get(name, miss))


# --- Shazam: fingerprinting the audio file --------------------------------

def test_recognize_parses_track_and_album(tmp_path, monkeypatch):
    """A Shazam hit gives us title, artist, album and the biggest cover URL."""
    audio = tmp_path / "song.m4a"
    audio.write_bytes(b"\x00\x01")

    async def fake_recognize(self, data, proxy=None, options=None):
        assert str(data).endswith("song.m4a")     # fingerprints OUR file
        return shazam_payload()

    monkeypatch.setattr("shazamio.Shazam.recognize", fake_recognize)
    meta = run(metadata._shazam_recognize(audio))
    assert meta["title"] == "Never Gonna Give You Up"
    assert meta["artist"] == "Rick Astley"
    assert meta["album"] == "Whenever You Need Somebody"
    assert meta["art_url"] == "https://img.test/cover.jpg"
    assert meta["source"] == "shazam"


def test_recognize_without_track_is_a_miss(tmp_path, monkeypatch):
    """Shazam answering "no match" is a miss, not an exception."""
    audio = tmp_path / "song.m4a"
    audio.write_bytes(b"\x00\x01")

    async def no_match(self, data, proxy=None, options=None):
        return {"matches": [], "timestamp": 1}

    monkeypatch.setattr("shazamio.Shazam.recognize", no_match)
    assert run(metadata._shazam_recognize(audio)) is None


def test_recognize_survives_api_errors(tmp_path, monkeypatch):
    """A network/timeout blow-up must not escape into the pipeline."""
    audio = tmp_path / "song.m4a"
    audio.write_bytes(b"\x00\x01")

    async def boom(self, data, proxy=None, options=None):
        raise RuntimeError("connection reset")

    monkeypatch.setattr("shazamio.Shazam.recognize", boom)
    assert run(metadata._shazam_recognize(audio)) is None


def test_missing_audio_file_skips_recognition(tmp_path, monkeypatch):
    async def must_not_run(self, data, proxy=None, options=None):
        raise AssertionError("must not fingerprint a file that isn't there")

    monkeypatch.setattr("shazamio.Shazam.recognize", must_not_run)
    assert run(metadata._shazam_recognize(tmp_path / "gone.m4a")) is None


# --- pipeline order and fallbacks ------------------------------------------

def test_pipeline_prefers_the_audio_fingerprint(monkeypatch, tmp_path):
    """All four hops available: only Shazam's fingerprint runs."""
    calls: list[str] = []

    async def rec(audio):
        calls.append("recognize")
        return {"title": "T", "artist": "A", "album": "", "art_url": None,
                "source": "shazam"}

    async def miss(name):
        calls.append(name)
        return None

    monkeypatch.setattr(metadata, "_shazam_recognize", rec)
    monkeypatch.setattr(metadata, "_deezer_search", lambda *a, **kw: miss("deezer"))
    monkeypatch.setattr(metadata, "_shazam_search", lambda *a, **kw: miss("search"))
    monkeypatch.setattr(metadata, "_genius_lyrics_search", lambda *a, **kw: miss("genius"))
    meta = run(fetch_metadata("T", "A", track_id="t1", audio_path=tmp_path / "a.mp3"))
    assert calls == ["recognize"]
    assert meta == {"title": "T", "artist": "A", "album": "", "art_path": None,
                    "art_url": None, "source": "shazam"}


def test_pipeline_asks_deezer_second(monkeypatch, tmp_path):
    """No fingerprint match: Deezer is method #2, before any Genius token check."""
    calls: list[str] = []

    async def rec(audio):
        calls.append("recognize")
        return None

    async def deezer(title, artist):
        calls.append(f"deezer:{artist} {title}".strip())
        return {"title": title, "artist": artist, "album": "Timeless",
                "art_url": "https://cdn-images.dzcdn.net/cover.jpg", "source": "deezer"}

    async def search(title, artist):
        calls.append("search")
        return None

    async def genius(lyrics):
        calls.append("genius")
        return None

    monkeypatch.setattr(metadata, "_shazam_recognize", rec)
    monkeypatch.setattr(metadata, "_deezer_search", deezer)
    monkeypatch.setattr(metadata, "_shazam_search", search)
    monkeypatch.setattr(metadata, "_genius_lyrics_search", genius)
    audio = tmp_path / "t.m4a"
    audio.write_bytes(b"\x00")
    meta = run(fetch_metadata("Timeless", "Home Free", track_id="t2", audio_path=audio))
    assert calls == ["recognize", "deezer:Home Free Timeless"]
    assert meta["source"] == "deezer" and meta["album"] == "Timeless"


def test_pipeline_falls_back_to_shazam_text_search(monkeypatch, tmp_path):
    """Deezer missing too: Shazam's text search still gets its shot before Genius."""
    calls: list[str] = []

    async def rec(audio):
        calls.append("recognize")
        return None

    async def deezer(title, artist):
        calls.append("deezer")
        return None

    async def search(title, artist):
        calls.append(f"search:{artist} {title}")
        return {"title": title, "artist": artist, "album": "Alb",
                "art_url": None, "source": "shazam-search"}

    async def genius(lyrics):
        calls.append("genius")
        return None

    monkeypatch.setattr(metadata, "_shazam_recognize", rec)
    monkeypatch.setattr(metadata, "_deezer_search", deezer)
    monkeypatch.setattr(metadata, "_shazam_search", search)
    monkeypatch.setattr(metadata, "_genius_lyrics_search", genius)
    meta = run(fetch_metadata("Timeless", "Home Free", track_id="t2b"))
    assert calls == ["deezer", "search:Home Free Timeless"]
    assert meta["source"] == "shazam-search" and meta["album"] == "Alb"


def test_pipeline_reaches_genius_lyrics_only_after_both_first(monkeypatch):
    """Genius is the last resort — and only with our lyrics to search for."""
    calls: list[str] = []

    async def rec(audio):
        calls.append("recognize")
        return None

    async def deezer(title, artist):
        calls.append("deezer")
        return None

    async def search(title, artist):
        calls.append("search")
        return None

    async def genius(lyrics):
        calls.append(f"genius:{lyrics[:20]}")
        return {"title": "T", "artist": "A", "album": "", "art_url": None,
                "source": "genius-lyrics"}

    monkeypatch.setattr(metadata, "_shazam_recognize", rec)
    monkeypatch.setattr(metadata, "_deezer_search", deezer)
    monkeypatch.setattr(metadata, "_shazam_search", search)
    monkeypatch.setattr(metadata, "_genius_lyrics_search", genius)
    meta = run(fetch_metadata("T", "A", track_id="t3", lyrics=LYRICS))
    assert calls == ["deezer", "search", f"genius:{LYRICS[:20]}"]
    assert meta["source"] == "genius-lyrics"


def test_pipeline_skips_genius_without_lyrics(monkeypatch):
    """No transcript yet (a fresh download): nothing to lyric-search, so it's skipped."""
    async def miss(*a, **kw):
        return None

    monkeypatch.setattr(metadata, "_shazam_recognize", miss)
    monkeypatch.setattr(metadata, "_deezer_search", miss)
    monkeypatch.setattr(metadata, "_shazam_search", miss)
    monkeypatch.setattr(metadata, "_genius_lyrics_search",
                        lambda *a, **kw: pytest.fail("no lyrics → no lyric search"))
    assert run(fetch_metadata("T", "A", track_id="t3b")) is None


# --- Deezer (method 2) -----------------------------------------------------

class _DeezerResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self):
        return self._payload


class _DeezerClient:
    """Stand-in httpx client returning one canned Deezer payload."""
    payload: dict = {"data": []}
    calls: list[tuple] = []

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, **kw):
        type(self).calls.append((url, kw.get("params")))
        return _DeezerResp(type(self).payload)


@pytest.fixture
def deezer(monkeypatch):
    _DeezerClient.payload = {"data": []}
    _DeezerClient.calls = []
    monkeypatch.setattr("httpx.AsyncClient", _DeezerClient)
    return _DeezerClient


def _deezer_track(title="Timeless", artist="Home Free", album="Timeless",
                  cover="https://cdn-images.dzcdn.net/images/cover/abc/1000x1000.jpg",
                  track_id="101", duration=200) -> dict:
    return {
        "id": track_id,
        "title": title,
        "artist": {"name": artist},
        "album": {"title": album, "cover_xl": cover},
        "duration": duration,
    }


def _deezer_hit(**kw) -> dict:
    return {"data": [_deezer_track(**kw)]}


def test_deezer_returns_title_artist_album_and_big_cover(deezer):
    deezer.payload = _deezer_hit()
    meta = run(metadata._deezer_search("Timeless", "Home Free"))
    assert meta == {
        "id": "101", "title": "Timeless", "artist": "Home Free",
        "album": "Timeless",
        "art_url": "https://cdn-images.dzcdn.net/images/cover/abc/1000x1000.jpg",
        "duration": 200, "source": "deezer",
    }
    url, params = deezer.calls[0]
    assert url.endswith("/search") and params == {"q": "Home Free Timeless", "limit": 25}
    assert len(deezer.calls) == 1                       # a hit stops the search


def test_deezer_needs_no_key(deezer, monkeypatch):
    """Deezer is keyless: it must not consult any settings or token."""
    monkeypatch.setattr(metadata, "_genius_token",
                        lambda: pytest.fail("Deezer must not ask for a token"))
    deezer.payload = _deezer_hit()
    assert run(metadata._deezer_search("Timeless", "Home Free"))["source"] == "deezer"


def test_deezer_rejects_a_different_song(deezer):
    """A hit for someone else's track is refused — a wrong cover is worse than none."""
    deezer.payload = _deezer_hit(title="Footloose", artist="Kenny Loggins")
    assert run(metadata._deezer_search("Mama Says - FULL MIX", "Unknown artist")) is None


def test_deezer_accepts_a_qualified_version_of_our_title(deezer):
    """\"Timeless (Deluxe)\" still answers a search for \"Timeless\"."""
    deezer.payload = _deezer_hit(title="Timeless (Deluxe)")
    meta = run(metadata._deezer_search("Timeless", "Home Free"))
    assert meta["title"] == "Timeless (Deluxe)"


def test_deezer_prefers_our_artist_over_a_same_titled_song(deezer):
    """Text search ranks a stranger's exact title above our artist's version."""
    deezer.payload = {"data": [
        {"title": "Timeless", "artist": {"name": "Some Cover Band"},
         "album": {"title": "Covers", "cover_xl": "https://img.test/wrong.jpg"}},
        {"title": "Timeless (Deluxe)", "artist": {"name": "Home Free"},
         "album": {"title": "Timeless", "cover_xl": "https://img.test/right.jpg"}},
    ]}
    meta = run(metadata._deezer_search("Timeless", "Home Free"))
    assert meta["artist"] == "Home Free"
    assert meta["art_url"] == "https://img.test/right.jpg"


def test_deezer_ignores_a_placeholder_artist(deezer):
    """"Unknown artist" must not earn an artist bonus for anybody."""
    deezer.payload = {"data": [
        {"title": "Mama Says", "artist": {"name": "Unknown artist"},
         "album": {"title": "A", "cover_xl": "https://img.test/a.jpg"}},
    ]}
    assert metadata._hit_score("Mama Says", "Unknown artist",
                               deezer.payload["data"][0]) == 2   # title only
    assert metadata._is_generic_artist("Unknown artist")
    assert not metadata._is_generic_artist("Home Free")


def test_deezer_refuses_a_strangers_cover_when_we_know_the_artist(deezer):
    """A known artist means art-that-isn't-ours is worse than no art at all."""
    deezer.payload = {"data": [
        {"title": "Broken Hallelujah", "artist": {"name": "Don Savant"},
         "album": {"title": "Home", "cover_xl": "https://img.test/wrong.jpg"}},
    ]}
    assert run(metadata._deezer_search("Hallelujah", "Home Free")) is None


def test_deezer_title_only_search_when_the_artist_is_unknown(deezer):
    """YouTube keeps artist as "Unknown artist" — there, the title is all we have."""
    deezer.payload = {"data": [
        {"title": "Timeless (Deluxe)", "artist": {"name": "Home Free"},
         "album": {"title": "Timeless", "cover_xl": "https://img.test/right.jpg"}},
    ]}
    meta = run(metadata._deezer_search("Timeless", "Unknown artist"))
    assert meta["artist"] == "Home Free" and meta["album"] == "Timeless"
    # …and the placeholder artist is kept out of the query itself
    assert deezer.calls[0][1] == {"q": "Timeless", "limit": 25}


def test_deezer_skips_bad_hits_to_find_a_good_one(deezer):
    deezer.payload = {"data": [
        {"title": "Timelessness", "artist": {"name": "X"}, "album": {}},
        *(_deezer_hit()["data"]),
    ]}
    assert run(metadata._deezer_search("Timeless", "Home Free"))["album"] == "Timeless"


def test_deezer_errors_are_a_quiet_miss(deezer, monkeypatch):
    """Rate limits, timeouts and junk payloads must not break the pipeline."""
    deezer.payload = {"error": {"type": "Exception", "message": "Quota exceeded"}}
    assert run(metadata._deezer_search("Timeless", "Home Free")) is None
    deezer.payload = ["not", "a", "dict"]
    assert run(metadata._deezer_search("Timeless", "Home Free")) is None

    class Boom(_DeezerClient):
        async def get(self, url, **kw):
            raise RuntimeError("connection reset")

    monkeypatch.setattr("httpx.AsyncClient", Boom)
    assert run(metadata._deezer_search("Timeless", "Home Free")) is None


def test_deezer_without_a_title_is_skipped(deezer):
    assert run(metadata._deezer_search("", "Home Free")) is None
    assert deezer.calls == []


# --- query hygiene: ask the API something it can answer --------------------

def test_query_drops_version_and_mix_tails():
    """Our titles come from YouTube/filenames; indexes don't use those tails."""
    assert metadata._clean_query("Mama Says - FULL MIX") == "Mama Says"
    assert metadata._clean_query("Song (Official Video)") == "Song"
    assert metadata._clean_query("Song - Official Music Video") == "Song"
    assert metadata._clean_query("Song - Live at Wembley") == "Song - Live at Wembley"
    assert metadata._clean_query("T-Pain Buy U a Drank") == "T-Pain Buy U a Drank"
    assert metadata._clean_query("Never Gonna Give You Up") == "Never Gonna Give You Up"
    assert metadata._clean_query("") == ""


def test_queries_try_the_raw_title_before_the_cleaned_one():
    assert metadata._deezer_queries("Mama Says - FULL MIX", "") == \
        ["Mama Says - FULL MIX", "Mama Says"]
    assert metadata._deezer_queries("Timeless", "Unknown artist") == ["Timeless"]
    # a known artist is tried first, but never *instead of* the bare title
    assert metadata._deezer_queries("Timeless", "Home Free") == \
        ["Home Free Timeless", "Timeless"]
    assert metadata._deezer_queries("Mama Says - FULL MIX", "Tom Plotkin") == \
        ["Tom Plotkin Mama Says - FULL MIX", "Mama Says - FULL MIX",
         "Tom Plotkin Mama Says", "Mama Says"]


def test_picker_still_lists_other_songs_when_our_artist_is_wrong(monkeypatch):
    """The artist-scoped query answers with one hit; the picker needs the rest."""
    asked: list[str] = []
    footloose = _deezer_track(title="Mama Says (You Can't Back Down)",
                              artist="Tom Plotkin", album="Footloose", track_id="4")
    others = [_deezer_track(title="Mama Says", artist=who, track_id=str(i))
              for i, who in enumerate(["Ibeyi", "The Beach Boys", "jael"])]

    class ByQuery(_DeezerClient):
        async def get(self, url, **kw):
            query = (kw.get("params") or {}).get("q")
            asked.append(query)
            if query.startswith("Tom Plotkin"):        # narrow: Deezer knows one
                return _DeezerResp({"data": [footloose]})
            return _DeezerResp({"data": [footloose, *others]})

    monkeypatch.setattr("httpx.AsyncClient", ByQuery)
    found = run(metadata.find_candidates("Mama Says - FULL MIX", "Tom Plotkin"))
    assert asked[0] == "Tom Plotkin Mama Says - FULL MIX"
    assert "Mama Says" in asked                          # the broad net was cast
    assert {c["artist"] for c in found} == {"Tom Plotkin", "Ibeyi",
                                             "The Beach Boys", "jael"}
    assert len(found) == 4                               # no duplicates either


def test_deezer_retries_with_the_cleaned_query(deezer, monkeypatch):
    """Asked verbatim Deezer returns junk; the cleaned title finds the song."""
    asked: list[str] = []

    class ByQuery(_DeezerClient):
        async def get(self, url, **kw):
            query = kw["params"]["q"]
            asked.append(query)
            if query == "Mama Says - FULL MIX":      # what the real API does
                return _DeezerResp({"data": [_deezer_track(title="CALLS (missed)",
                                                            artist="Juice")]})
            return _DeezerResp({"data": [_deezer_track(
                title="Mama Says", artist="jael", album="Mama Says")]})

    monkeypatch.setattr("httpx.AsyncClient", ByQuery)
    meta = run(metadata._deezer_search("Mama Says - FULL MIX", ""))
    assert asked == ["Mama Says - FULL MIX", "Mama Says"]
    assert meta["artist"] == "jael" and meta["art_url"].endswith("1000x1000.jpg")


# --- ambiguous titles: hand over a shortlist instead of guessing -----------

MAMA_SAYS = {"data": [
    _deezer_track(title="Mama Says (Recorded at Deezer, Paris)", artist="Ibeyi",
                  album="Deezer Sessions", track_id="1"),
    _deezer_track(title="Mama Says (Remastered 2001)", artist="The Beach Boys",
                  album="Wild Honey", track_id="2"),
    _deezer_track(title="Mama Says", artist="jael", album="Mama Says", track_id="3"),
    _deezer_track(title="Mama Says (You Can't Back Down)", artist="Tom Plotkin",
                  album="Footloose: The Musical (Original Broadway Cast Recording)",
                  track_id="4", duration=306),
]}


def test_ambiguous_title_is_not_guessed(deezer):
    """Four artists sing "Mama Says" — picking one silently would be a coin flip."""
    deezer.payload = MAMA_SAYS
    assert run(metadata._deezer_search("Mama Says - FULL MIX", "")) is None


def test_a_unique_title_is_still_applied_automatically(deezer):
    """One artist across every best hit is unambiguous — no click needed."""
    deezer.payload = {"data": [
        _deezer_track(title="Boots In It", artist="Jordan Rowe", track_id="9"),
        _deezer_track(title="Boots In It (Live)", artist="Jordan Rowe", track_id="10"),
    ]}
    meta = run(metadata._deezer_search("Boots In It", ""))
    assert meta["artist"] == "Jordan Rowe" and meta["id"] == "9"


def test_candidates_rank_the_right_song_into_the_shortlist(deezer):
    """The shortlist carries the Footloose cast recording the user wants."""
    deezer.payload = MAMA_SAYS
    found = run(metadata.find_candidates("Mama Says - FULL MIX", ""))
    assert [c["id"] for c in found] == ["1", "2", "3", "4"]   # Deezer's order on ties
    footloose = [c for c in found if c["id"] == "4"][0]
    assert footloose["artist"] == "Tom Plotkin"
    assert footloose["album"].startswith("Footloose: The Musical")
    assert footloose["art_url"].endswith("1000x1000.jpg")
    assert footloose["duration"] == 306


def test_candidates_dedupe_across_query_variants(deezer, monkeypatch):
    """Both the raw and the cleaned query run — one song, one row."""
    asked: list[str] = []

    class Echo(_DeezerClient):
        async def get(self, url, **kw):
            asked.append(kw["params"]["q"])
            return _DeezerResp(MAMA_SAYS)

    monkeypatch.setattr("httpx.AsyncClient", Echo)
    found = run(metadata.find_candidates("Mama Says - FULL MIX", ""))
    assert asked == ["Mama Says - FULL MIX", "Mama Says"]
    assert [c["id"] for c in found] == ["1", "2", "3", "4"]


def test_candidates_skip_hits_without_art_or_id(deezer):
    """A candidate we cannot show a cover for is useless in the picker."""
    deezer.payload = {"data": [
        {"id": "7", "title": "Mama Says", "artist": {"name": "jael"},
         "album": {"title": "Mama Says"}},
        _deezer_track(title="Mama Says", artist="The Tibbs", track_id="8"),
    ]}
    found = run(metadata.find_candidates("Mama Says", ""))
    assert [c["id"] for c in found] == ["8"]


def test_candidates_of_a_nonsense_title_are_empty(deezer):
    deezer.payload = {"data": [_deezer_track(title="Something Else", artist="Nobody")]}
    assert run(metadata.find_candidates("Mama Says - FULL MIX", "")) == []


# --- lyrics: the transcript picks the search line, LRCLIB judges candidates --

MAMA_SAYS_TRANSCRIPT = """
Everything I ever learned that gets me through the worst
I learned at my mama's knee
Now listen up, Mama says: Don't use a toaster while standin' in the shower (prrr!)
Now who can argue with that?
"""

BEACH_BOYS_LYRICS = """
All the guys that turned away
I've been thinking about the way
Mama says a lot of things
"""

FOOTLOOSE_LYRICS = """
Everything I ever learned that gets me through the worst
I learned at my mama's knee
Now listen up, mama says don't use a toaster while standing in the shower
The woman knows where it's at
"""


def test_lyrics_text_flattens_lines_or_strings():
    class _Line:
        def __init__(self, text):
            self.text = text

    assert metadata.lyrics_text([_Line("one"), _Line("two")]) == "one\ntwo"
    assert metadata.lyrics_text(["one", "two"]) == "one\ntwo"
    assert metadata.lyrics_text(None) == ""


def test_pick_lyric_line_prefers_a_long_distinctive_line():
    """Genius matches a phrase, so send the longest line that isn't a chorus."""
    picked = metadata.pick_lyric_line(MAMA_SAYS_TRANSCRIPT)
    assert picked.startswith("now listen up")
    assert "prrr" not in picked                     # ASR noise, not in any lyric sheet

    repeated = "hold on tight\nhold on tight\nhold on tight\na longer line that appears once here\n"
    assert metadata.pick_lyric_line(repeated).startswith("a longer line")
    assert metadata.pick_lyric_line("too short\n") == "too short"   # fallback: longest
    assert metadata.pick_lyric_line("") == ""


def test_lyrics_similarity_separates_right_song_from_same_titled_ones():
    """Calibrated thresholds live in a wide gap: 1.00 for ours, <0.5 for others."""
    same = metadata.lyrics_similarity(MAMA_SAYS_TRANSCRIPT, FOOTLOOSE_LYRICS)
    other = metadata.lyrics_similarity(MAMA_SAYS_TRANSCRIPT, BEACH_BOYS_LYRICS)
    assert same >= metadata.LYRICS_MATCH_AT
    assert other <= metadata.LYRICS_MISMATCH_AT
    assert metadata.lyrics_similarity(MAMA_SAYS_TRANSCRIPT, "") == 0.0


def test_lyrics_similarity_tolerates_asr_slips():
    """A few mis-heard words must not flip the verdict."""
    slipped = FOOTLOOSE_LYRICS.replace("toaster", "toast of").replace("knee", "needs")
    assert metadata.lyrics_similarity(MAMA_SAYS_TRANSCRIPT, slipped) >= \
        metadata.LYRICS_MATCH_AT


class _LrclibClient:
    """Serves canned lyric rows from LRCLIB's search endpoint."""
    by_title: dict = {}
    asked: list[dict] = []

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, **kw):
        params = kw.get("params") or {}
        type(self).asked.append((url, params))
        rows = type(self).by_title.get(params.get("track_name"), [])
        return _AnyResp(rows)


@pytest.fixture
def lrclib(monkeypatch):
    _LrclibClient.by_title = {}
    _LrclibClient.asked = []
    monkeypatch.setattr("httpx.AsyncClient", _LrclibClient)
    return _LrclibClient


class _RoutingClient:
    """One stubbed client serving both APIs — Deezer and LRCLIB — by URL."""
    deezer_payload: dict = {"data": []}
    lyric_rows: dict = {}

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, **kw):
        if "lrclib" in url:
            key = (kw.get("params") or {}).get("track_name")
            return _AnyResp(type(self).lyric_rows.get(key, []))
        return _AnyResp(type(self).deezer_payload)


@pytest.fixture
def deezer_and_lrclib(monkeypatch):
    """For the tests that need the search *and* the lyric check in one run."""
    _RoutingClient.deezer_payload = {"data": []}
    _RoutingClient.lyric_rows = {}
    monkeypatch.setattr("httpx.AsyncClient", _RoutingClient)
    return _RoutingClient


def test_lyrics_verdict_marks_both_a_match_and_a_veto(lrclib):
    candidate = {"title": "Mama Says (You Can't Back Down)", "artist": "Tom Plotkin"}
    lrclib.by_title = {"Mama Says": [{"plainLyrics": FOOTLOOSE_LYRICS}]}
    assert run(metadata._lyrics_verdict(MAMA_SAYS_TRANSCRIPT, candidate)) == "match"

    lrclib.by_title = {"Mama Says": [{"plainLyrics": BEACH_BOYS_LYRICS}]}
    assert run(metadata._lyrics_verdict(MAMA_SAYS_TRANSCRIPT, candidate)) == "mismatch"

    assert lrclib.asked[0][0].endswith("/api/search")
    assert lrclib.asked[0][1]["track_name"] == "Mama Says"      # qualifier stripped
    assert lrclib.asked[0][1]["artist_name"] == "Tom Plotkin"


def test_lyrics_verdict_is_none_without_lyrics_on_file(lrclib):
    """No lyrics means no verdict — we never veto what we cannot check."""
    lrclib.by_title = {"Mama Says": []}
    verdict = run(metadata._lyrics_verdict(MAMA_SAYS_TRANSCRIPT,
                                           {"title": "Mama Says", "artist": "Nobody"}))
    assert verdict is None


def test_lrclib_lookup_works_without_a_token(lrclib, monkeypatch):
    monkeypatch.setattr(metadata, "_genius_token",
                        lambda: pytest.fail("LRCLIB must not need a Genius token"))
    lrclib.by_title = {"Mama Says": [{"plainLyrics": FOOTLOOSE_LYRICS}]}
    assert run(metadata._lrclib_lyrics("Mama Says - FULL MIX", "")) == FOOTLOOSE_LYRICS
    assert "artist_name" not in lrclib.asked[0][1]      # blank artist isn't sent


# --- Genius lyric search ---------------------------------------------------

def test_genius_lyrics_search_sends_a_transcript_line(monkeypatch):
    """The query is a phrase from *our* transcript, not a title we guessed."""
    seen: dict = {}

    class FakeGenius:
        def __init__(self, *a, **kw):
            pass

        def search_lyrics(self, term):
            seen["term"] = term
            return {"response": {"sections": [{"hits": [
                {"type": "song", "result": {
                    "title": "Mama Says (You Can't Back Down)",
                    "primary_artist": {"name": "Tom Plotkin"},
                    "album": {"name": "Footloose: The Musical"},
                    "song_art_image_url": "https://images.genius.com/x.jpg",
                }},
            ]}]}}

    monkeypatch.setattr(metadata, "_genius_token", lambda: "token")
    monkeypatch.setattr("lyricsgenius.Genius", FakeGenius)
    meta = run(metadata._genius_lyrics_search(MAMA_SAYS_TRANSCRIPT))
    assert seen["term"].startswith("now listen up")
    assert meta["title"] == "Mama Says (You Can't Back Down)"
    assert meta["artist"] == "Tom Plotkin" and meta["source"] == "genius-lyrics"
    assert meta["art_url"] == "https://images.genius.com/x.jpg"


def test_genius_lyric_search_survives_a_flat_response(monkeypatch):
    """Genius has answered with bare ``response.hits`` too — handle both shapes."""
    class FakeGenius:
        def __init__(self, *a, **kw):
            pass

        def search_lyrics(self, term):
            return {"response": {"hits": [
                {"result": {"title": "Song", "primary_artist": {"name": "A"}}}]}}

    monkeypatch.setattr(metadata, "_genius_token", lambda: "token")
    monkeypatch.setattr("lyricsgenius.Genius", FakeGenius)
    assert run(metadata._genius_lyrics_search(LYRICS))["title"] == "Song"

    class Junk(FakeGenius):
        def search_lyrics(self, term):
            return {"response": {"sections": [{"hits": []}]}}

    monkeypatch.setattr("lyricsgenius.Genius", Junk)
    assert run(metadata._genius_lyrics_search(LYRICS)) is None


def test_same_recording_keeps_remixes_apart():
    """A fingerprint hit must not be deduped away by a remix of itself."""
    assert metadata.same_recording("Timeless", "Home Free", "Timeless", "Home Free")
    assert metadata.same_recording("Timeless!", "Home Free", "timeless", "Home Free")
    assert metadata.same_recording("Timeless", "", "Timeless", "Home Free")
    assert not metadata.same_recording("Timeless (Cake Mix)", "Rick Astley",
                                       "Timeless", "Rick Astley")
    assert not metadata.same_recording("Timeless", "Rick Astley",
                                       "Timeless", "Home Free")
    assert not metadata.same_recording("Timeless", "Home Free",
                                       "Other Song", "Home Free")


def test_genius_lyrics_search_needs_a_line(monkeypatch):
    monkeypatch.setattr(metadata, "_genius_token", lambda: "token")
    monkeypatch.setattr("lyricsgenius.Genius",
                        lambda *a, **kw: pytest.fail("nothing to search for"))
    assert run(metadata._genius_lyrics_search("")) is None


# --- the shortlist gets judged so the panel stays readable -----------------


def test_candidates_sort_confirmed_first_then_vetoed_last(deezer, monkeypatch):
    """Readability: the panel shows the confirmed song, then buries the vetoes."""
    deezer.payload = MAMA_SAYS
    verdicts = {"1": "mismatch", "2": None, "3": "match", "4": "mismatch"}

    async def fake_verdict(lyrics, candidate):
        assert lyrics == MAMA_SAYS_TRANSCRIPT
        return verdicts[candidate["id"]]

    monkeypatch.setattr(metadata, "_lyrics_verdict", fake_verdict)
    found = run(metadata.find_candidates("Mama Says - FULL MIX", "",
                                         lyrics=MAMA_SAYS_TRANSCRIPT))
    assert [(c["id"], c["match"]) for c in found] == [
        ("3", "match"), ("2", None), ("1", "mismatch"), ("4", "mismatch")]


def test_candidates_get_judged_against_lrclib(deezer_and_lrclib):
    client = deezer_and_lrclib
    client.deezer_payload = {"data": [_deezer_track(title="Mama Says", artist="jael",
                                                   track_id="3")]}
    client.lyric_rows = {"Mama Says": [{"plainLyrics": FOOTLOOSE_LYRICS}]}
    found = run(metadata.find_candidates("Mama Says", "", lyrics=MAMA_SAYS_TRANSCRIPT))
    assert found[0]["match"] == "match"


def test_candidates_are_tiered_by_artist_agreement(monkeypatch):
    """Strict = our artist agrees, i.e. the title+artist search would find it."""
    payload = {"data": [
        _deezer_track(title="Timeless", artist="The Weeknd", track_id="2"),
        _deezer_track(title="Timeless", artist="Home Free", track_id="1"),
        _deezer_track(title="Timeless (Live)", artist="Home Free", track_id="3"),
    ]}

    class Client(_DeezerClient):
        async def get(self, url, **kw):
            return _DeezerResp(payload)

    monkeypatch.setattr("httpx.AsyncClient", Client)
    found = run(metadata.find_candidates("Timeless", "Home Free"))
    assert {c["id"]: c["tier"] for c in found} == {
        "1": "strict", "3": "strict", "2": "broad"}
    assert [c["id"] for c in found][:2] == ["1", "3"]      # the strict tier leads


def test_without_an_artist_every_candidate_is_broad(monkeypatch):
    """No artist to agree with: the lyrics check becomes the only filter."""
    class Client(_DeezerClient):
        async def get(self, url, **kw):
            return _DeezerResp(MAMA_SAYS)

    monkeypatch.setattr("httpx.AsyncClient", Client)
    found = run(metadata.find_candidates("Mama Says - FULL MIX", ""))
    assert {c["tier"] for c in found} == {"broad"}


def test_candidates_without_lyrics_are_unjudged(deezer_and_lrclib):
    """A candidate LRCLIB knows nothing about stays visible, not vetoed."""
    client = deezer_and_lrclib
    client.deezer_payload = {"data": [_deezer_track(title="Mama Says", artist="jael")]}
    client.lyric_rows = {}
    found = run(metadata.find_candidates("Mama Says", "", lyrics=MAMA_SAYS_TRANSCRIPT))
    assert [c["match"] for c in found] == [None]


def test_candidates_are_not_judged_without_a_transcript(deezer, monkeypatch):
    """No lyrics of our own — nothing to judge by, so no verdicts at all."""
    deezer.payload = MAMA_SAYS
    monkeypatch.setattr(metadata, "_lyrics_verdict",
                        lambda *a, **kw: pytest.fail("no transcript → no check"))
    found = run(metadata.find_candidates("Mama Says - FULL MIX", ""))
    assert [c.get("match") for c in found] == [None] * 4


# --- applying a picked candidate -------------------------------------------

class _AnyResp:
    """A response that serves either JSON (the API) or bytes (the image CDN)."""

    def __init__(self, payload=None, content=b""):
        self._payload = payload
        self.content = content

    def raise_for_status(self) -> None:
        pass

    def json(self):
        return self._payload


class _ByUrlClient:
    """Routes /track/<id> to canned tracks and everything else to image bytes."""
    tracks: dict = {}
    asked: list[str] = []

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, **kw):
        type(self).asked.append(url)
        if "/track/" in url:
            track_id = url.rstrip("/").rsplit("/", 1)[-1]
            return _AnyResp(type(self).tracks.get(track_id,
                                                  {"error": {"message": "no"}}))
        return _AnyResp(content=JPEG)


@pytest.fixture
def deezer_by_id(monkeypatch):
    _ByUrlClient.tracks = {}
    _ByUrlClient.asked = []
    monkeypatch.setattr("httpx.AsyncClient", _ByUrlClient)
    return _ByUrlClient


FOOTLOOSE = _deezer_track(
    title="Mama Says (You Can't Back Down)", artist="Tom Plotkin",
    album="Footloose: The Musical (Original Broadway Cast Recording)",
    cover="https://cdn-images.dzcdn.net/images/cover/ff/1000x1000.jpg",
    track_id="4", duration=306)


# --- applying a match the pipeline found (no Deezer id) ---------------------

def test_pipeline_matches_are_parked_behind_a_token():
    """A client never names a URL: it hands back a token we issued."""
    meta = {"title": "Timeless", "artist": "Home Free",
            "art_url": "https://cdn/shazam.jpg", "source": "shazam"}
    token = metadata.remember_match("trackA", meta)
    assert token and metadata.take_match(token, "trackA") == meta
    assert metadata.take_match(token, "another-track") is None   # track-scoped
    assert metadata.take_match("nope", "trackA") is None
    assert metadata.take_match("", "trackA") is None


def test_apply_match_caches_the_pipeline_cover(art_dir, monkeypatch):
    monkeypatch.setattr("httpx.AsyncClient", _ByUrlClient)
    meta = {"title": "Timeless", "artist": "Home Free", "album": "Timeless",
            "art_url": "https://cdn-images.dzcdn.net/images/cover/aa/1000x1000.jpg",
            "source": "shazam"}
    result = run(metadata.apply_match("cand4", meta))
    assert result["source"] == "shazam" and result["title"] == "Timeless"
    assert art_for("cand4") is not None and art_for("cand4").read_bytes() == JPEG


def test_apply_match_without_a_cover_keeps_the_metadata(art_dir, monkeypatch):
    monkeypatch.setattr("httpx.AsyncClient", _ByUrlClient)
    result = run(metadata.apply_match("cand5", {"title": "T", "artist": "A",
                                                "source": "shazam"}))
    assert result["art_path"] is None and art_for("cand5") is None


def test_fetch_metadata_can_look_without_touching_the_track(art_dir, monkeypatch):
    """cache_art=False: the picker's entry for the pipeline match writes nothing."""
    async def rec(audio):
        return {"title": "T", "artist": "A", "album": "",
                "art_url": "https://img.test/c.jpg", "source": "shazam"}

    stub_hops(monkeypatch, rec)
    audio = art_dir / "x.m4a"
    audio.write_bytes(b"\x00")
    meta = run(fetch_metadata("T", "A", track_id="peek", audio_path=audio,
                              cache_art=False))
    assert meta["art_url"] == "https://img.test/c.jpg"     # we learn the match…
    assert art_for("peek") is None                          # …but cache no cover
    assert _Client.calls == []


def test_apply_candidate_caches_the_cover(art_dir, deezer_by_id):
    """Picking a candidate downloads that song's cover for the track."""
    deezer_by_id.tracks = {"4": FOOTLOOSE}
    meta = run(metadata.apply_candidate("cand1", "4"))
    assert meta["title"] == "Mama Says (You Can't Back Down)"
    assert meta["artist"] == "Tom Plotkin" and meta["album"].startswith("Footloose")
    assert meta["source"] == "deezer"
    assert art_for("cand1") is not None and art_for("cand1").read_bytes() == JPEG
    assert deezer_by_id.asked[0].endswith("/track/4")      # id re-resolved server-side


def test_apply_candidate_replaces_an_existing_cover(art_dir, deezer_by_id):
    """A user pick overrides what the pipeline cached earlier — even PNG for JPEG."""
    (art_dir / "art").mkdir(exist_ok=True)
    stale = art_dir / "art" / "cand2.png"
    stale.write_bytes(PNG)
    deezer_by_id.tracks = {"5": _deezer_track(cover="https://cdn.test/x.jpg",
                                              track_id="5")}

    run(metadata.apply_candidate("cand2", "5"))
    assert not stale.exists()                                  # old cover gone
    assert art_for("cand2").read_bytes() == JPEG               # new one wins
    assert art_for("cand2").suffix == ".jpg"


def test_apply_candidate_with_a_bad_id_reports_nothing(art_dir, deezer_by_id):
    assert run(metadata.apply_candidate("cand3", "999")) is None
    assert art_for("cand3") is None
    assert deezer_by_id.asked == [
        "https://api.deezer.com/track/999"]     # a dead id is a quiet miss
    assert run(metadata.apply_candidate("cand3", "")) is None
    assert len(deezer_by_id.asked) == 1         # an empty id never hits the API


def test_title_matching_tolerates_qualifiers_only():
    assert _titles_match("Timeless", "Timeless")
    assert _titles_match("Timeless", "Timeless (Deluxe)")
    assert _titles_match("Song - Official Video", "Song")
    assert not _titles_match("Timeless", "Timelessness")   # whole words only
    assert not _titles_match("Mama Says - FULL MIX", "Footloose")
    assert not _titles_match("", "Anything")
    assert _norm_title("[Official Music Video] Never Gonna Give You Up (HD)") == \
        "never gonna give you up"


def test_genius_without_token_is_skipped(monkeypatch):
    """No key configured: we stop quietly instead of erroring (old behavior)."""
    monkeypatch.delenv("GENIUS_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("GENIUS_API_TOKEN", raising=False)
    monkeypatch.setattr(metadata, "_genius_token", lambda: None)
    monkeypatch.setattr("lyricsgenius.Genius",
                        lambda *a, **kw: pytest.fail("Genius must not be called"))
    assert run(metadata._genius_lyrics_search(LYRICS)) is None


def test_both_methods_failing_returns_none(monkeypatch):
    """Nothing found → None, so the caller keeps its existing metadata."""
    async def miss(*a, **kw):
        return None

    stub_all_hops(monkeypatch)
    assert run(fetch_metadata("T", "A", track_id="t4", lyrics=LYRICS)) is None


# --- album art caching -----------------------------------------------------

def test_art_is_downloaded_and_cached(art_dir, monkeypatch):
    async def rec(audio):
        return {"title": "T", "artist": "A", "album": "",
                "art_url": "https://img.test/c.jpg", "source": "shazam"}

    stub_hops(monkeypatch, rec)
    audio = art_dir / "a.m4a"
    audio.write_bytes(b"\x00")
    meta = run(fetch_metadata("T", "A", track_id="abc", audio_path=audio))
    assert meta["art_path"].endswith(".jpg")
    assert art_for("abc") is not None and art_for("abc").read_bytes() == JPEG

    # a second fetch reuses the cached file instead of re-downloading
    run(fetch_metadata("T", "A", track_id="abc", audio_path=audio))
    assert _Client.calls == ["https://img.test/c.jpg"]


def test_png_cover_keeps_its_extension(art_dir, monkeypatch):
    """Genius/Shazam sometimes hand back PNG — don't mislabel it as jpeg."""
    assert _ext_for(PNG) == ".png" and _ext_for(JPEG) == ".jpg"
    _Client.body = PNG

    async def rec(audio):
        return {"title": "T", "artist": "A", "album": "",
                "art_url": "https://img.test/c.png", "source": "shazam"}

    stub_hops(monkeypatch, rec)
    audio = art_dir / "b.m4a"
    audio.write_bytes(b"\x00")
    meta = run(fetch_metadata("T", "A", track_id="png1", audio_path=audio))
    assert meta["art_path"].endswith(".png")
    assert art_for("png1").suffix == ".png"


def test_art_failure_keeps_the_metadata(art_dir, monkeypatch):
    """A dead image URL is cosmetic: the found title/artist still come back."""
    class Broken(_Client):
        async def get(self, url, **kw):
            raise RuntimeError("502 from the CDN")

    monkeypatch.setattr("httpx.AsyncClient", Broken)

    async def rec(audio):
        return {"title": "Real Title", "artist": "Real Artist", "album": "",
                "art_url": "https://img.test/dead.jpg", "source": "shazam"}

    stub_hops(monkeypatch, rec)
    audio = art_dir / "c.m4a"
    audio.write_bytes(b"\x00")
    meta = run(fetch_metadata("T", "A", track_id="dead", audio_path=audio))
    assert meta["title"] == "Real Title" and meta["art_path"] is None


def test_album_parsing_ignores_other_sections():
    track = {"sections": [
        {"type": "LYRICS"},                              # no metadata at all
        {"type": "SONG", "metadata": [{"title": "Artist", "text": "A"},
                                      {"title": "Album", "text": "Alb"}]},
    ]}
    assert _album_from_sections(track) == "Alb"
    assert _album_from_sections({}) == ""


# --- the post-transcription pass (pipeline) --------------------------------

def _pipeline_without_workers(store):
    """A Pipeline that never started its worker threads — enough for one method."""
    from app.pipeline import Pipeline

    pipeline = Pipeline.__new__(Pipeline)
    pipeline.store = store
    return pipeline


def test_pipeline_second_pass_searches_lyrics(monkeypatch):
    """After transcription the fetch is retried *with* our own lyrics."""
    from app.models import Line, Track
    from app.pipeline import store

    seen: dict = {}
    done = threading.Event()

    async def fake_fetch(title, artist, *, track_id=None, audio_path=None, lyrics=None):
        seen.update(lyrics=lyrics, title=title)
        done.set()
        return None

    monkeypatch.setattr("app.metadata.fetch_metadata", fake_fetch)
    monkeypatch.setattr("app.metadata.art_for", lambda track_id: None)
    track = Track(id="pipe1", url="", title="Mama Says - FULL MIX", artist="")
    lines = [Line(text="Everything I ever learned", start=0.0, progress=0.0, words=[])]

    _pipeline_without_workers(store)._fetch_metadata_async(track, lines=lines)
    assert done.wait(5), "the background lyric pass never ran"
    assert seen["lyrics"] == "Everything I ever learned"
    assert seen["title"] == "Mama Says - FULL MIX"


def test_pipeline_second_pass_is_skipped_when_art_exists(monkeypatch, tmp_path):
    """The pre-transcription fingerprint pass won, so don't fetch again."""
    from app.models import Line, Track
    from app.pipeline import store

    called = threading.Event()

    async def fake_fetch(*a, **kw):
        called.set()
        return None

    monkeypatch.setattr("app.metadata.fetch_metadata", fake_fetch)
    monkeypatch.setattr("app.metadata.art_for", lambda track_id: tmp_path / "a.jpg")
    track = Track(id="pipe2", url="", title="T", artist="A")
    lines = [Line(text="a distinctive transcript line", start=0.0, progress=0.0, words=[])]
    pipeline = _pipeline_without_workers(store)

    pipeline._fetch_metadata_async(track, lines=lines)
    assert not called.wait(0.6)                 # second pass: nothing to do

    pipeline._fetch_metadata_async(track)       # download-time pass always runs
    assert called.wait(5)


# --- API surface -----------------------------------------------------------

def test_fetch_metadata_endpoint_reports_art(monkeypatch):
    """The hover button's endpoint passes the audio file and returns artUrl."""
    from fastapi.testclient import TestClient

    from app.models import Track
    from app.pipeline import store
    from app.server import app

    seen: dict = {}

    async def fake_fetch(title, artist, *, track_id=None, audio_path=None, lyrics=None):
        seen.update(title=title, artist=artist, track_id=track_id,
                    audio_path=str(audio_path), lyrics=lyrics)
        return {"title": title, "artist": "Found Artist", "album": "Alb",
                "art_path": "data/art/x.jpg", "source": "shazam"}

    monkeypatch.setattr("app.metadata.fetch_metadata", fake_fetch)
    monkeypatch.setattr("app.server._has_art", lambda tid: True)
    track = Track(id="meta1", url="u", title="T", artist="Unknown artist",
                  audio_file=cfg.DATA_DIR / "audio" / "meta1.m4a")
    store.add_track(track)
    try:
        res = TestClient(app).post("/api/tracks/meta1/fetch-metadata")
        assert res.status_code == 200
        body = res.json()
        assert body["artUrl"] == "/api/tracks/meta1/art"
        assert body["source"] == "shazam"
        assert body["message"] == "Found via Shazam (audio fingerprint)"
        assert seen["audio_path"].endswith("meta1.m4a")   # fingerprintable
        assert store.get_track("meta1").artist == "Found Artist"
        assert seen["lyrics"] is None                      # no transcript on this track
    finally:
        store.delete_track("meta1")


def test_fetch_metadata_endpoint_without_a_hit_says_so(monkeypatch):
    from fastapi.testclient import TestClient

    from app.models import Track
    from app.pipeline import store
    from app.server import app

    async def miss(*a, **kw):
        return None

    async def no_candidates(*a, **kw):
        return []

    monkeypatch.setattr("app.metadata.fetch_metadata", miss)
    monkeypatch.setattr("app.metadata.find_candidates", no_candidates)
    track = Track(id="meta2", url="u", title="T", artist="A")
    store.add_track(track)
    try:
        res = TestClient(app).post("/api/tracks/meta2/fetch-metadata")
        assert res.status_code == 200
        body = res.json()
        assert body["artUrl"] is None
        assert body["candidates"] == []
        assert "No metadata found" in body["message"]
    finally:
        store.delete_track("meta2")


def test_fetch_metadata_endpoint_offers_candidates_when_unsure(monkeypatch):
    """A miss answers with the shortlist, so the user can pick instead of us guessing."""
    from fastapi.testclient import TestClient

    from app.models import Track
    from app.pipeline import store
    from app.server import app

    async def miss(*a, **kw):
        return None

    async def candidates(title, artist, *, lyrics=None):
        assert title == "Mama Says - FULL MIX"
        assert lyrics is None                       # this track has no transcript
        return [{
            "id": "4",
            "title": "Mama Says (You Can't Back Down)",
            "artist": "Tom Plotkin",
            "album": "Footloose: The Musical (Original Broadway Cast Recording)",
            "art_url": "https://cdn-images.dzcdn.net/images/cover/ff/1000x1000.jpg",
            "duration": 306,
        }]

    monkeypatch.setattr("app.metadata.fetch_metadata", miss)
    monkeypatch.setattr("app.metadata.find_candidates", candidates)
    track = Track(id="meta4", url="u", title="Mama Says - FULL MIX", artist="")
    store.add_track(track)
    try:
        body = TestClient(app).post("/api/tracks/meta4/fetch-metadata").json()
        assert body["artUrl"] is None
        assert body["candidates"][0]["id"] == "4"
        assert body["candidates"][0]["artist"] == "Tom Plotkin"
        assert body["candidates"][0]["artUrl"].endswith("1000x1000.jpg")
        assert "pick" in body["message"]
    finally:
        store.delete_track("meta4")


def test_redo_offers_the_pipeline_match_as_a_pick_not_an_answer(monkeypatch):
    """A fingerprint match is listed (and not applied) so the user can replace it."""
    from fastapi.testclient import TestClient

    from app.models import Track
    from app.pipeline import store
    from app.server import app

    seen: dict = {}

    async def confident(title, artist, *, track_id=None, audio_path=None,
                        lyrics=None, cache_art=True):
        seen.update(cache_art=cache_art, track_id=track_id)
        return {"title": title, "artist": "Home Free", "album": "Timeless",
                "art_url": "https://cdn-images.dzcdn.net/shazam.jpg",
                "source": "shazam", "art_path": None}

    async def candidates(title, artist, *, lyrics=None):
        return [{"id": "4", "title": "Mama Says (You Can't Back Down)",
                 "artist": "Tom Plotkin", "album": "Footloose: The Musical",
                 "art_url": "https://cdn/x.jpg", "duration": 306, "match": "match"}]

    monkeypatch.setattr("app.metadata.fetch_metadata", confident)
    monkeypatch.setattr("app.metadata.find_candidates", candidates)
    store.add_track(Track(id="meta7", url="u", title="Timeless", artist="Unknown artist"))
    try:
        body = TestClient(app).post("/api/tracks/meta7/fetch-metadata?pick=true").json()
        assert seen["cache_art"] is False             # read-only: no cover written
        assert body["artUrl"] is None                 # nothing applied silently
        assert store.get_track("meta7").artist == "Unknown artist"

        first = body["candidates"][0]
        assert first["source"] == "shazam"            # the fingerprint answer, first
        assert first["title"] == "Timeless" and first["token"]
        assert first["tier"] == "strict"              # an identification: never folded
        assert first["artUrl"] == "https://cdn-images.dzcdn.net/shazam.jpg"
        assert body["candidates"][1]["match"] == "match"    # then the shortlist
        assert 'identified "Timeless"' in body["message"]
    finally:
        store.delete_track("meta7")


def test_redo_does_not_list_the_pipeline_match_twice(monkeypatch):
    """When the search already names that same release, its Deezer row is the pick."""
    from fastapi.testclient import TestClient

    from app.models import Track
    from app.pipeline import store
    from app.server import app

    async def pipeline(title, artist, *, track_id=None, audio_path=None,
                       lyrics=None, cache_art=True):
        return {"title": "Timeless", "artist": "Home Free", "album": "Timeless",
                "art_url": "https://cdn/shazam.jpg", "source": "shazam"}

    async def candidates(title, artist, *, lyrics=None):
        return [{"id": "77", "title": "Timeless", "artist": "Home Free",
                 "album": "Timeless", "art_url": "https://cdn/deezer.jpg",
                 "duration": 200, "match": "match"}]

    monkeypatch.setattr("app.metadata.fetch_metadata", pipeline)
    monkeypatch.setattr("app.metadata.find_candidates", candidates)
    store.add_track(Track(id="meta8", url="u", title="Timeless", artist="Home Free"))
    try:
        body = TestClient(app).post("/api/tracks/meta8/fetch-metadata?pick=true").json()
        assert len(body["candidates"]) == 1
        assert body["candidates"][0]["id"] == "77"      # Deezer id, not a token
        assert body["candidates"][0]["token"] is None
    finally:
        store.delete_track("meta8")


def test_redo_with_nothing_to_offer_says_so(monkeypatch):
    """No search hits and no pipeline match: the redo reports that honestly."""
    from fastapi.testclient import TestClient

    from app.models import Track
    from app.pipeline import store
    from app.server import app

    async def nothing(*a, **kw):
        return None

    async def no_candidates(*a, **kw):
        return []

    monkeypatch.setattr("app.metadata.fetch_metadata", nothing)
    monkeypatch.setattr("app.metadata.find_candidates", no_candidates)
    store.add_track(Track(id="meta8b", url="u", title="Mama Says - FULL MIX", artist=""))
    try:
        body = TestClient(app).post("/api/tracks/meta8b/fetch-metadata?pick=true").json()
        assert body["candidates"] == []
        assert body["message"] == "No metadata found for this track"
    finally:
        store.delete_track("meta8b")


def test_plain_fetch_still_applies_a_confident_match(monkeypatch):
    """The hover button (and the upload-time pass) keep auto-picking."""
    from fastapi.testclient import TestClient

    from app.models import Track
    from app.pipeline import store
    from app.server import app

    async def candidates(*a, **kw):
        pytest.fail("no picker unless the user asked for one")

    async def pipeline(title, artist, *, track_id=None, audio_path=None, lyrics=None):
        return {"title": title, "artist": "Home Free", "album": "Timeless",
                "art_path": None, "source": "shazam"}

    monkeypatch.setattr("app.metadata.fetch_metadata", pipeline)
    monkeypatch.setattr("app.metadata.find_candidates", candidates)
    monkeypatch.setattr("app.server._has_art", lambda track_id: True)
    store.add_track(Track(id="meta9", url="u", title="Timeless", artist="Unknown artist"))
    try:
        body = TestClient(app).post("/api/tracks/meta9/fetch-metadata").json()
        assert body["artUrl"] == "/api/tracks/meta9/art"
        assert body["candidates"] == []
        assert body["message"] == "Found via Shazam (audio fingerprint)"
    finally:
        store.delete_track("meta9")


def test_apply_endpoint_caches_the_picked_cover(monkeypatch):
    """Picking a candidate fills in the artist we never had and caches the cover."""
    from fastapi.testclient import TestClient

    from app.models import Track
    from app.pipeline import store
    from app.server import app

    seen: dict = {}

    async def applied(track_id, candidate_id):
        seen.update(track_id=track_id, candidate_id=candidate_id)
        return {"id": "4", "title": "Mama Says (You Can't Back Down)",
                "artist": "Tom Plotkin", "album": "Footloose: The Musical",
                "art_url": "https://cdn/x.jpg", "duration": 306,
                "art_path": "data/art/meta5.jpg", "source": "deezer"}

    monkeypatch.setattr("app.metadata.apply_candidate", applied)
    monkeypatch.setattr("app.server._has_art", lambda tid: True)
    track = Track(id="meta5", url="u", title="Mama Says - FULL MIX", artist="")
    store.add_track(track)
    try:
        res = TestClient(app).post("/api/tracks/meta5/metadata/apply",
                                   json={"candidateId": "4"})
        body = res.json()
        assert res.status_code == 200
        assert seen == {"track_id": "meta5", "candidate_id": "4"}
        assert body["artUrl"] == "/api/tracks/meta5/art"
        assert body["artist"] == "Tom Plotkin"
        assert body["album"] == "Footloose: The Musical"
        assert body["message"] == "Applied Deezer"
        assert store.get_track("meta5").artist == "Tom Plotkin"   # artist filled in
        assert store.get_track("meta5").title == "Mama Says - FULL MIX"   # title untouched
    finally:
        store.delete_track("meta5")


def test_apply_endpoint_replaces_a_wrong_artist(monkeypatch):
    """Redoing a search onto a *different* song replaces the artist too.

    The old rule only filled an empty artist, so picking the right recording left
    the wrong artist sitting on the track — the cover changed and nothing else.
    """
    from fastapi.testclient import TestClient

    from app.models import Track
    from app.pipeline import store
    from app.server import app

    async def applied(track_id, candidate_id):
        return {"id": "4", "title": "Mama Says (You Can't Back Down)",
                "artist": "Tom Plotkin", "album": "Footloose: The Musical",
                "art_url": "https://cdn/x.jpg", "art_path": "data/art/meta6.jpg",
                "source": "deezer"}

    monkeypatch.setattr("app.metadata.apply_candidate", applied)
    monkeypatch.setattr("app.server._has_art", lambda tid: True)
    store.add_track(Track(id="meta6", url="u", title="Mama Says - FULL MIX",
                          artist="Ibeyi"))
    try:
        body = TestClient(app).post("/api/tracks/meta6/metadata/apply",
                                    json={"candidateId": "4"}).json()
        assert body["artist"] == "Tom Plotkin"
        assert store.get_track("meta6").artist == "Tom Plotkin"   # wrong artist corrected
        assert store.get_track("meta6").title == "Mama Says - FULL MIX"   # never renamed
    finally:
        store.delete_track("meta6")


def test_apply_endpoint_never_blanks_an_artist(monkeypatch):
    """A match that names nobody leaves the artist we already have alone."""
    from fastapi.testclient import TestClient

    from app.models import Track
    from app.pipeline import store
    from app.server import app

    async def applied(track_id, candidate_id):
        return {"id": "9", "title": "Timeless", "artist": "", "album": "",
                "art_url": "https://cdn/x.jpg", "art_path": "data/art/meta7.jpg",
                "source": "deezer"}

    monkeypatch.setattr("app.metadata.apply_candidate", applied)
    monkeypatch.setattr("app.server._has_art", lambda tid: True)
    store.add_track(Track(id="meta7", url="u", title="Timeless", artist="Home Free"))
    try:
        body = TestClient(app).post("/api/tracks/meta7/metadata/apply",
                                    json={"candidateId": "9"}).json()
        assert body["artist"] == "Home Free"
        assert store.get_track("meta7").artist == "Home Free"
    finally:
        store.delete_track("meta7")


def test_apply_endpoint_takes_a_pipeline_match_token(monkeypatch):
    """Picking the fingerprint entry applies *its* cover, via the token."""
    from fastapi.testclient import TestClient

    from app.models import Track
    from app.pipeline import store
    from app.server import app

    token = metadata.remember_match("meta10", {
        "title": "Timeless", "artist": "Home Free", "album": "Timeless",
        "art_url": "https://cdn/shazam.jpg", "source": "shazam"})

    async def applied(track_id, meta):
        assert meta["art_url"] == "https://cdn/shazam.jpg"   # server-side, not ours
        return {"title": meta["title"], "artist": meta["artist"],
                "album": meta["album"], "art_path": "data/art/meta10.jpg",
                "art_url": meta["art_url"], "source": meta["source"]}

    monkeypatch.setattr("app.metadata.apply_match", applied)
    monkeypatch.setattr("app.server._has_art", lambda track_id: True)
    store.add_track(Track(id="meta10", url="u", title="Timeless",
                          artist="Unknown artist"))
    try:
        body = TestClient(app).post("/api/tracks/meta10/metadata/apply",
                                    json={"matchToken": token}).json()
        assert body["artUrl"] == "/api/tracks/meta10/art"
        assert body["artist"] == "Home Free"
        assert body["message"] == "Applied Shazam (audio fingerprint)"
        assert store.get_track("meta10").artist == "Home Free"
    finally:
        store.delete_track("meta10")


def test_apply_endpoint_rejects_a_token_from_another_track(monkeypatch):
    from fastapi.testclient import TestClient

    from app.models import Track
    from app.pipeline import store
    from app.server import app

    token = metadata.remember_match("someone-else", {"title": "T", "artist": "A",
                                                     "art_url": "https://cdn/x.jpg"})
    store.add_track(Track(id="meta11", url="u", title="T", artist="A"))
    try:
        body = TestClient(app).post("/api/tracks/meta11/metadata/apply",
                                    json={"matchToken": token}).json()
        assert body["artUrl"] is None
        assert "expired" in body["message"]
    finally:
        store.delete_track("meta11")


def test_apply_endpoint_needs_something_to_apply():
    from fastapi.testclient import TestClient

    from app.models import Track
    from app.pipeline import store
    from app.server import app

    store.add_track(Track(id="meta12", url="u", title="T", artist="A"))
    try:
        res = TestClient(app).post("/api/tracks/meta12/metadata/apply", json={})
        assert res.status_code == 400
    finally:
        store.delete_track("meta12")


def test_apply_endpoint_reports_a_dead_id(monkeypatch):
    from fastapi.testclient import TestClient

    from app.models import Track
    from app.pipeline import store
    from app.server import app

    async def gone(track_id, candidate_id):
        return None

    monkeypatch.setattr("app.metadata.apply_candidate", gone)
    store.add_track(Track(id="meta6", url="u", title="T", artist="A"))
    try:
        res = TestClient(app).post("/api/tracks/meta6/metadata/apply",
                                   json={"candidateId": "999"})
        assert res.status_code == 200
        assert res.json()["artUrl"] is None
        assert "no longer available" in res.json()["message"]
    finally:
        store.delete_track("meta6")


def test_art_endpoint_404s_without_a_cover(monkeypatch):
    from fastapi.testclient import TestClient

    from app.server import app

    assert TestClient(app).get("/api/tracks/nope/art").status_code == 404


def test_art_url_is_versioned_so_a_new_cover_is_a_new_url(art_dir):
    """A replaced cover must reach the browser as a *new* URL.

    Same URL means the browser can keep painting the bytes it already has — the
    stale cover. A versioned one is refetched, and it is what lets the client
    cross-fade old into new (both are addressable at once).
    """
    from app.server import _art_url

    (art_dir / "art").mkdir(exist_ok=True)
    cover = art_dir / "art" / "ver1.jpg"
    cover.write_bytes(JPEG)
    first = _art_url("ver1")
    assert first.startswith("/api/tracks/ver1/art?v=")

    # a different cover for the same track: new bytes, so a new URL
    cover.write_bytes(JPEG + b"2")
    later = cover.stat().st_mtime + 5
    os.utime(cover, (later, later))
    assert _art_url("ver1") != first
    assert _art_url("ver1").split("?")[0] == "/api/tracks/ver1/art"

    assert _art_url("no-such-track") is None


def test_list_endpoint_hands_out_the_versioned_cover(art_dir):
    """The library cards read artUrl from this list, so it has to carry it."""
    from fastapi.testclient import TestClient

    from app.models import Track
    from app.pipeline import store
    from app.server import app

    (art_dir / "art").mkdir(exist_ok=True)
    (art_dir / "art" / "ver3.jpg").write_bytes(JPEG)
    store.add_track(Track(id="ver3", url="u", title="T", artist="A"))
    store.add_track(Track(id="ver4", url="u", title="U", artist="B"))
    try:
        rows = {r["id"]: r for r in TestClient(app).get("/api/tracks").json()}
        assert rows["ver3"]["hasArt"] is True
        assert rows["ver3"]["artUrl"].startswith("/api/tracks/ver3/art?v=")
        assert rows["ver4"]["hasArt"] is False
        assert rows["ver4"]["artUrl"] is None
    finally:
        store.delete_track("ver3")
        store.delete_track("ver4")
