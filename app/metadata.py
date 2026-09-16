"""Fetch track metadata (album art, artist, album) via Shazam → Deezer → Genius.

A lookup only applies a match it is *sure* of. When the title alone cannot
identify the song (five different recordings are called "Mama Says") the user
gets a shortlist via :func:`find_candidates` and picks one, which
:func:`apply_candidate` then caches. That is also the escape hatch for tracks no
fingerprint or search can name — a custom "FULL MIX", say.

Our own transcript is used twice: as the *query* for Genius's lyric search, and
as the *judge* that vetoes shortlist entries whose lyrics clearly differ
(LRCLIB, no token needed).

Hops, in order:

1. **Shazam** (``shazamio``) — fingerprints the local audio file, so it works
   for YouTube downloads and uploads that carry no metadata at all. Runs in the
   background on every new track. If fingerprinting finds nothing we retry with
   Shazam's text search using the title/artist we already have.
2. **Deezer** — a keyless public search (no account, no token). It is the best
   source of *art* for real songs: covers come back at 1000×1000. Tried before
   Shazam's text search because Shazam's text endpoint currently fails upstream
   (``FailedDecodeJson``) and because a hit here needs no credentials at all.
3. **Genius** (``lyricsgenius``) — only reached when everything above misses
   *and* the user configured a Genius API token. No token → we simply stop.

Anything found is display-only (cover art, artist, album). Art is cached under
``data/art/<track_id>.<ext>`` and served by ``GET /api/tracks/{id}/art``.
"""
from __future__ import annotations

import asyncio
import difflib
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Optional

import app.config as cfg

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shazam (primary — runs on every upload)
# ---------------------------------------------------------------------------

def _album_from_sections(track: dict) -> str:
    """Pull the album name out of Shazam's ``sections`` metadata blob."""
    for section in track.get("sections") or []:
        for item in section.get("metadata") or []:
            if (item.get("title") or "").lower() == "album":
                return item.get("text") or ""
    return ""


def _meta_from_shazam_track(track: dict, title: str, artist: str) -> dict:
    """Normalise a Shazam ``track`` object into our metadata dict."""
    images = track.get("images") or {}
    art_url = images.get("coverart") or images.get("coverarthq") or images.get("background")
    return {
        "title": track.get("title") or title,
        "artist": track.get("subtitle") or artist,
        "album": _album_from_sections(track),
        "art_url": art_url,
        "source": "shazam",
    }


async def _shazam_recognize(audio_path: Path) -> Optional[dict]:
    """Identify the track by fingerprinting the audio file (method 1)."""
    try:
        from shazamio import Shazam
    except ImportError:
        log.warning("shazamio not installed — skipping Shazam recognition")
        return None

    if not audio_path or not Path(audio_path).exists():
        return None

    try:
        result = await Shazam().recognize(str(audio_path))
    except Exception as exc:  # noqa: BLE001 - network/format problems are normal
        log.info("Shazam recognition failed for %s: %s", audio_path, exc)
        return None

    track = (result or {}).get("track")
    if not track:
        # "no match" responses come back with an empty/matches-only payload
        return None
    return _meta_from_shazam_track(track, "", "")


async def _shazam_search(title: str, artist: str) -> Optional[dict]:
    """Fallback for method 1: Shazam *text* search on title/artist."""
    if not (title or artist):
        return None
    try:
        from shazamio import Shazam
    except ImportError:
        return None

    try:
        query = f"{artist} {title}".strip()
        result = await Shazam().search_track(query=query, limit=1)
    except Exception as exc:  # noqa: BLE001
        log.info("Shazam search failed for %r: %s", title, exc)
        return None

    hits = ((result or {}).get("tracks") or {}).get("hits") or []
    if not hits:
        return None
    track = hits[0].get("track") or {}
    if not track:
        return None
    meta = _meta_from_shazam_track(track, title, artist)
    meta["source"] = "shazam-search"
    return meta


# ---------------------------------------------------------------------------
# Deezer (method 2 — keyless text search, no credentials required)
# ---------------------------------------------------------------------------

DEEZER_SEARCH_URL = "https://api.deezer.com/search"
DEEZER_TRACK_URL = "https://api.deezer.com/track/{id}"

# Words that carry no identifying weight in a title comparison.
_TITLE_NOISE = {
    "feat", "ft", "featuring", "official", "video", "audio", "lyric", "lyrics",
    "hd", "hq", "mv", "remastered", "remaster", "prod",
}


def _norm_title(text: str) -> str:
    """Lowercase a title, drop bracketed asides/punctuation, collapse spaces."""
    text = re.sub(r"[\(\[\{].*?[\)\]\}]", " ", text or "")
    text = re.sub(r"[^0-9a-z]+", " ", text.lower())
    return " ".join(tok for tok in text.split() if tok not in _TITLE_NOISE)


# Version/mix qualifiers that a search index does not use in its titles.
_QUALIFIER_WORDS = {
    "full", "mix", "remix", "version", "edit", "official", "music", "video",
    "audio", "lyric", "lyrics", "live", "cover", "karaoke", "instrumental",
    "hd", "hq", "4k", "extended", "radio", "acoustic", "session", "sessions",
    "remaster", "remastered", "slowed", "sped", "up", "nightcore", "demo",
    "reprise", "clean", "explicit", "clip", "visualizer", "performance",
}

_SEP_RE = re.compile(r"\s*[-\u2013\u2014|]\s*")


def _clean_query(title: str) -> str:
    """Strip version/mix tails for *searching*: "Mama Says - FULL MIX" → "Mama Says".

    Titles reach us from YouTube and filenames, so they carry qualifiers no
    index uses: asked verbatim, "Mama Says - FULL MIX" returns junk, while
    "Mama Says" finds the song. Only the query is cleaned — matching still
    compares a hit against the full stored title, so an unrelated hit cannot
    sneak in through a narrowed search.
    """
    base = re.sub(r"[\(\[\{].*?[\)\]\}]", " ", title or "")
    base = " ".join(base.split())
    cut = None
    for match in _SEP_RE.finditer(base):
        words = re.findall(r"[a-z0-9]+", base[match.end():].lower())
        if words and all(word in _QUALIFIER_WORDS for word in words):
            cut = match.start()
            break
    cleaned = (base[:cut] if cut is not None else base).strip()
    return cleaned or base


def _deezer_queries(title: str, artist: str) -> list[str]:
    """Query variants to try, best first (deduped, order preserved).

    A known artist is searched *alongside* the title first, then the title alone:
    the artist we hold can be wrong, and a cast recording answers
    "Tom Plotkin Mama Says" with a single hit, hiding every other song called
    "Mama Says" — which is exactly what the picker exists to offer. Safety is not
    affected, because a confident match still has to agree on the artist
    (``_is_confident``); the broad query only ever adds *candidates*.

    Placeholder artists are never sent: "Unknown artist Timeless" matches
    nothing at all, while the bare title finds the song.
    """
    queries: list[str] = []
    generic = _is_generic_artist(artist)
    for variant in (title, _clean_query(title)):
        variants = [variant] if generic else [f"{artist} {variant}", variant]
        for query in variants:
            query = " ".join(query.split())
            if query and query not in queries:
                queries.append(query)
    return queries


def _titles_match(ours: str, theirs: str) -> bool:
    """True when a text-search hit plausibly *is* the track we searched for.

    Deezer happily returns other people's songs for a vague query, and a wrong
    cover is worse than no cover, so ``fetch_metadata`` only accepts a hit whose
    title equals ours or contains ours as whole words (and vice versa) — that
    tolerates "Song" vs "Song (Deluxe)" / "Song - Live" while rejecting
    "Mama Says" answering a search for "Footloose".
    """
    a, b = _norm_title(ours), _norm_title(theirs)
    if not a or not b:
        return False
    if a == b:
        return True
    return f" {b} " in f" {a} " or f" {a} " in f" {b} "


# Artist strings that say nothing about identity — never score a match on them.
_GENERIC_ARTISTS = {"", "unknown", "unknown artist", "unknown artists",
                    "various", "various artists", "va", "artist", "various artist"}


def _is_generic_artist(text: str) -> bool:
    """True for placeholder artists ("Unknown artist", "", "Various Artists")."""
    return _norm_title(text) in _GENERIC_ARTISTS


def _artists_match(ours: str, theirs: str) -> bool:
    """True when a hit's artist plausibly is our artist (skips placeholders).

    Containment either way, so "Home Free - Topic" matches "Home Free".
    """
    a, b = _norm_title(ours), _norm_title(theirs)
    if not a or not b or a in _GENERIC_ARTISTS or b in _GENERIC_ARTISTS:
        return False
    return a == b or f" {b} " in f" {a} " or f" {a} " in f" {b} "


def _hit_score(title: str, artist: str, hit: dict) -> int:
    """Rank a Deezer hit: -1 = not our song. Higher is better.

    Title agreement is required; an artist agreement is worth twice an exact
    title, because text search otherwise prefers a *cover* by another name —
    searching "Rick Astley Never Gonna Give You Up" turns up an exact title by
    an unrelated artist well after Rick Astley's own (remix-titled) versions.
    """
    hit_title = hit.get("title") or ""
    if not _titles_match(title, hit_title):
        return -1
    score = 1
    if _norm_title(title) == _norm_title(hit_title):
        score += 1                                      # exact title
    if _artists_match(artist, (hit.get("artist") or {}).get("name") or ""):
        score += 2                                      # right artist
    return score


async def _deezer_hits(query: str) -> list[dict]:
    """One raw Deezer search. Returns ``[]`` on any network/parse problem."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(DEEZER_SEARCH_URL, params={"q": query, "limit": 25})
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:  # noqa: BLE001 - network/parse problems are normal
        log.info("Deezer search failed for %r: %s", query, exc)
        return []

    if not isinstance(data, dict) or data.get("error"):
        detail = data.get("error") if isinstance(data, dict) else f"unexpected {type(data).__name__}"
        log.info("Deezer search error for %r: %s", query, detail)
        return []
    return [hit for hit in (data.get("data") or []) if isinstance(hit, dict)]


def _rank_hits(title: str, artist: str, hits: list[dict]) -> list[tuple[int, dict]]:
    """Title-matching hits, best first (ties keep Deezer's own order)."""
    ranked = [(score, hit) for hit in hits
              if (score := _hit_score(title, artist, hit)) > 0]
    ranked.sort(key=lambda pair: -pair[0])
    return ranked


def _flat_text(text: str) -> str:
    """Lowercase, punctuation-free, whitespace-collapsed — nothing else removed."""
    return " ".join(re.sub(r"[^0-9a-z]+", " ", (text or "").lower()).split())


def same_recording(title_a: str, artist_a: str, title_b: str, artist_b: str) -> bool:
    """True when two (title, artist) pairs are *the same release*.

    Deliberately stricter than ``_titles_match``: qualifiers are kept, so
    "Never Gonna Give You Up (Cake Mix)" is not "Never Gonna Give You Up". The
    picker uses this only to avoid listing one song twice, and a fingerprint hit
    has to survive next to the remixes of itself.
    """
    if _flat_text(title_a) != _flat_text(title_b):
        return False
    if _is_generic_artist(artist_a) or _is_generic_artist(artist_b):
        return True                     # nothing to disagree about
    return _artists_match(artist_a, artist_b)


def _hit_fields(hit: dict, title: str, artist: str) -> dict:
    """Normalise one Deezer hit (search result or /track lookup) for us."""
    album = hit.get("album") or {}
    return {
        "id": str(hit.get("id") or ""),
        "title": hit.get("title") or title,
        "artist": (hit.get("artist") or {}).get("name") or artist,
        "album": album.get("title") or "",
        "art_url": (album.get("cover_xl") or album.get("cover_big")
                    or album.get("cover_medium")),
        "duration": hit.get("duration"),
    }


def _is_confident(title: str, artist: str, ranked: list[tuple[int, dict]]) -> bool:
    """Can the top hit be applied without asking the user?

    Two ways to be sure. Either we know the artist and it agrees (score ≥ 3, so
    a stranger's cover never lands on the track — "Broken Hallelujah" by Don
    Savant answering Home Free's "Hallelujah"), or the artist is a placeholder
    and every best-scoring hit is by the *same* artist. That second rule is what
    stops a tie from being a coin flip: "Mama Says" is a song by Ibeyi, the
    Beach Boys, jael and the Footloose cast recording, so there is no honest
    answer to pick automatically — the user chooses instead.
    """
    if not ranked:
        return False
    if not _is_generic_artist(artist):
        return ranked[0][0] >= 3
    top = ranked[0][0]
    names = {_norm_title((hit.get("artist") or {}).get("name") or "")
             for score, hit in ranked if score == top}
    return len(names) == 1


async def _deezer_search(title: str, artist: str) -> Optional[dict]:
    """Search Deezer for the track (method 2). Keyless; best art quality."""
    if not title:
        return None

    for query in _deezer_queries(title, artist):
        ranked = _rank_hits(title, artist, await _deezer_hits(query))
        if ranked and _is_confident(title, artist, ranked):
            meta = _hit_fields(ranked[0][1], title, artist)
            meta["source"] = "deezer"
            return meta
    log.info("Deezer found no confident match for %r / %r", title, artist)
    return None


async def find_candidates(title: str, artist: str, *, lyrics: str | None = None,
                          limit: int = 10, verify: bool = True) -> list[dict]:
    """The shortlist of Deezer songs that could be this track.

    Text search cannot answer "which 'Mama Says'?", so when the automatic hops
    give up we hand the user the candidates (best-scoring first, deduped) and
    let them pick. Only ever called for a miss, so the extra requests cost
    nothing on the happy path.

    Each candidate is tagged ``tier``: "strict" when its artist agrees with ours
    (the title+artist search would have found it, so it is the strongest evidence
    we have) and "broad" otherwise — the panel shows the strict survivors and
    folds the rest away, so a pile of same-titled songs stays out of the way.

    With ``lyrics`` we also *judge* each candidate against the transcript: the
    ones whose lyrics clearly differ get ``match: "mismatch"`` and are folded too.
    """
    if not title:
        return []
    ranked: list[tuple[int, dict]] = []
    for query in _deezer_queries(title, artist):
        ranked.extend(_rank_hits(title, artist, await _deezer_hits(query)))
    ranked.sort(key=lambda pair: -pair[0])              # stable across queries

    out, seen = [], set()
    for score, hit in ranked:
        fields = _hit_fields(hit, title, artist)
        if not fields["id"] or fields["id"] in seen or not fields["art_url"]:
            continue
        seen.add(fields["id"])
        # Two tiers of evidence. "strict": the hit's artist agrees with ours, so
        # the title+artist search would have found it — that is the first level
        # of filtering and it earns the main list. "broad": everything else the
        # title search turned up, which the picker folds away unless there is
        # nothing better (our artist is missing or simply wrong).
        fields["tier"] = "strict" if _artists_match(artist, fields["artist"]) else "broad"
        out.append(fields)
        if len(out) >= limit:
            break

    if verify and lyrics and out:
        verdicts = await asyncio.gather(
            *(_lyrics_verdict(lyrics, candidate) for candidate in out))
        for candidate, verdict in zip(out, verdicts):
            candidate["match"] = verdict
        # strict first, then by verdict: confirmed, unjudgeable, vetoed
        out.sort(key=lambda c: (c["tier"] != "strict",
                                {"match": 0, None: 1, "mismatch": 2}.get(c["match"], 1)))
    else:
        out.sort(key=lambda c: c["tier"] != "strict")
    return out


# Matches the pipeline found that aren't Deezer ids: a Shazam fingerprint hit
# carries its own cover URL. Applying one must not let the client name a URL, so
# the server parks the match and hands back a token instead.
_FOUND_MATCHES: dict[str, dict] = {}
_FOUND_TTL = 15 * 60        # seconds
_FOUND_MAX = 50


def remember_match(track_id: str, meta: dict) -> str:
    """Park a pipeline match and return the token that stands for it."""
    token = uuid.uuid4().hex
    _FOUND_MATCHES[token] = {"track_id": track_id, "meta": meta, "at": time.time()}
    now = time.time()
    for key, entry in list(_FOUND_MATCHES.items()):
        if now - entry["at"] > _FOUND_TTL:
            _FOUND_MATCHES.pop(key, None)
    while len(_FOUND_MATCHES) > _FOUND_MAX:            # oldest first
        oldest = min(_FOUND_MATCHES, key=lambda k: _FOUND_MATCHES[k]["at"])
        _FOUND_MATCHES.pop(oldest, None)
    return token


def take_match(token: str, track_id: str) -> Optional[dict]:
    """The parked match behind ``token`` — only for the track it was found for."""
    entry = _FOUND_MATCHES.get(str(token or ""))
    if not entry or entry["track_id"] != track_id:
        return None
    return entry["meta"]


async def apply_match(track_id: str, meta: dict) -> Optional[dict]:
    """Cache the cover of a match the pipeline found (token came from us)."""
    art_path = None
    if meta.get("art_url"):
        art_path = await _download_art(meta["art_url"], track_id, force=True)
    log.info("Applied %s match for track %s: %s (art: %s)",
             meta.get("source"), track_id, meta.get("title"), bool(art_path))
    return {
        "title": meta.get("title") or "",
        "artist": meta.get("artist") or "",
        "album": meta.get("album") or "",
        "art_url": meta.get("art_url"),
        "art_path": str(art_path) if art_path else None,
        "source": meta.get("source") or "match",
    }


async def apply_candidate(track_id: str, deezer_id: str) -> Optional[dict]:
    """Cache the cover of the candidate the user picked.

    The id is re-resolved through Deezer's own API server-side rather than
    trusting a client-supplied image URL, so nothing else can be pointed at the
    downloader. Replaces any existing cover (the user is correcting us).
    """
    import httpx

    deezer_id = str(deezer_id or "").strip()
    if not deezer_id:
        return None

    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(DEEZER_TRACK_URL.format(id=deezer_id))
            resp.raise_for_status()
            hit = resp.json()
    except Exception as exc:  # noqa: BLE001 - a bad/dead id is not fatal
        log.info("Deezer track lookup failed for %r: %s", deezer_id, exc)
        return None

    if not isinstance(hit, dict) or hit.get("error") or not hit.get("id"):
        log.info("Deezer track lookup found nothing for %r", deezer_id)
        return None

    meta = _hit_fields(hit, "", "")
    art_path = await _download_art(meta["art_url"], track_id, force=True) \
        if meta.get("art_url") else None
    meta["art_path"] = str(art_path) if art_path else None
    meta["source"] = "deezer"
    log.info("Applied Deezer candidate %s for track %s: %s — %s (art: %s)",
             deezer_id, track_id, meta["title"], meta["artist"], bool(art_path))
    return meta


# ---------------------------------------------------------------------------
# Genius (last fallback — only if everything above misses and a token is set)
# ---------------------------------------------------------------------------

def _genius_hits(response: object) -> list[dict]:
    """Flatten a Genius search response into song result dicts.

    Genius buries hits under ``response.sections[].hits`` for typed searches and
    ``response.hits`` otherwise, and wraps each one in ``{"result": {...}}``.
    """
    resp = response.get("response") if isinstance(response, dict) else None
    resp = resp if isinstance(resp, dict) else {}
    hits = [h for h in (resp.get("hits") or []) if isinstance(h, dict)]
    for section in resp.get("sections") or []:
        if isinstance(section, dict):
            hits.extend(h for h in (section.get("hits") or []) if isinstance(h, dict))
    out = []
    for hit in hits:
        result = hit.get("result")
        out.append(result if isinstance(result, dict) else hit)
    return out


def _meta_from_genius_song(song: dict) -> Optional[dict]:
    """Normalise one Genius song object into our metadata dict."""
    title = song.get("title") or song.get("name") or ""
    if not title:
        return None
    album = song.get("album")
    album_name = album.get("name") if isinstance(album, dict) else (album or "")
    art = (song.get("song_art_image_url") or song.get("song_art_image_thumbnail_url")
           or (album.get("cover_art_url") if isinstance(album, dict) else None))
    return {
        "title": title,
        "artist": (song.get("primary_artist") or {}).get("name") or "",
        "album": album_name,
        "art_url": art,
    }


async def _genius_lyrics_search(lyrics: str) -> Optional[dict]:
    """Search Genius **by lyric line** — the thing Genius is actually for.

    A phrase from our own transcript is a much stronger query than the title: it
    is unique to *this* recording, so it can name a song whose title, artist and
    spelling we never knew ("Mama Says - FULL MIX" → the Footloose cast
    recording). Genius's title/artist metadata search was dropped: it does
    Deezer's job with worse art, while this does a job nothing else can.

    Needs an API token; with none configured we stop, exactly as before.
    """
    token = _genius_token()
    if not token:
        log.info("No Genius token configured — skipping Genius lyric search")
        return None
    line = pick_lyric_line(lyrics)
    if not line:
        return None

    try:
        import lyricsgenius
    except ImportError:
        log.warning("lyricsgenius not installed — skipping Genius lyric search")
        return None

    def _search() -> Optional[dict]:
        # lyricsgenius is synchronous; run it off the event loop.
        genius = lyricsgenius.Genius(
            token, verbose=False, remove_section_headers=True,
            skip_non_songs=True, timeout=15,
        )
        for hit in _genius_hits(genius.search_lyrics(line)):
            meta = _meta_from_genius_song(hit)
            if meta:
                meta["source"] = "genius-lyrics"
                return meta
        return None

    try:
        return await asyncio.get_running_loop().run_in_executor(None, _search)
    except Exception as exc:  # noqa: BLE001
        log.info("Genius lyric search failed for %r: %s", line, exc)
        return None


def _genius_token() -> Optional[str]:
    """Read the Genius API token from env or app settings."""
    tok = os.getenv("GENIUS_ACCESS_TOKEN") or os.getenv("GENIUS_API_TOKEN")
    if tok:
        return tok
    try:
        from app.settings import settings
        s = settings.get()
        keys = s.get("keys") or {}
        return (keys.get("genius_key") or s.get("genius_key")
                or s.get("GENIUS_ACCESS_TOKEN") or None)
    except Exception:  # noqa: BLE001 - settings unavailable must not break fetching
        return None


# ---------------------------------------------------------------------------
# Lyrics — the transcript we already have, used two ways:
#   * as the *query* for Genius lyric search (needs a token)
#   * as the *judge* of a shortlist via LRCLIB (keyless, no token needed)
# ---------------------------------------------------------------------------

LRCLIB_SEARCH_URL = "https://lrclib.net/api/search"
LRCLIB_UA = "Refrain/0.1 (personal transcription app)"

# A line has to carry this many words to be worth searching for by itself.
_MIN_LINE_WORDS = 5
# How many of our lines to compare, and how many of the best of those to average.
_COMPARE_LINES = 12
_KEPT_LINES = 3
# Calibrated on real data: the same song scores 1.00 while unrelated songs top
# out around 0.48, so these thresholds sit in a wide empty gap.
LYRICS_MATCH_AT = 0.8
LYRICS_MISMATCH_AT = 0.55


def lyrics_text(lines) -> str:
    """Flatten a transcript (``Line`` objects or plain strings) into lyric text."""
    out = []
    for line in lines or []:
        text = getattr(line, "text", line)
        if text and str(text).strip():
            out.append(str(text).strip())
    return "\n".join(out)


def _norm_lyric_line(raw: str) -> str:
    """Lowercase, drop bracketed asides and punctuation, collapse whitespace."""
    line = re.sub(r"[\(\[\{].*?[\)\]\}]", " ", raw or "").replace("\u2019", "'")
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", line.lower()).split())


def _lyric_lines(text: str, min_words: int = _MIN_LINE_WORDS) -> list[str]:
    """Distinct, normalised lyric lines long enough to identify a song."""
    out, seen = [], set()
    for raw in (text or "").splitlines():
        line = _norm_lyric_line(raw)
        if len(line.split()) < min_words or line in seen:
            continue
        seen.add(line)
        out.append(line)
    return out


def pick_lyric_line(lyrics: str) -> str:
    """The most distinctive line of a transcript, for a lyric search.

    Lyric search engines match a phrase, so a long, unusual line lands the song
    while a short chorus line matches hundreds. Lines that repeat are demoted —
    repetition means a chorus, which is the part every cover shares — and if the
    transcript has no line long enough we just take the longest one.
    """
    counts: dict[str, int] = {}
    for raw in (lyrics or "").splitlines():
        line = _norm_lyric_line(raw)
        if not line or not any(c.isalpha() for c in line):
            continue
        counts[line] = counts.get(line, 0) + 1
    if not counts:
        return ""
    words = {line: len(line.split()) for line in counts}
    candidates = [line for line in counts if words[line] >= _MIN_LINE_WORDS]
    pool = candidates or list(counts)
    return max(pool, key=lambda line: (words[line] - 2 * (counts[line] - 1), words[line]))


async def _lrclib_lyrics(title: str, artist: str) -> Optional[str]:
    """Lyric text for a song from LRCLIB — free and keyless."""
    if not title:
        return None

    import httpx

    params = {"track_name": _clean_query(title)}
    if not _is_generic_artist(artist):
        params["artist_name"] = artist
    try:
        async with httpx.AsyncClient(timeout=12,
                                     headers={"User-Agent": LRCLIB_UA}) as client:
            resp = await client.get(LRCLIB_SEARCH_URL, params=params)
            resp.raise_for_status()
            rows = resp.json()
    except Exception as exc:  # noqa: BLE001 - a lyric lookup must never be fatal
        log.info("LRCLIB lookup failed for %r: %s", title, exc)
        return None

    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, dict) and row.get("plainLyrics"):
            return row["plainLyrics"]
    return None


def lyrics_similarity(ours: str, theirs: str) -> float:
    """How much of our transcript appears in a candidate's lyrics (0..1).

    Transcripts come from ASR, so this is deliberately fuzzy: every one of our
    lines is scored against the candidate's most similar line (difflib ratio
    over lines that share a couple of words) and the mean of the best few lines
    is the score. Measured on real material: the right song scores 1.00, while
    unrelated songs — including the *other* songs with the same title — stay
    under 0.48.
    """
    mine = _lyric_lines(ours)
    theirs_lines = _lyric_lines(theirs, min_words=4)
    if not mine or not theirs_lines:
        return 0.0

    pool = [(set(line.split()), line) for line in theirs_lines]
    per_line = []
    for line in mine[:_COMPARE_LINES]:
        words = set(line.split())
        best = 0.0
        for word_set, candidate in pool:
            if len(words & word_set) < 2:
                continue
            best = max(best, difflib.SequenceMatcher(None, line, candidate).ratio())
        per_line.append(best)
    top = sorted(per_line, reverse=True)[:_KEPT_LINES]
    return sum(top) / len(top)


async def _lyrics_verdict(lyrics: str, candidate: dict) -> Optional[str]:
    """Judge one candidate against our transcript: match, mismatch, or unknown.

    ``None`` means we could not judge it (LRCLIB has no lyrics for that song, or
    the two are too close to call) — such a candidate stays visible, since we
    refuse to veto something we cannot check.
    """
    theirs = await _lrclib_lyrics(candidate.get("title", ""), candidate.get("artist", ""))
    if not theirs:
        return None
    score = lyrics_similarity(lyrics, theirs)
    log.info("Lyrics check for %r: %.2f", candidate.get("title"), score)
    if score >= LYRICS_MATCH_AT:
        return "match"
    if score <= LYRICS_MISMATCH_AT:
        return "mismatch"
    return None


# ---------------------------------------------------------------------------
# Combined lookup + art download
# ---------------------------------------------------------------------------

async def fetch_metadata(
    title: str,
    artist: str,
    *,
    track_id: str | None = None,
    audio_path: Path | str | None = None,
    lyrics: str | None = None,
    cache_art: bool = True,
) -> Optional[dict]:
    """Run the metadata pipeline (Shazam → Deezer → Shazam text → Genius lyrics).

    ``lyrics`` is our own transcript, when we have one: it is what lets the last
    hop search Genius by lyric line instead of by a title we may have made up.

    Returns ``{title, artist, album, art_path, source}`` — ``art_path`` is the
    cached cover on disk — or ``None`` when nothing was found. A missing Genius
    token or a total miss is not an error: the caller keeps its existing
    (possibly empty) metadata.
    """
    meta = None
    if audio_path:
        meta = await _shazam_recognize(Path(audio_path))
    if meta is None:
        meta = await _deezer_search(title, artist)
    if meta is None:
        meta = await _shazam_search(title, artist)
    if meta is None and lyrics:
        # Genius is a lyric site, so it is only worth asking once we have lyrics
        meta = await _genius_lyrics_search(lyrics)
    if meta is None:
        log.info("No metadata found for %r / %r", title, artist)
        return None

    art_path = None
    if meta.get("art_url") and track_id and cache_art:
        # cache_art=False: the caller only wants to *know* the match (a picker's
        # "here's what the pipeline found" entry must not touch the track)
        art_path = await _download_art(meta["art_url"], track_id)

    found_title = meta.get("title") or title
    found_artist = meta.get("artist") or artist
    log.info("Metadata for %r via %s: %s — %s (art: %s)",
             title, meta.get("source"), found_title, found_artist, bool(art_path))
    return {
        "title": found_title,
        "artist": found_artist,
        "album": meta.get("album") or "",
        "art_path": str(art_path) if art_path else None,
        # kept so a caller can offer this match as a pick later (a Shazam hit's
        # cover URL is the only way to get that art — it has no Deezer id)
        "art_url": meta.get("art_url"),
        "source": meta.get("source", "unknown"),
    }


_EXT_BY_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"GIF8", ".gif"),
    (b"RIFF", ".webp"),
)


def _ext_for(data: bytes) -> str:
    """Pick a file extension from the image's magic bytes (default jpg)."""
    for magic, ext in _EXT_BY_MAGIC:
        if data.startswith(magic):
            return ext
    return ".jpg"


async def _download_art(url: str, track_id: str, *, force: bool = False) -> Optional[Path]:
    """Download cover art into ``data/art/<track_id><ext>`` and return it.

    ``force`` replaces an existing cover (a user-picked candidate); without it a
    cached file wins, so a background fetch never re-downloads the same image.
    """
    import httpx

    art_dir = cfg.DATA_DIR / "art"
    art_dir.mkdir(parents=True, exist_ok=True)
    if not force and art_for(track_id):
        return art_for(track_id)          # already cached

    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.content
        if not data:
            return None
        dest = art_dir / f"{track_id}{_ext_for(data)}"
        if force:
            # art_for() prefers .jpg first, so a stale cover of another
            # extension would keep shadowing the replacement.
            for stale in art_dir.glob(f"{track_id}.*"):
                if stale != dest:
                    stale.unlink(missing_ok=True)
        dest.write_bytes(data)
        return dest
    except Exception as exc:  # noqa: BLE001 - art is decorative, never fatal
        log.warning("Failed to download art from %s: %s", url, exc)
        return None


def art_for(track_id: str) -> Optional[Path]:
    """Return the cached art file for a track (any image extension), if any."""
    art_dir = cfg.DATA_DIR / "art"
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
        candidate = art_dir / f"{track_id}{ext}"
        if candidate.exists():
            return candidate
    return None


MEDIA_TYPES = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp", ".gif": "image/gif",
}
