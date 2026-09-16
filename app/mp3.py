"""MP3 export: the track's audio with its tags and its lyrics baked in.

Three lyric layers go into one MP3, because no single frame covers every
player:

* ``SYLT`` — word-level karaoke. Absolute-millisecond tuples
  (``("word ", 1200)``), so a player can animate as it sings. Trailing spaces
  are part of the tokens: without them players render words bunched together.
* ``USLT`` — plain, human-readable lyrics with every timestamp stripped, for
  the players (and tag editors) that show a static lyric pane.
* ``COMM`` — the raw Enhanced LRC script, so the timing information survives as
  text: re-importable, greppable, and immune to a player that ignores SYLT.

Plus the ordinary tags (title, artist, album) and the cached cover as an APIC
front-cover frame. Source audio is copied when it already is an MP3 and
transcoded through ffmpeg when it is not (m4a, wav, …), so a download is always
a real MP3 whatever the downloader handed us.
"""
from __future__ import annotations

import logging
import re
import shutil
import tempfile
from pathlib import Path

from mutagen.id3 import (APIC, COMM, ID3, SYLT, TALB, TIT2, TPE1, USLT,
                         Encoding, ID3NoHeaderError)

from app.models import Line

log = logging.getLogger(__name__)

# SYLT wants these spelled out: format 2 = timestamps are absolute
# milliseconds, type 1 = the content is lyrics.
SYLT_TIMESTAMP_MS = 2
SYLT_CONTENT_LYRICS = 1
# No language detection exists in the pipeline, so everything is tagged "eng".
LANG = "eng"
# mp3 is a lossy target: give a transcoded source room to breathe.
TRANSCODE_BITRATE = "256k"

# Leading bracketed tags on a line: our own [ti:]/[ar:]/[re:] headers and the
# [mm:ss.xx] line timestamps (2 or 3 hundredth digits — take both flavors).
_LEADING_TAGS_RE = re.compile(r"^\s*(?:\[[^\]]*\]\s*)+")
# <mm:ss.xx> word-level timestamps, wherever they sit.
_WORD_TAGS_RE = re.compile(r"<\s*\d{1,2}:\d{2}(?:[.:]\d{1,3})?\s*>")


def clean_lyrics(text: str) -> str:
    """Strip the timing and header tags, then tidy the lines.

    This is the USLT half: timestamps and `[ti:]`-style headers belong in SYLT
    and in the raw script, not in the text a listener reads.
    """
    out = []
    for raw in (text or "").splitlines():
        line = _WORD_TAGS_RE.sub("", _LEADING_TAGS_RE.sub("", raw)).strip()
        if line:
            out.append(line)
    return "\n".join(out)


def sylt_tokens(lines: list[Line]) -> list[tuple[str, int]]:
    """Word-level tuples for SYLT: ``[("Hello ", 1200), ("world ", 1600)]``.

    Each token keeps a trailing space so a renderer that lays words out itself
    does not bunch them together; a line's last token ends in a newline instead,
    which is how a line break is expressed inside a single-text blob.
    """
    out: list[tuple[str, int]] = []
    for line in lines:
        for i, word in enumerate(line.words):
            text = (word.text or "").strip()
            if not text:
                continue
            last = i == len(line.words) - 1
            out.append((f"{text}\n" if last else f"{text} ",
                        max(0, int(round(word.start * 1000)))))
    return out


def embed_lyrics(path: Path, *, title: str, artist: str, album: str,
                 lines: list[Line], enhanced_lrc: str,
                 art: Path | None = None) -> None:
    """Write the tags in place: metadata, cover, then the three lyric layers."""
    try:
        tags = ID3(path)
    except ID3NoHeaderError:
        tags = ID3()      # a fresh encode has no tag block yet

    tags.setall("TIT2", [TIT2(encoding=Encoding.UTF8, text=[title])])
    tags.setall("TPE1", [TPE1(encoding=Encoding.UTF8, text=[artist])])
    if album:
        tags.setall("TALB", [TALB(encoding=Encoding.UTF8, text=[album])])

    if art is not None and art.exists():
        mime = "image/png" if art.suffix.lower() == ".png" else "image/jpeg"
        tags.setall("APIC", [APIC(encoding=Encoding.UTF8, mime=mime, type=3,
                                  desc="Cover (front)", data=art.read_bytes())])

    # 1) word-level sync (the karaoke layer)
    tags.setall("SYLT", [SYLT(encoding=Encoding.UTF8, lang=LANG,
                              format=SYLT_TIMESTAMP_MS, type=SYLT_CONTENT_LYRICS,
                              text=sylt_tokens(lines))])
    # 2) clean text (the read-along layer)
    tags.setall("USLT", [USLT(encoding=Encoding.UTF8, lang=LANG, desc="Lyrics",
                              text=clean_lyrics(enhanced_lrc))])
    # 3) the enhanced-LRC script itself (the fidelity backup)
    tags.setall("COMM", [COMM(encoding=Encoding.UTF8, lang=LANG, desc="Comment",
                              text=enhanced_lrc)])
    tags.save(path)


def _write_mp3(source: Path, target: Path) -> None:
    """Materialise ``target`` as an MP3: copy when it already is one, else transcode."""
    if source.suffix.lower() == ".mp3":
        shutil.copyfile(source, target)
        return
    try:
        from pydub import AudioSegment
    except ImportError as exc:                     # pragma: no cover - env guard
        raise RuntimeError(
            "pydub is required to convert this track to MP3 "
            "(pip install pydub audioop-lts)") from exc
    try:
        AudioSegment.from_file(source).export(target, format="mp3",
                                              bitrate=TRANSCODE_BITRATE)
    except FileNotFoundError as exc:               # ffmpeg missing
        raise RuntimeError(
            "ffmpeg is required to convert this track to MP3 and was not found "
            "on PATH") from exc
    except Exception as exc:  # noqa: BLE001 - pydub raises its own decode errors
        raise RuntimeError(f"Could not convert {source.suffix or 'audio'} to "
                          f"MP3: {exc}") from exc


def export_mp3(source: Path, *, title: str, artist: str, album: str = "",
               lines: list[Line], enhanced_lrc: str,
               art: Path | None = None) -> bytes:
    """The finished MP3 as bytes: audio + tags + all three lyric layers."""
    if not source.exists():
        raise RuntimeError("Track audio is missing")
    with tempfile.TemporaryDirectory(prefix="refrain-mp3-") as tmp:
        out = Path(tmp) / "track.mp3"
        _write_mp3(source, out)
        # tag a real file rather than a buffer: mutagen's tag surgery is
        # implemented against seekable files and this is the well-trodden path
        embed_lyrics(out, title=title, artist=artist, album=album, lines=lines,
                     enhanced_lrc=enhanced_lrc, art=art)
        data = out.read_bytes()
    log.info("Exported MP3 for %r (%d bytes, %d words)", title, len(data),
             sum(len(l.words) for l in lines))
    return data
