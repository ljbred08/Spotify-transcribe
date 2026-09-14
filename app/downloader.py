"""Download audio: Spotify URLs via spotdl metadata+matching, else yt-dlp direct."""
from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import app.config as cfg


def _import_spotdl():
    """Import spotdl pieces lazily so the module loads without spotdl too."""
    from spotdl.providers.audio.youtube import YouTube
    from spotdl.providers.audio.ytmusic import YouTubeMusic
    from ytmusicapi import YTMusic

    return YouTube, YouTubeMusic, YTMusic


YouTube, YouTubeMusic, YTMusic = _import_spotdl()


@dataclass
class DownloadResult:
    audio_path: Path
    title: str
    artist: str
    duration: float
    thumbnail_url: str | None = None


class DownloadError(RuntimeError):
    pass


_SPOTIFY_RE = re.compile(r"open\.spotify\.com/(track|album|playlist)", re.I)


def is_spotify_url(url: str) -> bool:
    return bool(_SPOTIFY_RE.search(url))


def _ytdlp_cmd() -> list[str]:
    # Prefer the module so the pinned dependency is what actually runs.
    base = [sys.executable, "-m", "yt_dlp"]
    if not _module_available():
        base = ["yt-dlp"]
    cmd = list(base)
    if cfg.YTDLP_COOKIES_FILE:
        p = Path(cfg.YTDLP_COOKIES_FILE)
        if p.exists():
            cmd += ["--cookies", str(p)]
    if cfg.YTDLP_EXTRA:
        cmd += cfg.YTDLP_EXTRA
    return cmd


def _module_available() -> bool:
    try:
        import yt_dlp  # noqa: F401
        return True
    except ImportError:
        return False


def _run_ytdlp(url: str, dest_dir: Path, job_id: str) -> DownloadResult:
    """Download best audio as m4a via yt-dlp; returns metadata + final path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_tpl = str(dest_dir / f"{job_id}.%(ext)s")
    cmd = _ytdlp_cmd() + [
        "-x", "--audio-format", "m4a",
        "--audio-quality", "0",
        "-o", out_tpl,
        "--print-json",
        "--no-playlist",
        "--no-warnings",
        url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-6:]
        raise DownloadError("\n".join(tail) or "yt-dlp failed with no output")

    info = _print_json(proc)
    if not info:
        raise DownloadError("yt-dlp produced no JSON metadata")

    path = Path(info.get("filepath") or dest_dir / f"{job_id}.m4a")
    if not path.exists():
        # ffmpeg may have transcoded to a different container name
        candidates = sorted(dest_dir.glob(f"{job_id}.*"),
                            key=lambda p: p.stat().st_size, reverse=True)
        if not candidates:
            raise DownloadError("Downloaded file not found on disk")
        path = candidates[0]

    artist = info.get("artist") or info.get("uploader") or "Unknown artist"
    return DownloadResult(
        audio_path=path,
        title=info.get("title") or "Untitled",
        artist=artist,
        duration=float(info.get("duration") or 0.0),
        thumbnail_url=info.get("thumbnail"),
    )


def _print_json(proc: subprocess.CompletedProcess) -> dict | None:
    import json
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


# --- Spotify path (spotdl metadata + matching, then yt-dlp) -----------------

_SPOTIFY_CLIENT_READY = False


def _ensure_spotify_client() -> None:
    """Initialize spotdl's Spotify client once (its public default creds).

    spotdl ships its own client_id/secret (the same ones its CLI uses), so
    users don't need their own Spotify developer app. Set SPOTIPY_CLIENT_ID
    / SPOTIPY_CLIENT_SECRET env vars to override.
    """
    global _SPOTIFY_CLIENT_READY
    if _SPOTIFY_CLIENT_READY:
        return
    import os

    from spotdl.utils.config import DEFAULT_CONFIG
    from spotdl.utils.spotify import SpotifyClient

    SpotifyClient.init(
        os.environ.get("SPOTIPY_CLIENT_ID") or DEFAULT_CONFIG["client_id"],
        os.environ.get("SPOTIPY_CLIENT_SECRET") or DEFAULT_CONFIG["client_secret"],
        user_auth=False,
        no_cache=True,
        headless=True,
        use_cache_file=False,
        max_retries=3,
    )
    _SPOTIFY_CLIENT_READY = True


class _FixedYTMusic(YouTubeMusic):
    """YouTubeMusic with the default-language client.

    spotdl 4.5.2 hardcodes ``YTMusic(language="de")``, which currently returns
    ZERO results from the YouTube Music search endpoint. The default-language
    client works, so patch it here (upstream may fix; subclassing keeps us
    compatible either way).
    """

    @staticmethod
    def _create_client() -> YTMusic:  # type: ignore[override]
        return YTMusic()


def _spotify_song(url: str) -> "Song":
    """Fetch canonical Spotify metadata for a track URL as a spotdl Song."""
    _ensure_spotify_client()
    from spotdl.types.song import Song

    return Song.from_url(url)


def _resolve_youtube_url(song: "Song") -> str:
    """Resolve a spotdl Song to a YouTube Music url via spotdl's matcher.

    Tries the YT Music provider first (structured metadata, verified-song
    results), then the plain YouTube provider (ytsearch-based) as fallback.
    spotdl's matching scores title/artist/album/duration together and hard-
    skips candidates whose artist or name don't overlap — the failure mode
    that picked wrong videos before.
    """
    errors: list[str] = []
    for provider_cls in (_FixedYTMusic, YouTube):
        try:
            provider = provider_cls(output_format="m4a")
            result_url = provider.search(song)
        except Exception as exc:  # noqa: BLE001 - provider can raise mid-search
            errors.append(f"{provider_cls.__name__}: {exc}")
            continue
        if result_url:
            return result_url
        errors.append(f"{provider_cls.__name__}: no confident match")
    raise DownloadError(
        f"No confident YouTube match for '{song.artist} - {song.name}' "
        f"({'; '.join(errors)})")


def _download_spotify(url: str, dest_dir: Path, job_id: str) -> DownloadResult:
    song = _spotify_song(url)
    video_url = _resolve_youtube_url(song)
    result = _run_ytdlp(video_url, dest_dir, job_id)
    # keep canonical Spotify metadata (artist/title/cover), keep resolved duration
    result.title = song.name or result.title
    result.artist = song.artist or result.artist
    result.thumbnail_url = song.cover_url or result.thumbnail_url
    return result


def download_audio(url: str, dest_dir: Path, job_id: str) -> DownloadResult:
    """Route Spotify URLs through spotdl metadata; everything else via yt-dlp."""
    if is_spotify_url(url):
        return _download_spotify(url, dest_dir, job_id)
    return _run_ytdlp(url, dest_dir, job_id)
