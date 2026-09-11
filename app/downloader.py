"""Download audio: Spotify URLs via sfotipy metadata+matching, else yt-dlp direct."""
from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import app.config as cfg


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


# --- Spotify path (sfotipy metadata + ytmusicapi match, then yt-dlp) ---------

def _spotify_track(url: str) -> dict:
    """Fetch canonical Spotify metadata for a track URL."""
    from sfotipy import downloader as sf
    tracks, dicts = sf.get_track_metadata(url)
    if not dicts:
        raise DownloadError("Could not fetch Spotify metadata for this URL "
                            "(is it a valid track link?)")
    return dicts[0]


def _resolve_youtube_id(track: dict) -> str:
    """Resolve a Spotify track dict to a YouTube( Music) video id.

    Uses sfotipy's ytmusicapi matcher with its strict threshold, then falls
    back to a duration-aware best-of search so near-matches like differing
    artist spellings still resolve.
    """
    from ytmusicapi import YTMusic
    from sfotipy import downloader as sf, matcher as sfm

    yt = YTMusic()
    vid = sfm.find_best_match(track, source="auto", timeout=15)
    if vid:
        return vid

    # Fallback: score results ourselves, leaning on duration agreement.
    expected = int(track.get("duration_seconds") or 0)
    best_id, best_score = None, 0.0
    for flt in ("songs", "videos"):
        try:
            results = yt.search(f"{track.get('artist')} {track.get('title')}",
                                filter=flt) or []
        except Exception:
            continue
        for item in results:
            video_id = item.get("videoId")
            if not video_id:
                continue
            cand = sfm._result_to_dict(item)
            score = sf.score_match(cand, track.get("title", ""),
                                   track.get("artist", ""), expected)
            # reward duration agreement (±3s), which is track-fingerprint-ish
            dur = int(cand.get("duration") or 0)
            if expected and dur:
                if abs(dur - expected) <= 3:
                    score += 0.25
                elif abs(dur - expected) > 20:
                    score -= 0.2
            if score > best_score:
                best_score, best_id = score, video_id
    if best_id and best_score >= 0.5:
        return best_id
    raise DownloadError(
        f"No confident YouTube match for '{track.get('artist')} - {track.get('title')}'")


def _download_spotify(url: str, dest_dir: Path, job_id: str) -> DownloadResult:
    track = _spotify_track(url)
    video_id = _resolve_youtube_id(track)
    result = _run_ytdlp(f"https://www.youtube.com/watch?v={video_id}",
                        dest_dir, job_id)
    # keep canonical Spotify metadata (artist/title), keep resolved duration
    result.title = track.get("title") or result.title
    result.artist = track.get("artist") or result.artist
    result.thumbnail_url = track.get("cover_url") or result.thumbnail_url
    return result


def download_audio(url: str, dest_dir: Path, job_id: str) -> DownloadResult:
    """Route Spotify URLs through sfotipy metadata; everything else via yt-dlp."""
    if is_spotify_url(url):
        return _download_spotify(url, dest_dir, job_id)
    return _run_ytdlp(url, dest_dir, job_id)
