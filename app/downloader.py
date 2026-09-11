"""Download audio via yt-dlp (Spotify URLs are resolved by its extractors)."""
from __future__ import annotations

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


class DownloadError(RuntimeError):
    pass


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


def download_audio(url: str, dest_dir: Path, job_id: str) -> DownloadResult:
    """Download best audio as m4a; returns metadata + final path."""
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

    info = None
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            import json
            try:
                info = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
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
    )
