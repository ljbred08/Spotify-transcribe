"""Line assembly from word timings + exporters (LRC, SRT, VTT, plain text).

Line splitting strategy: break at gaps > LINE_GAP seconds or when a line
exceeds MAX_CHARS. Lines longer than MAX_CHARS are split near the middle at
a word boundary so LRC karaoke lines stay readable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from app.models import Line, Word
from app.transcribe import TranscriptionResult

LINE_GAP = 0.7      # seconds of silence that forces a new line
MAX_CHARS = 42      # classic LRC line length


@dataclass
class AssemblyResult:
    lines: list[Line]
    text: str


def assemble_lines(result: TranscriptionResult) -> AssemblyResult:
    words = result.words
    if not words:
        return AssemblyResult(lines=[], text="")

    # 1) group words into lines
    groups: list[list[dict]] = []
    current: list[dict] = []
    prev_end = words[0]["start"]
    for w in words:
        gap = w["start"] - prev_end
        text_len = sum(len(x["text"]) + 1 for x in current)
        if current and (gap > LINE_GAP or text_len > MAX_CHARS):
            groups.append(current)
            current = []
        current.append(w)
        prev_end = w["end"]
    if current:
        groups.append(current)

    # 2) split oversized groups near the middle at a word boundary
    final_groups: list[list[dict]] = []
    for g in groups:
        while sum(len(x["text"]) + 1 for x in g) > MAX_CHARS * 1.4:
            target = sum(len(x["text"]) + 1 for x in g) / 2
            acc, cut = 0, 0
            for i, x in enumerate(g):
                acc += len(x["text"]) + 1
                if acc >= target:
                    cut = max(1, i + 1)
                    break
            left, right = g[:cut], g[cut:]
            final_groups.append(left)
            g = right
        if g:
            final_groups.append(g)

    lines = [
        Line(
            text=" ".join(w["text"] for w in g),
            start=round(g[0]["start"], 3),
            progress=0.0,
            words=[Word(text=w["text"], start=w["start"], end=w["end"]) for w in g],
        )
        for g in final_groups
    ]

    text = "\n".join(l.text for l in lines)
    return AssemblyResult(lines=lines, text=text)


# --- exporters ----------------------------------------------------------------

def _fmt_lrc_time(t: float) -> str:
    minutes = int(t // 60)
    seconds = t - minutes * 60
    return f"{minutes:02d}:{seconds:05.2f}"


def _fmt_timestamp(t: float, ms: bool) -> str:
    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    seconds = t % 60
    if ms:
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def to_lrc(track_title: str, track_artist: str, lines: list[Line]) -> str:
    header = [f"[ti:{track_title}]", f"[ar:{track_artist}]"]
    body = [f"[{_fmt_lrc_time(l.start)}]{l.text}" for l in lines]
    return "\n".join(header + body) + "\n"


def to_lrc_enhanced(track_title: str, track_artist: str, lines: list[Line]) -> str:
    """Enhanced LRC (A2): word-level <mm:ss.xx> start tags per word."""
    out = [f"[ti:{track_title}]", f"[ar:{track_artist}]", "[re:Refrain]"]
    for ln in lines:
        if not ln.words:
            continue
        body = "".join(f"<{_fmt_lrc_time(w.start)}>{w.text} " for w in ln.words)
        out.append(f"[{_fmt_lrc_time(ln.start)}]{body}")
    return "\n".join(out) + "\n"


def to_srt(lines: list[Line]) -> str:
    out = []
    for i, line in enumerate(lines, start=1):
        end = line.words[-1].end if line.words else line.start + 2.0
        end = max(end, line.start + 0.4)
        out.append(f"{i}\n{_fmt_timestamp(line.start, ms=True)} --> "
                   f"{_fmt_timestamp(end, ms=True)}\n{line.text}\n")
    return "\n".join(out)


def to_vtt(lines: list[Line]) -> str:
    out = ["WEBVTT", ""]
    for i, line in enumerate(lines, start=1):
        end = line.words[-1].end if line.words else line.start + 2.0
        end = max(end, line.start + 0.4)
        out.append(f"{i}\n{_fmt_timestamp(line.start, ms=False)} --> "
                   f"{_fmt_timestamp(end, ms=False)}\n{line.text}\n")
    return "\n".join(out)


def to_txt(lines: list[Line]) -> str:
    return "\n".join(l.text for l in lines) + "\n"


def to_json_export(track_title: str, track_artist: str, lines: list[Line]) -> str:
    payload = {
        "title": track_title,
        "artist": track_artist,
        "lines": [
            {"text": l.text, "start": l.start,
             "end": l.words[-1].end if l.words else l.start}
            for l in lines
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def export(fmt: str, track_title: str, track_artist: str,
           lines: list[Line]) -> str:
    dispatch = {
        "lrc": lambda: to_lrc(track_title, track_artist, lines),
        "lrc-enhanced": lambda: to_lrc_enhanced(track_title, track_artist, lines),
        "srt": lambda: to_srt(lines),
        "vtt": lambda: to_vtt(lines),
        "txt": lambda: to_txt(lines),
        "json": lambda: to_json_export(track_title, track_artist, lines),
    }
    fn = dispatch.get(fmt)
    if fn is None:
        raise ValueError(f"Unknown export format: {fmt}")
    return fn()
