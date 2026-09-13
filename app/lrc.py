"""Enhanced LRC (A2) parsing + serialization — the canonical transcript format.

Transcripts are stored as Enhanced LRC text (word-level <mm:ss.xx> tags), so
the format we get from audio LLMs is exactly what we keep on disk: portable,
human-inspectable, and re-parsed on every load — improvements to this parser
apply retroactively to every stored transcript without retranscribing.
"""
from __future__ import annotations

import re

_LINE_TS = re.compile(r"^\[(\d{1,3}):(\d{1,2})(?:[.:](\d{1,3}))?\]")
_WORD_TS = re.compile(r"<(\d{1,3}):(\d{1,2})(?:[.:](\d{1,3}))?>")
# bare bracket timestamps that appear MID-text (not as a line tag) — some
# models (e.g. gemini flash-lite) emit these instead of proper <> word tags
_ANY_BRACKET = re.compile(r"\[(\d{1,3}):(\d{1,2})(?:[.:](\d{1,3}))?\]")
_OFFSET_TAG = re.compile(r"^\[offset:\s*([+-]?\d+)\s*\]", re.I)


def _tag_seconds(mm: str, ss: str, frac: str | None) -> float:
    t = int(mm) * 60 + int(ss)
    if frac:
        t += int(frac) / (10 ** len(frac))
    return t


def fmt_time(t: float) -> str:
    """Seconds -> LRC `mm:ss.xx` (centisecond precision)."""
    t = max(0.0, t)
    m = int(t // 60)
    s = t - m * 60
    return f"{m:02d}:{s:05.2f}"


def parse_enhanced(raw: str | None) -> list[dict] | None:
    """Parse Enhanced LRC (word-level <mm:ss.xx> tags) into lines of words.

    Word tags carry START times only, so a word's end is the next tag's time
    (next line's first tag for line-final words). A trailing/empty tag at the
    end of a line is the previous word's explicit END time (Gemini emits
    those). Plain LRC lines without word tags get their text evenly
    interpolated across the line's time span.
    Returns None when nothing timed could be extracted.
    """
    if not raw or not raw.strip():
        return None
    content = "\n".join(ln for ln in raw.splitlines()
                        if not ln.strip().startswith("```"))
    offset_ms = 0.0
    # (line_start, [(word_start, token, explicit_end|None), ...]) per line
    entries: list[tuple[float, list[tuple[float, str, float | None]]]] = []
    def _angle(m: re.Match) -> str:
        inner = f"{m.group(1)}:{m.group(2)}"
        if m.group(3):
            inner += f".{m.group(3)}"
        return f"<{inner}>"

    for ln in content.splitlines():
        s = ln.strip()
        om = _OFFSET_TAG.match(s)
        if om:
            offset_ms = float(om.group(1))
            continue
        # normalize MID-TEXT bracket timestamps into <> word tags (some
        # models emit [mm:ss:cs] between words instead of proper LRC); a
        # leading line-position tag is left for _LINE_TS below
        s = _ANY_BRACKET.sub(lambda m: m.group(0) if m.start() == 0
                             else _angle(m), s)
        m = _LINE_TS.match(s)
        if not m:
            # tolerate tag-only lines (no leading [mm:ss.xx] line tag): the
            # first word tag carries the line start
            wm0 = _WORD_TS.search(s)
            if not wm0:
                continue  # metadata tag ([ti:], [ar:], …) or noise
            t0 = _tag_seconds(*wm0.groups())
            rest = (s[:wm0.start()].strip() + " " + s[wm0.end():]).strip()
            # re-inject the words that preceded the first tag at line start
            tags = list(_WORD_TS.finditer(rest))
        else:
            t0 = _tag_seconds(*m.groups())
            rest = s[m.end():]
            tags = list(_WORD_TS.finditer(rest))
        # (start, token, explicit_end|None) per word
        wlist: list[tuple[float, str, float | None]] = []
        if tags:
            lead = rest[:tags[0].start()].strip()
            if lead:
                wlist.extend((t0, tok, None) for tok in lead.split())
            for i, wm in enumerate(tags):
                wt = _tag_seconds(*wm.groups())
                nxt = tags[i + 1].start() if i + 1 < len(tags) else len(rest)
                chunk = rest[wm.end():nxt].strip()
                if chunk:
                    wlist.extend((wt, tok, None) for tok in chunk.split())
                elif wlist:
                    # trailing/empty tag: explicit end for the last open word
                    for j in range(len(wlist) - 1, -1, -1):
                        if wlist[j][2] is None:
                            wlist[j] = (wlist[j][0], wlist[j][1], wt)
                            break
        else:
            wlist.extend((t0, tok, None) for tok in rest.split())
        if wlist:
            entries.append((t0, wlist))
    if not entries:
        return None
    entries.sort(key=lambda x: x[0])
    adj = offset_ms / 1000.0
    lines: list[dict] = []
    for i, (_t0, wlist) in enumerate(entries):
        ws = []
        for j, (wt, tok, xend) in enumerate(wlist):
            start = wt - adj
            if xend is not None:
                end = xend - adj
            elif j + 1 < len(wlist):
                end = wlist[j + 1][0] - adj
            elif i + 1 < len(entries):
                end = entries[i + 1][1][0][0] - adj
            else:
                end = start + 1.0
            end = max(end, start)
            ws.append({"t": tok, "s": round(start, 3), "e": round(end, 3)})
        lines.append({"text": " ".join(w["t"] for w in ws), "words": ws})
    return lines


def serialize(title: str, artist: str, lines) -> str:
    """Lines (app.models.Line) -> Enhanced LRC text with word tags.

    Every word gets its start tag; each line ends with a trailing end-time
    tag for its final word (round-trips through parse_enhanced exactly).
    """
    out = [f"[ti:{title}]", f"[ar:{artist}]", "[re:Refrain]"]
    for ln in lines:
        if not ln.words:
            continue
        body = "".join(f"<{fmt_time(w.start)}>{w.text} " for w in ln.words)
        body += f"<{fmt_time(ln.words[-1].end)}>"
        out.append(f"[{fmt_time(ln.start)}]{body}")
    return "\n".join(out) + "\n"
