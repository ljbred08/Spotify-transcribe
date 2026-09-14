"""Sync improvement: retime a good transcript from a transcription-only reference.

The active transcript is the source of truth for WORDS and LINE BREAKS; the
reference transcript (a transcribe-only model such as Groq Whisper, whose
timestamps are measured rather than generated) is the source of truth for
TIMINGS. The two are aligned with a global Needleman-Wunsch pass over the
whole word sequence — whole-transcript, because a misheard word only maps to
its counterpart when the surrounding context lines up:

    good:  "We're no strangers to love"
    ref:   "We strangers to love"
    →      "We're"←We, "no" (missed in ref) interpolated between its
           retimed neighbours, "strangers to love" retimed.

    good:  "...rules and so do I"
    ref:   "...rules but so do I"
    →      "and"←"but": they don't fuzzy-match at all, but aligning the whole
           transcript forces the pair (both-missing-here makes no sense).

    good:  "...and so do I"
    ref:   "...and so too do I"
    →      the invented "too" gets no old-word partner and is dropped.

Scoring: a word pair scores (2·sim − 1) with sim = fuzzy similarity in [0,1],
a gap costs −1. A forced pair of dissimilar words (−1) therefore always beats
leaving BOTH sides as gaps (−2), while a spurious reference word between two
good matches is dropped (−1) because keeping both words matched (−1 −1) is
worse. Choruses stay in order automatically — global alignment preserves
sequence, so repeat occurrences map occurrence-to-occurrence.
"""
from __future__ import annotations

from difflib import SequenceMatcher

from app.models import Line, Word

# quality gate: below this fraction of genuinely-matched good words the
# alignment is considered bogus (wrong song / garbage reference) and refused
MIN_COVERAGE = 0.35
# hard cell cap for the DP matrix (≈ 2400 x 2500 words)
_MAX_CELLS = 6_000_000
_GAP = -1.0


class SyncError(RuntimeError):
    pass


def _norm(text: str) -> str:
    """Lowercase; keep alphanumerics and internal apostrophes."""
    out = []
    for ch in (text or "").lower():
        if ch.isalnum() or (ch == "'" and out):
            out.append(ch)
    return "".join(out).strip("'")


def _sim(a: str, b: str) -> float:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


# traceback directions
_DIAG, _UP, _LEFT = 1, 2, 3


def _align(old: list[str], new: list[str]) -> list[tuple[str, int, int, float]]:
    """Global NW alignment. Returns ops in order:

    ("pair", old_idx, new_idx, sim)  — retiming pair (any similarity)
    ("keep", old_idx, -1, 0.0)       — old word with no reference partner:
                                       keep its existing timing
    ("drop", -1, new_idx, 0.0)       — reference word with no old partner:
                                       invented/hallucinated, ignore it
    """
    n, m = len(old), len(new)
    if n * m > _MAX_CELLS:
        raise SyncError("Song is too long to align")

    cache: dict[tuple[str, str], float] = {}

    def sim(i: int, j: int) -> float:
        key = (old[i], new[j])
        v = cache.get(key)
        if v is None:
            v = _sim(*key)
            cache[key] = v
        return v

    # rolling score row + full direction matrix (1 byte/cell)
    prev = [j * _GAP for j in range(m + 1)]
    dirs = [bytearray(m + 1) for _ in range(n + 1)]
    for j in range(1, m + 1):
        dirs[0][j] = _LEFT
    for i in range(1, n + 1):
        cur = [0.0] * (m + 1)
        cur[0] = prev[0] + _GAP
        dirs[i][0] = _UP
        for j in range(1, m + 1):
            diag = prev[j - 1] + (2.0 * sim(i - 1, j - 1) - 1.0)
            up = prev[j] + _GAP        # old word missing in reference
            left = cur[j - 1] + _GAP   # reference word missing in old
            best, d = diag, _DIAG
            if up > best:
                best, d = up, _UP
            if left > best:
                best, d = left, _LEFT
            cur[j] = best
            dirs[i][j] = d
        prev = cur

    ops: list[tuple[str, int, int, float]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dirs[i][j] == _DIAG:
            s = sim(i - 1, j - 1)
            ops.append(("pair", i - 1, j - 1, s))
            i -= 1
            j -= 1
        elif i > 0 and dirs[i][j] == _UP:
            ops.append(("keep", i - 1, -1, 0.0))
            i -= 1
        elif j > 0:
            ops.append(("drop", -1, j - 1, 0.0))
            j -= 1
        else:  # j == 0: remaining old words
            ops.append(("keep", i - 1, -1, 0.0))
            i -= 1
    ops.reverse()
    return ops


def _clamp_monotonic(words: list[Word]) -> list[Word]:
    """Guard against degenerate reference timings inside one line.

    Transcribers emit monotonic timestamps, so this is normally a no-op; it
    only fires when a reference word would start before its predecessor ends
    (or ends before it starts), which would make the karaoke sweep jump
    backwards. Times are nudged forward, never earlier.
    """
    prev_end = -1.0
    for w in words:
        if w.end < w.start:
            w.end = w.start + 0.05
        if w.start < prev_end:
            shift = prev_end - w.start
            w.start += shift
            w.end = max(w.end + shift, w.start + 0.05)
        prev_end = w.end
    return words


def improve_sync(lines: list[Line], ref_words: list[dict], *,
                 min_coverage: float = MIN_COVERAGE) -> tuple[list[Line], dict]:
    """Retime `lines` (good transcript) using timings from `ref_words`.

    ref_words: [{text, start, end}] sorted-or-not, from a transcribe-only
    backend. Returns (new_lines, stats) — the input is not mutated.
    Raises SyncError when the reference matches the transcript too poorly.
    """
    ref = [{"text": str(w.get("text", "")), "start": float(w.get("start", 0.0)),
            "end": float(w.get("end", 0.0))} for w in ref_words]
    if not ref:
        raise SyncError("Reference transcript has no timed words")

    flat: list[tuple[int, int, Word]] = [   # (line_idx, word_idx_in_line, Word)
        (li, wi, w)
        for li, line in enumerate(lines)
        for wi, w in enumerate(line.words)]
    if not flat:
        raise SyncError("Current transcript has no words to retime")

    ops = _align([w.text for _, _, w in flat], [w["text"] for w in ref])

    # working copies of every old word
    words = [[Word(text=w.text, start=w.start, end=w.end) for w in line.words]
             for line in lines]
    matched = forced = kept = dropped = 0
    last_end: float | None = None      # end of the last word with final timing
    pending: list[tuple[int, int]] = []  # (line, word) indices awaiting timing

    def _fill_pending(right_start: float | None) -> None:
        """Time the missed words sitting between two anchored words.

        Per spec: a missed word KEEPS its existing timestamp whenever that
        still sits between its retimed neighbours (within tolerance) — it was
        never the problem. Stale timings are only repaired when they would
        visibly break the karaoke sweep: outside the surrounding slot, or
        dangling past the last anchor (outro keeps before the audio ends).
        """
        nonlocal last_end
        if not pending:
            return
        n = len(pending)
        if right_start is not None:
            span = (right_start - (last_end if last_end is not None else 0.0))
            room = span * 1.6 + 0.6      # generous tolerance for the slot
            if last_end is not None \
                    and all(0 <= (words[li][wi].start - last_end) <= room
                            for li, wi in pending):
                pending.clear()           # old timings still fit — keep them
                return
            if last_end is None:
                # song-start keeps: keep when they don't overshoot the anchor
                if all(0 <= (words[li][wi].end) <= right_start
                       for li, wi in pending):
                    pending.clear()
                    return
                for k, (li, wi) in enumerate(pending):
                    w = words[li][wi]
                    w.start = round(max(0.0, right_start - 0.25 * (n - k)), 3)
                    w.end = round(max(w.start + 0.2,
                                      min(right_start, w.start + 0.25)), 3)
                pending.clear()
                return
            # stale: spread between the anchors
            slot = span / (n + 1)
            for k, (li, wi) in enumerate(pending):
                w = words[li][wi]
                s0 = last_end + slot * (k + 1)
                w.start = round(s0, 3)
                w.end = round(s0 + min(slot * 0.8, 1.0), 3)
            pending.clear()
            return
        # trailing keeps (song outro): keep when they plausibly continue the
        # timeline; otherwise continue after the last anchor
        base = last_end if last_end is not None else 0.0
        if last_end is not None and all(words[li][wi].start >= base
                                        for li, wi in pending):
            pending.clear()
            return
        for k, (li, wi) in enumerate(pending):
            w = words[li][wi]
            w.start = round(base + 0.3 * k, 3)
            w.end = round(w.start + 0.25, 3)
        pending.clear()

    for kind, oi, nj, s in ops:
        if kind == "pair":
            li, wi, ow = flat[oi]
            rw = ref[nj]
            _fill_pending(rw["start"])
            words[li][wi] = Word(text=ow.text, start=rw["start"], end=rw["end"])
            last_end = rw["end"]
            if s >= 0.5:
                matched += 1
            else:
                forced += 1
        elif kind == "keep":
            pending.append((flat[oi][0], flat[oi][1]))
            kept += 1
        else:
            dropped += 1
    _fill_pending(None)

    coverage = matched / len(flat)
    if coverage < min_coverage:
        raise SyncError(
            f"Reference timings don't match this transcript well enough "
            f"({matched}/{len(flat)} words, {coverage:.0%}) — try a different "
            f"transcription model")

    new_lines: list[Line] = []
    for li, line in enumerate(lines):
        ws = _clamp_monotonic(words[li])
        start = ws[0].start if ws else line.start
        new_lines.append(Line(text=line.text, start=start, progress=0.0,
                              words=ws))
    stats = {
        "old_words": len(flat),
        "ref_words": len(ref),
        "matched": matched,
        "forced": forced,
        "kept": kept,
        "dropped": dropped,
        "coverage": round(coverage, 3),
    }
    return new_lines, stats
