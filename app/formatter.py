"""LLM-powered lyric formatting: punctuation, capitalization, line breaks.

Design: the ASR returns one continuous word stream with timings. A small local
LLM (via ollama or any OpenAI-compatible server) reformats it into short
karaoke-style lines. Because karaoke requires exact word↔timing mapping, the
LLM output is *verified* against the input words and any chunk that fails
validation falls back to the heuristic splitter. The LLM therefore only ever
improves presentation; timing integrity is guaranteed deterministically.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import httpx

from app.models import Line, Word

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
# NOTE: LFM2.5-350M was evaluated and cannot do faithful copy-tasks (stops
# after ~25 tokens, rewrites words). qwen2.5:3b is the smallest tested model
# that reformats reliably; any chat model works via FORMATTER_MODEL.
FORMATTER_MODEL = os.environ.get("FORMATTER_MODEL", "qwen2.5:3b")
CHUNK_WORDS = 80       # words per LLM request (keeps outputs short + reliable)
MAX_LINE_CHARS = 52    # karaoke line length cap
# words an LLM may legitimately add (stylistic connectives / ad-libs)
_OK_INSERTS = {"oh", "ooh", "yeah", "yea", "hey", "la", "na", "uh", "ah",
               "come", "on", "now", "then", "so", "and", "but", "cmon"}
_MAX_INSERT_RUN = 2    # longer invented runs = hallucination → reject chunk
_SENTENCE_GAP = 0.9    # silence gap after a phrase that suggests a sentence end

# Task framing matters: asking a small model to "reformat" a transcript of a
# famous song makes it recite the REAL lyrics from memory instead of copying
# (verified with qwen2.5:3b on the rickroll — pronouns dropped, verses invented).
# Asking it to insert line-break markers keeps the copy mechanical: it produced
# an exact token-preserving result on the same input.
_PROMPT = """Insert line-break markers " / " into this song transcript at natural lyric line boundaries. Also add punctuation and capitalization, but copy every word EXACTLY as given — never add, remove, or change words. The transcript is automatic and may contain odd words; keep them anyway. Output only the result.

{chunk}"""

# Used when the whole-chunk pass fails (e.g. the model recites famous lyrics
# from memory instead of copying): format each timing-derived phrase alone.
# Tiny phrases leave no room for memory contamination, and any rewritten
# phrase simply keeps its raw text — polish where possible, never wrong.
_PHRASE_PROMPT = """Add punctuation and capitalization to this short transcribed lyric phrase. Copy every word EXACTLY as given — never add, remove, or change words. It may contain odd words; keep them. Output only the formatted phrase on one line.

{phrase}"""


@dataclass
class FormatResult:
    lines: list[Line]
    llm_lines: int      # lines that came from the LLM
    fallback_lines: int  # lines from the heuristic fallback


def _norm(tok: str) -> str:
    return re.sub(r"[^a-z0-9']", "", tok.lower())


def llm_available() -> bool:
    """True when the formatter model is reachable and pulled."""
    try:
        with httpx.Client(timeout=2.0) as c:
            tags = c.get(f"{OLLAMA_URL}/api/tags").json()
        names = {m.get("name", "") for m in tags.get("models", [])}
        return any(n.startswith(FORMATTER_MODEL) for n in names)
    except Exception:
        return False


def _chat(chunk_text: str, *, phrase: bool = False) -> str | None:
    prompt = _PHRASE_PROMPT if phrase else _PROMPT
    try:
        with httpx.Client(timeout=240.0) as c:
            r = c.post(f"{OLLAMA_URL}/api/chat", json={
                "model": FORMATTER_MODEL,
                "messages": [{"role": "user", "content": prompt.format(
                    chunk=chunk_text, phrase=chunk_text)}],
                "stream": False,
                "keep_alive": "30m",
                "options": {"temperature": 0.0, "num_ctx": 4096, "num_predict": 500},
            })
        return r.json()["message"]["content"]
    except Exception:
        return None


def _chunks(words: list[Word], size: int) -> list[list[Word]]:
    return [words[i:i + size] for i in range(0, len(words), size)]


def _verify_and_align(chunk: list[Word], llm_text: str) -> list[list[Word]] | None:
    """Map the LLM's lines back onto input words.

    Returns a list of word-lists (one per line), or None when the LLM output
    is not a faithful reformatting of the chunk. Inserted words are tolerated
    (kept, with interpolated timings); any DROPPED input word rejects the
    whole chunk — karaoke must not silently lose words.
    """
    # the LLM marks line boundaries with " / " — turn them into real breaks
    # (handle both embedded markers and native line wraps)
    raw_lines: list[str] = []
    for ln in llm_text.splitlines():
        for part in ln.split(" / "):
            part = part.strip()
            if part:
                raw_lines.append(part)
    if not raw_lines:
        return None
    out_lines: list[list[Word]] = []
    inp = 0
    n_in = len(chunk)
    vocab = {_norm(w.text) for w in chunk if _norm(w.text)}
    insert_run = 0

    for line_text in raw_lines:
        out_words: list[Word] = []
        for tok in line_text.split():
            nt = _norm(tok)
            if not nt:
                continue
            # find the next matching input word within a small lookahead
            matched = -1
            for j in range(inp, min(inp + 6, n_in)):
                if _norm(chunk[j].text) == nt:
                    matched = j
                    break
            if matched == -1:
                for j in range(inp, min(inp + 6, n_in)):
                    a, b = _norm(chunk[j].text), nt
                    if a and b and (a.startswith(b) or b.startswith(a)):
                        matched = j
                        break
            if matched > inp:
                return None  # lookahead skipped input words = the LLM dropped them
            if matched == -1:
                # LLM inserted a word: only tolerate stylistic connectives or
                # repeats of words from this chunk, in short runs
                if (nt not in _OK_INSERTS and nt not in vocab) or insert_run >= _MAX_INSERT_RUN:
                    return None
                prev_end = chunk[inp - 1].end if inp > 0 else chunk[0].start
                next_start = chunk[inp].start if inp < n_in else prev_end + 0.2
                w = Word(text=tok, start=prev_end, end=next_start)
                w.inserted = True  # type: ignore[attr-defined]
                out_words.append(w)
                insert_run += 1
                continue
            insert_run = 0
            out_words.append(chunk[matched])
            inp = matched + 1
        if out_words:
            out_lines.append(out_words)
        if inp >= n_in:
            break

    # unmatched trailing input words = the LLM stopped early → reject
    if inp < n_in:
        return None

    # trim hallucinated tails: interpolated words after the last real match
    # of the final line are invention, not formatting
    if out_lines:
        last = out_lines[-1]
        while last and getattr(last[-1], "inserted", False):
            last.pop()
        if not last:
            out_lines.pop()
    if not out_lines:
        return None

    # degenerate one-word-per-line outputs (only meaningful for long chunks)
    if n_in >= 12 and len(out_lines) > n_in // 2:
        return None
    return out_lines


def _split_long(line_words: list[Word]) -> list[list[Word]]:
    """Split over-long lines at word boundaries."""
    out: list[list[Word]] = []
    cur: list[Word] = []
    for w in line_words:
        cur.append(w)
        if sum(len(x.text) + 1 for x in cur) >= MAX_LINE_CHARS:
            out.append(cur)
            cur = []
    if cur:
        if out and len(cur) <= 2:
            out[-1].extend(cur)  # avoid 1-2 word orphan lines
        else:
            out.append(cur)
    return out


def _heuristic_lines(chunk: list[Word]) -> list[list[Word]]:
    """Silence-gap line splitting (same rule as exporters, chunk-local)."""
    groups: list[list[Word]] = []
    cur: list[Word] = []
    prev_end = chunk[0].start
    for w in chunk:
        if cur and (w.start - prev_end > 0.7
                    or sum(len(x.text) + 1 for x in cur) > MAX_LINE_CHARS):
            groups.append(cur)
            cur = []
        cur.append(w)
        prev_end = w.end
    if cur:
        groups.append(cur)
    return groups


def _to_lines(groups: list[list[Word]]) -> list[Line]:
    lines = []
    for g in groups:
        if not g:
            continue
        lines.append(Line(
            text=" ".join(w.text for w in g),
            start=round(g[0].start, 3),
            progress=0.0,
            words=list(g),
        ))
    return lines


def format_words(words: list[Word]) -> FormatResult:
    """Format a timed word stream into karaoke lines using the LLM.

    Three tiers per 80-word chunk:
      1. chunk-level LLM pass (adds punctuation + line breaks, verified)
      2. per-phrase LLM pass on timing-derived phrases (for chunks where the
         model recites famous lyrics instead of copying)
      3. raw phrases (heuristically split) — presentation unchanged, always safe
    """
    if not words:
        return FormatResult(lines=[], llm_lines=0, fallback_lines=0)

    # annotate each word with the start of the NEXT word in the stream, so
    # phrase-end heuristics can see silence gaps (last word -> None)
    for i, w in enumerate(words):
        w._next_start = words[i + 1].start if i + 1 < len(words) else None  # type: ignore[attr-defined]

    finished: list[Line] = []
    llm_count = 0
    fallback_count = 0

    for chunk in _chunks(words, CHUNK_WORDS):
        text = " ".join(w.text for w in chunk)
        llm_text = _chat(text)
        mapped = _verify_and_align(chunk, llm_text) if llm_text else None
        if mapped:
            finished.extend(_to_lines(_split_long_group(mapped)))
            llm_count += len(mapped)
            continue
        # Tier 2/3: timings own the line breaks; LLM only polishes each phrase.
        for g in _heuristic_lines(chunk):
            polished = _polish_phrase(g)
            if polished is not None:
                finished.append(polished)
                llm_count += 1
            else:
                finished.append(_to_lines([g])[0])
                fallback_count += 1

    return FormatResult(lines=finished,
                        llm_lines=llm_count, fallback_lines=fallback_count)


def _polish_phrase(group: list[Word]) -> Line | None:
    """LLM-format one timing-derived phrase; None = keep it raw.

    The returned line always carries the ORIGINAL Word objects (with their
    timings); only the display text changes.
    """
    if len(group) < 2:
        return None  # nothing to punctuate on a single word
    src = " ".join(w.text for w in group)
    out = _chat(src, phrase=True)
    if not out:
        return None
    out = out.strip().splitlines()[0].strip()
    # word-level faithfulness check: same tokens, same order, no adds/drops
    src_toks = [_norm(t) for t in src.split()]
    out_toks = [_norm(t) for t in out.split()]
    if [t for t in src_toks if t] != [t for t in out_toks if t]:
        return None
    # single-sentence phrases only: mid-phrase sentence breaks would need
    # re-splitting which we don't do here
    if out.count(". ") + out.count("? ") + out.count("! ") > 1:
        return None
    # fold the polish onto the words themselves (verification proved the
    # token sequences match 1:1) so the player, which renders word spans,
    # shows the same punctuation/caps as line.text and the exports do
    polished_tokens = [t for t in out.split() if _norm(t)]
    for w, tok in zip(group, polished_tokens):
        w.text = tok
    if not out.endswith((".", "?", "!")) and _ends_sentence(group):
        out = out.rstrip(",;:") + "."
    return Line(
        text=out,
        start=round(group[0].start, 3),
        progress=0.0,
        words=list(group),
    )


def _capitalize_first(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _ends_sentence(group: list[Word]) -> bool:
    """Heuristic: does this phrase end where a sentence likely ends?

    A big silence gap (or the track's end) right after the final word means
    the singer finished a thought — a good place for a period.
    """
    if not group:
        return False
    end = group[-1].end
    nxt = getattr(group[-1], "_next_start", None)
    return nxt is None or (nxt - end) > _SENTENCE_GAP


def _split_long_group(mapped: list[list[Word]]) -> list[list[Word]]:
    out: list[list[Word]] = []
    for line in mapped:
        out.extend(_split_long(line))
    return out
