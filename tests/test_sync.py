"""Aligner scenario tests: the user's exact examples + hard cases."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.models import Line, Word
from app.sync import SyncError, _align, _norm, improve_sync


def line(text: str, start: float, words: list[tuple[str, float, float]]) -> Line:
    return Line(text=text, start=start, progress=0.0,
                words=[Word(text=t, start=s, end=e) for t, s, e in words])


def ref(*triples: tuple[str, float, float]) -> list[dict]:
    return [{"text": t, "start": s, "end": e} for t, s, e in triples]


def word_texts(lines: list[Line]) -> list[str]:
    return [w.text for ln in lines for w in ln.words]


def starts(lines: list[Line]) -> list[float]:
    return [w.start for ln in lines for w in ln.words]


def test_norm_strips_punct_keeps_apostrophe():
    assert _norm("We're,") == "we're"
    assert _norm("(Go!)" ) == "go"


def test_missing_word_keeps_old_timing():
    """ref misses "no" and drift is small, so its old timestamp still fits
    between the retimed neighbours — it stays untouched."""
    good = [line("We're no strangers to love", 10.0,
                 [("We're", 10.0, 10.5), ("no", 10.5, 10.8),
                  ("strangers", 10.8, 11.6), ("to", 11.6, 11.8),
                  ("love", 11.8, 12.4)])]
    new, st = improve_sync(good, ref(
        ("We", 9.9, 10.4), ("strangers", 10.9, 11.7), ("to", 11.7, 11.9),
        ("love", 11.9, 12.5)))
    assert word_texts(new) == ["We're", "no", "strangers", "to", "love"]
    ws = new[0].words
    assert (ws[0].start, ws[0].end) == (9.9, 10.4)       # retimed
    assert (ws[1].start, ws[1].end) == (10.5, 10.8)      # kept (still fits)
    assert ws[2].start == 10.9                            # retimed
    assert st["kept"] == 1 and st["matched"] == 4


def test_match_percent_reports_reference_agreement():
    """stats carry a 0-100 match accuracy = matched / old words, so the UI
    can show how closely the reference transcription agreed ("We" pairs with
    "We're" at sim 0.5 -> counts as matched)."""
    good = [line("We're no strangers to love", 10.0,
                 [("We're", 10.0, 10.5), ("no", 10.5, 10.8),
                  ("strangers", 10.8, 11.6), ("to", 11.6, 11.8),
                  ("love", 11.8, 12.4)])]
    _, st = improve_sync(good, ref(
        ("We", 9.9, 10.4), ("strangers", 10.9, 11.7), ("to", 11.7, 11.9),
        ("love", 11.9, 12.5)))
    assert st["matched"] == 4 and st["forced"] == 0
    assert st["coverage"] == round(4 / 5, 3)
    assert round(st["coverage"] * 100) == 80
    # forced (misheard) pairs do NOT count toward the match percentage
    _, st2 = improve_sync(
        [line("and so do I", 0.0, [("and", 0.0, 0.3), ("so", 0.3, 0.6),
                                    ("do", 0.6, 0.8), ("I", 0.8, 1.0)])],
        ref(("and", 0.0, 0.3), ("but", 0.3, 0.6), ("do", 0.6, 0.8),
            ("I", 0.8, 1.0)))
    assert st2["forced"] == 1 and st2["matched"] == 3
    assert round(st2["coverage"] * 100) == 75


def test_stale_missed_word_is_interpolated():
    """ref misses a word whose old timestamp no longer fits the retimed
    neighbourhood — it is interpolated between the anchors instead."""
    good = [line("one two three", 0.0,
                 [("one", 0.0, 0.4), ("two", 0.4, 0.8), ("three", 0.8, 1.2)])]
    new, st = improve_sync(good, ref(
        ("one", 50.0, 50.4), ("three", 52.0, 52.4)))
    ws = new[0].words
    assert (ws[0].start, ws[0].end) == (50.0, 50.4)
    assert (ws[2].start, ws[2].end) == (52.0, 52.4)
    # "two" now sits between its retimed neighbours, not at its stale 0.4
    assert 50.4 < ws[1].start < ws[1].end < 52.0
    assert st["kept"] == 1


def test_misheard_word_forced_pair():
    """"and" vs "but": dissimilar words pair via whole-transcript context."""
    good = [
        line("You know the rules and so do I", 20.0,
             [("You", 20.0, 20.2), ("know", 20.2, 20.5), ("the", 20.5, 20.6),
              ("rules", 20.6, 21.2), ("and", 21.2, 21.4), ("so", 21.4, 21.6),
              ("do", 21.6, 21.8), ("I", 21.8, 22.1)]),
    ]
    new, st = improve_sync(good, ref(
        ("You", 40.0, 40.2), ("know", 40.2, 40.5), ("the", 40.5, 40.6),
        ("rules", 40.6, 41.2), ("but", 41.2, 41.4), ("so", 41.4, 41.6),
        ("do", 41.6, 41.8), ("I", 41.8, 42.1)))
    assert word_texts(new)[4] == "and"
    assert (new[0].words[4].start, new[0].words[4].end) == (41.2, 41.4)
    assert st["forced"] == 1


def test_invented_word_dropped():
    """ref invents "too": it must be dropped, timings shift correctly."""
    good = [line("You know the rules and so do I", 0.0,
                 [("You", 0.0, 0.2), ("know", 0.2, 0.5), ("the", 0.5, 0.6),
                  ("rules", 0.6, 1.2), ("and", 1.2, 1.4), ("so", 1.4, 1.6),
                  ("do", 1.6, 1.8), ("I", 1.8, 2.1)])]
    new, st = improve_sync(good, ref(
        ("You", 5.0, 5.2), ("know", 5.2, 5.5), ("the", 5.5, 5.6),
        ("rules", 5.6, 6.2), ("and", 6.2, 6.4), ("so", 6.4, 6.6),
        ("too", 6.6, 6.7), ("do", 6.7, 6.9), ("I", 6.9, 7.2)))
    assert word_texts(new) == ["You", "know", "the", "rules", "and", "so",
                               "do", "I"]
    assert st["dropped"] == 1
    assert (new[0].words[6].start, new[0].words[6].end) == (6.7, 6.9)


def test_chorus_repeats_map_in_order():
    """9 chorus occurrences align occurrence-to-occurrence, never crossing."""
    good = []
    for i in range(3):
        base = i * 50.0
        good.append(line("boots in it", base,
                         [("boots", base, base + 0.5),
                          ("in", base + 0.5, base + 0.7),
                          ("it", base + 0.7, base + 1.0)]))
    rw = []
    for i in range(3):
        base = 100.0 + i * 47.0
        rw += [("boots", base, base + 0.5), ("in", base + 0.5, base + 0.7),
               ("it", base + 0.7, base + 1.0)]
    new, st = improve_sync(good, ref(*rw))
    for i, ln in enumerate(new):
        assert ln.words[0].start == pytest.approx(100.0 + i * 47.0)
    assert st["matched"] == 9


def test_line_starts_follow_first_word():
    good = [
        line("First line here", 1.0, [("First", 1.0, 1.4), ("line", 1.4, 1.7),
                                      ("here", 1.7, 2.0)]),
        line("Second line", 3.0, [("Second", 3.0, 3.5), ("line", 3.5, 3.8)]),
    ]
    new, _ = improve_sync(good, ref(
        ("First", 11.0, 11.4), ("line", 11.4, 11.7), ("here", 11.7, 12.0),
        ("Second", 13.0, 13.5), ("line", 13.5, 13.8)))
    assert new[0].start == 11.0 and new[1].start == 13.0


def test_words_stay_monotonic_within_line():
    good = [line("a b c d", 0.0, [("a", 0.0, 0.3), ("b", 0.3, 0.6),
                                  ("c", 0.6, 0.9), ("d", 0.9, 1.2)])]
    # reference gives "b" a window that starts after "c"'s — clamp keeps order
    new, _ = improve_sync(good, ref(
        ("a", 10.0, 10.3), ("b", 11.0, 11.4), ("c", 10.8, 10.95),
        ("d", 11.2, 11.5)))
    s = starts(new)
    assert s == sorted(s)


def test_coverage_gate_rejects_garbage():
    good = [line("totally different words entirely", 0.0,
                 [("totally", 0.0, 0.4), ("different", 0.4, 0.9),
                  ("words", 0.9, 1.2), ("entirely", 1.2, 1.8)])]
    with pytest.raises(SyncError):
        improve_sync(good, ref(("song", 5.0, 5.5), ("nothing", 6.0, 6.5),
                               ("alike", 7.0, 7.4)))


def test_empty_inputs_raise():
    with pytest.raises(SyncError):
        improve_sync([], ref(("x", 0.0, 0.5)))
    with pytest.raises(SyncError):
        improve_sync([line("hi", 0.0, [("hi", 0.0, 0.5)])], [])


def test_align_ops_order_and_shapes():
    ops = _align(["we're", "no", "strangers"], ["we", "strangers"])
    kinds = [o[0] for o in ops]
    assert kinds == ["pair", "keep", "pair"]
    assert ops[0][1] == 0 and ops[0][2] == 0      # we're <- we
    assert ops[1][1] == 1 and ops[1][2] == -1     # no kept
    assert ops[2][1] == 2 and ops[2][2] == 1      # strangers <- strangers


def test_insertion_between_matches():
    """ref inserts a word between two good matches — the insertion is dropped."""
    ops = _align(["one", "two"], ["one", "extra", "two"])
    kinds = [o[0] for o in ops]
    assert kinds == ["pair", "drop", "pair"]
