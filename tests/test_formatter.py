"""Formatter tests: LLM verification, interpolation, fallback safety."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.formatter import (_norm, _verify_and_align, _split_long,
                           _heuristic_lines, format_words)
from app.models import Word


def W(*pairs):
    return [Word(text=t, start=s, end=e) for t, s, e in pairs]


def test_norm_strips_punct():
    assert _norm("It's,") == "it's"
    assert _norm("“Hello.”") == "hello"
    assert _norm("...") == ""


def test_clean_llm_output_aligns():
    chunk = W(("we're", 0.0, 0.3), ("no", 0.3, 0.5), ("strangers", 0.5, 0.9),
              ("to", 0.9, 1.0), ("love", 1.0, 1.4), ("you", 1.5, 1.7),
              ("know", 1.7, 2.0), ("the", 2.0, 2.1), ("rules", 2.1, 2.5))
    llm = "We're no strangers to love\nYou know the rules"
    out = _verify_and_align(chunk, llm)
    assert out is not None
    assert [w.text for w in out[0]] == ["we're", "no", "strangers", "to", "love"]
    assert [w.text for w in out[1]] == ["you", "know", "the", "rules"]
    # timings preserved
    assert out[0][0].start == 0.0
    assert out[1][-1].end == 2.5


def test_punctuated_tokens_match():
    chunk = W(("love", 1.0, 1.4), ("you", 1.5, 1.7))
    llm = "Love.\nYou?"
    out = _verify_and_align(chunk, llm)
    assert out is not None
    assert out[0][0].start == 1.0 and out[1][0].end == 1.7


def test_inserted_word_gets_interpolated_timing():
    # one inserted word in a longer chunk stays under the insertion ratio
    chunk = W(("love", 1.0, 1.4), ("you", 1.5, 1.7), ("know", 1.8, 2.1),
              ("the", 2.2, 2.3), ("rules", 2.4, 2.8), ("and", 2.9, 3.0),
              ("so", 3.1, 3.2), ("do", 3.3, 3.4), ("i", 3.5, 3.6),
              ("too", 3.7, 3.9))
    llm = "Love, oh love\nYou know the rules and so do I too"
    out = _verify_and_align(chunk, llm)
    assert out is not None
    flat = [w for line in out for w in line]
    oh = next(w for w in flat if w.text == "oh")
    assert 1.0 <= oh.start <= oh.end <= 1.7
    assert getattr(oh, "inserted", False) is True
    # every real word survived with timing
    assert [w.text for w in chunk] == [w.text for w in chunk]  # sanity
    real = [w.text for l in out for w in l if not getattr(w, "inserted", False)]
    assert real == [w.text for w in chunk]


def test_word_dropping_llm_falls_back():
    """A dropped word (even when lookahead could jump over it) → reject."""
    chunk = W(("a", 0.0, 0.2), ("b", 0.2, 0.4), ("c", 0.4, 0.6), ("d", 0.6, 0.8))
    llm = "A b d"  # "c" dropped
    out = _verify_and_align(chunk, llm)
    assert out is None


def test_early_stopped_llm_falls_back():
    """LLM output that ends before the input does → reject."""
    chunk = W(("a", 0.0, 0.2), ("b", 0.2, 0.4), ("c", 0.4, 0.6), ("d", 0.6, 0.8))
    llm = "A b"  # stopped early
    out = _verify_and_align(chunk, llm)
    assert out is None


def test_split_long_caps_line_length():
    words = W(*[(f"word{i:02d}", i * 0.1, i * 0.1 + 0.09) for i in range(20)])
    lines = _split_long(words)
    assert all(sum(len(w.text) + 1 for w in ln) <= 58 for ln in lines)
    assert len(lines) >= 2


def test_heuristic_splits_on_gaps():
    chunk = W(("one", 0.0, 0.3), ("two", 0.4, 0.6),
              ("three", 3.0, 3.3), ("four", 3.4, 3.6))
    groups = _heuristic_lines(chunk)
    assert len(groups) == 2


def test_format_words_falls_back_without_llm(monkeypatch):
    """With the LLM unavailable, format_words still returns valid timed lines."""
    words = W(("hello", 0.0, 0.3), ("world", 0.4, 0.7),
              ("again", 3.0, 3.3), ("friend", 3.4, 3.8))
    import app.formatter as f
    monkeypatch.setattr(f, "_chat", lambda text, *, phrase=False: None)
    result = f.format_words(words)
    assert len(result.lines) == 2
    assert result.llm_lines == 0
    assert result.fallback_lines == 2
    assert result.lines[0].words[0].start == 0.0
    assert result.lines[1].words[-1].end == 3.8
