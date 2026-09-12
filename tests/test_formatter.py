"""Formatter tests: re-emit-with-breaks protocol + lyric conventions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import formatter as F
from app.models import Word


def W(*pairs):
    return [Word(text=t, start=s, end=e) for t, s, e in pairs]


def test_polish_token_strips_terminal_periods_keeps_question():
    assert F._polish_token("love.", line_start=False) == "love"
    assert F._polish_token("love?!", line_start=False) == "love?!"
    assert F._polish_token("why?.", line_start=True) == "Why?"
    assert F._polish_token("well,", line_start=False) == "well,"
    assert F._polish_token("gone..", line_start=False) == "gone"


def test_polish_token_capitalization():
    assert F._polish_token("we're", line_start=True) == "We're"
    assert F._polish_token("YOU", line_start=False) == "YOU"   # acronym kept
    assert F._polish_token("Get", line_start=False) == "get"   # stray ASR cap
    assert F._polish_token("i", line_start=False) == "I"
    assert F._polish_token("i'm", line_start=False) == "I'm"
    assert F._polish_token("love", line_start=False) == "love"


def test_break_lines_accepts_exact_reemission(monkeypatch):
    chunk = W(("we're", 0.0, 0.3), ("no", 0.3, 0.5), ("strangers", 0.5, 0.9),
              ("to", 0.9, 1.0), ("love", 1.0, 1.4), ("you", 1.5, 1.7),
              ("know", 1.7, 2.0))
    out_text = "We're no strangers to love\nyou know"
    monkeypatch.setattr(F, "_chat", lambda prompt: out_text)
    lines = F._break_lines(chunk)
    assert lines is not None
    assert [[w.text for w in ln] for ln in lines] == [
        ["we're", "no", "strangers", "to", "love"], ["you", "know"]]
    # timings preserved: same Word objects
    assert lines[0][0].start == 0.0 and lines[1][-1].end == 2.0


def test_break_lines_rejects_word_changes(monkeypatch):
    chunk = W(("a", 0.0, 0.2), ("b", 0.2, 0.4), ("c", 0.4, 0.6))
    monkeypatch.setattr(F, "_chat", lambda prompt: "A b d\nc")       # word added
    assert F._break_lines(chunk) is None
    monkeypatch.setattr(F, "_chat", lambda prompt: "A c")            # word dropped
    assert F._break_lines(chunk) is None
    monkeypatch.setattr(F, "_chat", lambda prompt: "A x b c")        # word changed
    assert F._break_lines(chunk) is None
    monkeypatch.setattr(F, "_chat", lambda prompt: None)             # offline
    assert F._break_lines(chunk) is None


def test_break_lines_retries_then_falls_back(monkeypatch):
    words = [(f"w{i}", i * 0.1, i * 0.1 + 0.09) for i in range(12)]
    chunk = W(*words)
    calls = []

    def fake_chat(prompt):
        calls.append(prompt)
        # first prompt (sync framing) gets an undivided answer, second (typist)
        # a properly divided one
        one_line = " ".join(f"W{i}" for i in range(12))
        two_lines = " ".join(f"W{i}" for i in range(6)) + "\n" + " ".join(f"W{i}" for i in range(6, 12))
        return one_line if len(calls) == 1 else two_lines

    monkeypatch.setattr(F, "_chat", fake_chat)
    lines = F._break_lines(chunk)
    assert lines is not None and len(lines) == 2
    assert len(calls) == 2


def test_format_words_llm_path_polishes_display(monkeypatch):
    words = W(("we're", 0.0, 0.3), ("no", 0.3, 0.5), ("strangers", 0.5, 0.9),
              ("to", 0.9, 1.0), ("love.", 1.0, 1.4), ("you", 1.5, 1.7),
              ("know", 1.7, 2.0), ("THE", 2.0, 2.1), ("rules", 2.1, 2.5))
    monkeypatch.setattr(
        F, "_chat",
        lambda prompt: "We're no strangers to love.\nYou know THE rules")
    result = F.format_words(words)
    assert result.fallback_lines == 0
    assert [ln.text for ln in result.lines] == [
        "We're no strangers to love", "You know THE rules"]
    # word spans carry the same polish (player renders word spans)
    assert [w.text for w in result.lines[0].words] == [
        "We're", "no", "strangers", "to", "love"]
    assert result.lines[1].words[0].start == 1.5


def test_format_words_falls_back_without_llm(monkeypatch):
    words = W(("hello", 0.0, 0.3), ("world", 0.4, 0.7),
              ("again", 3.0, 3.3), ("friend", 3.4, 3.8))
    monkeypatch.setattr(F, "_chat", lambda prompt: None)
    result = F.format_words(words)
    assert len(result.lines) == 2
    assert result.llm_lines == 0
    assert result.fallback_lines == 2
    # conventions still applied by the deterministic layer
    assert result.lines[0].text == "Hello world"
    assert result.lines[1].words[-1].end == 3.8


def test_split_long_caps_line_length():
    words = W(*[(f"word{i:02d}", i * 0.1, i * 0.1 + 0.09) for i in range(20)])
    lines = F._split_long(words)
    assert all(sum(len(w.text) + 1 for w in ln) <= 58 for ln in lines)
    assert len(lines) >= 2


def test_heuristic_splits_on_gaps():
    chunk = W(("one", 0.0, 0.3), ("two", 0.4, 0.6),
              ("three", 3.0, 3.3), ("four", 3.4, 3.6))
    groups = F._heuristic_lines(chunk)
    assert len(groups) == 2


def test_discover_candidates_order_and_dedup():
    urls = F.discover_candidates(hint="192.168.1.50:11434",
                                 current="http://127.0.0.1:11434")
    assert urls[0] == "192.168.1.50:11434".join(["http://", ""]) or urls[0].startswith("http://192.168.1.50")
    assert len(urls) == len(set(urls))
    assert urls[-1].startswith("http://127.0.0.1")
