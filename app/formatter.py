"""LLM-powered lyric formatting: line breaking + capitalization cleanup.

Design ("re-emit with breaks"): the ASR returns a continuous word stream with
timings. The LLM's ONLY job is to re-emit the exact same words with line breaks
inserted so each line reads like a printed lyric booklet (a roughly
grammatically complete thought). It never writes, removes, or "corrects" words:
the output is verified token-exact against the input and otherwise rejected and
retried, then falls back to timing-based line splitting. This framing keeps
small models from reciting famous lyrics from memory (verified: the task is
copying, and the sync rationale makes "helpful" correction counterproductive).

Lyric conventions are applied deterministically afterwards (never by the LLM):
- no terminal periods (standard for lyrics); ? and ! and commas are kept
- every line starts capitalized; "I" and I'm/I'll/... stay capitalized
- stray mid-line ASR capitalization is lower-cased (acronyms kept)

Timing integrity is untouched throughout: only display text changes, never the
Word objects' start/end.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

import httpx

from app.models import Line, Word

# URL + model live in persisted settings (provider/model editable in the
# models panel); API keys come from env or UI-saved settings.
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_FORMATTER_MODEL = "qwen2.5:3b"
CHUNK_WORDS = 80       # words per LLM request (short outputs stay reliable)
MAX_LINE_CHARS = 52    # karaoke line length cap
_SENTENCE_GAP = 0.9    # silence gap after a phrase that suggests a sentence end

_BREAK_PROMPT = """This text is a speech-recognition transcript used by a synchronization system. Reproduce it EXACTLY, word for word in the same order - even words that look wrong, misspelled, or misheard MUST be kept exactly as written, because changing any word breaks the sync. Your ONLY task is inserting line breaks so each line is a roughly grammatically complete thought. Do NOT correct, complete, or improve the text.

Transcript:
{chunk}

Output only the reproduced transcript with line breaks, word-identical to the input."""


@dataclass
class FormatResult:
    lines: list[Line]
    llm_lines: int      # lines whose breaks came from the LLM
    fallback_lines: int  # lines from the timing-based fallback


def _norm(tok: str) -> str:
    return re.sub(r"[^a-z0-9']", "", tok.lower())


def _clean_token(text: str) -> str:
    """Strip punctuation/casing noise from an ASR word, keeping apostrophes."""
    base = re.sub(r"[^\w'’]", "", text, flags=re.UNICODE)
    return base or text


def _llm_config() -> "tuple[str, str, str]":
    """(provider, url_or_base, model) from persisted settings.

    provider is ollama | gemini | openrouter | groq | openai; url_or_base is
    the ollama URL for ollama, else the provider's OpenAI-compatible base.
    """
    try:
        from app.settings import settings
        s = settings.get()
        provider = s.get("llm_provider") or "ollama"
        if provider == "ollama":
            return provider, s.get("llm_url") or DEFAULT_OLLAMA_URL, \
                s.get("llm_model_ollama") or DEFAULT_FORMATTER_MODEL
        model = s.get("llm_model") or ""
        if not model:
            from app.cloud_models import formatter_default
            model = formatter_default(provider)
        if provider == "gemini":
            return provider, "https://generativelanguage.googleapis.com/v1beta/openai", model
        if provider == "openrouter":
            return provider, "https://openrouter.ai/api/v1", model
        if provider == "groq":
            return provider, "https://api.groq.com/openai/v1", model
        if provider == "openai":
            return provider, "https://api.openai.com/v1", model
        return "ollama", DEFAULT_OLLAMA_URL, DEFAULT_FORMATTER_MODEL
    except Exception:
        return "ollama", DEFAULT_OLLAMA_URL, DEFAULT_FORMATTER_MODEL


_status_cache: dict = {"at": 0.0, "data": None}
_STATUS_TTL = 5.0  # seconds; ollama polling happens on every settings fetch


def llm_status() -> dict:
    """Reachability + the active config, for the UI's formatter block."""
    provider, url, model = _llm_config()
    models: list[str] = []
    online = False
    if provider == "ollama":
        try:
            with httpx.Client(timeout=2.5) as c:
                tags = c.get(f"{url}/api/tags").json()
            models = sorted(m.get("name", "") for m in tags.get("models", []))
            online = True
        except Exception:
            online = False
        # exact or tag-prefix match ("qwen2.5:3b" also matches "qwen2.5:3b-instruct")
        installed = model in models or any(m.startswith(model) for m in models)
    else:
        from app.cloud_models import FORMATTER_MODELS, provider_key
        online = bool(provider_key(provider))
        installed = online
        models = [name for _api, name in FORMATTER_MODELS.get(provider, [])]
    return {"provider": provider, "url": url, "model": model, "online": online,
            "installed": installed, "models": models}


def llm_status_cached() -> dict:
    now = time.time()
    if _status_cache["data"] is None or now - _status_cache["at"] > _STATUS_TTL:
        _status_cache["data"] = llm_status()
        _status_cache["at"] = now
    return _status_cache["data"]


def invalidate_status_cache() -> None:
    _status_cache["at"] = 0.0


def probe_ollama(url: str, timeout: float = 0.8) -> bool:
    """Cheap reachability check for one base URL."""
    try:
        with httpx.Client(timeout=timeout) as c:
            tags = c.get(f"{url.rstrip('/')}/api/tags").json()
        return isinstance(tags, dict) and "models" in tags
    except Exception:
        return False


def discover_candidates(hint: str | None = None,
                        current: str | None = None) -> list[str]:
    """URLs to try when auto-discovering an ollama server, best first."""
    urls: list[str] = []

    def add(u: str) -> None:
        u = u.strip().rstrip("/")
        if u and u not in urls:
            urls.append(u)

    from urllib.parse import urlsplit
    if hint:
        raw = hint.strip()
        if raw and "://" not in raw:
            raw = f"http://{raw}"
        add(raw)
        sp = urlsplit(raw)
        host = sp.hostname or "127.0.0.1"
        for port in (11434, 11435):
            add(f"http://{host}:{port}")
    if current:
        add(current)
    for u in ("http://127.0.0.1:11434", "http://localhost:11434",
              "http://127.0.0.1:11435"):
        add(u)
    return urls


def discover_server(hint: str | None = None,
                    current: str | None = None) -> str | None:
    """Return the first reachable ollama URL, or None."""
    for url in discover_candidates(hint=hint, current=current):
        if probe_ollama(url):
            return url.rstrip("/")
    return None


def llm_available() -> bool:
    """True when the formatter model is reachable and pulled."""
    st = llm_status_cached()
    return st["online"] and st["installed"]


def _chat(prompt: str) -> str | None:
    """One chat completion via the configured provider (ollama or cloud)."""
    provider, url, model = _llm_config()
    try:
        with httpx.Client(timeout=240.0) as c:
            if provider == "ollama":
                r = c.post(f"{url}/api/chat", json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "keep_alive": "30m",
                    "options": {"temperature": 0.0, "num_ctx": 4096, "num_predict": 500},
                })
                return r.json()["message"]["content"]
            # OpenAI-compatible cloud providers
            from app.cloud_models import provider_key
            key = provider_key(provider)
            if not key:
                return None
            r = c.post(f"{url}/chat/completions", headers={
                "Authorization": f"Bearer {key}",
            }, json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            })
            if r.status_code != 200:
                return None
            return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return None


def _chunks(words: list[Word], size: int) -> list[list[Word]]:
    return [words[i:i + size] for i in range(0, len(words), size)]


_DANGLING = {"the", "and", "of", "to", "a", "an", "my", "your", "our",
             "for", "with", "on", "in", "but", "or", "so", "if", "as",
             "that", "it", "is", "are", "was", "were", "gonna", "wanna"}


def _break_quality(lines: list[list[Word]]) -> bool:
    """Heuristic quality bar for a proposed line mapping.

    Rejects answers that are technically word-exact but lazy: most lines
    ending on dangling connectives, or average line length way over booklet
    scale. Used to decide whether to try the second prompt.
    """
    if not lines:
        return False
    ends = [_norm(ln[-1].text) for ln in lines]
    dangle = sum(1 for e in ends if e in _DANGLING)
    avg = sum(len(ln) for ln in lines) / len(lines)
    return dangle <= max(1, len(lines) // 3) and avg <= 9.5


def _break_lines(chunk: list[Word]) -> list[list[Word]] | None:
    """Ask the LLM to re-emit the chunk with line breaks; verify token-exact.

    Returns lists of the ORIGINAL Word objects grouped into lines, or None when
    every attempt fails verification (then the caller uses timing-based
    splitting). Two attempts: the sync-framing prompt, then a plainer variant.
    A word-exact but lazy mapping (see _break_quality) triggers the retry too;
    if both attempts are word-exact, the better-quality one wins.
    """
    text = " ".join(w.text for w in chunk)
    src = [t for t in (_norm(x) for x in text.split()) if t]
    prompts = (
        _BREAK_PROMPT.format(chunk=text),
        "Copy the following text exactly, word for word, adding line breaks "
        "where a reader would pause for a new sentence or clause. You are a "
        "typist, not an editor: do not fix anything, even obvious errors. The "
        f"output must be word-identical to the input.\n\nText:\n{text}\n\n"
        "Output only the text with your inserted line breaks.",
    )
    best: list[list[Word]] | None = None
    for prompt in prompts:
        raw = _chat(prompt)
        if not raw:
            continue
        out = [t for t in (_norm(x) for x in raw.split()) if t]
        if out != src:
            continue  # model changed/dropped/added words -> unusable
        lines: list[list[Word]] = []
        i = 0
        for ln in raw.splitlines():
            toks = [t for t in ln.split() if _norm(t)]
            if not toks:
                continue
            if i + len(toks) > len(chunk):
                break
            lines.append(chunk[i:i + len(toks)])
            i += len(toks)
        if i != len(chunk) or not lines:
            continue
        # a single undivided line for a long chunk is token-exact but
        # useless (the model skipped the actual task) -> keep trying
        if len(lines) == 1 and len(chunk) >= 12:
            continue
        if _break_quality(lines):
            return lines
        if best is None:
            best = lines
    return best


def _polish_token(text: str, *, line_start: bool) -> str:
    """Apply lyric conventions to one word's display text.

    - drop terminal periods (lyrics don't use them); keep ? ! , etc.
    - capitalize line starts, "I", and I'm/I'll/...; lower-case stray
      mid-line capitals (all-caps words are treated as acronyms and kept)
    """
    base = text
    if base.endswith("?.") or base.endswith("!."):
        base = base[:-1]
    while base.endswith(".") and not base.endswith("..."):
        base = base[:-1]
    core = _clean_token(base)
    if not core:
        return text  # punctuation-only token: leave it alone
    # keep the trailing punctuation that survived the period strip
    tail = base[len(base.rstrip("?!.,;:!?…")):] if base != core else ""
    low = core.lower()
    if line_start or low == "i" or low.startswith(("i'", "i’")):
        core = core[0].upper() + core[1:]
    elif not core.isupper():
        core = core[0].lower() + core[1:]
    return core + tail


def _decorate(groups: list[list[Word]]) -> list[list[Word]]:
    """Apply _polish_token to every word, in place, line-aware."""
    for g in groups:
        for j, w in enumerate(g):
            w.text = _polish_token(w.text, line_start=(j == 0))
    return groups


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


def _split_long_group(mapped: list[list[Word]]) -> list[list[Word]]:
    out: list[list[Word]] = []
    for line in mapped:
        out.extend(_split_long(line))
    return out


def _heuristic_lines(chunk: list[Word]) -> list[list[Word]]:
    """Timing-based line splitting: break after long silences or long lines."""
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
    """Format a timed word stream into karaoke lines.

    Per 80-word chunk: the LLM re-emits the exact words with line breaks
    (verified token-exact, retried once, then timing-based fallback); lyric
    conventions (no terminal periods, corrected caps) are applied
    deterministically to every line either way. The LLM can therefore never
    add, drop, or rewrite a word — it only proposes where lines break.
    """
    if not words:
        return FormatResult(lines=[], llm_lines=0, fallback_lines=0)

    finished: list[Line] = []
    llm_count = 0
    fallback_count = 0

    for chunk in _chunks(words, CHUNK_WORDS):
        mapped = _break_lines(chunk)
        if mapped:
            groups = _split_long_group(mapped)
            llm_count += len(groups)
        else:
            groups = _split_long_group(_heuristic_lines(chunk))
            fallback_count += len(groups)
        finished.extend(_to_lines(_decorate(groups)))

    return FormatResult(lines=finished,
                        llm_lines=llm_count, fallback_lines=fallback_count)
