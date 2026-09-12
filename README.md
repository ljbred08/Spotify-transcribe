# Refrain 🎵

Paste a song link → get a karaoke-style synced transcript you can read along with,
then export as **LRC / SRT / VTT / TXT / JSON**.

```
Spotify URL ──▶ sfotipy metadata ──▶ ytmusicapi match ──▶ yt-dlp ─┐
YouTube URL ──────────────────────────────────────────▶ yt-dlp ─┤
                                                                 ▼
Parakeet v3 (ONNX, word timestamps) ◀── audio (m4a → 16 kHz WAV)
                                                                 │
                                  synced player UI ◀── line assembly
                                  (web, local)          + exporters
```

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (bash: source .venv/Scripts/activate)
pip install -e ".[dev]"

copy .env.example .env          # optional: keys + backend selection
uvicorn app.server:app --reload
```

Open http://127.0.0.1:8000 — paste a URL, hit **Transcribe →**.

> First local transcription downloads the Parakeet v3 ONNX model from Hugging
> Face (a one-time ~2.4 GB fetch, cached afterwards).

## How it works

1. **Download** — Spotify URLs are resolved via **sfotipy** (canonical track
   metadata from the Spotify page) then matched to a YouTube Music video with
   **ytmusicapi** (duration-aware fallback included); YouTube and other URLs go
   straight to yt-dlp. Audio is pulled as m4a via ffmpeg.
2. **Transcribe** — backend priority when `TRANSCRIBE_BACKEND=auto`:
   **Parakeet v3 → Groq → OpenAI → faster-whisper**. Parakeet runs locally through
   **onnx-asr** (~36× realtime on CPU, multilingual, word-level timestamps, no
   PyTorch needed). Very long audio falls back to VAD-chunked recognition.
3. **Align** — words are grouped into lines at silence gaps / length limits, so
   every line is karaoke-ready with per-word sweep.
4. **Read along** — the player highlights the active line, sweeps each word green,
   shows a click-to-seek waveform, and lets you click any line to jump there.

## Configuration (`.env`)

| Var | Default | Notes |
| --- | --- | --- |
| `TRANSCRIBE_BACKEND` | `auto` | initial backend; overridden by the app's persisted settings |
| `GROQ_API_KEY` | — | free-tier fast transcription with word timestamps |
| `OPENAI_API_KEY` | — | fallback cloud backend |

## Models & backend selection

The **⚙ Models & backend** panel in the app (bottom of the sidebar) manages everything:

- **Download / delete** local ONNX models (stored under `data/models/<id>/`, so deleting
  is instant and safe)
- **Pick the active model** — every English/multilingual onnx-asr model with word-timestamp
  support (the whisper exports return no timestamps, so they can't drive karaoke)
- **Pick the backend** — auto, local, Groq, or OpenAI

Choices persist in `data/settings.json` and apply to the next job (no restart).

| Model | Size | Good at |
| --- | --- | --- |
| `parakeet-v3` | ~2.5 GB | Best all-round: fast TDT decoder, handles singing, 25 languages |
| `parakeet-v2` | ~2.5 GB | English-only predecessor of v3 |
| `parakeet-unified-en` | ~2.5 GB | Unified offline/streaming RNNT with built-in punctuation & capitalization (community export) |
| `parakeet-ctc-0.6b` | ~2.5 GB | CTC decoder variant (English) |
| `parakeet-rnnt-0.6b` | ~2.5 GB | RNN-T decoder variant (English) |
| `canary-1b-v2` | ~4 GB | Largest Canary; spoken word — weak on singing |
| `canary-1b-flash` | ~3.6 GB | Spoken word (podcasts); weak on singing |
| `canary-180m-flash` | ~780 MB | Tiny + fast spoken word; weak on singing |
| `gigaam-multilingual-ctc` | ~900 MB | Compact multilingual CTC (70+ languages, best on CIS) |
| `gigaam-multilingual-large-ctc` | ~2.3 GB | Large multilingual CTC (70+ languages) |

Community exports without onnx-asr's standard layout (like `parakeet-unified-en`) are
normalized automatically after download: graph files renamed, a `config.json` synthesized,
and the vocab rebuilt from the authoritative token map (including the `<blk>` blank token
and SentencePiece word-boundary markers the export strips).

## Re-transcribing & versions

Every track has a **↻ Re-transcribe** button in the player: re-run ASR on the *existing*
audio with any downloaded model (pick it in the dropdown next to the button) — useful for
A/B-ing models or upgrading when a better one lands. No re-download of the song.

Each transcription run creates a **new version** (`v1`, `v2`, …) alongside the previous
ones — nothing is overwritten, so a worse re-run can simply be rolled back. Version pills
next to the button switch the active version; every export uses the active version.

## LLM lyric formatting

With [ollama](https://ollama.com) running, transcripts are auto-polished (punctuation,
capitalization, sentence-aware line breaks) by a small local model — default `qwen2.5:3b`
(`FORMATTER_MODEL` in `.env`, needs ~2 GB). Three safety tiers guarantee the words and
timings are never corrupted by the LLM:

1. chunk-level reformatting, verified word-by-word (order-preserving),
2. per-phrase polish on timing-derived phrases (used when the model "recites" famous
   songs from memory instead of copying — e.g. the rickroll),
3. raw phrases, unchanged.

Line breaks come from **word timings** (silence gaps), not from the LLM, so karaoke
sweep stays in sync regardless of which tier produced a line. Toggle in the Models panel
(`llm_format`); off = deterministic silence-gap splitting.

## Configuration (`.env`)

| Var | Default | Notes |
| --- | --- | --- |
| `TRANSCRIBE_BACKEND` | `auto` | `auto` / `parakeet` / `groq` / `openai` / `local` |
| `APP_DATA_DIR` | `data` | where audio, transcripts, models, settings live |
| `YTDLP_COOKIES_FILE` | — | e.g. `cookies.txt` if yt-dlp gets blocked |
| `YTDLP_EXTRA` | — | extra yt-dlp flags, comma-separated |

## API

| Method | Route | Purpose |
| --- | --- | --- |
| POST | `/api/jobs` `{url}` | queue a download+transcribe job |
| GET | `/api/jobs/{id}` | poll job status |
| GET | `/api/tracks` | library listing |
| GET | `/api/tracks/{id}` | track + transcript lines with word timings |
| GET | `/api/tracks/{id}/audio` | stream audio |
| GET | `/api/tracks/{id}/export?fmt=lrc\|srt\|vtt\|txt\|json` | download transcript |
| DELETE | `/api/tracks/{id}` | remove track and files |

## Notes on legality & reliability

- yt-dlp resolves *available* audio sources; if a specific track has no available
  match, the download fails with yt-dlp's reason. Nothing here circumvents DRM —
  Spotify's own Ogg streams are encrypted and never touched.
- For reliability, drop a `cookies.txt` (YouTube cookies exported via a browser
  extension) in the repo root and set `YTDLP_COOKIES_FILE=cookies.txt`.

## UI brainstorm (implemented ✅ / ideas 💡)

✅ Karaoke word-sweep highlighting, click-line-to-seek
✅ Click-to-seek bar-waveform with gradient progress
✅ Auto-scroll that yields to manual scrolling for 4s
✅ Playback speed control (0.75–1.5×)
✅ One-click exports: LRC / SRT / VTT / TXT / JSON
💡 **Album art hero** — pull cover art via yt-dlp thumbnail into the hero + library cards
💡 **Follow-mode toggle + font-size slider** for comfortable desk reading
💡 **Search-in-transcript** (⌘K palette) with jump-to-match
💡 **Two-pane compare view** — original transcription vs. synced playback for edits
💡 **Edit mode** — click a word to fix its text; re-export instantly
💡 **Spec-boost LRC** — write per-word `<mm:ss.xx>` enhanced LRC for players that support it
💡 **Shareable static page** — export a self-contained HTML file with audio + timings embedded
💡 **Batch mode** — paste a playlist URL, queue the whole thing with progress per track
💡 **Apple Music-style blur art background** driven by the album colors
💡 **Loop a line** — tap a line to loop it until you tap away (great for language learning)
