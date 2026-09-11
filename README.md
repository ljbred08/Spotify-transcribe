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
| `TRANSCRIBE_BACKEND` | `auto` | `auto` / `parakeet` / `groq` / `openai` / `local` |
| `PARAKEET_MODEL` | `nemo-parakeet-tdt-0.6b-v3` | any onnx-asr-supported model name |
| `GROQ_API_KEY` | — | free-tier fast transcription with word timestamps |
| `OPENAI_API_KEY` | — | fallback cloud backend |
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
