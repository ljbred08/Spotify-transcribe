# Refrain 🎵

Paste a song link → get a karaoke-style synced transcript you can read along with,
then export as **LRC / SRT / VTT / TXT / JSON**.

```
Spotify URL ──▶ spotdl metadata ──▶ spotdl match ──────▶ yt-dlp ─┐
YouTube URL ────────────────────────────────────────────▶ yt-dlp ─┤
                                                                 ▼
              cloud transcription ◀── audio (m4a)
   audio-LLM (one step, formatted + synced)  or  transcriber + LLM formatter
                                                                 │
                                  synced player UI ◀── line assembly
                                  (web, local)          + exporters
```

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (bash: source .venv/Scripts/activate)
pip install -e ".[dev]"

copy .env.example .env          # put your API key(s) here
uvicorn app.server:app --reload
```

Open http://127.0.0.1:8000 — paste a URL, hit **Transcribe →**.

> Set at least one of `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, `GROQ_API_KEY`
> or `OPENAI_API_KEY` (or paste keys in the app — they're saved to settings).
> Audio LLMs listen and return formatted, word-synced lyrics in one step.

## How it works

1. **Download** — Spotify URLs are resolved via **spotdl** (canonical track
   metadata from Spotify, using spotdl's public client — override with
   `SPOTIPY_CLIENT_ID`/`SPOTIPY_CLIENT_SECRET` if you like) then matched to a
   YouTube Music video by spotdl's battle-tested matcher (title/artist/album/
   duration scoring, with a plain-YouTube fallback); YouTube and other URLs go
   straight to yt-dlp. Audio is pulled as m4a via ffmpeg.
2. **Transcribe** — two cloud paths, chosen per model in the Models panel:
   - **Audio LLMs** (Gemini via Google/OpenRouter, GPT Audio, Voxtral, Muse Spark,
     Inkling, Nemotron Omni, MiMo — see the in-app list): one call sends the song
     (plus title/artist) and asks for **Enhanced LRC** with word-level timestamps —
     the model does transcription, line breaking, capitalization and lyric
     conventions itself, in the format it already knows. If the reply isn't
     parseable LRC, it retries once with a strict JSON lyric-sheet prompt.
   - **Transcribers** (Whisper large v3 / turbo via Groq, Whisper or GPT-4o-transcribe
     via OpenAI): return words with timestamps; a separate LLM — Gemini, OpenRouter,
     Groq, OpenAI or local ollama — formats them into lines, verified token-exact.
3. **Align** — words are grouped into lines at silence gaps / length limits, so
   every line is karaoke-ready with per-word sweep.
4. **Read along** — the player highlights the active line, sweeps each word green,
   shows a click-to-seek waveform, and lets you click any line to jump there.

## Configuration (`.env`)

| Var | Default | Notes |
| --- | --- | --- |
| `GEMINI_API_KEY` | — | enables Gemini audio models + Gemini formatter (one-step transcription + formatting) |
| `OPENROUTER_API_KEY` | — | enables the OpenRouter omni list (Gemini aliases, GPT Audio, Voxtral, …) + formatter |
| `GROQ_API_KEY` | — | enables Whisper large v3/turbo transcription + Groq formatters |
| `OPENAI_API_KEY` | — | enables Whisper / GPT-4o transcription + OpenAI audio/formatter models |
| `GENIUS_ACCESS_TOKEN` | — | optional *last* metadata method: search the Genius **lyric** index with a line from your own transcript when Shazam and Deezer can't identify a track (Deezer and the LRCLIB lyric check need no key) |
| `APP_DATA_DIR` | `data` | where audio, transcripts, settings live |
| `YTDLP_COOKIES_FILE` | — | e.g. `cookies.txt` if yt-dlp gets blocked |
| `YTDLP_EXTRA` | — | extra yt-dlp flags, comma-separated |

## Models & backend selection

The **⚙ Transcription models** panel in the app (bottom of the sidebar) manages everything.
The catalog is **dynamic**: models whose API key is present show as available; picking one
is instant (nothing to download). Choices persist in `data/settings.json` and apply to
the next job (no restart). The model dropdown is grouped: **Audio LLMs** (listen +
format in one step) vs **Transcribers** (measured word timings + a formatter LLM).
Only models that can return word-level timestamps are listed — plain-text STT endpoints
can't drive the karaoke word sweep and are deliberately excluded.

| Model | Kind | Good at |
| --- | --- | --- |
| `gemini:gemini-flash-latest` | audio LLM | one-step: newest Gemini Flash; listens, returns formatted synced lyrics |
| `gemini:gemini-pro-latest` | audio LLM | one-step: newest Gemini Pro; highest quality, slower |
| `gemini:gemini-flash-lite-latest` | audio LLM | one-step: lightweight/fast Gemini Flash Lite |
| `gemini:gemini-3.5-transcribe` | transcriber | Google's dedicated ASR; measured word timings via the Interactions API |
| `openrouter:~google/gemini-flash-latest` | audio LLM | newest Gemini Flash via OpenRouter |
| `openrouter:~google/gemini-pro-latest` | audio LLM | newest Gemini Pro via OpenRouter |
| `openrouter:google/gemini-3.5-flash-lite` | audio LLM | lightweight Gemini 3.5 Flash via OpenRouter (no `latest` alias there) |
| `openrouter:openai/gpt-audio` · `…-mini` | audio LLM | OpenAI native audio engines |
| `openrouter:mistralai/voxtral-small-24b-2507` | audio LLM | Mistral audio model; strong speech transcription |
| `openrouter:meta/muse-spark-1.3` (+ contributor) | audio LLM | Meta multimodal audio (contributor = discounted, trains on prompts) |
| `openrouter:thinkingmachines/inkling` (+ small) | audio LLM | open-weight MoE models with native audio understanding |
| `openrouter:nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` · `xiaomi/mimo-v2.5` | audio LLM | lightweight omni models |
| `openrouter:deepgram/nova-3`, `x-ai/grok-stt-1.0`, `mistralai/voxtral-small-24b-2507-stt`, `mistralai/voxtral-mini-transcribe`, `microsoft/mai-transcribe-2`, `qwen/qwen3-asr-flash…`, `qwen/qwen3-asr-0.6b`, `fish-audio/transcribe-1`, `openai/whisper-large-v3`(+`-turbo`) | transcriber | OpenRouter STT models with word-level timestamps via `/audio/transcriptions` (need formatter block) |
| `groq:whisper-large-v3` / `…-turbo` | transcriber | fast word-level timestamps (needs formatter block) |
| `openai:whisper-1` / `gpt-4o-transcribe` / `…-mini-transcribe` | transcriber | word/segment timestamps (needs formatter block) |

When a **transcriber** is selected, the panel also shows the **LLM lyric formatting**
section (the formatter stage); select an **audio LLM** and it hides — the model already
formats. Formatters: local ollama (free), Groq or OpenAI chat models.

## Re-transcribing & versions

Every track has a **↻ Re-transcribe** button in the player: re-run transcription on the
*existing* audio with any available cloud model (pick it in the dropdown next to the
button) — useful for A/B-ing models. No re-download of the song.

Each transcription run creates a **new version** (`v1`, `v2`, …) alongside the previous
ones — nothing is overwritten, so a worse re-run can simply be rolled back. Version pills
next to the button switch the active version; every export uses the active version.

## LLM lyric formatting

Used on the **transcriber** path (audio LLMs return formatted lines themselves). The
formatter provider is selectable: local [ollama](https://ollama.com) (default
`qwen2.5:3b`, needs ~2 GB), Gemini, OpenRouter, Groq (`llama-3.3-70b-versatile`) or
OpenAI (`gpt-4o-mini`).
The LLM's only job is to re-emit the exact transcript with line breaks inserted so each
line reads like a printed lyric booklet; the result is verified **token-exact** against
the input (retrying with a plainer prompt when the model touches words), then lyric
conventions are applied deterministically: no terminal periods, kept `?`/`!`/commas,
corrected capitalization. The LLM can therefore never add, drop, or rewrite a word —
timing integrity is guaranteed by construction. Toggle in the Models panel (`llm_format`).

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
✅ **Album art hero** — cover art in the hero + library cards, fetched in order: Shazam
   fingerprints the local file, then a keyless **Deezer** search (1000×1000 covers),
   then Shazam's text search, then Genius **searched by a line of your own lyrics**
   (needs a token; runs after transcription, since that is when the lyrics exist).
   Covers are cached in `data/art/`, and a hit is only accepted when it plausibly
   *is* the track (title match, plus the artist when we know it) — a wrong cover is
   worse than none. Version tails like `- FULL MIX` are stripped from the *query*
   before searching, and when a title alone can't identify a song (five recordings
   are called "Mama Says") the shortlist appears in the status card instead of a
   guess. The picker filters in two levels: the main list is the **strict tier**
   (hits whose artist agrees with ours, so the title+artist search would have
   found them) minus the entries your transcript contradicts — checked against
   LRCLIB's lyrics, which is the *only* filter when we have no artist — and
   everything else, broad hits and lyrics-vetoed ones alike, sits under *More
   results*. A confirmed entry is badged "lyrics match". Any cover that has no art yet shows a hover
   **download** button to run the lookup (handy for tracks added before this
   existed, and the way in for a custom mix no fingerprint can name), and
   **right-clicking any cover** — hero or library card — offers *Redo metadata
   search*, which always opens the picker so even a wrong pick can be replaced.
   That picker lists what the search found **and** the pipeline's own answer,
   badged by how it was found (*fingerprint match* / *title match* / *lyrics
   search*), so a Shazam-only identification is just as selectable — and
   replaceable — as a Deezer one. Applying a pick brings its **artist** across
   too, even when the track already had one: picking a different recording is
   the user saying the artist we hold is wrong. The title is never touched — a
   metadata fetch does not rename anything. A cover never pops in: it is decoded
   first and then **cross-fades** in over the placeholder, or over the cover it
   replaces (art URLs are versioned by the file's mtime, so the outgoing and
   incoming bitmaps are addressable at the same time).
💡 **Follow-mode toggle + font-size slider** for comfortable desk reading
💡 **Search-in-transcript** (⌘K palette) with jump-to-match
💡 **Two-pane compare view** — original transcription vs. synced playback for edits
💡 **Edit mode** — click a word to fix its text; re-export instantly
💡 **Spec-boost LRC** — write per-word `<mm:ss.xx>` enhanced LRC for players that support it
💡 **Shareable static page** — export a self-contained HTML file with audio + timings embedded
💡 **Batch mode** — paste a playlist URL, queue the whole thing with progress per track
💡 **Apple Music-style blur art background** driven by the album colors
💡 **Loop a line** — tap a line to loop it until you tap away (great for language learning)
