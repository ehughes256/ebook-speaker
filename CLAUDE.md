# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in OPENAI_API_KEY, ELEVENLABS_API_KEY, and DB_* values
python manage.py migrate
python manage.py runserver
```

The app uses **PostgreSQL** (configured in `config/settings.py`, read from `DB_*` env
vars). Create a matching database/role before `migrate`, e.g.:

```bash
createdb reader   # then set DB_NAME / DB_USER / DB_PASSWORD / DB_HOST / DB_PORT in .env
```

`ffmpeg` must be on `PATH` (used for audiobook compilation and ElevenLabs WAV conversion).

GPU-only dependencies (not in requirements.txt — install separately on the target machine):
```bash
pip install qwen-tts torch chatterbox-tts
pip install flash-attn --no-build-isolation  # optional, improves Qwen TTS speed
```

Voice **samples** can be produced via the ElevenLabs API (no GPU). Local Qwen TTS and
full audiobook **compilation** (Chatterbox) require a CUDA GPU.

## Commands

```bash
# Run all tests
python -m pytest -v

# Run a single test
python -m pytest tests/test_ingestion.py::test_compute_hash_deterministic -v

# Run a test file
python -m pytest tests/test_pipeline.py -v

# Django checks
python manage.py check
python manage.py migrate
```

## Architecture

The app accepts an ebook (PDF/EPUB) or text, runs a three-pass LLM + TTS pipeline to
produce an annotated narration script with per-character voice samples, and can then
compile the whole book into a single multi-voice MP3 audiobook. Output is
content-addressed by SHA-256, so re-uploading the same content is an instant cache hit
and interrupted runs resume.

### Request flow

1. **Upload** (`/`) → POST to `/process/` → computes SHA-256 of normalized text →
   `get_or_create` a `ProcessedBook` → writes `raw.txt` (and per-chapter `raw.txt` files
   for multi-chapter books) → fetches cover art → redirects to `/progress/<hash>/`
2. **Progress** (`/progress/<hash>/`) → page opens `EventSource` to `/stream/<hash>/` →
   runs the pipeline synchronously, streaming SSE events → JS redirects to
   `/results/<hash>/` on `done`. `stream_view` marks the book `done`/`failed` at the end.
3. **Results** (`/results/<hash>/`) → renders speakers + annotated script; speakers can be
   edited and individual voices regenerated; full-audiobook compilation is launched here.
4. **Listen** (`/listen/`, `/listen/<hash>/`) → library of books that have a compiled
   `full.mp3`, with a player per chapter.

Other routes: `/compile/<hash>/[<chapter>/]` + `/compile/<hash>/[<chapter>/]stream/`
(SSE compilation), `/audio/<hash>/[<chapter>/]` (serve `full.mp3`), `/voice/.../regenerate/`,
`/speaker/.../update/`, `/cover/<hash>/`, `/delete/<hash>/`.

### Pipeline (`reader/pipeline.py`)

Two generators yield SSE strings; `stream_view` picks based on whether `chapters.json`
exists:

- **`run_pipeline()`** — single-document books.
- **`run_book_pipeline()`** — multi-chapter books; the speaker cast accumulates across
  chapters and each chapter writes into `chapters/NN/`.

Both run three passes:

- **Pass 1** — `llm.extract_speakers()` per chunk → `merge_speakers()` → canonical speaker
  list (name, sex, age, nationality, traits, aliases).
- **Pass 2** — `llm.annotate_chunk()` per chunk with the speaker list →
  `output.normalize_speaker_names()` rewrites tags (incl. aliases) to match `speakers.txt`
  → writes `speakers.txt` and `annotated.txt`.
- **Pass 3** — `tts.generate_voice_sample()` per speaker → writes `voices/<slug>.wav`.

Each pass is skipped (resumed) when its output files already exist. Pass 3 failures emit
`voice_warning` events and are skipped; the pipeline always ends with `done`.

### Voice generation (`reader/tts.py`)

`generate_voice_sample()` uses **ElevenLabs Voice Design** (preferred when
`ELEVENLABS_API_KEY` is set; description built by `build_elevenlabs_description`), and
falls back to the local **Qwen3-TTS VoiceDesign** model (`get_tts_model`, CUDA;
description built by `build_instruct`). ElevenLabs returns MP3, converted to WAV via
`ffmpeg`.

### Compilation (`reader/compile.py`)

`run_compile()` (a separate SSE generator, triggered from results) parses `annotated.txt`
into per-line segments, synthesizes each line in the matching character's cloned voice
using **Chatterbox TTS** (`get_chatterbox_model`, CUDA; mood maps to an `exaggeration`
value), converts to MP3 in batches, and concatenates into `compiled/full.mp3` with
`ffmpeg`. Missing character voices fall back to the narrator voice.

### Key modules

| Module | Responsibility |
|---|---|
| `reader/ingestion.py` | Parse PDF/EPUB/txt → text; SHA-256 hash; chapter splitting (`split_text_chapters`, `split_epub_chapters`); cover art (`extract_epub_cover`, `fetch_openlibrary_cover`) |
| `reader/chunker.py` | Split text into ≤3000-token chunks with 200-token overlap context |
| `reader/llm.py` | OpenAI calls (model = `OPENAI_MODEL`, default `gpt-5.4-mini`): `extract_speakers`, `annotate_chunk`, `merge_speakers`, `split_segment`, `generate_delivery_style` |
| `reader/output.py` | Read/write `speakers.txt` & `annotated.txt`; `normalize_speaker_names`; `chapter_dir_path`; `update_speaker_attrs` |
| `reader/tts.py` | Voice-sample generation (ElevenLabs → Qwen fallback) and Chatterbox helpers (`get_chatterbox_model`, `synthesize_line`); `slugify_name` |
| `reader/compile.py` | Line-by-line synthesis and `ffmpeg` MP3 assembly into `compiled/full.mp3` |
| `reader/pipeline.py` | Orchestrates the three passes (`run_pipeline`, `run_book_pipeline`) as SSE generators |
| `reader/views.py` | Django views; `stream_view` / `compile_stream_view` wrap generators in `StreamingHttpResponse` |

### Output directory layout

```
outputs/<sha256>/
  raw.txt          # normalized input text (re-read by stream_view to run/resume the pipeline)
  cover            # cover image (JPEG or PNG), if found
  speakers.txt     # NAME[,alias…] | sex=X | age=Y | nationality=Z | traits=Z
  annotated.txt    # single-doc books: [NARRATOR] … / [NAME | mood=X] "dialogue"
  chapters.json    # multi-chapter books: [{index, title}, …]
  voices/
    narrator.wav
    <slug>.wav     # one per character
  compiled/
    full.mp3       # single-doc compiled audiobook
  chapters/        # multi-chapter books
    01/
      raw.txt
      annotated.txt
      compiled/full.mp3
    02/
      …
```

In `speakers.txt`, any name, alias, or field value containing a delimiter is
backslash-escaped — `\` → `\\`, `|` → `\p`, `,` → `\c`, newline → `\n` — so each speaker
stays on one physical line and survives a write→read round-trip (`reader/output.py`).

### Settings

`config/settings.py` uses `python-decouple`. Key custom settings (all from env/`.env`):
- `OPENAI_API_KEY` — used by `reader/llm.py` (required)
- `OPENAI_MODEL` — defaults to `gpt-5.4-mini`
- `ELEVENLABS_API_KEY` — used by `reader/tts.py` (required to load; empty forces Qwen fallback)
- `DB_ENGINE` / `DB_NAME` / `DB_USER` / `DB_PASSWORD` / `DB_HOST` / `DB_PORT` — PostgreSQL connection
- `DEBUG` — defaults to **`False`**; set `DEBUG=True` in `.env` for local dev
- `SECRET_KEY` — required when `DEBUG` is `False` (raises `ImproperlyConfigured` if unset); a throwaway dev fallback is used only when `DEBUG=True`
- `ALLOWED_HOSTS` — comma-separated (`Csv`), defaults to `127.0.0.1,localhost`. The app has no login; access control relies on binding to localhost only
- `MAX_UPLOAD_MB` — defaults to `50`; caps upload file and pasted-text size and sets `DATA_UPLOAD_MAX_MEMORY_SIZE`
- `OUTPUTS_DIR` — defaults to `BASE_DIR / "outputs"`

### Testing notes

All OpenAI and TTS calls are mocked in tests — no real API keys or GPU needed. The
`settings` fixture (pytest-django) overrides `OUTPUTS_DIR` to a `tmp_path` in
output/pipeline tests. Because TTS helpers are imported at module level, patch them on
the module that uses them: pipeline tests patch `get_tts_model` and
`generate_voice_sample` on `reader.pipeline`; compile tests patch `get_chatterbox_model`
and `synthesize_line` on `reader.compile`.
