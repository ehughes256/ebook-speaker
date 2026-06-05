# Reader — Multi-Voice Audiobook Narrator

Reader turns an ebook (PDF, EPUB) or pasted text into a fully narrated, multi-voice
audiobook. It runs the text through a three-pass LLM + TTS pipeline that identifies
every speaking character, annotates the prose into a tagged narration script, designs
a distinct voice for each character, and can compile the whole book into a single
multi-voice MP3.

Everything is **content-addressed**: the same input always maps to the same output
directory, so re-uploading a book is an instant cache hit, and interrupted runs resume
where they left off.

## Features

- **Ingest PDF, EPUB, or plain text** — uploads or pasted text are normalized to plain
  text and hashed (SHA-256) into a stable output directory.
- **Automatic chapter splitting** — EPUBs are split via their table of contents; plain
  text is split on `Chapter`/`Part` headings. Multi-chapter books carry a cumulative,
  consistent cast of characters across chapters.
- **Speaker extraction** — the LLM identifies each character with sex, age, nationality,
  personality traits, and aliases.
- **Narration annotation** — every passage is tagged `[NARRATOR]` or
  `[CHARACTER | mood=X]`, with attribution and action cleanly separated from dialogue.
- **Per-character voice design** — a representative voice sample is generated for each
  character (and the narrator) from their attributes.
- **Full audiobook compilation** — every line is synthesized in its character's voice
  with mood-driven expressiveness, then concatenated into `compiled/full.mp3`.
- **Cover art** — pulled from the EPUB or fetched from Open Library by title.
- **Live progress** — the pipeline streams progress to the browser over Server-Sent
  Events (SSE); a "Listen" library lists every finished audiobook.
- **Editable speakers** — character attributes, aliases, and voices can be adjusted and
  regenerated from the results page.

## How it works

Upload → `/process/` computes the content hash, creates a `ProcessedBook`, writes
`raw.txt` (and per-chapter `raw.txt` files for multi-chapter books), then redirects to a
progress page. The progress page opens an `EventSource` to `/stream/<hash>/`, which runs
the pipeline synchronously and streams SSE events until `done`, then redirects to the
results view.

The pipeline (`reader/pipeline.py`) runs three passes:

1. **Pass 1 — Speakers.** Text is chunked (≤3000 tokens, 200-token overlap) and
   `llm.extract_speakers()` runs per chunk. `merge_speakers()` produces the canonical
   cast. For multi-chapter books, the cast accumulates chapter by chapter.
2. **Pass 2 — Annotation.** `llm.annotate_chunk()` tags each chunk against the speaker
   list; `output.normalize_speaker_names()` rewrites tags (including aliases) to match
   `speakers.txt` exactly. Writes `speakers.txt` and `annotated.txt`.
3. **Pass 3 — Voice design.** `tts.generate_voice_sample()` produces one WAV per speaker
   under `voices/`. Failures (e.g. no GPU and no API key) emit `voice_warning` events and
   are skipped — the pipeline always ends with `done`.

**Compilation** (`reader/compile.py`, triggered from the results page) is a separate
step: it parses `annotated.txt` into per-line segments, synthesizes each line in the
matching character's cloned voice (mood modulates expressiveness), converts to MP3 in
batches, and concatenates into `compiled/full.mp3`.

### Voice generation: two paths

| Step | Engine | Requirements |
|---|---|---|
| Voice **samples** (Pass 3) | ElevenLabs Voice Design, falling back to local Qwen3-TTS VoiceDesign | ElevenLabs API key **or** a CUDA GPU |
| Audiobook **compilation** | Chatterbox TTS (clones each sample) | CUDA GPU |

You can produce annotated scripts and per-character voice samples with just API keys.
Compiling the full audiobook requires a local GPU.

## Tech stack

- **Django 5.2** web app (`config/` project, `reader/` app), **PostgreSQL** database
- **OpenAI** for speaker extraction, annotation, and delivery-style prompts
- **ElevenLabs Voice Design** for voice samples (no GPU needed)
- **Qwen3-TTS VoiceDesign** and **Chatterbox TTS** for local, GPU-based synthesis
- **pypdf / ebooklib / BeautifulSoup** for ingestion, **tiktoken** for chunking
- **ffmpeg** for MP3 conversion and concatenation
- **python-decouple** for configuration, **pytest / pytest-django** for tests

## Prerequisites

- Python 3.10+
- PostgreSQL (the connection is configured in `config/settings.py`)
- `ffmpeg` on `PATH` (required for audiobook compilation and ElevenLabs WAV conversion)
- An OpenAI API key, and either an ElevenLabs API key or a CUDA GPU for voices
- A CUDA GPU for full audiobook compilation

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # then fill in the keys below

python manage.py migrate
python manage.py runserver
```

GPU-only dependencies are **not** in `requirements.txt` — install them on the target
machine:

```bash
pip install qwen-tts torch chatterbox-tts
pip install flash-attn --no-build-isolation   # optional, speeds up Qwen TTS
```

### Configuration

Settings are read from the environment / `.env` via `python-decouple`:

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | yes | — | LLM calls in `reader/llm.py` |
| `ELEVENLABS_API_KEY` | yes | — | Voice sample generation (set empty to force the Qwen fallback) |
| `OPENAI_MODEL` | no | `gpt-5.4-mini` | Model used for all LLM calls |
| `SECRET_KEY` | no | dev placeholder | Django secret key |
| `DEBUG` | no | `True` | Django debug mode |

Both `OPENAI_API_KEY` and `ELEVENLABS_API_KEY` are read without a default, so they must be
present in the environment for the app to start. The database connection (name, user,
password, host) is defined directly in `config/settings.py`; adjust it to match your
local PostgreSQL instance. Outputs are written under `OUTPUTS_DIR` (`<BASE_DIR>/outputs`).

## Usage

1. Open `http://127.0.0.1:8000/`.
2. Paste text or upload a `.txt`, `.pdf`, or `.epub` file and submit.
3. Watch live progress as the three passes run.
4. On the results page, review the cast and annotated script, tweak speaker attributes,
   regenerate individual voices, and kick off full-audiobook compilation.
5. Visit `/listen/` for a library of finished audiobooks with playable chapters.

## Output directory layout

```
outputs/<sha256>/
  raw.txt                 # normalized input text (re-read to resume the pipeline)
  cover                   # cover image (JPEG or PNG), if found
  speakers.txt            # NAME[,alias…] | sex=X | age=Y | nationality=Z | traits=…
  annotated.txt           # single-doc books: [NARRATOR] … / [NAME | mood=X] "dialogue"
  chapters.json           # multi-chapter books: [{index, title}, …]
  voices/
    narrator.wav
    <slug>.wav            # one per character
  compiled/
    full.mp3              # single-doc compiled audiobook
  chapters/               # multi-chapter books
    01/
      raw.txt
      annotated.txt
      compiled/full.mp3
    02/
      …
```

## Project structure

| Path | Responsibility |
|---|---|
| `reader/ingestion.py` | Parse PDF/EPUB/txt → text; SHA-256 hashing; chapter & cover extraction |
| `reader/chunker.py` | Split text into ≤3000-token chunks with overlap context |
| `reader/llm.py` | OpenAI calls: `extract_speakers`, `annotate_chunk`, `merge_speakers`, `split_segment` |
| `reader/output.py` | Read/write `speakers.txt` & `annotated.txt`; normalize names; parse annotated lines |
| `reader/tts.py` | Voice-sample generation (ElevenLabs → Qwen fallback); Chatterbox helpers |
| `reader/compile.py` | Line-by-line synthesis and MP3 assembly into `full.mp3` |
| `reader/pipeline.py` | Orchestrates the three passes (`run_pipeline`, `run_book_pipeline`) as SSE generators |
| `reader/views.py` | Django views; SSE streaming via `StreamingHttpResponse` |
| `config/` | Django project settings, URLs, WSGI/ASGI |

## Testing

OpenAI and TTS calls are mocked, so no API keys or GPU are needed to run the suite.

```bash
# Run all tests
python -m pytest -v

# Run a single test file
python -m pytest tests/test_pipeline.py -v

# Run a single test
python -m pytest tests/test_ingestion.py::test_compute_hash_deterministic -v

# Django checks
python manage.py check
```

## License

Released under the GNU General Public License v3.0 — see [LICENSE](LICENSE).
