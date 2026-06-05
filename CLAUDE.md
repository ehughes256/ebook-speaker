# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then set OPENAI_API_KEY
python manage.py migrate
python manage.py runserver
```

GPU-only dependencies (not in requirements.txt — install separately on the target machine):
```bash
pip install qwen-tts torch
pip install flash-attn --no-build-isolation  # optional, improves TTS speed
```

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

The app accepts an ebook or text, runs a three-pass LLM + TTS pipeline, and serves an annotated narration script with character voice samples.

### Request flow

1. **Upload** (`/`) → POST to `/process/` → computes SHA-256 of content → `get_or_create` a `ProcessedBook` → writes `raw.txt` to `outputs/<hash>/` → redirects to `/progress/<hash>/`
2. **Progress** (`/progress/<hash>/`) → page opens `EventSource` to `/stream/<hash>/` → runs the pipeline synchronously, streaming SSE events → JS redirects to `/results/<hash>/` on `done`
3. **Results** (`/results/<hash>/`) → reads output files → renders two-panel view (speakers + annotated script)

Identical content always maps to the same output directory (content-addressed). Re-uploading the same book hits the cached result immediately.

### Pipeline (`reader/pipeline.py`)

`run_pipeline()` is a generator yielding SSE strings. Three passes:

- **Pass 1** — `llm.extract_speakers()` per chunk → `merge_speakers()` → canonical speaker list
- **Pass 2** — `llm.annotate_chunk()` per chunk with speaker list → `output.normalize_speaker_names()` ensures tag names match `speakers.txt` exactly → writes `speakers.txt` and `annotated.txt`
- **Pass 3** — `tts.get_tts_model()` (lazy singleton, CUDA) → `tts.generate_voice_sample()` per speaker → writes `voices/<slug>.wav`

Pass 3 failures (`RuntimeError`, `ImportError` when GPU unavailable) emit `voice_warning` events and are skipped; the pipeline always ends with `done`.

### Key modules

| Module | Responsibility |
|---|---|
| `reader/ingestion.py` | Parse PDF/EPUB/txt → plain text; SHA-256 hash |
| `reader/chunker.py` | Split text into ≤3000-token chunks with 200-token overlap context |
| `reader/llm.py` | OpenAI GPT-4o calls: `extract_speakers`, `annotate_chunk`, `merge_speakers` |
| `reader/output.py` | Read/write `speakers.txt`, `annotated.txt`; parse annotated lines; `normalize_speaker_names` |
| `reader/tts.py` | Qwen VoiceDesign model (lazy singleton); `build_instruct` from speaker attributes; `generate_voice_sample` → WAV |
| `reader/pipeline.py` | Orchestrates all three passes as a generator |
| `reader/views.py` | Django views; `stream_view` wraps pipeline in `StreamingHttpResponse` |

### Output directory layout

```
outputs/<sha256>/
  raw.txt          # original input text (used by stream_view to re-run pipeline)
  speakers.txt     # pipe-delimited: NAME | sex=X | age=Y | traits=Z
  annotated.txt    # [NARRATOR] ... / [NAME | mood=X] "dialogue"
  voices/
    narrator.wav
    <slug>.wav     # one per character
```

### Settings

`config/settings.py` uses `python-decouple`. Key custom settings:
- `OPENAI_API_KEY` — used by `reader/llm.py`
- `OUTPUTS_DIR` — defaults to `BASE_DIR / "outputs"`

### Testing notes

All OpenAI and TTS calls are mocked in tests — no real API keys needed. The `settings` fixture (pytest-django) is used in output/pipeline tests to override `OUTPUTS_DIR` to a `tmp_path`. Pipeline tests must patch both `get_tts_model` and `generate_voice_sample` from `reader.pipeline` (not `reader.tts`) since they're imported at module level.
