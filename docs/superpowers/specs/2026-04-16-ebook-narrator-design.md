# Ebook Narrator — Design Spec

**Date:** 2026-04-16  
**Status:** Approved

---

## Overview

A Django web application that accepts an ebook (EPUB, PDF) or plain text and produces an annotated narration script. The LLM (OpenAI GPT-4o) identifies all speaking characters and their attributes, then annotates every segment of the text with the speaker and mood. Output is written to a content-addressed directory and previewed inline in the browser.

---

## Architecture

A single Django project (`config`) with one app (`reader`), organized into four layers:

1. **Ingestion** — normalizes input (file upload or text paste) to plain text using `pypdf` (PDF), `ebooklib` (EPUB), or direct string (plain text).
2. **Pipeline** — SHA-256 hash of raw content → DB lookup for cache hit → if new, runs two-pass LLM pipeline → writes output files.
3. **Streaming** — a Django `StreamingHttpResponse` SSE endpoint emits progress events consumed by the browser via `EventSource`. Processing runs in the request thread.
4. **Preview** — server-rendered view reads output files and displays a two-panel speaker/annotated-text layout.

No Celery, no Redis, no frontend framework.

---

## Data Model

```
ProcessedBook
  content_hash   CharField(unique)   SHA-256 hex of raw input bytes/string
  title          CharField           Extracted from filename or first line of text
  status         CharField           pending | processing | done | failed
  output_path    CharField           Relative path: outputs/<hash>/
  error_message  TextField           Blank if no error
  created_at     DateTimeField
  updated_at     DateTimeField
```

---

## Output Format

Files are written to `outputs/<hash>/` relative to the Django project root.

### `speakers.txt`

One speaker per line, pipe-delimited:

```
NARRATOR | sex=unknown | age=unknown
Elizabeth Bennet | sex=female | age=early 20s | traits=witty, independent
Mr. Darcy | sex=male | age=late 20s | traits=proud, reserved
```

### `annotated.txt`

Every segment prefixed with a tag. Dialogue carries speaker name and mood; all other text is attributed to NARRATOR:

```
[NARRATOR] It was a truth universally acknowledged...
[ELIZABETH BENNET | mood=teasing] "I am perfectly convinced by it."
[NARRATOR] He made no answer.
[MR. DARCY | mood=cold] "I have no wish to speak to her."
```

Speaker names in `annotated.txt` match entries in `speakers.txt` exactly (case-insensitive normalized).

---

## LLM Pipeline

Uses OpenAI GPT-4o via the official `openai` SDK.

### Chunking

- Target: ~3,000 tokens (~12,000 characters) per chunk
- Split on paragraph boundaries where possible
- 200-token overlap at chunk edges included as read-only context for the LLM; overlap text is **not** included in the Pass 2 annotation output to avoid duplicate segments
- Tokenization via `tiktoken`
- Texts shorter than one chunk are processed as a single chunk

### Pass 1 — Speaker Extraction

Each chunk is sent with a prompt asking for every speaking character with: name, sex, approximate age range, personality traits. The prompt instructs the LLM to use consistent, full names (e.g., "Elizabeth Bennet" not "Elizabeth"). Results from all chunks are merged by exact name, case-insensitive. The merged list becomes the canonical `speakers.txt` and is used as input to Pass 2.

### Pass 2 — Dialogue Annotation

Each chunk is sent with the full merged speaker list. The prompt asks for every segment to be prefixed with `[SPEAKER NAME | mood=X]` (dialogue) or `[NARRATOR]` (non-dialogue), using only names from the provided list. Annotated chunks are concatenated in order to produce `annotated.txt`.

### Progress Events (SSE)

| Event | Meaning |
|---|---|
| `parsing` | File ingestion and normalization complete |
| `chunk_progress N M` | Chunk N of M has completed both passes |
| `done` | Output files written successfully |
| `error <message>` | Failure at any pipeline stage |

---

## Web UI

### Views

| URL | Purpose |
|---|---|
| `/` | Upload form (file picker + textarea, mutually exclusive) |
| `/process/` | POST handler — hash check, DB record creation, redirect |
| `/progress/<hash>/` | Progress page with SSE-driven progress bar |
| `/stream/<hash>/` | SSE stream endpoint (`text/event-stream`) |
| `/results/<hash>/` | Two-panel results preview |

### Flow

1. User uploads file or pastes text at `/`. Only one input type is active at a time — JS disables the inactive input when the other is used; the server validates that exactly one is present and rejects the form otherwise.
2. POST to `/process/` computes hash:
   - Cache hit (`status=done`) → redirect to `/results/<hash>/`
   - Already processing → redirect to `/progress/<hash>/`
   - New → create `ProcessedBook`, redirect to `/progress/<hash>/`
3. Progress page opens `EventSource` to `/stream/<hash>/`, which runs the pipeline and emits events
4. On `done` event, JS redirects to `/results/<hash>/`
5. Results page renders speaker table (left panel) and annotated text (right panel), both independently scrollable

### Results Styling

- Speaker tags rendered in bold
- Mood annotations in italic
- Narrator segments in gray
- Server-rendered HTML, no frontend framework

---

## Error Handling

| Scenario | Behavior |
|---|---|
| Corrupted PDF / invalid EPUB | Caught at ingestion; `status=failed`; error shown on progress page |
| OpenAI API error (rate limit, timeout) | Retry once with exponential backoff; if still failing, mark failed |
| Duplicate upload while processing | Redirect to existing progress page; no second pipeline run |
| Text shorter than one chunk | Processed as single chunk; no special case needed |

---

## Tech Stack

| Concern | Library / Tool |
|---|---|
| Web framework | Django 5.x |
| PDF parsing | `pypdf` |
| EPUB parsing | `ebooklib` |
| OpenAI client | `openai` (official SDK) |
| Tokenization | `tiktoken` |
| Frontend | Plain HTML + vanilla JS (`EventSource`) |
| Database | SQLite (dev) |
| Static files | Django `staticfiles` |

---

## Out of Scope (v1)

- Text-to-speech audio generation
- User accounts / authentication
- Concurrent-user scaling (Celery/Redis)
- Languages other than English
