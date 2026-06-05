# Chapter Processing Design

**Date:** 2026-04-17

## Overview

Process multi-chapter books one chapter at a time, storing per-chapter output in subdirectories while keeping the speaker list and voice files shared across all chapters. Single-chapter books (no chapters detected) use the existing layout unchanged.

## Approach

Single-chapter books route to the existing `run_pipeline` — no changes to their behaviour or file layout. Multi-chapter books route to a new `run_book_pipeline` that processes chapters sequentially, accumulating speakers across chapters and generating voices only for new speakers as they are discovered.

---

## Section 1 — Chapter Detection (`ingestion.py`)

Two new functions:

**`split_epub_chapters(file_bytes) -> list[dict]`**
Re-reads the epub spine. Returns one `{"title": str, "text": str}` dict per spine item. Title comes from the item's first non-empty line, falling back to `"Chapter N"`. Spine items under ~100 words are merged into the next one to skip front matter, TOC pages, and other short structural items.

**`split_text_chapters(text) -> list[dict] | None`**
Splits on lines matching chapter heading patterns (case-insensitive, line must be short and surrounded by whitespace):
- `Chapter 1`, `Chapter One`, `Chapter I`
- `CHAPTER 1`, `CHAPTER ONE`, `CHAPTER I`
- `Part I`, `PART 1`

Returns `None` if no chapter markers are found, signalling a single-chapter book.

**`process_view` integration:**
After `normalize_input`, attempt chapter detection:
- EPUB: call `split_epub_chapters`
- PDF/txt: call `split_text_chapters`

If chapters are found, write a `chapters.json` manifest to `outputs/<hash>/`:
```json
[{"index": 1, "title": "Chapter 1 — The Man in Black"}, ...]
```

If no chapters found (or single-chapter), do not write `chapters.json` — existing flow is used.

---

## Section 2 — Pipeline Orchestration (`pipeline.py`)

**`run_book_pipeline(content_hash, chapters, title)`** — new generator yielding SSE strings.

For each chapter `i` of `N`:

1. Yield `data: chapter_start {i} {N} {chapter_title}\n\n`
2. Write `outputs/<hash>/chapters/NN/raw.txt`
3. **Pass 1 — Speaker extraction:** Extract speakers from chapter chunks. Pass the cumulative known-speakers list from `speakers.txt` to `extract_speakers` so the LLM uses consistent name spellings. Merge new speakers into `speakers.txt` (append only — existing entries untouched).
4. **Pass 2 — Annotation:** Annotate chapter chunks using the full cumulative speaker list (all speakers found so far across all chapters). Write `outputs/<hash>/chapters/NN/annotated.txt`.
5. **Pass 3 — Voice generation:** Generate voice samples only for speakers whose WAV does not yet exist in `voices/`.
6. Yield `data: chapter_done {i} {N}\n\n`

After all chapters: yield `data: done\n\n`.

**`run_pipeline`** is unchanged. **`stream_view`** reads `chapters.json` to decide which to call.

**`extract_speakers` signature update:** Add `known_speakers: list[dict] = None` parameter. When provided, inject a known-characters block into `_SPEAKER_PROMPT` so the LLM reuses existing name spellings:

**`_SPEAKER_PROMPT` update:** Add a known-speakers block to the extraction prompt so the LLM reuses existing name spellings:
```
Known characters already identified (use exact spellings):
- Brown (male, 30s)
- The Gunslinger (male, middle-aged)
```

**Progress page** handles `chapter_start` and `chapter_done` SSE events, displaying "Processing chapter 2 of 5 — The Storm" with a progress bar spanning all chapters.

---

## Section 3 — File Structure

```
outputs/<hash>/
  raw.txt                  # full text (unchanged)
  chapters.json            # chapter manifest (multi-chapter only)
  speakers.txt             # cumulative across all chapters
  voices/                  # all voice WAVs, shared
    narrator.wav
    brown.wav
  chapters/
    01/
      raw.txt
      annotated.txt
      compiled/
        full.mp3
    02/
      raw.txt
      annotated.txt
      compiled/
        full.mp3
```

Chapter directories are zero-padded to the digit width of the total chapter count (`01`–`09`, `10`+, etc.).

Single-chapter books use the existing root-level layout — no `chapters/` directory, no `chapters.json`.

---

## Section 4 — Views and URLs

**`stream_view`** — reads `chapters.json` if present; calls `run_book_pipeline`, otherwise calls `run_pipeline` unchanged.

**`results_view`** — reads `chapters.json` if present; loads per-chapter `annotated.txt` files; passes chapter list and per-chapter annotated lines to the template.

**`chapter_content_view`** (new) — returns JSON `{"lines": [...], "has_full_audio": bool}` for a given chapter. `lines` is a list of `{"type": "narrator"|"dialogue"|"raw", "speaker": str|null, "mood": str|null, "text": str}` dicts — the same shape as `read_annotated()` output. URL: `GET /results/<hash>/chapter/<n>/`

**`compile_view` / `compile_stream_view`** — gain a `chapter` URL parameter. New URL: `compile/<hash>/<chapter>/` and `compile/<hash>/<chapter>/stream/`. Compile reads annotated.txt from `chapters/<NN>/` and writes `compiled/full.mp3` there. Existing `/compile/<hash>/` (no chapter) continues to work for single-chapter books.

**`full_audio_view`** — gains a `chapter` parameter. New URL: `audio/<hash>/<chapter>/` serves `chapters/<NN>/compiled/full.mp3`. Existing `/audio/<hash>/` continues to work for single-chapter books.

**`update_speaker_view`** — unchanged; always reads/writes root `speakers.txt`.

---

## Section 5 — Results Page (Chapter Selector)

When `chapters.json` exists, a chapter selector appears above the annotated panel.

- ≤8 chapters: button row
- >8 chapters: `<select>` dropdown

Clicking a chapter:
1. Fetches `chapter_content_view` JSON for that chapter
2. Replaces annotated panel content
3. Updates "Compile audio" link to `/compile/<hash>/<chapter>/`
4. Updates audio player `src` to `/audio/<hash>/<chapter>/` (shown only if `has_full_audio` is true)

The speakers pane stays fixed — shows all speakers from root `speakers.txt` regardless of selected chapter.

Single-chapter books show no selector and render exactly as today.

---

## Error Handling

- Fatal error in any chapter's pipeline → yield `data: error ...\n\n`, stop. Completed chapters are preserved.
- Voice generation failure for a speaker → emit `voice_warning`, continue (existing behaviour).
- Missing `chapters/<NN>/annotated.txt` when loading results → show empty annotated panel for that chapter, log warning.
