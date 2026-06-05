# Audiobook Reader Design

**Date:** 2026-04-23

## Overview

A dedicated audiobook reader UI at `/listen/` and `/listen/<hash>/`, separate from the upload/processing/results workflow. Only shows books that have at least one compiled `full.mp3`. Supports per-chapter playback with automatic advance to the next chapter on completion, and localStorage-based resume.

---

## Section 1 — Book List (`/listen/`)

`listen_list_view` queries all `ProcessedBook` with `status="done"`. For each, it checks whether compiled audio exists:
- Single-chapter: `outputs/<hash>/compiled/full.mp3`
- Multi-chapter: any `outputs/<hash>/chapters/NN/compiled/full.mp3`

Only books with at least one compiled `full.mp3` are included. The template renders a list of book titles; multi-chapter books also show a chapter count. Each entry links to `/listen/<hash>/`.

A "Listen" link is added to the upload page nav so users can reach the reader from the main UI.

---

## Section 2 — Player Page (`/listen/<hash>/`)

`listen_book_view` loads the `ProcessedBook`, then branches:
- **Single-chapter** (no `chapters.json`): one entry in the chapter list titled with the book title, audio URL is `/audio/<hash>/`.
- **Multi-chapter** (`chapters.json` exists): reads chapter metadata, finds which chapters have `chapters/NN/compiled/full.mp3`, audio URL is `/audio/<hash>/<N>/`.

If no audio exists at all, returns 404.

The template has two areas:

**Sidebar** — lists every chapter that has compiled audio, numbered and titled. The active chapter is highlighted. Clicking a chapter loads it into the player.

**Player area** — a native HTML5 `<audio controls>` element. Large, full-width.

**Auto-advance** — when the `ended` event fires, JS loads the next chapter with audio, scrolls the sidebar to highlight it, and calls `play()`. If no next chapter exists, playback stops.

**Resume** — on page load, JS reads `localStorage` for the key `reader_pos_<hash>`. If found, it restores the last chapter and seeks to the last saved position. The `timeupdate` event saves position to `localStorage` every 5 seconds.

---

## Section 3 — Views and URLs

**New views in `reader/views.py`:**

```python
def listen_list_view(request):
    # Queries done books, filters to those with compiled audio
    # Returns: books list with title, hash, chapter_count

def listen_book_view(request, content_hash):
    # Loads book + chapters with compiled audio
    # 404 if no audio exists
    # Returns: book, chapters [{index, title, audio_url}]
```

**New URL patterns in `reader/urls.py`:**
```
path("listen/", views.listen_list_view, name="listen"),
path("listen/<str:content_hash>/", views.listen_book_view, name="listen_book"),
```

**New templates:**
- `reader/templates/reader/listen_list.html`
- `reader/templates/reader/listen_book.html`

---

## Data Flow

```
ProcessedBook (status=done)
  → check compiled/full.mp3 or chapters/NN/compiled/full.mp3
  → listen_list_view filters to books with audio
  → user selects book
  → listen_book_view loads chapters.json + checks each chapter for full.mp3
  → template renders chapter list + first-chapter audio src
  → JS handles auto-advance + localStorage resume
```

Audio files are served by the existing `full_audio_view` endpoints:
- `/audio/<hash>/` for single-chapter
- `/audio/<hash>/<chapter>/` for per-chapter

---

## Error Handling

- Book with no compiled audio → 404 from `listen_book_view`
- Chapter with no audio → omitted from the sidebar (not shown)
- localStorage unavailable → graceful degradation, no resume
