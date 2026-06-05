# Audiobook Reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated audiobook reader at `/listen/` and `/listen/<hash>/` that lists compiled books and plays them chapter-by-chapter with auto-advance and localStorage resume.

**Architecture:** Two new views (`listen_list_view`, `listen_book_view`) query `ProcessedBook` and check the filesystem for compiled `full.mp3` files. Two new templates render the book list and player. A "Listen" nav link is added to the upload and results pages.

**Tech Stack:** Django views, HTML5 `<audio>`, localStorage, existing `full_audio_view` endpoints

---

## File Map

| Action | File |
|--------|------|
| Modify | `reader/views.py` |
| Modify | `reader/urls.py` |
| Create | `reader/templates/reader/listen_list.html` |
| Create | `reader/templates/reader/listen_book.html` |
| Modify | `reader/templates/reader/upload.html` |
| Modify | `reader/templates/reader/results.html` |
| Modify | `tests/test_views.py` |

---

## Task 1: Views

**Files:**
- Modify: `reader/views.py`
- Modify: `tests/test_views.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_views.py`:

```python
class ListenListViewTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_listen_list_returns_200(self):
        response = self.client.get(reverse("reader:listen"))
        self.assertEqual(response.status_code, 200)

    def test_listen_list_only_shows_books_with_audio(self):
        import tempfile, json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.settings(OUTPUTS_DIR=tmp_dir):
                # Book WITH audio
                b1 = ProcessedBook.objects.create(
                    content_hash="aaa111",
                    title="Has Audio",
                    status="done",
                    output_path="outputs/aaa111/",
                )
                p1 = Path(tmp_dir) / "aaa111" / "compiled"
                p1.mkdir(parents=True)
                (p1 / "full.mp3").write_bytes(b"fake")

                # Book WITHOUT audio
                ProcessedBook.objects.create(
                    content_hash="bbb222",
                    title="No Audio",
                    status="done",
                    output_path="outputs/bbb222/",
                )
                Path(tmp_dir, "bbb222").mkdir(parents=True)

                response = self.client.get(reverse("reader:listen"))
                available = response.context["available"]
        self.assertEqual(len(available), 1)
        self.assertEqual(available[0]["book"].content_hash, "aaa111")

    def test_listen_list_multi_chapter_counts_audio_chapters(self):
        import tempfile, json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.settings(OUTPUTS_DIR=tmp_dir):
                book = ProcessedBook.objects.create(
                    content_hash="ccc333",
                    title="Multi",
                    status="done",
                    output_path="outputs/ccc333/",
                )
                out = Path(tmp_dir) / "ccc333"
                out.mkdir()
                chapters_meta = [{"index": 1, "title": "Ch 1"}, {"index": 2, "title": "Ch 2"}]
                (out / "chapters.json").write_text(json.dumps(chapters_meta))
                # Only chapter 1 has audio
                ch1 = out / "chapters" / "01" / "compiled"
                ch1.mkdir(parents=True)
                (ch1 / "full.mp3").write_bytes(b"fake")

                response = self.client.get(reverse("reader:listen"))
                available = response.context["available"]
        self.assertEqual(len(available), 1)
        self.assertEqual(available[0]["chapter_count"], 1)


class ListenBookViewTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_listen_book_single_chapter(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.settings(OUTPUTS_DIR=tmp_dir):
                book = ProcessedBook.objects.create(
                    content_hash="ddd444",
                    title="Single",
                    status="done",
                    output_path="outputs/ddd444/",
                )
                p = Path(tmp_dir) / "ddd444" / "compiled"
                p.mkdir(parents=True)
                (p / "full.mp3").write_bytes(b"fake")

                response = self.client.get(reverse("reader:listen_book", args=["ddd444"]))
        self.assertEqual(response.status_code, 200)
        chapters = response.context["chapters"]
        self.assertEqual(len(chapters), 1)
        self.assertIn("/audio/ddd444/", chapters[0]["audio_url"])

    def test_listen_book_multi_chapter(self):
        import tempfile, json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.settings(OUTPUTS_DIR=tmp_dir):
                book = ProcessedBook.objects.create(
                    content_hash="eee555",
                    title="Multi",
                    status="done",
                    output_path="outputs/eee555/",
                )
                out = Path(tmp_dir) / "eee555"
                out.mkdir()
                meta = [{"index": 1, "title": "Ch 1"}, {"index": 2, "title": "Ch 2"}]
                (out / "chapters.json").write_text(json.dumps(meta))
                for i in (1, 2):
                    p = out / "chapters" / f"0{i}" / "compiled"
                    p.mkdir(parents=True)
                    (p / "full.mp3").write_bytes(b"fake")

                response = self.client.get(reverse("reader:listen_book", args=["eee555"]))
        self.assertEqual(response.status_code, 200)
        chapters = response.context["chapters"]
        self.assertEqual(len(chapters), 2)
        self.assertIn("/audio/eee555/1/", chapters[0]["audio_url"])

    def test_listen_book_404_when_no_audio(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.settings(OUTPUTS_DIR=tmp_dir):
                ProcessedBook.objects.create(
                    content_hash="fff666",
                    title="No Audio",
                    status="done",
                    output_path="outputs/fff666/",
                )
                Path(tmp_dir, "fff666").mkdir(parents=True)
                response = self.client.get(reverse("reader:listen_book", args=["fff666"]))
        self.assertEqual(response.status_code, 404)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/ehughes/code/claude/mnt/reader && python -m pytest tests/test_views.py::ListenListViewTest tests/test_views.py::ListenBookViewTest -v 2>&1 | tail -10
```

Expected: FAIL — URL not found or view not defined

- [ ] **Step 3: Add the two views to `reader/views.py`**

Add after the existing `upload_view` function:

```python
def listen_list_view(request):
    books = ProcessedBook.objects.filter(status="done").order_by("-created_at")
    available = []
    for book in books:
        out_dir = Path(settings.OUTPUTS_DIR) / book.content_hash
        chapters_path = out_dir / "chapters.json"
        if chapters_path.exists():
            chapters_meta = json.loads(chapters_path.read_text(encoding="utf-8"))
            n = len(chapters_meta)
            pad = max(2, len(str(n)))
            audio_count = sum(
                1 for i in range(1, n + 1)
                if (out_dir / "chapters" / str(i).zfill(pad) / "compiled" / "full.mp3").exists()
            )
            if audio_count > 0:
                available.append({"book": book, "chapter_count": audio_count, "is_multi": True})
        else:
            if (out_dir / "compiled" / "full.mp3").exists():
                available.append({"book": book, "chapter_count": 1, "is_multi": False})
    return render(request, "reader/listen_list.html", {"available": available})


def listen_book_view(request, content_hash):
    from django.urls import reverse as _reverse
    book = get_object_or_404(ProcessedBook, content_hash=content_hash, status="done")
    out_dir = Path(settings.OUTPUTS_DIR) / content_hash
    chapters_path = out_dir / "chapters.json"

    if chapters_path.exists():
        chapters_meta = json.loads(chapters_path.read_text(encoding="utf-8"))
        n = len(chapters_meta)
        pad = max(2, len(str(n)))
        chapters = []
        for ch in chapters_meta:
            idx = ch["index"]
            if (out_dir / "chapters" / str(idx).zfill(pad) / "compiled" / "full.mp3").exists():
                chapters.append({
                    "index": idx,
                    "title": ch["title"],
                    "audio_url": _reverse("reader:full_audio_chapter", args=[content_hash, idx]),
                })
    else:
        if not (out_dir / "compiled" / "full.mp3").exists():
            raise Http404
        chapters = [{
            "index": 1,
            "title": book.title,
            "audio_url": _reverse("reader:full_audio", args=[content_hash]),
        }]

    if not chapters:
        raise Http404

    return render(request, "reader/listen_book.html", {"book": book, "chapters": chapters})
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/ehughes/code/claude/mnt/reader && python -m pytest tests/test_views.py::ListenListViewTest tests/test_views.py::ListenBookViewTest -v 2>&1 | tail -10
```

Expected: Still FAIL — URL not yet registered. Move to Task 2.

---

## Task 2: URLs and nav links

**Files:**
- Modify: `reader/urls.py`
- Modify: `reader/templates/reader/upload.html`
- Modify: `reader/templates/reader/results.html`

- [ ] **Step 1: Add URL patterns to `reader/urls.py`**

Add after the existing `compile_stream_chapter` line:

```python
path("listen/", views.listen_list_view, name="listen"),
path("listen/<str:content_hash>/", views.listen_book_view, name="listen_book"),
```

- [ ] **Step 2: Add "Listen" link to upload page**

In `reader/templates/reader/upload.html`, find:

```html
<h1>Ebook Narrator</h1>
<p class="subtitle">Upload an ebook or paste text to generate an annotated narration script.</p>
```

Replace with:

```html
<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px;">
  <h1 style="margin:0;">Ebook Narrator</h1>
  <a href="{% url 'reader:listen' %}" style="font-size:0.85rem;color:#555;text-decoration:none;">▶ Listen</a>
</div>
<p class="subtitle">Upload an ebook or paste text to generate an annotated narration script.</p>
```

- [ ] **Step 3: Add "Listen" link to results page header**

In `reader/templates/reader/results.html`, find:

```html
  <a href="{% if current_chapter %}{% url 'reader:compile_chapter' content_hash=book.content_hash chapter=current_chapter %}{% else %}{% url 'reader:compile' content_hash=book.content_hash %}{% endif %}" id="compile-link">Compile audio</a>
  <a href="{% url 'reader:upload' %}">← New book</a>
```

Replace with:

```html
  <a href="{% url 'reader:listen_book' book.content_hash %}">▶ Listen</a>
  <a href="{% if current_chapter %}{% url 'reader:compile_chapter' content_hash=book.content_hash chapter=current_chapter %}{% else %}{% url 'reader:compile' content_hash=book.content_hash %}{% endif %}" id="compile-link">Compile audio</a>
  <a href="{% url 'reader:upload' %}">← New book</a>
```

- [ ] **Step 4: Run Django check and view tests**

```bash
cd /Users/ehughes/code/claude/mnt/reader && python manage.py check && python -m pytest tests/test_views.py::ListenListViewTest tests/test_views.py::ListenBookViewTest -v 2>&1 | tail -10
```

Expected: Tests still fail with TemplateDoesNotExist — that's correct, templates come next.

---

## Task 3: Book list template

**Files:**
- Create: `reader/templates/reader/listen_list.html`

- [ ] **Step 1: Create `reader/templates/reader/listen_list.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Audiobook Library</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 60px auto; padding: 0 20px; }
  .header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 32px; }
  h1 { font-size: 1.6rem; margin: 0; }
  .header a { font-size: 0.85rem; color: #555; text-decoration: none; }
  .book-list { list-style: none; padding: 0; margin: 0; }
  .book-item { display: flex; justify-content: space-between; align-items: center; padding: 16px 0; border-bottom: 1px solid #eee; }
  .book-item:last-child { border-bottom: none; }
  .book-title { font-size: 1rem; font-weight: 600; color: #1a1a1a; text-decoration: none; }
  .book-title:hover { text-decoration: underline; }
  .book-meta { font-size: 0.82rem; color: #999; margin-top: 3px; }
  .play-btn { padding: 6px 16px; background: #1a1a1a; color: #fff; border: none; border-radius: 4px; font-size: 0.85rem; text-decoration: none; cursor: pointer; }
  .play-btn:hover { background: #333; }
  .empty { color: #999; font-size: 0.9rem; margin-top: 40px; text-align: center; }
</style>
</head>
<body>
<div class="header">
  <h1>Library</h1>
  <a href="{% url 'reader:upload' %}">← Upload</a>
</div>

{% if available %}
<ul class="book-list">
  {% for entry in available %}
  <li class="book-item">
    <div>
      <a class="book-title" href="{% url 'reader:listen_book' entry.book.content_hash %}">{{ entry.book.title }}</a>
      <div class="book-meta">
        {{ entry.chapter_count }} chapter{{ entry.chapter_count|pluralize }} with audio
      </div>
    </div>
    <a class="play-btn" href="{% url 'reader:listen_book' entry.book.content_hash %}">▶ Play</a>
  </li>
  {% endfor %}
</ul>
{% else %}
<p class="empty">No audiobooks available yet. Upload a book and compile its audio first.</p>
{% endif %}
</body>
</html>
```

- [ ] **Step 2: Run Django check**

```bash
cd /Users/ehughes/code/claude/mnt/reader && python manage.py check
```

Expected: no issues

---

## Task 4: Player template

**Files:**
- Create: `reader/templates/reader/listen_book.html`

- [ ] **Step 1: Create `reader/templates/reader/listen_book.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ book.title }} — Listen</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: system-ui, sans-serif; margin: 0; padding: 0; height: 100vh; display: flex; flex-direction: column; }
  header { padding: 12px 24px; border-bottom: 1px solid #ddd; display: flex; justify-content: space-between; align-items: center; flex-shrink: 0; }
  header h1 { font-size: 1.1rem; margin: 0; }
  header a { font-size: 0.85rem; color: #555; text-decoration: none; margin-left: 16px; }
  .player-bar { padding: 12px 24px; border-bottom: 1px solid #ddd; background: #fafafa; flex-shrink: 0; }
  .player-bar audio { width: 100%; height: 36px; }
  .now-playing { font-size: 0.78rem; color: #999; margin-bottom: 6px; }
  .layout { display: flex; flex: 1; overflow: hidden; }
  .sidebar { width: 280px; min-width: 200px; border-right: 1px solid #ddd; overflow-y: auto; flex-shrink: 0; }
  .chapter-item { padding: 12px 16px; cursor: pointer; border-bottom: 1px solid #f0f0f0; font-size: 0.88rem; color: #333; display: flex; align-items: center; gap: 10px; }
  .chapter-item:hover { background: #f5f5f5; }
  .chapter-item.active { background: #1a1a1a; color: #fff; }
  .chapter-num { font-size: 0.75rem; opacity: 0.5; min-width: 24px; }
  .chapter-title { flex: 1; }
  .main { flex: 1; display: flex; align-items: center; justify-content: center; color: #ccc; font-size: 0.9rem; padding: 24px; }
</style>
</head>
<body>
<header>
  <h1>{{ book.title }}</h1>
  <div>
    <a href="{% url 'reader:results' book.content_hash %}">Script</a>
    <a href="{% url 'reader:listen' %}">← Library</a>
  </div>
</header>

<div class="player-bar">
  <div class="now-playing" id="now-playing">Select a chapter to begin</div>
  <audio id="player" controls style="width:100%"></audio>
</div>

<div class="layout">
  <div class="sidebar" id="sidebar">
    {% for ch in chapters %}
    <div class="chapter-item" data-index="{{ ch.index }}" data-url="{{ ch.audio_url }}" data-title="{{ ch.title|escapejs }}">
      <span class="chapter-num">{{ ch.index }}</span>
      <span class="chapter-title">{{ ch.title }}</span>
    </div>
    {% endfor %}
  </div>
  <div class="main" id="main-hint">Choose a chapter from the list</div>
</div>

<script>
(function () {
  var STORAGE_KEY = 'reader_pos_{{ book.content_hash }}';
  var player = document.getElementById('player');
  var nowPlaying = document.getElementById('now-playing');
  var items = Array.from(document.querySelectorAll('.chapter-item'));

  function setActive(item) {
    items.forEach(function(i) { i.classList.remove('active'); });
    item.classList.add('active');
    item.scrollIntoView({ block: 'nearest' });
  }

  function loadChapter(item, autoplay) {
    setActive(item);
    player.src = item.dataset.url;
    nowPlaying.textContent = item.dataset.title;
    document.getElementById('main-hint').style.display = 'none';
    if (autoplay) {
      player.play();
    }
    savePos(item.dataset.index, 0);
  }

  function nextChapter(currentIndex) {
    var next = items.find(function(i) { return parseInt(i.dataset.index) > parseInt(currentIndex); });
    return next || null;
  }

  function savePos(chapterIndex, time) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ chapter: chapterIndex, time: time }));
    } catch(e) {}
  }

  function loadPos() {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY));
    } catch(e) { return null; }
  }

  // Chapter click
  items.forEach(function(item) {
    item.addEventListener('click', function() { loadChapter(item, true); });
  });

  // Auto-advance
  player.addEventListener('ended', function() {
    var active = document.querySelector('.chapter-item.active');
    var next = active ? nextChapter(active.dataset.index) : null;
    if (next) {
      loadChapter(next, true);
    }
  });

  // Save position every 5 seconds
  player.addEventListener('timeupdate', function() {
    var active = document.querySelector('.chapter-item.active');
    if (active && Math.floor(player.currentTime) % 5 === 0 && player.currentTime > 0) {
      savePos(active.dataset.index, player.currentTime);
    }
  });

  // Resume from localStorage
  var saved = loadPos();
  if (saved) {
    var resumeItem = items.find(function(i) { return String(i.dataset.index) === String(saved.chapter); });
    if (resumeItem) {
      loadChapter(resumeItem, false);
      player.addEventListener('loadedmetadata', function onLoad() {
        player.currentTime = saved.time || 0;
        player.removeEventListener('loadedmetadata', onLoad);
      });
    }
  } else if (items.length > 0) {
    loadChapter(items[0], false);
  }
})();
</script>
</body>
</html>
```

- [ ] **Step 2: Run Django check and full test suite**

```bash
cd /Users/ehughes/code/claude/mnt/reader && python manage.py check && python -m pytest tests/test_views.py::ListenListViewTest tests/test_views.py::ListenBookViewTest -v 2>&1 | tail -15
```

Expected: all listen tests PASS

- [ ] **Step 3: Run full suite**

```bash
cd /Users/ehughes/code/claude/mnt/reader && python -m pytest -q --ignore=tests/test_views.py 2>&1 | tail -3
```

Expected: all PASS, 1 skipped

- [ ] **Step 4: Manual smoke test**

Start the server and verify:
1. `http://localhost:8000/listen/` — shows only books with compiled audio
2. Click a book — player page loads with chapter list
3. Click a chapter — audio plays
4. Audio ends — next chapter auto-loads and plays
5. Refresh page — resumes from saved chapter and position
6. Upload page shows "▶ Listen" link in top right
7. Results page header shows "▶ Listen" link
