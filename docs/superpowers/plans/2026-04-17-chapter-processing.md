# Chapter Processing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Process multi-chapter books one chapter at a time, storing per-chapter output under `chapters/NN/` while keeping `speakers.txt` and `voices/` at the root and accumulating speakers across chapters.

**Architecture:** Single-chapter books use the existing layout and `run_pipeline` unchanged. Multi-chapter books are detected at upload time (EPUB spine or regex), written to `chapters.json`, and processed by a new `run_book_pipeline` that iterates chapters sequentially, passing cumulative speaker context to the LLM at every step. The results page gains a chapter selector that loads annotated content lazily.

**Tech Stack:** Django, ebooklib, BeautifulSoup, tiktoken, OpenAI GPT-4o, existing TTS stack

---

## File Map

| Action | File | What changes |
|--------|------|--------------|
| Modify | `reader/ingestion.py` | Add `split_epub_chapters`, `split_text_chapters` |
| Modify | `reader/output.py` | Fix `write_speakers` to preserve NARRATOR attrs |
| Modify | `reader/llm.py` | Add `known_speakers` param to `extract_speakers` |
| Modify | `reader/pipeline.py` | Add `run_book_pipeline` |
| Modify | `reader/compile.py` | Add `chapter` param to `run_compile` |
| Modify | `reader/views.py` | Update `process_view`, `stream_view`, `results_view`; add `chapter_content_view`, update compile/audio views |
| Modify | `reader/urls.py` | Add chapter-aware URL patterns |
| Modify | `reader/templates/reader/progress.html` | Handle chapter SSE events |
| Modify | `reader/templates/reader/results.html` | Add chapter selector |
| Modify | `tests/test_ingestion.py` | Tests for chapter detection |
| Modify | `tests/test_output.py` | Tests for updated `write_speakers` |
| Modify | `tests/test_llm.py` | Tests for `known_speakers` param |
| Create | `tests/test_pipeline_book.py` | Tests for `run_book_pipeline` |

---

## Task 1: Chapter detection

**Files:**
- Modify: `reader/ingestion.py`
- Modify: `tests/test_ingestion.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_ingestion.py`:

```python
from reader.ingestion import split_text_chapters, split_epub_chapters


def test_split_text_chapters_detects_chapter_headings():
    text = "Chapter 1\n\nFirst chapter content.\n\nChapter 2\n\nSecond chapter content."
    chapters = split_text_chapters(text)
    assert chapters is not None
    assert len(chapters) == 2
    assert chapters[0]["title"] == "Chapter 1"
    assert "First chapter content." in chapters[0]["text"]
    assert chapters[1]["title"] == "Chapter 2"


def test_split_text_chapters_returns_none_when_no_chapters():
    text = "Just some plain text without any chapter markers."
    assert split_text_chapters(text) is None


def test_split_text_chapters_handles_roman_numerals():
    text = "Chapter I\n\nFirst.\n\nChapter II\n\nSecond."
    chapters = split_text_chapters(text)
    assert chapters is not None
    assert len(chapters) == 2


def test_split_text_chapters_handles_part_headings():
    text = "Part I\n\nContent.\n\nPart II\n\nMore content."
    chapters = split_text_chapters(text)
    assert chapters is not None
    assert len(chapters) == 2


def test_split_epub_chapters_returns_one_per_spine_item():
    long_text = "Word " * 150  # > 100 words
    mock_item1 = MagicMock()
    mock_item1.get_content.return_value = f"<html><body><h1>Chapter One</h1><p>{long_text}</p></body></html>".encode()
    mock_item2 = MagicMock()
    mock_item2.get_content.return_value = f"<html><body><h1>Chapter Two</h1><p>{long_text}</p></body></html>".encode()
    mock_book = MagicMock()
    mock_book.get_items_of_type.return_value = [mock_item1, mock_item2]

    with patch("reader.ingestion.epub.read_epub", return_value=mock_book):
        with patch("reader.ingestion.tempfile.NamedTemporaryFile") as mock_tmpfile:
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_file.name = "/tmp/fake.epub"
            mock_tmpfile.return_value = mock_file
            with patch("reader.ingestion.os.unlink"):
                chapters = split_epub_chapters(b"PK fake epub")

    assert len(chapters) == 2
    assert chapters[0]["title"] == "Chapter One"
    assert chapters[1]["title"] == "Chapter Two"


def test_split_epub_chapters_merges_short_items():
    short_text = "Word " * 50  # < 100 words
    long_text = "Word " * 150
    mock_item1 = MagicMock()
    mock_item1.get_content.return_value = f"<html><body><p>{short_text}</p></body></html>".encode()
    mock_item2 = MagicMock()
    mock_item2.get_content.return_value = f"<html><body><h1>Real Chapter</h1><p>{long_text}</p></body></html>".encode()
    mock_book = MagicMock()
    mock_book.get_items_of_type.return_value = [mock_item1, mock_item2]

    with patch("reader.ingestion.epub.read_epub", return_value=mock_book):
        with patch("reader.ingestion.tempfile.NamedTemporaryFile") as mock_tmpfile:
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_file.name = "/tmp/fake.epub"
            mock_tmpfile.return_value = mock_file
            with patch("reader.ingestion.os.unlink"):
                chapters = split_epub_chapters(b"PK fake epub")

    assert len(chapters) == 1
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_ingestion.py::test_split_text_chapters_detects_chapter_headings tests/test_ingestion.py::test_split_epub_chapters_returns_one_per_spine_item -v
```

Expected: FAIL with `ImportError: cannot import name 'split_text_chapters'`

- [ ] **Step 3: Implement in `reader/ingestion.py`**

Add after the existing imports:

```python
import re as _re

_CHAPTER_RE = _re.compile(
    r'^((?:chapter|part)\s+(?:\d+|[ivxlcdmIVXLCDM]+|one|two|three|four|five|six|'
    r'seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|'
    r'seventeen|eighteen|nineteen|twenty)[^\n]*)',
    _re.IGNORECASE | _re.MULTILINE,
)


def split_text_chapters(text: str) -> list[dict] | None:
    matches = list(_CHAPTER_RE.finditer(text))
    if not matches:
        return None
    chapters = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chapter_text = text[start:end].strip()
        title = match.group(1).strip()[:80]
        chapters.append({"title": title, "text": chapter_text})
    return chapters or None


def split_epub_chapters(file_bytes: bytes) -> list[dict]:
    with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as f:
        f.write(file_bytes)
        tmp_path = f.name
    try:
        book = epub.read_epub(tmp_path)
        raw_items = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text = soup.get_text().strip()
            if text:
                raw_items.append(text)
    finally:
        os.unlink(tmp_path)

    if not raw_items:
        return []

    # Merge items under 100 words into the next one (front matter, TOC, etc.)
    merged: list[str] = []
    buffer = ""
    for item in raw_items:
        if buffer:
            buffer += "\n\n" + item
            if len(buffer.split()) >= 100:
                merged.append(buffer)
                buffer = ""
        elif len(item.split()) < 100:
            buffer = item
        else:
            merged.append(item)
    if buffer:
        if merged:
            merged[-1] += "\n\n" + buffer
        else:
            merged.append(buffer)

    result = []
    for i, text in enumerate(merged, start=1):
        first_line = next((l.strip() for l in text.splitlines() if l.strip()), "")
        title = first_line[:80] if first_line and len(first_line) <= 80 else f"Chapter {i}"
        result.append({"title": title, "text": text})
    return result
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_ingestion.py -v
```

Expected: all PASS

- [ ] **Step 5: Run full suite**

```bash
python -m pytest -q
```

Expected: all PASS, 1 skipped

- [ ] **Step 6: Skip git — no git in this repo**

---

## Task 2: Fix `write_speakers` and add `known_speakers` to `extract_speakers`

**Files:**
- Modify: `reader/output.py`
- Modify: `reader/llm.py`
- Modify: `tests/test_output.py`
- Modify: `tests/test_llm.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_output.py`:

```python
def test_write_speakers_preserves_narrator_attributes(tmp_path, settings):
    settings.OUTPUTS_DIR = tmp_path
    out_dir = ensure_output_dir("abc123")
    custom_narrator = {"name": "NARRATOR", "sex": "female", "age": "elderly", "traits": "wise"}
    write_speakers([custom_narrator, {"name": "Alice", "sex": "female", "age": "30s"}], out_dir)
    result = read_speakers(out_dir)
    narrator = next(s for s in result if s["name"] == "NARRATOR")
    assert narrator["sex"] == "female"
    assert narrator["age"] == "elderly"
```

Append to `tests/test_llm.py`:

```python
def test_extract_speakers_includes_known_speakers_in_prompt():
    chunk = {"content": '"Hello," said Alice.', "context": ""}
    known = [{"name": "Alice", "sex": "female", "age": "30s"}]
    with patch("reader.llm._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_openai_response('{"speakers": []}')
        extract_speakers(chunk, known_speakers=known)
        prompt = mock_client.chat.completions.create.call_args[1]["messages"][0]["content"]
    assert "Alice" in prompt
    assert "female" in prompt


def test_extract_speakers_no_known_speakers_unchanged():
    chunk = {"content": '"Hi," said Bob.', "context": ""}
    with patch("reader.llm._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_openai_response('{"speakers": []}')
        extract_speakers(chunk, known_speakers=None)
        prompt = mock_client.chat.completions.create.call_args[1]["messages"][0]["content"]
    assert "Known characters" not in prompt
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_output.py::test_write_speakers_preserves_narrator_attributes tests/test_llm.py::test_extract_speakers_includes_known_speakers_in_prompt -v
```

Expected: FAIL

- [ ] **Step 3: Fix `write_speakers` in `reader/output.py`**

Replace the existing `write_speakers`:

```python
def write_speakers(speakers: list[dict], out_dir: Path) -> None:
    narrator = next(
        (s for s in speakers if s["name"] == "NARRATOR"),
        {"name": "NARRATOR", "sex": "unknown", "age": "unknown"},
    )
    others = [s for s in speakers if s["name"] != "NARRATOR"]
    lines = []
    for s in [narrator] + others:
        parts = [s["name"], f"sex={s.get('sex', 'unknown')}", f"age={s.get('age', 'unknown')}"]
        if s.get("nationality"):
            parts.append(f"nationality={s['nationality']}")
        if s.get("traits"):
            parts.append(f"traits={s['traits']}")
        lines.append(" | ".join(parts))
    (out_dir / "speakers.txt").write_text("\n".join(lines), encoding="utf-8")
```

- [ ] **Step 4: Update `_SPEAKER_PROMPT` and `extract_speakers` in `reader/llm.py`**

Replace the existing `_SPEAKER_PROMPT` and `extract_speakers`:

```python
_SPEAKER_PROMPT = """\
Identify every character who speaks in the following passage.
For each character provide:
- name: their full, consistent name as used in the text (e.g. "Elizabeth Bennet" not just "Elizabeth")
- sex: "male", "female", or "unknown"
- age: approximate age range (e.g. "early 20s", "middle-aged", "elderly", "unknown")
- nationality: likely nationality or regional accent inferred from the text, setting, or your knowledge of this character (e.g. "Russian", "American Southern", "British RP", "Irish", "unknown")
- traits: 2-4 personality traits inferred from their speech and actions or anything you know about this character

Return ONLY a JSON object in this exact shape, no commentary:
{{"speakers": [{{"name": "...", "sex": "...", "age": "...", "nationality": "...", "traits": "..."}}]}}

If no characters speak, return: {{"speakers": []}}
{known_block}{context_block}
PASSAGE:
{text}"""


def extract_speakers(chunk: dict, known_speakers: list[dict] | None = None) -> list[dict]:
    context_block = (
        f"\n[Prior context — for reference only]\n{chunk['context']}\n"
        if chunk["context"]
        else ""
    )
    known_block = ""
    if known_speakers:
        names = "\n".join(
            f"- {s['name']} ({s.get('sex', '?')}, {s.get('age', '?')})"
            for s in known_speakers
        )
        known_block = (
            f"\nKnown characters already identified (use exact name spellings if you see them):\n{names}\n"
        )
    prompt = _SPEAKER_PROMPT.format(
        known_block=known_block,
        context_block=context_block,
        text=chunk["content"],
    )
    response = _client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    data = json.loads(response.choices[0].message.content)
    return data.get("speakers", [])
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_output.py tests/test_llm.py -v
```

Expected: all PASS

- [ ] **Step 6: Run full suite**

```bash
python -m pytest -q
```

Expected: all PASS, 1 skipped

---

## Task 3: `run_book_pipeline`

**Files:**
- Modify: `reader/pipeline.py`
- Create: `tests/test_pipeline_book.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_pipeline_book.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from reader.pipeline import run_book_pipeline

CHAPTERS = [
    {"index": 1, "title": "Chapter 1", "text": '"Hello," said Alice. It was quiet.'},
    {"index": 2, "title": "Chapter 2", "text": '"Goodbye," said Bob. He left.'},
]

MOCK_SPEAKERS = [{"name": "Alice", "sex": "female", "age": "30s", "traits": "brave", "nationality": "British"}]
MOCK_ANNOTATED = '[NARRATOR] It was quiet.\n[ALICE | mood=happy] "Hello."'


@patch("reader.pipeline.generate_voice_sample")
@patch("reader.pipeline.get_tts_model")
@patch("reader.pipeline.write_annotated")
@patch("reader.pipeline.write_speakers")
@patch("reader.pipeline.annotate_chunk")
@patch("reader.pipeline.extract_speakers")
def test_book_pipeline_yields_done(
    mock_extract, mock_annotate, mock_write_s, mock_write_a,
    mock_get_model, mock_gen_voice, tmp_path, settings
):
    settings.OUTPUTS_DIR = tmp_path
    mock_extract.return_value = MOCK_SPEAKERS
    mock_annotate.return_value = MOCK_ANNOTATED
    mock_get_model.return_value = MagicMock()
    events = list(run_book_pipeline("abc123", CHAPTERS, "Test Book"))
    assert any("done" in e for e in events)


@patch("reader.pipeline.generate_voice_sample")
@patch("reader.pipeline.get_tts_model")
@patch("reader.pipeline.write_annotated")
@patch("reader.pipeline.write_speakers")
@patch("reader.pipeline.annotate_chunk")
@patch("reader.pipeline.extract_speakers")
def test_book_pipeline_yields_chapter_start_and_done_events(
    mock_extract, mock_annotate, mock_write_s, mock_write_a,
    mock_get_model, mock_gen_voice, tmp_path, settings
):
    settings.OUTPUTS_DIR = tmp_path
    mock_extract.return_value = MOCK_SPEAKERS
    mock_annotate.return_value = MOCK_ANNOTATED
    mock_get_model.return_value = MagicMock()
    events = list(run_book_pipeline("abc123", CHAPTERS, "Test Book"))
    starts = [e for e in events if "chapter_start" in e]
    dones = [e for e in events if "chapter_done" in e]
    assert len(starts) == 2
    assert len(dones) == 2


@patch("reader.pipeline.generate_voice_sample")
@patch("reader.pipeline.get_tts_model")
@patch("reader.pipeline.write_annotated")
@patch("reader.pipeline.write_speakers")
@patch("reader.pipeline.annotate_chunk")
@patch("reader.pipeline.extract_speakers")
def test_book_pipeline_writes_per_chapter_annotated(
    mock_extract, mock_annotate, mock_write_s, mock_write_a,
    mock_get_model, mock_gen_voice, tmp_path, settings
):
    settings.OUTPUTS_DIR = tmp_path
    mock_extract.return_value = MOCK_SPEAKERS
    mock_annotate.return_value = MOCK_ANNOTATED
    mock_get_model.return_value = MagicMock()
    list(run_book_pipeline("abc123", CHAPTERS, "Test Book"))
    # write_annotated called once per chapter with the chapter_dir
    assert mock_write_a.call_count == 2
    dirs_used = [call[0][1] for call in mock_write_a.call_args_list]
    assert any("01" in str(d) for d in dirs_used)
    assert any("02" in str(d) for d in dirs_used)


@patch("reader.pipeline.generate_voice_sample")
@patch("reader.pipeline.get_tts_model")
@patch("reader.pipeline.write_annotated")
@patch("reader.pipeline.write_speakers")
@patch("reader.pipeline.annotate_chunk")
@patch("reader.pipeline.extract_speakers")
def test_book_pipeline_passes_known_speakers_to_extract(
    mock_extract, mock_annotate, mock_write_s, mock_write_a,
    mock_get_model, mock_gen_voice, tmp_path, settings
):
    settings.OUTPUTS_DIR = tmp_path
    mock_extract.return_value = MOCK_SPEAKERS
    mock_annotate.return_value = MOCK_ANNOTATED
    mock_get_model.return_value = MagicMock()
    list(run_book_pipeline("abc123", CHAPTERS, "Test Book"))
    # Second chapter's extract_speakers should receive known_speakers from chapter 1
    second_chapter_calls = mock_extract.call_args_list[1:]  # calls after first chunk of ch1
    assert any(
        call[1].get("known_speakers") or (len(call[0]) > 1 and call[0][1])
        for call in second_chapter_calls
    )


@patch("reader.pipeline.generate_voice_sample")
@patch("reader.pipeline.get_tts_model")
@patch("reader.pipeline.write_annotated")
@patch("reader.pipeline.write_speakers")
@patch("reader.pipeline.annotate_chunk")
@patch("reader.pipeline.extract_speakers")
def test_book_pipeline_skips_voice_for_existing_speakers(
    mock_extract, mock_annotate, mock_write_s, mock_write_a,
    mock_get_model, mock_gen_voice, tmp_path, settings
):
    settings.OUTPUTS_DIR = tmp_path
    mock_extract.return_value = MOCK_SPEAKERS
    mock_annotate.return_value = MOCK_ANNOTATED
    mock_get_model.return_value = MagicMock()
    # Pre-create alice.wav so her voice should not be regenerated
    voices_dir = tmp_path / "abc123" / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    (voices_dir / "alice.wav").write_bytes(b"fake")
    list(run_book_pipeline("abc123", CHAPTERS, "Test Book"))
    generated_names = [c[0][0]["name"] for c in mock_gen_voice.call_args_list]
    assert "Alice" not in generated_names
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_pipeline_book.py -v
```

Expected: FAIL with `ImportError: cannot import name 'run_book_pipeline'`

- [ ] **Step 3: Implement `run_book_pipeline` in `reader/pipeline.py`**

Add to the existing imports in `reader/pipeline.py`:

```python
from reader.output import ensure_output_dir, normalize_speaker_names, write_speakers, write_annotated, read_speakers
```

Add after `run_pipeline`:

```python
def run_book_pipeline(content_hash: str, chapters: list[dict], title: str):
    """Generator that processes a multi-chapter book and yields SSE event strings."""
    try:
        out_dir = ensure_output_dir(content_hash)
        voices_dir = out_dir / "voices"
        voices_dir.mkdir(exist_ok=True)
        total_chapters = len(chapters)
        pad = len(str(total_chapters))

        # Load existing speakers (preserving any custom NARRATOR attributes)
        if (out_dir / "speakers.txt").exists():
            all_existing = read_speakers(out_dir)
            narrator_entry = next(
                (s for s in all_existing if s["name"] == "NARRATOR"),
                {"name": "NARRATOR", "sex": "unknown", "age": "unknown", "traits": ""},
            )
            known_speakers = [s for s in all_existing if s["name"] != "NARRATOR"]
        else:
            narrator_entry = {"name": "NARRATOR", "sex": "unknown", "age": "unknown", "traits": ""}
            known_speakers = []

        for i, chapter in enumerate(chapters, start=1):
            chapter_title = chapter["title"]
            chapter_text = chapter["text"]
            chapter_dir = out_dir / "chapters" / str(i).zfill(pad)
            chapter_dir.mkdir(parents=True, exist_ok=True)

            yield f"data: chapter_start {i} {total_chapters} {chapter_title}\n\n"
            (chapter_dir / "raw.txt").write_text(chapter_text, encoding="utf-8")

            # Pass 1: extract speakers, merge into cumulative list
            chunks = chunk_text(chapter_text)
            chapter_total = len(chunks)
            new_from_chapter = []
            for chunk in chunks:
                speakers = extract_speakers(chunk, known_speakers=known_speakers)
                new_from_chapter.extend(speakers)

            merged_chapter = merge_speakers([new_from_chapter])
            known_names_lower = {s["name"].lower() for s in known_speakers}
            truly_new = [s for s in merged_chapter if s["name"].lower() not in known_names_lower]
            known_speakers = known_speakers + truly_new
            write_speakers([narrator_entry] + known_speakers, out_dir)

            # Pass 2: annotate chapter with full cumulative speaker list
            annotated_chunks = []
            for j, chunk in enumerate(chunks, start=1):
                annotated = annotate_chunk(chunk, known_speakers)
                annotated_chunks.append(annotated)
                yield f"data: chunk_progress {j} {chapter_total}\n\n"

            annotated_chunks = normalize_speaker_names(annotated_chunks, known_speakers)
            write_annotated(annotated_chunks, chapter_dir)

            # Pass 3: generate voices only for speakers without an existing WAV
            try:
                get_tts_model()
                yield "data: voices_start\n\n"
                speakers_for_tts = []
                if not (voices_dir / "narrator.wav").exists():
                    speakers_for_tts.append(narrator_entry)
                for speaker in known_speakers:
                    slug = slugify_name(speaker["name"])
                    if not (voices_dir / f"{slug}.wav").exists():
                        speakers_for_tts.append(speaker)
                total_voices = len(speakers_for_tts)
                for k, speaker in enumerate(speakers_for_tts, start=1):
                    try:
                        logger.info("Generating voice for %s", speaker["name"])
                        generate_voice_sample(speaker, voices_dir)
                    except Exception as exc:
                        logger.exception("Failed to generate voice for %s", speaker["name"])
                        yield f"data: voice_warning Failed voice for {speaker['name']}: {exc}\n\n"
                    yield f"data: voice_progress {k} {total_voices}\n\n"
            except Exception as exc:
                logger.exception("Voice generation unavailable")
                yield f"data: voice_warning Voice generation unavailable: {exc}\n\n"

            yield f"data: chapter_done {i} {total_chapters}\n\n"

        yield "data: done\n\n"

    except Exception as exc:
        logger.exception("Book pipeline failed")
        yield f"data: error {exc}\n\n"
```

Also add `slugify_name` to the pipeline imports:

```python
from reader.tts import generate_voice_sample, get_tts_model, slugify_name
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_pipeline_book.py -v
```

Expected: all PASS

- [ ] **Step 5: Run full suite**

```bash
python -m pytest -q
```

Expected: all PASS, 1 skipped

---

## Task 4: `process_view` and `stream_view`

**Files:**
- Modify: `reader/views.py`
- Modify: `tests/test_views.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_views.py`:

```python
class ProcessViewChapterTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_post_chaptered_text_writes_chapters_json(self):
        import tempfile, json
        text = "Chapter 1\n\nFirst chapter.\n\nChapter 2\n\nSecond chapter."
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.settings(OUTPUTS_DIR=tmp_dir):
                self.client.post(reverse("reader:process"), {"input_text": text})
        book = ProcessedBook.objects.first()
        from pathlib import Path
        chapters_path = Path(tmp_dir) / book.content_hash / "chapters.json"
        assert chapters_path.exists()
        data = json.loads(chapters_path.read_text())
        assert len(data) == 2
        assert data[0]["title"] == "Chapter 1"

    def test_post_plain_text_does_not_write_chapters_json(self):
        import tempfile
        text = "Just some plain text without chapters."
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.settings(OUTPUTS_DIR=tmp_dir):
                self.client.post(reverse("reader:process"), {"input_text": text})
        book = ProcessedBook.objects.first()
        from pathlib import Path
        chapters_path = Path(tmp_dir) / book.content_hash / "chapters.json"
        assert not chapters_path.exists()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_views.py::ProcessViewChapterTest -v
```

Expected: FAIL

- [ ] **Step 3: Update `process_view` in `reader/views.py`**

Add import at the top of views.py:

```python
import json
from reader.ingestion import compute_hash, normalize_input, split_text_chapters, split_epub_chapters
from reader.pipeline import run_pipeline, run_book_pipeline
```

Replace the `process_view` function:

```python
def process_view(request):
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")

    input_text = request.POST.get("input_text", "").strip()
    input_file = request.FILES.get("input_file")

    if not input_text and not input_file:
        return HttpResponseBadRequest("Provide text or a file")

    if input_text and input_file:
        return HttpResponseBadRequest("Provide text or a file, not both")

    try:
        if input_file:
            file_bytes = input_file.read()
            text, title = normalize_input(file_bytes=file_bytes, filename=input_file.name, text=None)
            ext = Path(input_file.name).suffix.lower()
            chapters = split_epub_chapters(file_bytes) if ext == ".epub" else split_text_chapters(text)
        else:
            text, title = normalize_input(file_bytes=None, filename=None, text=input_text)
            chapters = split_text_chapters(text)
    except ValueError as exc:
        return HttpResponseBadRequest(str(exc))

    content_hash = compute_hash(text)
    book, created = ProcessedBook.objects.get_or_create(
        content_hash=content_hash,
        defaults={
            "title": title,
            "status": "pending",
            "output_path": f"outputs/{content_hash}/",
        },
    )

    if created:
        out_dir = ensure_output_dir(content_hash)
        (out_dir / "raw.txt").write_text(text, encoding="utf-8")
        if chapters and len(chapters) > 1:
            chapters_data = [
                {"index": i + 1, "title": c["title"], "text": c["text"]}
                for i, c in enumerate(chapters)
            ]
            (out_dir / "chapters.json").write_text(
                json.dumps(chapters_data, ensure_ascii=False), encoding="utf-8"
            )

    if book.status == "done":
        return redirect("reader:results", content_hash=content_hash)

    return redirect("reader:progress", content_hash=content_hash)
```

- [ ] **Step 4: Update `stream_view` in `reader/views.py`**

Replace the `stream_view` function:

```python
def stream_view(request, content_hash):
    book = get_object_or_404(ProcessedBook, content_hash=content_hash)

    if book.status == "done":
        def _done():
            yield "data: done\n\n"
        resp = StreamingHttpResponse(_done(), content_type="text/event-stream")
        resp["Cache-Control"] = "no-cache"
        resp["X-Accel-Buffering"] = "no"
        return resp

    book.status = "processing"
    book.save(update_fields=["status", "updated_at"])

    out_dir = Path(settings.OUTPUTS_DIR) / content_hash
    raw_path = out_dir / "raw.txt"
    text = raw_path.read_text(encoding="utf-8") if raw_path.exists() else ""
    chapters_path = out_dir / "chapters.json"

    if chapters_path.exists():
        chapters = json.loads(chapters_path.read_text(encoding="utf-8"))

        def _event_stream():
            success = True
            for event in run_book_pipeline(content_hash, chapters, book.title):
                yield event
                if "error" in event:
                    success = False
            ProcessedBook.objects.filter(content_hash=content_hash).update(
                status="done" if success else "failed"
            )
    else:
        def _event_stream():
            success = True
            for event in run_pipeline(content_hash, text, book.title):
                yield event
                if "error" in event:
                    success = False
            ProcessedBook.objects.filter(content_hash=content_hash).update(
                status="done" if success else "failed"
            )

    resp = StreamingHttpResponse(_event_stream(), content_type="text/event-stream")
    resp["Cache-Control"] = "no-cache"
    resp["X-Accel-Buffering"] = "no"
    return resp
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_views.py -v
```

Expected: all PASS

- [ ] **Step 6: Run full suite**

```bash
python -m pytest -q
```

Expected: all PASS, 1 skipped

---

## Task 5: `results_view`, `chapter_content_view`, and URL updates

**Files:**
- Modify: `reader/views.py`
- Modify: `reader/urls.py`

- [ ] **Step 1: Update `results_view` in `reader/views.py`**

Replace the existing `results_view`:

```python
def results_view(request, content_hash):
    book = get_object_or_404(ProcessedBook, content_hash=content_hash, status="done")
    out_dir = Path(settings.OUTPUTS_DIR) / content_hash
    speakers = read_speakers(out_dir)
    for speaker in speakers:
        slug = slugify_name(speaker["name"])
        speaker["slug"] = slug
        speaker["has_voice"] = (out_dir / "voices" / f"{slug}.wav").exists()

    chapters_path = out_dir / "chapters.json"
    if chapters_path.exists():
        chapters_meta = json.loads(chapters_path.read_text(encoding="utf-8"))
        pad = len(str(len(chapters_meta)))
        first_chapter_dir = out_dir / "chapters" / "1".zfill(pad)
        annotated_lines = read_annotated(first_chapter_dir)
        has_full_audio = (first_chapter_dir / "compiled" / "full.mp3").exists()
        chapters_data = [{"index": c["index"], "title": c["title"]} for c in chapters_meta]
        current_chapter = 1
    else:
        chapters_data = None
        current_chapter = None
        annotated_lines = read_annotated(out_dir)
        has_full_audio = (out_dir / "compiled" / "full.mp3").exists()

    return render(request, "reader/results.html", {
        "book": book,
        "speakers": speakers,
        "annotated_lines": annotated_lines,
        "has_full_audio": has_full_audio,
        "chapters": chapters_data,
        "current_chapter": current_chapter,
    })
```

- [ ] **Step 2: Add `chapter_content_view` to `reader/views.py`**

Add before `compile_view`:

```python
def chapter_content_view(request, content_hash, chapter):
    get_object_or_404(ProcessedBook, content_hash=content_hash, status="done")
    out_dir = Path(settings.OUTPUTS_DIR) / content_hash
    chapters_path = out_dir / "chapters.json"
    if not chapters_path.exists():
        raise Http404
    chapters_meta = json.loads(chapters_path.read_text(encoding="utf-8"))
    pad = len(str(len(chapters_meta)))
    chapter_dir = out_dir / "chapters" / str(chapter).zfill(pad)
    annotated_lines = read_annotated(chapter_dir)
    has_full_audio = (chapter_dir / "compiled" / "full.mp3").exists()
    lines_data = [
        {"type": line["type"], "speaker": line["speaker"], "mood": line["mood"], "text": line["text"]}
        for line in annotated_lines
    ]
    return JsonResponse({"lines": lines_data, "has_full_audio": has_full_audio})
```

- [ ] **Step 3: Add URLs to `reader/urls.py`**

Add after the existing `compile_stream` pattern:

```python
path("results/<str:content_hash>/chapter/<int:chapter>/", views.chapter_content_view, name="chapter_content"),
path("compile/<str:content_hash>/<int:chapter>/", views.compile_view, name="compile_chapter"),
path("compile/<str:content_hash>/<int:chapter>/stream/", views.compile_stream_view, name="compile_stream_chapter"),
path("audio/<str:content_hash>/<int:chapter>/", views.full_audio_view, name="full_audio_chapter"),
```

- [ ] **Step 4: Update `compile_view`, `compile_stream_view`, `full_audio_view` to accept optional `chapter`**

Replace the three existing view functions:

```python
def compile_view(request, content_hash, chapter=None):
    book = get_object_or_404(ProcessedBook, content_hash=content_hash, status="done")
    return render(request, "reader/compile.html", {"book": book, "chapter": chapter})


def compile_stream_view(request, content_hash, chapter=None):
    get_object_or_404(ProcessedBook, content_hash=content_hash, status="done")
    from reader.compile import run_compile

    resp = StreamingHttpResponse(
        run_compile(content_hash, chapter=chapter), content_type="text/event-stream"
    )
    resp["Cache-Control"] = "no-cache"
    resp["X-Accel-Buffering"] = "no"
    return resp


def full_audio_view(request, content_hash, chapter=None):
    get_object_or_404(ProcessedBook, content_hash=content_hash, status="done")
    out_dir = Path(settings.OUTPUTS_DIR) / content_hash
    if chapter is not None:
        chapters_path = out_dir / "chapters.json"
        if not chapters_path.exists():
            raise Http404
        chapters_meta = json.loads(chapters_path.read_text(encoding="utf-8"))
        pad = len(str(len(chapters_meta)))
        mp3_path = out_dir / "chapters" / str(chapter).zfill(pad) / "compiled" / "full.mp3"
    else:
        mp3_path = out_dir / "compiled" / "full.mp3"
    if not mp3_path.exists():
        raise Http404
    return FileResponse(mp3_path.open("rb"), content_type="audio/mpeg")
```

- [ ] **Step 5: Run Django check**

```bash
python manage.py check
```

Expected: `System check identified no issues (0 silenced).`

- [ ] **Step 6: Run full suite**

```bash
python -m pytest -q
```

Expected: all PASS, 1 skipped

---

## Task 6: Compile with chapter support

**Files:**
- Modify: `reader/compile.py`

- [ ] **Step 1: Update `run_compile` to accept `chapter` param**

In `reader/compile.py`, replace the `run_compile` signature and the first lines that determine `out_dir` and read speakers/annotated:

```python
def run_compile(content_hash: str, chapter: int | None = None):
    """Generator that synthesizes each annotated segment with the speaker's cloned voice and yields SSE strings."""
    try:
        import soundfile as sf

        out_dir = Path(settings.OUTPUTS_DIR) / content_hash

        if chapter is not None:
            chapters_path = out_dir / "chapters.json"
            chapters_meta = json.loads(chapters_path.read_text(encoding="utf-8"))
            pad = len(str(len(chapters_meta)))
            work_dir = out_dir / "chapters" / str(chapter).zfill(pad)
        else:
            work_dir = out_dir

        speakers = read_speakers(out_dir)  # always from root
        annotated_text = (work_dir / "annotated.txt").read_text(encoding="utf-8")
        segments = _parse_segments(annotated_text)
        total = len(segments)

        if total == 0:
            yield "data: done\n\n"
            return

        pad_files = len(str(total))
        voices_dir = out_dir / "voices"  # always from root
        compiled_dir = work_dir / "compiled"
        compiled_dir.mkdir(exist_ok=True)
        # ... rest of existing run_compile body unchanged from get_chatterbox_model() onward
```

The full replacement for `run_compile` (preserving all existing logic):

```python
def run_compile(content_hash: str, chapter: int | None = None):
    """Generator that synthesizes each annotated segment with the speaker's cloned voice and yields SSE strings."""
    try:
        import json as _json
        import soundfile as sf

        out_dir = Path(settings.OUTPUTS_DIR) / content_hash

        if chapter is not None:
            chapters_path = out_dir / "chapters.json"
            chapters_meta = _json.loads(chapters_path.read_text(encoding="utf-8"))
            n_chapters = len(chapters_meta)
            pad = len(str(n_chapters))
            work_dir = out_dir / "chapters" / str(chapter).zfill(pad)
        else:
            work_dir = out_dir

        speakers = read_speakers(out_dir)
        annotated_text = (work_dir / "annotated.txt").read_text(encoding="utf-8")
        segments = _parse_segments(annotated_text)
        total = len(segments)

        if total == 0:
            yield "data: done\n\n"
            return

        pad_files = len(str(total))
        voices_dir = out_dir / "voices"
        compiled_dir = work_dir / "compiled"
        compiled_dir.mkdir(exist_ok=True)

        get_chatterbox_model()

        speaker_wavs = {}
        narrator_wav = None
        for speaker in speakers:
            name = speaker["name"]
            wav_path = voices_dir / f"{slugify_name(name)}.wav"
            if wav_path.exists():
                speaker_wavs[name] = wav_path
                if name == "NARRATOR":
                    narrator_wav = wav_path
            else:
                logger.warning("No WAV for speaker %r (expected %s)", name, wav_path)

        logger.info("speaker_wavs keys: %s", list(speaker_wavs.keys()))
        speaker_wavs_ci = {k.lower(): v for k, v in speaker_wavs.items()}

        if narrator_wav is None:
            yield "data: error Narrator voice file not found. Generate voices first.\n\n"
            return

        pending_wavs = []

        for i, segment in enumerate(segments, start=1):
            try:
                speaker_key = segment["speaker"].lower()
                ref_wav = speaker_wavs_ci.get(speaker_key, narrator_wav)
                slug = slugify_name(segment["speaker"])
                logger.info("segment %d: speaker=%r ref=%s", i, segment["speaker"], ref_wav.name)

                exaggeration = _MOOD_EXAGGERATION.get(segment.get("mood", "").lower(), 0.5)
                wav, sr = synthesize_line(segment["text"], ref_wav, exaggeration=exaggeration)
                filename = f"{str(i).zfill(pad_files)}_{slug}.wav"
                wav_path = compiled_dir / filename
                sf.write(str(wav_path), wav, sr)
                pending_wavs.append(wav_path)
            except Exception as exc:
                logger.exception("Failed to synthesize segment %d", i)
                yield f"data: compile_warning Segment {i} failed: {exc}\n\n"

            yield f"data: compile_progress {i} {total}\n\n"

            if len(pending_wavs) == CONVERT_BATCH_SIZE:
                yield f"data: compile_converting {len(pending_wavs)}\n\n"
                _convert_batch(pending_wavs)
                pending_wavs = []

        if pending_wavs:
            yield f"data: compile_converting {len(pending_wavs)}\n\n"
            _convert_batch(pending_wavs)

        yield "data: compile_finalizing\n\n"
        try:
            _concatenate_mp3s(compiled_dir)
        except Exception as exc:
            logger.exception("Failed to concatenate MP3s")
            yield f"data: compile_warning Could not create full.mp3: {exc}\n\n"

        yield "data: done\n\n"

    except Exception as exc:
        logger.exception("Compile pipeline failed")
        yield f"data: error {exc}\n\n"
```

- [ ] **Step 2: Update compile.html to use chapter-aware stream URL**

In `reader/templates/reader/compile.html`, replace the streamUrl line:

```html
const streamUrl = "{% if chapter %}{% url 'reader:compile_stream_chapter' content_hash=book.content_hash chapter=chapter %}{% else %}{% url 'reader:compile_stream' content_hash=book.content_hash %}{% endif %}";
```

- [ ] **Step 3: Run Django check and full suite**

```bash
python manage.py check && python -m pytest -q
```

Expected: no issues, all PASS, 1 skipped

---

## Task 7: Update `progress.html` for chapter events

**Files:**
- Modify: `reader/templates/reader/progress.html`

- [ ] **Step 1: Update the progress page SSE handler**

In `reader/templates/reader/progress.html`, replace the entire `<script>` block:

```html
<script>
const streamUrl = "{% url 'reader:stream' content_hash=book.content_hash %}";
const resultsUrl = "{% url 'reader:results' content_hash=book.content_hash %}";

const source = new EventSource(streamUrl);
const bar = document.getElementById('bar');
const statusMsg = document.getElementById('status-msg');
const log = document.getElementById('log');
const warnings = document.getElementById('warnings');

var totalChapters = 1;
var currentChapter = 0;

source.onmessage = function(e) {
  const data = e.data.trim();
  if (data === 'parsing') {
    statusMsg.textContent = 'Parsing document...';
    bar.style.width = '5%';
  } else if (data.startsWith('chapter_start ')) {
    const parts = data.split(' ');
    currentChapter = parseInt(parts[1]);
    totalChapters = parseInt(parts[2]);
    const chTitle = parts.slice(3).join(' ');
    statusMsg.textContent = 'Chapter ' + currentChapter + ' of ' + totalChapters + ': ' + chTitle;
    bar.style.width = Math.round(((currentChapter - 1) / totalChapters) * 90) + '%';
  } else if (data.startsWith('chapter_done ')) {
    const parts = data.split(' ');
    bar.style.width = Math.round((parseInt(parts[1]) / parseInt(parts[2])) * 90) + '%';
  } else if (data === 'done') {
    source.close();
    statusMsg.textContent = 'Done! Redirecting...';
    bar.style.width = '100%';
    setTimeout(() => window.location.href = resultsUrl, 600);
  } else if (data.startsWith('chunk_progress ')) {
    const parts = data.split(' ');
    const n = parseInt(parts[1]);
    const total = parseInt(parts[2]);
    if (totalChapters > 1) {
      const base = ((currentChapter - 1) / totalChapters) * 90;
      const span = (1 / totalChapters) * 90 * 0.6;
      bar.style.width = Math.round(base + (n / total) * span) + '%';
    } else {
      bar.style.width = Math.round(10 + (n / total) * 83) + '%';
    }
    statusMsg.textContent = totalChapters > 1
      ? 'Ch ' + currentChapter + ' — annotating chunk ' + n + ' of ' + total + '...'
      : 'Annotating chunk ' + n + ' of ' + total + '...';
    log.textContent = '';
  } else if (data === 'voices_start') {
    statusMsg.textContent = 'Generating character voices...';
  } else if (data.startsWith('voice_progress ')) {
    const parts = data.split(' ');
    statusMsg.textContent = 'Generating voice ' + parts[1] + ' of ' + parts[2] + '...';
  } else if (data.startsWith('voice_warning ')) {
    const p = document.createElement('p');
    p.textContent = '⚠ ' + data.slice('voice_warning '.length);
    warnings.appendChild(p);
  } else if (data.startsWith('error ')) {
    source.close();
    const span = document.createElement('span');
    span.className = 'error';
    span.textContent = 'Error: ' + data.slice(6);
    statusMsg.textContent = '';
    statusMsg.appendChild(span);
  }
};

source.onerror = function() {
  source.close();
  const span = document.createElement('span');
  span.className = 'error';
  span.textContent = 'Connection lost. Please try again.';
  statusMsg.textContent = '';
  statusMsg.appendChild(span);
};
</script>
```

- [ ] **Step 2: Run Django check**

```bash
python manage.py check
```

Expected: no issues

---

## Task 8: Results page chapter selector

**Files:**
- Modify: `reader/templates/reader/results.html`

- [ ] **Step 1: Add chapter selector styles**

In `reader/templates/reader/results.html`, add to the `<style>` block before the closing `</style>`:

```css
  .chapter-bar { padding: 8px 24px; border-bottom: 1px solid #ddd; background: #f5f5f5; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .chapter-bar span { font-size: 0.78rem; color: #999; text-transform: uppercase; letter-spacing: 0.05em; }
  .btn-chapter { font-size: 0.82rem; padding: 4px 12px; border: 1px solid #ccc; border-radius: 3px; background: #fff; cursor: pointer; color: #333; }
  .btn-chapter.active { background: #1a1a1a; color: #fff; border-color: #1a1a1a; }
  .btn-chapter:hover:not(.active) { border-color: #888; }
  .chapter-select { font-size: 0.85rem; border: 1px solid #ccc; border-radius: 3px; padding: 4px 8px; }
```

- [ ] **Step 2: Add chapter selector HTML**

In `reader/templates/reader/results.html`, add after the audio bar block and before `<div class="panels">`:

```html
{% if chapters %}
<div class="chapter-bar">
  <span>Chapter</span>
  {% if chapters|length <= 8 %}
    {% for ch in chapters %}
    <button class="btn-chapter {% if ch.index == current_chapter %}active{% endif %}"
            data-chapter="{{ ch.index }}"
            data-content-url="{% url 'reader:chapter_content' book.content_hash ch.index %}"
            data-compile-url="{% url 'reader:compile_chapter' book.content_hash ch.index %}"
            data-audio-url="{% url 'reader:full_audio_chapter' book.content_hash ch.index %}">
      {{ ch.title }}
    </button>
    {% endfor %}
  {% else %}
  <select class="chapter-select" id="chapter-select">
    {% for ch in chapters %}
    <option value="{{ ch.index }}"
            data-content-url="{% url 'reader:chapter_content' book.content_hash ch.index %}"
            data-compile-url="{% url 'reader:compile_chapter' book.content_hash ch.index %}"
            data-audio-url="{% url 'reader:full_audio_chapter' book.content_hash ch.index %}"
            {% if ch.index == current_chapter %}selected{% endif %}>
      {{ ch.title }}
    </option>
    {% endfor %}
  </select>
  {% endif %}
</div>
{% endif %}
```

- [ ] **Step 3: Update the "Compile audio" and audio links to be dynamic**

In `reader/templates/reader/results.html`, replace the header compile link and audio bar:

```html
<header>
  <h1>{{ book.title }}</h1>
  <a href="{% if current_chapter %}{% url 'reader:compile_chapter' content_hash=book.content_hash chapter=current_chapter %}{% else %}{% url 'reader:compile' content_hash=book.content_hash %}{% endif %}" id="compile-link">Compile audio</a>
  <a href="{% url 'reader:upload' %}">← New book</a>
</header>
{% if has_full_audio %}
<div class="audio-bar" id="audio-bar">
  <audio controls src="{% if current_chapter %}{% url 'reader:full_audio_chapter' book.content_hash current_chapter %}{% else %}{% url 'reader:full_audio' book.content_hash %}{% endif %}" id="audio-player"></audio>
</div>
{% else %}
<div class="audio-bar" id="audio-bar" style="display:none">
  <audio controls id="audio-player"></audio>
</div>
{% endif %}
```

- [ ] **Step 4: Add chapter switching JavaScript**

In `reader/templates/reader/results.html`, add before the closing `})();` of the existing script:

```javascript
  function escHtml(str) {
    return String(str || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  function renderLines(lines) {
    return lines.map(function(line) {
      if (line.type === 'narrator') {
        return '<p class="line narrator">' + escHtml(line.text) + '</p>';
      } else if (line.type === 'dialogue') {
        var mood = line.mood ? '<span class="mood-tag"> (' + escHtml(line.mood) + ')</span>' : '';
        return '<p class="line dialogue"><span class="speaker-tag">' + escHtml(line.speaker) + '</span>' +
               mood + '<span class="dialogue-text">' + escHtml(line.text) + '</span></p>';
      } else {
        return '<p class="line raw">' + escHtml(line.text) + '</p>';
      }
    }).join('');
  }

  function loadChapter(contentUrl, compileUrl, audioUrl) {
    fetch(contentUrl)
      .then(function(r) { return r.json(); })
      .then(function(data) {
        var panel = document.querySelector('.panel-annotated');
        var heading = panel.querySelector('h2');
        panel.innerHTML = '';
        panel.appendChild(heading);
        panel.insertAdjacentHTML('beforeend', renderLines(data.lines));

        var compileLink = document.getElementById('compile-link');
        if (compileLink) compileLink.href = compileUrl;

        var audioBar = document.getElementById('audio-bar');
        var audioPlayer = document.getElementById('audio-player');
        if (data.has_full_audio) {
          audioPlayer.src = audioUrl;
          audioBar.style.display = '';
        } else {
          audioBar.style.display = 'none';
        }
      });
  }

  document.querySelectorAll('.btn-chapter').forEach(function(btn) {
    btn.addEventListener('click', function() {
      document.querySelectorAll('.btn-chapter').forEach(function(b) { b.classList.remove('active'); });
      btn.classList.add('active');
      loadChapter(btn.dataset.contentUrl, btn.dataset.compileUrl, btn.dataset.audioUrl);
    });
  });

  var chapterSelect = document.getElementById('chapter-select');
  if (chapterSelect) {
    chapterSelect.addEventListener('change', function() {
      var opt = chapterSelect.options[chapterSelect.selectedIndex];
      loadChapter(opt.dataset.contentUrl, opt.dataset.compileUrl, opt.dataset.audioUrl);
    });
  }
```

- [ ] **Step 5: Run Django check and full suite**

```bash
python manage.py check && python -m pytest -q
```

Expected: no issues, all PASS, 1 skipped
