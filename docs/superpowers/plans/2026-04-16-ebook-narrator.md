# Ebook Narrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Django web app that parses ebooks/text, uses OpenAI GPT-4o to identify speakers and annotate dialogue, and displays a two-panel narration script inline.

**Architecture:** Single Django project (`config`) with one app (`reader`). File upload triggers a synchronous two-pass LLM pipeline streamed to the browser via SSE. Output is written to a content-addressed directory under `outputs/<sha256>/` and served as a server-rendered two-panel preview.

**Tech Stack:** Django 5, pypdf, ebooklib, beautifulsoup4, openai SDK, tiktoken, vanilla JS (EventSource), SQLite

---

## File Map

```
config/
  __init__.py
  settings.py          # Django settings, OPENAI_API_KEY, OUTPUTS_DIR
  urls.py              # Root URL conf
  wsgi.py

reader/
  __init__.py
  apps.py
  models.py            # ProcessedBook model
  ingestion.py         # Parse PDF/EPUB/text → plain text + SHA-256 hash
  chunker.py           # Split text into chunks with overlap context
  llm.py               # Pass 1 (speaker extraction) + Pass 2 (annotation) via OpenAI
  output.py            # Write/read speakers.txt and annotated.txt
  pipeline.py          # Orchestrates passes, yields SSE event strings
  views.py             # All five views
  urls.py              # App URL conf
  migrations/
    __init__.py
    0001_initial.py
  templates/reader/
    upload.html
    progress.html
    results.html

tests/
  __init__.py
  conftest.py          # pytest fixtures (tmp output dir, sample texts)
  test_ingestion.py
  test_chunker.py
  test_llm.py
  test_output.py
  test_pipeline.py
  test_views.py

outputs/               # Generated at runtime, gitignored
manage.py
requirements.txt
.env.example
.gitignore
pytest.ini
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `pytest.ini`
- Create: `config/settings.py`
- Create: `config/urls.py`
- Create: `config/__init__.py`
- Create: `config/wsgi.py`
- Create: `manage.py`

- [ ] **Step 1: Install Django and scaffold the project**

```bash
cd /Users/ehughes/code/claude/reader
pip install django==5.2
django-admin startproject config .
```

Expected output: `config/` directory and `manage.py` created.

- [ ] **Step 2: Write requirements.txt**

```
django>=5.2,<6.0
pypdf>=4.3
ebooklib>=0.18
beautifulsoup4>=4.12
openai>=1.30
tiktoken>=0.7
python-decouple>=3.8
pytest>=8.0
pytest-django>=4.8
```

- [ ] **Step 3: Install all dependencies**

```bash
pip install -r requirements.txt
```

- [ ] **Step 4: Write .env.example**

```
OPENAI_API_KEY=sk-...
DEBUG=True
SECRET_KEY=change-me
```

- [ ] **Step 5: Write .gitignore**

```
__pycache__/
*.pyc
.env
outputs/
db.sqlite3
*.egg-info/
.venv/
```

- [ ] **Step 6: Write pytest.ini**

```ini
[pytest]
DJANGO_SETTINGS_MODULE = config.settings
python_files = tests/test_*.py
```

- [ ] **Step 7: Update config/settings.py**

Replace the generated file with:

```python
from pathlib import Path
from decouple import config

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = config("SECRET_KEY", default="dev-secret-key-change-in-prod")
DEBUG = config("DEBUG", default=True, cast=bool)
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.staticfiles",
    "reader",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

STATIC_URL = "/static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

OPENAI_API_KEY = config("OPENAI_API_KEY", default="")
OUTPUTS_DIR = BASE_DIR / "outputs"
```

- [ ] **Step 8: Create the reader app**

```bash
python manage.py startapp reader
```

- [ ] **Step 9: Verify Django starts**

```bash
python manage.py check
```

Expected: `System check identified no issues (0 silenced).`

- [ ] **Step 10: Commit**

```bash
git init
git add .
git commit -m "feat: scaffold Django project and reader app"
```

---

## Task 2: ProcessedBook Model

**Files:**
- Modify: `reader/models.py`
- Create: `reader/migrations/0001_initial.py` (via makemigrations)
- Create: `tests/test_views.py` (model smoke test only, full view tests in Task 10)

- [ ] **Step 1: Write the failing test**

Create `tests/__init__.py` (empty) and `tests/test_views.py`:

```python
import pytest
from reader.models import ProcessedBook

@pytest.mark.django_db
def test_processedbook_creation():
    book = ProcessedBook.objects.create(
        content_hash="a" * 64,
        title="Test Book",
        status="pending",
        output_path="outputs/" + "a" * 64 + "/",
    )
    assert book.pk is not None
    assert book.error_message == ""
    assert book.status == "pending"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_views.py::test_processedbook_creation -v
```

Expected: `FAILED` — `No module named 'reader.models'` or similar import error.

- [ ] **Step 3: Write the model**

Replace `reader/models.py`:

```python
from django.db import models


class ProcessedBook(models.Model):
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("processing", "Processing"),
        ("done", "Done"),
        ("failed", "Failed"),
    ]

    content_hash = models.CharField(max_length=64, unique=True)
    title = models.CharField(max_length=255)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    output_path = models.CharField(max_length=500)
    error_message = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.title} ({self.status})"
```

- [ ] **Step 4: Add reader to INSTALLED_APPS and create migration**

`config/settings.py` already has `"reader"` — confirm it's there, then:

```bash
python manage.py makemigrations reader
python manage.py migrate
```

Expected: `Applying reader.0001_initial... OK`

- [ ] **Step 5: Run the test to verify it passes**

```bash
pytest tests/test_views.py::test_processedbook_creation -v
```

Expected: `PASSED`

- [ ] **Step 6: Commit**

```bash
git add reader/models.py reader/migrations/ tests/
git commit -m "feat: add ProcessedBook model"
```

---

## Task 3: Ingestion — Text Parsing and Hashing

**Files:**
- Create: `reader/ingestion.py`
- Create: `tests/test_ingestion.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ingestion.py`:

```python
import hashlib
import pytest
from reader.ingestion import compute_hash, normalize_input


def test_compute_hash_deterministic():
    text = "Hello, world!"
    expected = hashlib.sha256(text.encode()).hexdigest()
    assert compute_hash(text) == expected


def test_compute_hash_differs_for_different_text():
    assert compute_hash("foo") != compute_hash("bar")


def test_normalize_plain_text_returns_text_and_title():
    text = "My Story\n\nOnce upon a time."
    content, title = normalize_input(file_bytes=None, filename=None, text=text)
    assert content == text
    assert title == "My Story"


def test_normalize_txt_file():
    raw = b"A short story.\n\n\"Hello,\" she said."
    content, title = normalize_input(file_bytes=raw, filename="mystory.txt", text=None)
    assert content == raw.decode("utf-8")
    assert title == "mystory"


def test_normalize_requires_file_or_text():
    with pytest.raises(ValueError, match="Either file or text"):
        normalize_input(file_bytes=None, filename=None, text=None)


def test_normalize_rejects_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported file type"):
        normalize_input(file_bytes=b"data", filename="book.mobi", text=None)


def test_normalize_text_title_from_first_line():
    text = "Chapter One\n\nIt was a dark night."
    _, title = normalize_input(file_bytes=None, filename=None, text=text)
    assert title == "Chapter One"


def test_normalize_text_fallback_title():
    text = "   \n\nNo leading title."
    _, title = normalize_input(file_bytes=None, filename=None, text=text)
    assert title == "Untitled"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_ingestion.py -v
```

Expected: All `FAILED` — `ModuleNotFoundError: No module named 'reader.ingestion'`

- [ ] **Step 3: Implement ingestion.py (text/hash portion)**

Create `reader/ingestion.py`:

```python
import hashlib
from pathlib import Path


def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _title_from_text(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:80]
    return "Untitled"


def normalize_input(
    file_bytes: bytes | None,
    filename: str | None,
    text: str | None,
) -> tuple[str, str]:
    if text is not None:
        return text, _title_from_text(text)

    if file_bytes is None or filename is None:
        raise ValueError("Either file or text must be provided")

    ext = Path(filename).suffix.lower()
    title = Path(filename).stem

    if ext == ".txt":
        return file_bytes.decode("utf-8", errors="replace"), title
    elif ext == ".pdf":
        return _parse_pdf(file_bytes), title
    elif ext == ".epub":
        return _parse_epub(file_bytes), title
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _parse_pdf(file_bytes: bytes) -> str:
    raise NotImplementedError("PDF parsing implemented in Task 4")


def _parse_epub(file_bytes: bytes) -> str:
    raise NotImplementedError("EPUB parsing implemented in Task 4")
```

- [ ] **Step 4: Run the text/hash tests**

```bash
pytest tests/test_ingestion.py -v -k "not pdf and not epub"
```

Expected: All non-PDF/EPUB tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add reader/ingestion.py tests/test_ingestion.py
git commit -m "feat: add ingestion module (text parsing and hash)"
```

---

## Task 4: Ingestion — PDF and EPUB Parsing

**Files:**
- Modify: `reader/ingestion.py`
- Modify: `tests/test_ingestion.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ingestion.py`:

```python
from unittest.mock import MagicMock, patch


def test_normalize_pdf_calls_pypdf(tmp_path):
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Page one text."
    mock_reader = MagicMock()
    mock_reader.pages = [mock_page]

    with patch("reader.ingestion.pypdf.PdfReader", return_value=mock_reader):
        content, title = normalize_input(
            file_bytes=b"%PDF-fake",
            filename="novel.pdf",
            text=None,
        )

    assert content == "Page one text."
    assert title == "novel"


def test_normalize_pdf_joins_multiple_pages(tmp_path):
    pages = [MagicMock(), MagicMock()]
    pages[0].extract_text.return_value = "Page one."
    pages[1].extract_text.return_value = "Page two."
    mock_reader = MagicMock()
    mock_reader.pages = pages

    with patch("reader.ingestion.pypdf.PdfReader", return_value=mock_reader):
        content, _ = normalize_input(
            file_bytes=b"%PDF-fake",
            filename="novel.pdf",
            text=None,
        )

    assert "Page one." in content
    assert "Page two." in content


def test_normalize_epub_extracts_text():
    mock_item = MagicMock()
    mock_item.get_content.return_value = b"<html><body><p>Chapter text.</p></body></html>"
    mock_book = MagicMock()
    mock_book.get_items_of_type.return_value = [mock_item]

    with patch("reader.ingestion.epub.read_epub", return_value=mock_book):
        with patch("reader.ingestion.tempfile.NamedTemporaryFile"):
            with patch("reader.ingestion.os.unlink"):
                content, title = normalize_input(
                    file_bytes=b"PK fake epub",
                    filename="book.epub",
                    text=None,
                )

    assert "Chapter text." in content
    assert title == "book"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_ingestion.py::test_normalize_pdf_calls_pypdf tests/test_ingestion.py::test_normalize_epub_extracts_text -v
```

Expected: `FAILED` — `NotImplementedError: PDF parsing implemented in Task 4`

- [ ] **Step 3: Implement PDF and EPUB parsing in ingestion.py**

Replace the `_parse_pdf` and `_parse_epub` stubs and add required imports:

```python
import hashlib
import io
import os
import tempfile
from pathlib import Path

import pypdf
from bs4 import BeautifulSoup
from ebooklib import epub
import ebooklib


def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _title_from_text(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:80]
    return "Untitled"


def normalize_input(
    file_bytes: bytes | None,
    filename: str | None,
    text: str | None,
) -> tuple[str, str]:
    if text is not None:
        return text, _title_from_text(text)

    if file_bytes is None or filename is None:
        raise ValueError("Either file or text must be provided")

    ext = Path(filename).suffix.lower()
    title = Path(filename).stem

    if ext == ".txt":
        return file_bytes.decode("utf-8", errors="replace"), title
    elif ext == ".pdf":
        return _parse_pdf(file_bytes), title
    elif ext == ".epub":
        return _parse_epub(file_bytes), title
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _parse_pdf(file_bytes: bytes) -> str:
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _parse_epub(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as f:
        f.write(file_bytes)
        tmp_path = f.name
    try:
        book = epub.read_epub(tmp_path)
        texts = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), "html.parser")
            texts.append(soup.get_text())
        return "\n\n".join(texts)
    finally:
        os.unlink(tmp_path)
```

- [ ] **Step 4: Run all ingestion tests**

```bash
pytest tests/test_ingestion.py -v
```

Expected: All `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add reader/ingestion.py tests/test_ingestion.py
git commit -m "feat: add PDF and EPUB parsing to ingestion module"
```

---

## Task 5: Chunker

**Files:**
- Create: `reader/chunker.py`
- Create: `tests/test_chunker.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_chunker.py`:

```python
import pytest
from reader.chunker import chunk_text


SHORT_TEXT = "Hello world.\n\nThis is a short story.\n\nThe end."


def test_short_text_returns_single_chunk():
    chunks = chunk_text(SHORT_TEXT)
    assert len(chunks) == 1
    assert chunks[0]["content"] == SHORT_TEXT.strip() or SHORT_TEXT in chunks[0]["content"]


def test_single_chunk_has_empty_context():
    chunks = chunk_text(SHORT_TEXT)
    assert chunks[0]["context"] == ""


def test_large_text_splits_into_multiple_chunks():
    # Build a text that definitely exceeds one chunk (3000 tokens ≈ 12000 chars)
    paragraph = "A" * 500 + "\n\n"
    big_text = paragraph * 40  # ~20000 chars
    chunks = chunk_text(big_text)
    assert len(chunks) >= 2


def test_second_chunk_has_context_from_first():
    paragraph = "A" * 500 + "\n\n"
    big_text = paragraph * 40
    chunks = chunk_text(big_text)
    assert len(chunks[1]["context"]) > 0


def test_chunks_cover_all_content():
    paragraph = "Para {:03d}.\n\n"
    big_text = "".join(paragraph.format(i) for i in range(200))
    chunks = chunk_text(big_text)
    all_content = " ".join(c["content"] for c in chunks)
    # Every paragraph should appear in some chunk's content
    for i in range(200):
        assert f"Para {i:03d}" in all_content


def test_empty_text_returns_empty_list():
    assert chunk_text("") == []


def test_whitespace_only_returns_empty_list():
    assert chunk_text("   \n\n   ") == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_chunker.py -v
```

Expected: All `FAILED` — `ModuleNotFoundError: No module named 'reader.chunker'`

- [ ] **Step 3: Implement chunker.py**

Create `reader/chunker.py`:

```python
import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")
CHUNK_TOKENS = 3000
OVERLAP_TOKENS = 200


def chunk_text(text: str) -> list[dict]:
    """Split text into chunks. Returns list of {"content": str, "context": str}."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    raw_chunks = _split_paragraphs(paragraphs)

    result = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0:
            context = ""
        else:
            prev_tokens = _ENCODING.encode(raw_chunks[i - 1])
            overlap = prev_tokens[-OVERLAP_TOKENS:]
            context = _ENCODING.decode(overlap)
        result.append({"content": chunk, "context": context})
    return result


def _split_paragraphs(paragraphs: list[str]) -> list[str]:
    chunks = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = len(_ENCODING.encode(para))
        if current and current_tokens + para_tokens > CHUNK_TOKENS:
            chunks.append("\n\n".join(current))
            current = []
            current_tokens = 0
        current.append(para)
        current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))
    return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_chunker.py -v
```

Expected: All `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add reader/chunker.py tests/test_chunker.py
git commit -m "feat: add chunker with tiktoken-based paragraph splitting"
```

---

## Task 6: LLM — Speaker Extraction (Pass 1)

**Files:**
- Create: `reader/llm.py`
- Create: `tests/test_llm.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_llm.py`:

```python
import json
import pytest
from unittest.mock import patch, MagicMock
from reader.llm import extract_speakers, merge_speakers


def _mock_openai_response(content: str):
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


SPEAKER_JSON = json.dumps({
    "speakers": [
        {"name": "Alice", "sex": "female", "age": "30s", "traits": "brave, kind"},
        {"name": "Bob", "sex": "male", "age": "40s", "traits": "gruff, loyal"},
    ]
})


def test_extract_speakers_parses_llm_response():
    chunk = {"content": "\"Hello,\" said Alice. \"Indeed,\" said Bob.", "context": ""}
    with patch("reader.llm._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_openai_response(SPEAKER_JSON)
        speakers = extract_speakers(chunk)
    assert len(speakers) == 2
    assert speakers[0]["name"] == "Alice"
    assert speakers[1]["sex"] == "male"


def test_extract_speakers_includes_context_in_prompt():
    chunk = {"content": "\"Yes,\" she said.", "context": "Prior scene text."}
    with patch("reader.llm._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_openai_response('{"speakers": []}')
        extract_speakers(chunk)
        call_args = mock_client.chat.completions.create.call_args
        prompt = call_args[1]["messages"][0]["content"]
    assert "Prior scene text." in prompt


def test_extract_speakers_returns_empty_on_no_speakers():
    chunk = {"content": "It was a stormy night.", "context": ""}
    with patch("reader.llm._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_openai_response('{"speakers": []}')
        result = extract_speakers(chunk)
    assert result == []


def test_merge_speakers_deduplicates_by_name_case_insensitive():
    all_speakers = [
        [{"name": "Alice", "sex": "female", "age": "30s", "traits": "brave"}],
        [{"name": "alice", "sex": "female", "age": "30s", "traits": "brave"}],
        [{"name": "Bob", "sex": "male", "age": "40s", "traits": "gruff"}],
    ]
    merged = merge_speakers(all_speakers)
    names = [s["name"] for s in merged]
    assert len(merged) == 2
    assert "Alice" in names
    assert "Bob" in names


def test_merge_speakers_first_occurrence_wins():
    all_speakers = [
        [{"name": "Alice", "sex": "female", "age": "20s", "traits": "shy"}],
        [{"name": "ALICE", "sex": "female", "age": "30s", "traits": "bold"}],
    ]
    merged = merge_speakers(all_speakers)
    assert merged[0]["age"] == "20s"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_llm.py -v -k "extract or merge"
```

Expected: All `FAILED` — `ModuleNotFoundError: No module named 'reader.llm'`

- [ ] **Step 3: Implement speaker extraction in llm.py**

Create `reader/llm.py`:

```python
import json
from django.conf import settings
from openai import OpenAI

_client = OpenAI(api_key=settings.OPENAI_API_KEY)

_SPEAKER_PROMPT = """\
Identify every character who speaks in the following passage.
For each character provide:
- name: their full, consistent name as used in the text (e.g. "Elizabeth Bennet" not just "Elizabeth")
- sex: "male", "female", or "unknown"
- age: approximate age range (e.g. "early 20s", "middle-aged", "elderly", "unknown")
- traits: 2-3 personality traits inferred from their speech and actions

Return ONLY a JSON object in this exact shape, no commentary:
{{"speakers": [{{"name": "...", "sex": "...", "age": "...", "traits": "..."}}]}}

If no characters speak, return: {{"speakers": []}}
{context_block}
PASSAGE:
{text}"""

_ANNOTATE_PROMPT = """\
Annotate the following passage for narration. Apply a prefix tag to every segment.

Rules:
- Dialogue spoken by a known character: [CHARACTER NAME | mood=X] followed by the text
- All other text (narration, action, description): [NARRATOR] followed by the text
- Use ONLY names from the character list below, spelled exactly as listed
- mood must be a single descriptive word (e.g. happy, angry, sad, nervous, cold, teasing, formal)
- Return the annotated passage only, no commentary, no explanation

Known characters:
{speaker_list}

PASSAGE:
{text}"""


def extract_speakers(chunk: dict) -> list[dict]:
    context_block = (
        f"\n[Prior context — for reference only]\n{chunk['context']}\n"
        if chunk["context"]
        else ""
    )
    prompt = _SPEAKER_PROMPT.format(context_block=context_block, text=chunk["content"])
    response = _client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    data = json.loads(response.choices[0].message.content)
    return data.get("speakers", [])


def merge_speakers(all_speakers: list[list[dict]]) -> list[dict]:
    seen: dict[str, dict] = {}
    for chunk_speakers in all_speakers:
        for speaker in chunk_speakers:
            key = speaker["name"].lower()
            if key not in seen:
                seen[key] = speaker
    return list(seen.values())


def annotate_chunk(chunk: dict, speakers: list[dict]) -> str:
    raise NotImplementedError("Annotation implemented in Task 7")
```

- [ ] **Step 4: Run speaker extraction tests**

```bash
pytest tests/test_llm.py -v -k "extract or merge"
```

Expected: All `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add reader/llm.py tests/test_llm.py
git commit -m "feat: add LLM speaker extraction (pass 1)"
```

---

## Task 7: LLM — Dialogue Annotation (Pass 2)

**Files:**
- Modify: `reader/llm.py`
- Modify: `tests/test_llm.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_llm.py`:

```python
from reader.llm import annotate_chunk

SPEAKERS = [
    {"name": "Alice", "sex": "female", "age": "30s", "traits": "brave"},
    {"name": "Bob", "sex": "male", "age": "40s", "traits": "gruff"},
]

ANNOTATED_RESPONSE = (
    '[NARRATOR] It was a quiet evening.\n'
    '[ALICE | mood=nervous] "Are you sure about this?"\n'
    '[BOB | mood=gruff] "Absolutely," he said.'
)


def test_annotate_chunk_returns_llm_response():
    chunk = {"content": 'It was a quiet evening. "Are you sure?" "Absolutely."', "context": ""}
    with patch("reader.llm._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_openai_response(ANNOTATED_RESPONSE)
        result = annotate_chunk(chunk, SPEAKERS)
    assert result == ANNOTATED_RESPONSE


def test_annotate_chunk_includes_speaker_names_in_prompt():
    chunk = {"content": '"Hello," she said.', "context": ""}
    with patch("reader.llm._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_openai_response("[NARRATOR] text")
        annotate_chunk(chunk, SPEAKERS)
        call_args = mock_client.chat.completions.create.call_args
        prompt = call_args[1]["messages"][0]["content"]
    assert "Alice" in prompt
    assert "Bob" in prompt


def test_annotate_chunk_uses_zero_temperature():
    chunk = {"content": "text", "context": ""}
    with patch("reader.llm._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_openai_response("[NARRATOR] text")
        annotate_chunk(chunk, SPEAKERS)
        call_args = mock_client.chat.completions.create.call_args
    assert call_args[1]["temperature"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_llm.py::test_annotate_chunk_returns_llm_response -v
```

Expected: `FAILED` — `NotImplementedError: Annotation implemented in Task 7`

- [ ] **Step 3: Implement annotate_chunk in llm.py**

Replace the `annotate_chunk` stub:

```python
def annotate_chunk(chunk: dict, speakers: list[dict]) -> str:
    speaker_list = "\n".join(f"- {s['name']}" for s in speakers)
    prompt = _ANNOTATE_PROMPT.format(speaker_list=speaker_list, text=chunk["content"])
    response = _client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content
```

- [ ] **Step 4: Run all LLM tests**

```bash
pytest tests/test_llm.py -v
```

Expected: All `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add reader/llm.py tests/test_llm.py
git commit -m "feat: add LLM dialogue annotation (pass 2)"
```

---

## Task 8: Output Writer

**Files:**
- Create: `reader/output.py`
- Create: `tests/test_output.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_output.py`:

```python
import re
import pytest
from pathlib import Path
from reader.output import (
    ensure_output_dir,
    write_speakers,
    write_annotated,
    read_speakers,
    read_annotated,
    parse_annotated_line,
)

SPEAKERS = [
    {"name": "Alice", "sex": "female", "age": "30s", "traits": "brave, kind"},
    {"name": "Bob", "sex": "male", "age": "40s", "traits": "gruff"},
]

ANNOTATED_CHUNKS = [
    "[NARRATOR] It was quiet.\n[ALICE | mood=nervous] \"Hello?\"",
    "[BOB | mood=cold] \"Stay back.\"\n[NARRATOR] He stepped forward.",
]


def test_ensure_output_dir_creates_directory(tmp_path, settings):
    settings.OUTPUTS_DIR = tmp_path
    out_dir = ensure_output_dir("abc123")
    assert out_dir.is_dir()
    assert out_dir.name == "abc123"


def test_write_and_read_speakers(tmp_path, settings):
    settings.OUTPUTS_DIR = tmp_path
    out_dir = ensure_output_dir("abc123")
    write_speakers(SPEAKERS, out_dir)
    result = read_speakers(out_dir)
    assert len(result) == 2
    assert result[0]["name"] == "Alice"
    assert result[0]["sex"] == "female"
    assert result[1]["name"] == "Bob"


def test_speakers_file_includes_narrator(tmp_path, settings):
    settings.OUTPUTS_DIR = tmp_path
    out_dir = ensure_output_dir("abc123")
    write_speakers(SPEAKERS, out_dir)
    text = (out_dir / "speakers.txt").read_text()
    assert "NARRATOR" in text


def test_write_and_read_annotated(tmp_path, settings):
    settings.OUTPUTS_DIR = tmp_path
    out_dir = ensure_output_dir("abc123")
    write_annotated(ANNOTATED_CHUNKS, out_dir)
    lines = read_annotated(out_dir)
    assert len(lines) > 0
    assert any(line["type"] == "narrator" for line in lines)
    assert any(line["type"] == "dialogue" for line in lines)


def test_parse_annotated_line_narrator():
    line = "[NARRATOR] It was a dark night."
    result = parse_annotated_line(line)
    assert result["type"] == "narrator"
    assert result["text"] == "It was a dark night."
    assert result["speaker"] is None
    assert result["mood"] is None


def test_parse_annotated_line_dialogue():
    line = '[ALICE | mood=nervous] "Hello?"'
    result = parse_annotated_line(line)
    assert result["type"] == "dialogue"
    assert result["speaker"] == "ALICE"
    assert result["mood"] == "nervous"
    assert result["text"] == '"Hello?"'


def test_parse_annotated_line_unrecognized_falls_back_to_raw():
    line = "Some plain text with no tag."
    result = parse_annotated_line(line)
    assert result["type"] == "raw"
    assert result["text"] == line
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_output.py -v
```

Expected: All `FAILED` — `ModuleNotFoundError: No module named 'reader.output'`

- [ ] **Step 3: Implement output.py**

Create `reader/output.py`:

```python
import re
from pathlib import Path
from django.conf import settings as django_settings


_LINE_RE = re.compile(r"^\[([^\]]+)\]\s*(.*)", re.DOTALL)
_SPEAKER_TAG_RE = re.compile(r"^([^|]+?)(?:\s*\|\s*mood=(.+))?$")


def ensure_output_dir(content_hash: str) -> Path:
    out_dir = Path(django_settings.OUTPUTS_DIR) / content_hash
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_speakers(speakers: list[dict], out_dir: Path) -> None:
    lines = ["NARRATOR | sex=unknown | age=unknown"]
    for s in speakers:
        parts = [s["name"], f"sex={s.get('sex', 'unknown')}", f"age={s.get('age', 'unknown')}"]
        if s.get("traits"):
            parts.append(f"traits={s['traits']}")
        lines.append(" | ".join(parts))
    (out_dir / "speakers.txt").write_text("\n".join(lines), encoding="utf-8")


def write_annotated(chunks: list[str], out_dir: Path) -> None:
    full_text = "\n".join(chunks)
    (out_dir / "annotated.txt").write_text(full_text, encoding="utf-8")


def read_speakers(out_dir: Path) -> list[dict]:
    text = (out_dir / "speakers.txt").read_text(encoding="utf-8")
    result = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if not parts:
            continue
        entry: dict = {"name": parts[0]}
        for part in parts[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                entry[k.strip()] = v.strip()
        result.append(entry)
    return result


def read_annotated(out_dir: Path) -> list[dict]:
    text = (out_dir / "annotated.txt").read_text(encoding="utf-8")
    return [parse_annotated_line(line) for line in text.splitlines() if line.strip()]


def parse_annotated_line(line: str) -> dict:
    m = _LINE_RE.match(line.strip())
    if not m:
        return {"type": "raw", "speaker": None, "mood": None, "text": line}

    tag = m.group(1).strip()
    text = m.group(2).strip()

    if tag == "NARRATOR":
        return {"type": "narrator", "speaker": None, "mood": None, "text": text}

    sm = _SPEAKER_TAG_RE.match(tag)
    if sm:
        return {
            "type": "dialogue",
            "speaker": sm.group(1).strip(),
            "mood": sm.group(2).strip() if sm.group(2) else None,
            "text": text,
        }

    return {"type": "raw", "speaker": None, "mood": None, "text": line}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_output.py -v
```

Expected: All `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add reader/output.py tests/test_output.py
git commit -m "feat: add output writer (speakers.txt and annotated.txt)"
```

---

## Task 9: Pipeline Orchestrator

**Files:**
- Create: `reader/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pipeline.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from reader.pipeline import run_pipeline


SHORT_TEXT = (
    "It was a quiet evening.\n\n"
    '"Hello?" said Alice.\n\n'
    '"Stay back," Bob replied.'
)

MOCK_SPEAKERS = [{"name": "Alice", "sex": "female", "age": "30s", "traits": "brave"}]
MOCK_ANNOTATED = '[NARRATOR] It was quiet.\n[ALICE | mood=nervous] "Hello?"'


def _patch_pipeline(mock_extract, mock_annotate, mock_write_s, mock_write_a):
    mock_extract.return_value = MOCK_SPEAKERS
    mock_annotate.return_value = MOCK_ANNOTATED


@patch("reader.pipeline.write_annotated")
@patch("reader.pipeline.write_speakers")
@patch("reader.pipeline.annotate_chunk")
@patch("reader.pipeline.extract_speakers")
def test_pipeline_yields_done_event(mock_extract, mock_annotate, mock_write_s, mock_write_a, tmp_path, settings):
    settings.OUTPUTS_DIR = tmp_path
    _patch_pipeline(mock_extract, mock_annotate, mock_write_s, mock_write_a)

    events = list(run_pipeline("abc123", SHORT_TEXT, "Test Book"))
    assert any("done" in e for e in events)


@patch("reader.pipeline.write_annotated")
@patch("reader.pipeline.write_speakers")
@patch("reader.pipeline.annotate_chunk")
@patch("reader.pipeline.extract_speakers")
def test_pipeline_yields_chunk_progress(mock_extract, mock_annotate, mock_write_s, mock_write_a, tmp_path, settings):
    settings.OUTPUTS_DIR = tmp_path
    _patch_pipeline(mock_extract, mock_annotate, mock_write_s, mock_write_a)

    events = list(run_pipeline("abc123", SHORT_TEXT, "Test Book"))
    progress_events = [e for e in events if "chunk_progress" in e]
    assert len(progress_events) >= 1


@patch("reader.pipeline.write_annotated")
@patch("reader.pipeline.write_speakers")
@patch("reader.pipeline.annotate_chunk")
@patch("reader.pipeline.extract_speakers")
def test_pipeline_calls_extract_for_each_chunk(mock_extract, mock_annotate, mock_write_s, mock_write_a, tmp_path, settings):
    settings.OUTPUTS_DIR = tmp_path
    _patch_pipeline(mock_extract, mock_annotate, mock_write_s, mock_write_a)

    list(run_pipeline("abc123", SHORT_TEXT, "Test Book"))
    assert mock_extract.call_count >= 1
    assert mock_annotate.call_count >= 1


@patch("reader.pipeline.write_annotated")
@patch("reader.pipeline.write_speakers")
@patch("reader.pipeline.annotate_chunk")
@patch("reader.pipeline.extract_speakers")
def test_pipeline_yields_error_on_exception(mock_extract, mock_annotate, mock_write_s, mock_write_a, tmp_path, settings):
    settings.OUTPUTS_DIR = tmp_path
    mock_extract.side_effect = RuntimeError("API down")

    events = list(run_pipeline("abc123", SHORT_TEXT, "Test Book"))
    assert any("error" in e for e in events)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pipeline.py -v
```

Expected: All `FAILED` — `ModuleNotFoundError: No module named 'reader.pipeline'`

- [ ] **Step 3: Implement pipeline.py**

Create `reader/pipeline.py`:

```python
from reader.chunker import chunk_text
from reader.llm import extract_speakers, merge_speakers, annotate_chunk
from reader.output import ensure_output_dir, write_speakers, write_annotated


def run_pipeline(content_hash: str, text: str, title: str):
    """Generator that runs the two-pass LLM pipeline and yields SSE event strings."""
    try:
        yield "data: parsing\n\n"

        chunks = chunk_text(text)
        total = len(chunks)

        # Pass 1: extract speakers from each chunk
        all_speakers = []
        for chunk in chunks:
            speakers = extract_speakers(chunk)
            all_speakers.append(speakers)

        merged_speakers = merge_speakers(all_speakers)

        # Pass 2: annotate each chunk
        annotated_chunks = []
        for i, chunk in enumerate(chunks, start=1):
            annotated = annotate_chunk(chunk, merged_speakers)
            annotated_chunks.append(annotated)
            yield f"data: chunk_progress {i} {total}\n\n"

        # Write output
        out_dir = ensure_output_dir(content_hash)
        write_speakers(merged_speakers, out_dir)
        write_annotated(annotated_chunks, out_dir)

        yield "data: done\n\n"

    except Exception as exc:
        yield f"data: error {exc}\n\n"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_pipeline.py -v
```

Expected: All `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add reader/pipeline.py tests/test_pipeline.py
git commit -m "feat: add pipeline orchestrator with SSE event generator"
```

---

## Task 10: Views — Upload and Process

**Files:**
- Modify: `reader/views.py`
- Create: `reader/urls.py`
- Modify: `config/urls.py`
- Modify: `tests/test_views.py`

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_views.py` with:

```python
import pytest
from django.test import TestCase, Client
from django.urls import reverse
from reader.models import ProcessedBook
from reader.ingestion import compute_hash


class UploadViewTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_get_upload_returns_200(self):
        response = self.client.get(reverse("reader:upload"))
        self.assertEqual(response.status_code, 200)

    def test_get_upload_contains_form(self):
        response = self.client.get(reverse("reader:upload"))
        self.assertContains(response, "<form")


class ProcessViewTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_post_text_creates_processedbook(self):
        response = self.client.post(
            reverse("reader:process"),
            {"input_text": "Hello world. \"Hi,\" she said."},
        )
        self.assertEqual(ProcessedBook.objects.count(), 1)

    def test_post_text_redirects_to_progress(self):
        response = self.client.post(
            reverse("reader:process"),
            {"input_text": "Hello world."},
        )
        book = ProcessedBook.objects.first()
        self.assertRedirects(
            response,
            reverse("reader:progress", args=[book.content_hash]),
            fetch_redirect_response=False,
        )

    def test_post_same_text_twice_reuses_record(self):
        text = "Identical content."
        self.client.post(reverse("reader:process"), {"input_text": text})
        self.client.post(reverse("reader:process"), {"input_text": text})
        self.assertEqual(ProcessedBook.objects.count(), 1)

    def test_post_done_book_redirects_to_results(self):
        text = "Already done."
        content_hash = compute_hash(text)
        ProcessedBook.objects.create(
            content_hash=content_hash,
            title="Done",
            status="done",
            output_path=f"outputs/{content_hash}/",
        )
        response = self.client.post(reverse("reader:process"), {"input_text": text})
        self.assertRedirects(
            response,
            reverse("reader:results", args=[content_hash]),
            fetch_redirect_response=False,
        )

    def test_post_empty_input_returns_400(self):
        response = self.client.post(reverse("reader:process"), {})
        self.assertEqual(response.status_code, 400)

    def test_post_both_inputs_returns_400(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        response = self.client.post(
            reverse("reader:process"),
            {
                "input_text": "some text",
                "input_file": SimpleUploadedFile("test.txt", b"file content", content_type="text/plain"),
            },
        )
        self.assertEqual(response.status_code, 400)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_views.py -v
```

Expected: Several `FAILED` — URL resolution errors because views/urls don't exist yet.

- [ ] **Step 3: Write the upload and process views**

Replace `reader/views.py`:

```python
from django.http import HttpResponse, HttpResponseBadRequest, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.csrf import csrf_exempt

from reader.ingestion import compute_hash, normalize_input
from reader.models import ProcessedBook
from reader.output import read_annotated, read_speakers
from reader.pipeline import run_pipeline


def upload_view(request):
    return render(request, "reader/upload.html")


def process_view(request):
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")

    input_text = request.POST.get("input_text", "").strip()
    input_file = request.FILES.get("input_file")

    if not input_text and not input_file:
        return HttpResponseBadRequest("Provide text or a file")

    try:
        if input_file:
            file_bytes = input_file.read()
            filename = input_file.name
            text, title = normalize_input(file_bytes=file_bytes, filename=filename, text=None)
        else:
            text, title = normalize_input(file_bytes=None, filename=None, text=input_text)
    except ValueError as exc:
        return HttpResponseBadRequest(str(exc))

    content_hash = compute_hash(text)

    book, created = ProcessedBook.objects.get_or_create(
        content_hash=content_hash,
        defaults={"title": title, "status": "pending", "output_path": f"outputs/{content_hash}/"},
    )

    if book.status == "done":
        return redirect("reader:results", content_hash=content_hash)

    return redirect("reader:progress", content_hash=content_hash)


def progress_view(request, content_hash):
    book = get_object_or_404(ProcessedBook, content_hash=content_hash)
    return render(request, "reader/progress.html", {"book": book})


def stream_view(request, content_hash):
    book = get_object_or_404(ProcessedBook, content_hash=content_hash)

    if book.status == "done":
        def immediate_done():
            yield "data: done\n\n"
        response = StreamingHttpResponse(immediate_done(), content_type="text/event-stream")
        response["Cache-Control"] = "no-cache"
        response["X-Accel-Buffering"] = "no"
        return response

    book.status = "processing"
    book.save(update_fields=["status", "updated_at"])

    from reader.ingestion import normalize_input as _norm
    text_content = _get_text(book)

    def event_stream():
        success = True
        for event in run_pipeline(content_hash, text_content, book.title):
            yield event
            if event.startswith("data: error"):
                success = False
        if success:
            ProcessedBook.objects.filter(content_hash=content_hash).update(status="done")
        else:
            ProcessedBook.objects.filter(content_hash=content_hash).update(status="failed")

    response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


def results_view(request, content_hash):
    book = get_object_or_404(ProcessedBook, content_hash=content_hash, status="done")
    from pathlib import Path
    from django.conf import settings
    out_dir = Path(settings.OUTPUTS_DIR) / content_hash
    speakers = read_speakers(out_dir)
    annotated_lines = read_annotated(out_dir)
    return render(request, "reader/results.html", {
        "book": book,
        "speakers": speakers,
        "annotated_lines": annotated_lines,
    })


def _get_text(book: ProcessedBook) -> str:
    from pathlib import Path
    from django.conf import settings
    annotated_path = Path(settings.OUTPUTS_DIR) / book.content_hash / "annotated.txt"
    if annotated_path.exists():
        return annotated_path.read_text(encoding="utf-8")
    return ""
```

Wait — `stream_view` needs the original text to re-run the pipeline. But we only stored the hash, not the text. Fix: store the raw text in a `raw.txt` file in the output dir during `process_view`, then read it back in `stream_view`.

Update `process_view` to write `raw.txt`:

```python
from pathlib import Path
from django.conf import settings
from reader.output import ensure_output_dir

# Inside process_view, after computing content_hash:
if created:
    out_dir = ensure_output_dir(content_hash)
    (out_dir / "raw.txt").write_text(text, encoding="utf-8")
```

Update `_get_text` to read `raw.txt`:

```python
def _get_text(book: ProcessedBook) -> str:
    from pathlib import Path
    from django.conf import settings
    raw_path = Path(settings.OUTPUTS_DIR) / book.content_hash / "raw.txt"
    if raw_path.exists():
        return raw_path.read_text(encoding="utf-8")
    return ""
```

The full corrected `views.py` (replace entirely):

```python
from pathlib import Path

from django.conf import settings
from django.http import HttpResponseBadRequest, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render

from reader.ingestion import compute_hash, normalize_input
from reader.models import ProcessedBook
from reader.output import ensure_output_dir, read_annotated, read_speakers
from reader.pipeline import run_pipeline


def upload_view(request):
    return render(request, "reader/upload.html")


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
            text, title = normalize_input(
                file_bytes=file_bytes, filename=input_file.name, text=None
            )
        else:
            text, title = normalize_input(file_bytes=None, filename=None, text=input_text)
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

    if book.status == "done":
        return redirect("reader:results", content_hash=content_hash)

    return redirect("reader:progress", content_hash=content_hash)


def progress_view(request, content_hash):
    book = get_object_or_404(ProcessedBook, content_hash=content_hash)
    return render(request, "reader/progress.html", {"book": book})


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

    raw_path = Path(settings.OUTPUTS_DIR) / content_hash / "raw.txt"
    text = raw_path.read_text(encoding="utf-8") if raw_path.exists() else ""

    def _event_stream():
        success = True
        for event in run_pipeline(content_hash, text, book.title):
            yield event
            if "error" in event:
                success = False
        status = "done" if success else "failed"
        ProcessedBook.objects.filter(content_hash=content_hash).update(status=status)

    resp = StreamingHttpResponse(_event_stream(), content_type="text/event-stream")
    resp["Cache-Control"] = "no-cache"
    resp["X-Accel-Buffering"] = "no"
    return resp


def results_view(request, content_hash):
    book = get_object_or_404(ProcessedBook, content_hash=content_hash, status="done")
    out_dir = Path(settings.OUTPUTS_DIR) / content_hash
    speakers = read_speakers(out_dir)
    annotated_lines = read_annotated(out_dir)
    return render(request, "reader/results.html", {
        "book": book,
        "speakers": speakers,
        "annotated_lines": annotated_lines,
    })
```

- [ ] **Step 4: Write reader/urls.py**

```python
from django.urls import path
from reader import views

app_name = "reader"

urlpatterns = [
    path("", views.upload_view, name="upload"),
    path("process/", views.process_view, name="process"),
    path("progress/<str:content_hash>/", views.progress_view, name="progress"),
    path("stream/<str:content_hash>/", views.stream_view, name="stream"),
    path("results/<str:content_hash>/", views.results_view, name="results"),
]
```

- [ ] **Step 5: Update config/urls.py**

```python
from django.urls import include, path

urlpatterns = [
    path("", include("reader.urls")),
]
```

- [ ] **Step 6: Run the view tests**

```bash
pytest tests/test_views.py -v
```

Expected: Upload and process tests `PASSED`.

- [ ] **Step 7: Commit**

```bash
git add reader/views.py reader/urls.py config/urls.py tests/test_views.py
git commit -m "feat: add upload, process, progress, stream, and results views"
```

---

## Task 11: Templates

**Files:**
- Create: `reader/templates/reader/upload.html`
- Create: `reader/templates/reader/progress.html`
- Create: `reader/templates/reader/results.html`

No unit tests for templates — they are validated by running the server and confirming render (see Task 14).

- [ ] **Step 1: Create template directories**

```bash
mkdir -p reader/templates/reader
```

- [ ] **Step 2: Write upload.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ebook Narrator — Upload</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 640px; margin: 60px auto; padding: 0 20px; }
  h1 { font-size: 1.6rem; margin-bottom: 8px; }
  .subtitle { color: #666; margin-bottom: 32px; }
  .tab-bar { display: flex; gap: 0; margin-bottom: 0; }
  .tab { padding: 10px 20px; cursor: pointer; border: 1px solid #ccc; background: #f5f5f5; font-size: 0.9rem; }
  .tab.active { background: #fff; border-bottom-color: #fff; font-weight: 600; }
  .panel { border: 1px solid #ccc; padding: 20px; }
  textarea { width: 100%; box-sizing: border-box; height: 200px; font-family: monospace; font-size: 0.85rem; resize: vertical; }
  input[type=file] { width: 100%; }
  button[type=submit] { margin-top: 16px; padding: 10px 24px; background: #1a1a1a; color: #fff; border: none; border-radius: 4px; cursor: pointer; font-size: 1rem; }
  button[type=submit]:hover { background: #333; }
  .hidden { display: none; }
</style>
</head>
<body>
<h1>Ebook Narrator</h1>
<p class="subtitle">Upload an ebook or paste text to generate an annotated narration script.</p>

<form method="post" action="{% url 'reader:process' %}" enctype="multipart/form-data">
  {% csrf_token %}
  <div class="tab-bar">
    <div class="tab active" id="tab-text" onclick="showTab('text')">Paste Text</div>
    <div class="tab" id="tab-file" onclick="showTab('file')">Upload File</div>
  </div>
  <div class="panel" id="panel-text">
    <label for="input_text">Paste your text below:</label><br><br>
    <textarea id="input_text" name="input_text" placeholder="Paste the text of your book or story here..."></textarea>
  </div>
  <div class="panel hidden" id="panel-file">
    <label for="input_file">Choose a file (.txt, .pdf, .epub):</label><br><br>
    <input type="file" id="input_file" name="input_file" accept=".txt,.pdf,.epub">
  </div>
  <button type="submit">Analyze &amp; Annotate</button>
</form>

<script>
function showTab(tab) {
  document.getElementById('panel-text').classList.toggle('hidden', tab !== 'text');
  document.getElementById('panel-file').classList.toggle('hidden', tab !== 'file');
  document.getElementById('tab-text').classList.toggle('active', tab === 'text');
  document.getElementById('tab-file').classList.toggle('active', tab === 'file');
  // Clear inactive input to prevent sending both
  if (tab === 'text') document.getElementById('input_file').value = '';
  else document.getElementById('input_text').value = '';
}
</script>
</body>
</html>
```

- [ ] **Step 3: Write progress.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Processing — {{ book.title }}</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 600px; margin: 80px auto; padding: 0 20px; text-align: center; }
  h1 { font-size: 1.4rem; }
  .status { color: #555; margin: 12px 0 32px; }
  .bar-wrap { background: #eee; border-radius: 8px; height: 12px; overflow: hidden; margin-bottom: 16px; }
  .bar { height: 100%; background: #1a1a1a; border-radius: 8px; width: 5%; transition: width 0.4s ease; }
  .log { font-size: 0.85rem; color: #777; min-height: 24px; }
  .error { color: #c00; font-weight: 600; }
</style>
</head>
<body>
<h1>{{ book.title }}</h1>
<p class="status" id="status-msg">Connecting...</p>
<div class="bar-wrap"><div class="bar" id="bar"></div></div>
<p class="log" id="log"></p>

<script>
const streamUrl = "{% url 'reader:stream' content_hash=book.content_hash %}";
const resultsUrl = "{% url 'reader:results' content_hash=book.content_hash %}";

const source = new EventSource(streamUrl);
const bar = document.getElementById('bar');
const statusMsg = document.getElementById('status-msg');
const log = document.getElementById('log');

source.onmessage = function(e) {
  const data = e.data.trim();
  if (data === 'parsing') {
    statusMsg.textContent = 'Parsing document...';
    bar.style.width = '10%';
  } else if (data === 'done') {
    source.close();
    statusMsg.textContent = 'Done! Redirecting...';
    bar.style.width = '100%';
    setTimeout(() => window.location.href = resultsUrl, 600);
  } else if (data.startsWith('chunk_progress ')) {
    const parts = data.split(' ');
    const n = parseInt(parts[1]);
    const total = parseInt(parts[2]);
    const pct = Math.round(10 + (n / total) * 85);
    bar.style.width = pct + '%';
    statusMsg.textContent = `Annotating chunk ${n} of ${total}...`;
    log.textContent = '';
  } else if (data.startsWith('error ')) {
    source.close();
    statusMsg.innerHTML = '<span class="error">Error: ' + data.slice(6) + '</span>';
  }
};

source.onerror = function() {
  source.close();
  statusMsg.innerHTML = '<span class="error">Connection lost. Please try again.</span>';
};
</script>
</body>
</html>
```

- [ ] **Step 4: Write results.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ book.title }} — Narration Script</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: system-ui, sans-serif; margin: 0; padding: 0; height: 100vh; display: flex; flex-direction: column; }
  header { padding: 16px 24px; border-bottom: 1px solid #ddd; display: flex; justify-content: space-between; align-items: center; }
  header h1 { font-size: 1.2rem; margin: 0; }
  header a { font-size: 0.85rem; color: #555; text-decoration: none; }
  .panels { display: flex; flex: 1; overflow: hidden; }
  .panel { overflow-y: auto; padding: 20px 24px; }
  .panel-speakers { width: 280px; min-width: 220px; border-right: 1px solid #ddd; background: #fafafa; flex-shrink: 0; }
  .panel-annotated { flex: 1; }
  .speaker-card { margin-bottom: 16px; padding: 12px; background: #fff; border: 1px solid #e0e0e0; border-radius: 6px; }
  .speaker-name { font-weight: 700; font-size: 0.9rem; }
  .speaker-attrs { font-size: 0.78rem; color: #666; margin-top: 4px; }
  .line { margin-bottom: 6px; line-height: 1.6; font-size: 0.9rem; }
  .line.narrator { color: #888; }
  .line.dialogue .speaker-tag { font-weight: 700; font-size: 0.78rem; letter-spacing: 0.02em; }
  .line.dialogue .mood-tag { font-style: italic; font-size: 0.78rem; color: #888; }
  .line.dialogue .dialogue-text { margin-left: 4px; }
  .line.raw { color: #aaa; font-style: italic; font-size: 0.8rem; }
  h2 { font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.06em; color: #999; margin-bottom: 16px; }
</style>
</head>
<body>
<header>
  <h1>{{ book.title }}</h1>
  <a href="{% url 'reader:upload' %}">← New book</a>
</header>
<div class="panels">
  <div class="panel panel-speakers">
    <h2>Characters</h2>
    {% for speaker in speakers %}
    <div class="speaker-card">
      <div class="speaker-name">{{ speaker.name }}</div>
      <div class="speaker-attrs">
        {% if speaker.sex %}{{ speaker.sex }}{% endif %}
        {% if speaker.age %} · {{ speaker.age }}{% endif %}
        {% if speaker.traits %}<br>{{ speaker.traits }}{% endif %}
      </div>
    </div>
    {% endfor %}
  </div>
  <div class="panel panel-annotated">
    <h2>Annotated Script</h2>
    {% for line in annotated_lines %}
      {% if line.type == "narrator" %}
        <p class="line narrator">{{ line.text }}</p>
      {% elif line.type == "dialogue" %}
        <p class="line dialogue">
          <span class="speaker-tag">{{ line.speaker }}</span>
          {% if line.mood %}<span class="mood-tag"> ({{ line.mood }})</span>{% endif %}
          <span class="dialogue-text">{{ line.text }}</span>
        </p>
      {% else %}
        <p class="line raw">{{ line.text }}</p>
      {% endif %}
    {% endfor %}
  </div>
</div>
</body>
</html>
```

- [ ] **Step 5: Commit**

```bash
git add reader/templates/
git commit -m "feat: add upload, progress, and results templates"
```

---

## Task 12: Integration Smoke Test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write the integration test**

Create `tests/test_integration.py`:

```python
import tempfile
from unittest.mock import patch
from django.test import TestCase, override_settings
from django.urls import reverse
from reader.models import ProcessedBook


MOCK_SPEAKERS = [{"name": "Alice", "sex": "female", "age": "30s", "traits": "brave"}]
MOCK_ANNOTATED = '[NARRATOR] Once upon a time.\n[ALICE | mood=happy] "Hello!"'


class FullFlowTest(TestCase):
    @patch("reader.pipeline.annotate_chunk", return_value=MOCK_ANNOTATED)
    @patch("reader.pipeline.extract_speakers", return_value=MOCK_SPEAKERS)
    def test_upload_to_results_full_flow(self, mock_extract, mock_annotate):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.settings(OUTPUTS_DIR=tmp_dir):
                # Upload
                response = self.client.post(
                    reverse("reader:process"),
                    {"input_text": 'Once upon a time.\n\n"Hello!" said Alice.'},
                )
                self.assertIn(response.status_code, [301, 302])
                book = ProcessedBook.objects.get()
                self.assertEqual(book.status, "pending")

                # Consume stream (forces pipeline to run)
                stream_response = self.client.get(
                    reverse("reader:stream", args=[book.content_hash])
                )
                content = b"".join(stream_response.streaming_content).decode()
                self.assertIn("done", content)

                # Results page
                book.refresh_from_db()
                self.assertEqual(book.status, "done")
                results_response = self.client.get(
                    reverse("reader:results", args=[book.content_hash])
                )
                self.assertEqual(results_response.status_code, 200)
                self.assertContains(results_response, "Alice")
```

- [ ] **Step 2: Run the integration test**

```bash
pytest tests/test_integration.py -v
```

Expected: `PASSED`.

- [ ] **Step 3: Run the full test suite**

```bash
pytest -v
```

Expected: All tests `PASSED`.

- [ ] **Step 4: Start the dev server and verify manually**

```bash
python manage.py migrate
OPENAI_API_KEY=sk-... python manage.py runserver
```

Navigate to `http://127.0.0.1:8000/`, paste a short paragraph of dialogue, click "Analyze & Annotate", and verify:
- Progress bar animates
- Results page shows speaker table and annotated script

- [ ] **Step 5: Commit**

```bash
git add tests/test_integration.py
git commit -m "feat: add integration smoke test for full upload-to-results flow"
```

---

## Summary

| Task | Deliverable |
|---|---|
| 1 | Django project scaffold |
| 2 | ProcessedBook model + migration |
| 3 | Ingestion — text hashing and plain text |
| 4 | Ingestion — PDF and EPUB parsing |
| 5 | Chunker with tiktoken |
| 6 | LLM speaker extraction (Pass 1) |
| 7 | LLM dialogue annotation (Pass 2) |
| 8 | Output file writer and reader |
| 9 | Pipeline orchestrator (SSE generator) |
| 10 | Views + URL routing |
| 11 | HTML templates |
| 12 | Integration smoke test + manual verification |
