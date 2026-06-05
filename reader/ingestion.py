import hashlib
import io
import json
import os
import re as _re
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import pypdf
from bs4 import BeautifulSoup
from ebooklib import epub
import ebooklib


def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _title_from_text(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return "Untitled"

    first_line = lines[0].strip()
    if first_line:
        return first_line[:80]
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


def extract_epub_cover(file_bytes: bytes) -> bytes | None:
    """Extract the cover image from an EPUB file. Returns raw image bytes or None."""
    with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as f:
        f.write(file_bytes)
        tmp_path = f.name
    try:
        book = epub.read_epub(tmp_path)
        # Try metadata cover reference first
        cover_meta = book.get_metadata("OPF", "cover")
        if cover_meta:
            cover_id = cover_meta[0][1].get("content", "")
            for item in book.get_items():
                if item.id == cover_id and item.media_type.startswith("image/"):
                    return item.get_content()
        # Fall back to any image named "cover"
        for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
            name = (item.get_name() or "").lower()
            if "cover" in name:
                return item.get_content()
        return None
    except Exception:
        return None
    finally:
        os.unlink(tmp_path)


def fetch_openlibrary_cover(title: str) -> bytes | None:
    """Search OpenLibrary by title and return cover image bytes, or None."""
    try:
        query = urllib.parse.urlencode({"title": title, "fields": "cover_i", "limit": "1"})
        with urllib.request.urlopen(
            f"https://openlibrary.org/search.json?{query}", timeout=5
        ) as resp:
            docs = json.loads(resp.read()).get("docs", [])
        if not docs or not docs[0].get("cover_i"):
            return None
        cover_id = docs[0]["cover_i"]
        with urllib.request.urlopen(
            f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg", timeout=5
        ) as resp:
            return resp.read()
    except Exception:
        return None


def save_cover(out_dir: Path, image_bytes: bytes) -> None:
    """Save cover image bytes to out_dir/cover, detecting JPEG or PNG."""
    (out_dir / "cover").write_bytes(image_bytes)


_CHAPTER_RE = _re.compile(
    r'^((?:chapter|part)\s+(?:\d+|[ivxlcdmIVXLCDM]+|one|two|three|four|five|six|'
    r'seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|'
    r'seventeen|eighteen|nineteen|twenty)[^\n]*'
    r'|\d{1,3}:\s+\S[^\n]{0,79})',
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
    """Split an EPUB into chapters. Uses the TOC for titles; falls back to spine merging."""
    with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as f:
        f.write(file_bytes)
        tmp_path = f.name
    try:
        book = epub.read_epub(tmp_path)

        # Build a map: document basename → plain text
        doc_by_name: dict[str, str] = {}
        ordered_texts: list[str] = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text = soup.get_text(separator="\n").strip()
            key = os.path.basename(item.get_name() or "")
            if key:
                doc_by_name[key] = text
            if text:
                ordered_texts.append(text)
    finally:
        os.unlink(tmp_path)

    if not ordered_texts:
        return []

    # Primary: use the epub's Table of Contents for accurate titles
    chapters = _epub_chapters_from_toc(book.toc, doc_by_name)
    if chapters:
        return chapters

    # Fallback: merge spine documents by word count
    return _epub_chapters_from_spine(ordered_texts)


def _epub_chapters_from_toc(toc, doc_by_name: dict) -> list[dict]:
    """Build chapters from the epub TOC. Top-level entries only → one chapter each."""
    chapters = []
    for item in toc:
        if isinstance(item, tuple):
            section, _ = item   # ignore sub-entries; they belong to this chapter
            title = getattr(section, "title", "") or ""
            href = getattr(section, "href", "") or ""
        else:
            title = getattr(item, "title", "") or ""
            href = getattr(item, "href", "") or ""

        if not href:
            continue
        doc_name = os.path.basename(href.split("#")[0])
        text = doc_by_name.get(doc_name, "")
        if len(text.split()) < 50:
            continue  # skip TOC pages, copyright notices, etc.
        chapters.append({
            "title": title.strip() or f"Chapter {len(chapters) + 1}",
            "text": text,
        })
    return chapters


def _epub_chapters_from_spine(texts: list[str]) -> list[dict]:
    """Fallback: merge short spine documents and infer titles from first headings."""
    merged: list[str] = []
    buffer = ""
    for text in texts:
        if buffer:
            buffer += "\n\n" + text
            if len(buffer.split()) >= 100:
                merged.append(buffer)
                buffer = ""
        elif len(text.split()) < 100:
            buffer = text
        else:
            merged.append(text)
    if buffer:
        if merged:
            merged[-1] += "\n\n" + buffer
        else:
            merged.append(buffer)

    result = []
    for i, text in enumerate(merged, start=1):
        # Use first short line as title only if it looks like a heading
        first_line = next((l.strip() for l in text.splitlines() if l.strip()), "")
        if first_line and len(first_line) <= 80 and len(first_line.split()) <= 8:
            title = first_line
        else:
            title = f"Chapter {i}"
        result.append({"title": title, "text": text})
    return result


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
