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
        with patch("reader.ingestion.tempfile.NamedTemporaryFile") as mock_tmpfile:
            # NamedTemporaryFile returns a context manager with a .name attribute
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_file.name = "/tmp/fake.epub"
            mock_tmpfile.return_value = mock_file
            with patch("reader.ingestion.os.unlink"):
                content, title = normalize_input(
                    file_bytes=b"PK fake epub",
                    filename="book.epub",
                    text=None,
                )

    assert "Chapter text." in content
    assert title == "book"


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


def test_split_text_chapters_handles_numeric_colon_headings():
    text = (
        "OceanofPDF.com\n\n"
        "01: INITIATION\n"
        "The tower, which was not supposed to be there.\n\n"
        "02: INTEGRATION\n"
        "The second chapter content.\n"
    )
    chapters = split_text_chapters(text)
    assert chapters is not None
    assert len(chapters) == 2
    assert chapters[0]["title"] == "01: INITIATION"
    assert "tower" in chapters[0]["text"]
    assert chapters[1]["title"] == "02: INTEGRATION"


def _make_epub_mock(items_data):
    """Helper: build a mock epub book with no TOC and given document items."""
    mock_items = []
    for i, (name, content) in enumerate(items_data):
        item = MagicMock()
        item.get_name.return_value = name
        item.get_content.return_value = content.encode()
        mock_items.append(item)
    mock_book = MagicMock()
    mock_book.toc = []  # no TOC → use fallback
    mock_book.get_items_of_type.return_value = mock_items
    return mock_book


def _run_epub_split(mock_book):
    with patch("reader.ingestion.epub.read_epub", return_value=mock_book):
        with patch("reader.ingestion.tempfile.NamedTemporaryFile") as mock_tmpfile:
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            mock_file.name = "/tmp/fake.epub"
            mock_tmpfile.return_value = mock_file
            with patch("reader.ingestion.os.unlink"):
                return split_epub_chapters(b"PK fake epub")


def test_split_epub_chapters_returns_one_per_spine_item():
    long_text = "Word " * 150
    mock_book = _make_epub_mock([
        ("chapter1.xhtml", f"<html><body><h1>Chapter One</h1><p>{long_text}</p></body></html>"),
        ("chapter2.xhtml", f"<html><body><h1>Chapter Two</h1><p>{long_text}</p></body></html>"),
    ])
    chapters = _run_epub_split(mock_book)
    assert len(chapters) == 2


def test_split_epub_chapters_uses_toc_titles():
    long_text = "Word " * 150
    mock_book = _make_epub_mock([
        ("ch1.xhtml", f"<html><body><p>{long_text}</p></body></html>"),
        ("ch2.xhtml", f"<html><body><p>{long_text}</p></body></html>"),
    ])
    # Configure TOC entries
    link1 = MagicMock()
    link1.title = "The Beginning"
    link1.href = "ch1.xhtml"
    link2 = MagicMock()
    link2.title = "The End"
    link2.href = "ch2.xhtml"
    mock_book.toc = [link1, link2]
    chapters = _run_epub_split(mock_book)
    assert len(chapters) == 2
    assert chapters[0]["title"] == "The Beginning"
    assert chapters[1]["title"] == "The End"


def test_split_epub_chapters_merges_short_items():
    short_text = "Word " * 50  # < 100 words
    long_text = "Word " * 150
    mock_book = _make_epub_mock([
        ("front.xhtml", f"<html><body><p>{short_text}</p></body></html>"),
        ("ch1.xhtml", f"<html><body><h1>Real Chapter</h1><p>{long_text}</p></body></html>"),
    ])
    chapters = _run_epub_split(mock_book)
    assert len(chapters) == 1


def test_split_epub_chapters_toc_skips_short_entries():
    """TOC entries whose documents are too short (TOC pages, copyright) are omitted."""
    long_text = "Word " * 150
    mock_book = _make_epub_mock([
        ("toc.xhtml", "<html><body><p>Table of Contents</p></body></html>"),
        ("ch1.xhtml", f"<html><body><p>{long_text}</p></body></html>"),
    ])
    link_toc = MagicMock()
    link_toc.title = "Contents"
    link_toc.href = "toc.xhtml"
    link_ch = MagicMock()
    link_ch.title = "Chapter One"
    link_ch.href = "ch1.xhtml"
    mock_book.toc = [link_toc, link_ch]
    chapters = _run_epub_split(mock_book)
    assert len(chapters) == 1
    assert chapters[0]["title"] == "Chapter One"
