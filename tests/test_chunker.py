import pytest
from reader.chunker import chunk_text


SHORT_TEXT = "Hello world.\n\nThis is a short story.\n\nThe end."


def test_short_text_returns_single_chunk():
    chunks = chunk_text(SHORT_TEXT)
    assert len(chunks) == 1
    assert SHORT_TEXT.strip() in chunks[0]["content"] or chunks[0]["content"] in SHORT_TEXT


def test_single_chunk_has_empty_context():
    chunks = chunk_text(SHORT_TEXT)
    assert chunks[0]["context"] == ""


def test_large_text_splits_into_multiple_chunks():
    # ~20000 chars, definitely exceeds 3000-token chunk
    paragraph = "A" * 500 + "\n\n"
    big_text = paragraph * 70  # 70 * 63 tokens = 4410 tokens, exceeds 3000
    chunks = chunk_text(big_text)
    assert len(chunks) >= 2


def test_second_chunk_has_context_from_first():
    paragraph = "A" * 500 + "\n\n"
    big_text = paragraph * 70  # 70 * 63 tokens = 4410 tokens, exceeds 3000
    chunks = chunk_text(big_text)
    assert len(chunks[1]["context"]) > 0


def test_chunks_cover_all_content():
    paragraph = "Para {:03d}.\n\n"
    big_text = "".join(paragraph.format(i) for i in range(200))
    chunks = chunk_text(big_text)
    all_content = " ".join(c["content"] for c in chunks)
    for i in range(200):
        assert f"Para {i:03d}" in all_content


def test_empty_text_returns_empty_list():
    assert chunk_text("") == []


def test_whitespace_only_returns_empty_list():
    assert chunk_text("   \n\n   ") == []
