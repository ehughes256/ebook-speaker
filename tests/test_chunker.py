import pytest
from reader.chunker import chunk_text, _ENCODING, CHUNK_TOKENS


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


def test_oversized_single_sentence_splits_under_chunk_limit():
    # One sentence with no sentence-ending punctuation, exceeding CHUNK_TOKENS.
    single_sentence = "word " * (CHUNK_TOKENS * 2)
    assert len(_ENCODING.encode(single_sentence)) > CHUNK_TOKENS
    chunks = chunk_text(single_sentence)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(_ENCODING.encode(c["content"])) <= CHUNK_TOKENS


from reader.chunker import unwrap_hard_breaks


def test_unwrap_joins_mid_sentence_linebreaks():
    wrapped = (
        "Of all of my answers, “Breakfast” annoyed her the most. I could see it\n"
        "in the corners of her mouth fighting a downward turn, her rigid stance, the\n"
        "coldness in her eyes. But she kept her control."
    )
    expected = (
        "Of all of my answers, “Breakfast” annoyed her the most. I could see it "
        "in the corners of her mouth fighting a downward turn, her rigid stance, the "
        "coldness in her eyes. But she kept her control."
    )
    assert unwrap_hard_breaks(wrapped) == expected


def test_unwrap_preserves_sentence_end_newline():
    text = "First sentence.\nSecond sentence on its own line."
    # The newline follows sentence-ending punctuation, so it is kept.
    assert unwrap_hard_breaks(text) == text


def test_unwrap_preserves_blank_line_paragraph_breaks():
    text = "A wrapped line that keeps\ngoing.\n\nA separate paragraph."
    assert unwrap_hard_breaks(text) == "A wrapped line that keeps going.\n\nA separate paragraph."


def test_unwrap_noop_on_clean_text():
    text = "Hello world.\n\nThis is a short story.\n\nThe end."
    assert unwrap_hard_breaks(text) == text


def test_chunk_text_unwraps_hard_breaks():
    chunks = chunk_text("The cat sat on the\nmat and looked around.")
    assert "The cat sat on the mat and looked around." in chunks[0]["content"]


def test_empty_text_returns_empty_list():
    assert chunk_text("") == []


def test_whitespace_only_returns_empty_list():
    assert chunk_text("   \n\n   ") == []
