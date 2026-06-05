import re
import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")
CHUNK_TOKENS = 3000
OVERLAP_TOKENS = 200

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?…])\s+')


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


def _split_long_paragraph(para: str) -> list[str]:
    """Split a paragraph that exceeds CHUNK_TOKENS at sentence boundaries."""
    sentences = _SENTENCE_SPLIT_RE.split(para)
    parts = []
    current = ""
    current_tokens = 0
    for sentence in sentences:
        s_tokens = len(_ENCODING.encode(sentence))
        if current and current_tokens + s_tokens + 1 > CHUNK_TOKENS:
            parts.append(current.strip())
            current = sentence
            current_tokens = s_tokens
        else:
            current = (current + " " + sentence).strip() if current else sentence
            current_tokens += s_tokens
    if current:
        parts.append(current.strip())
    return parts or [para]


def _split_paragraphs(paragraphs: list[str]) -> list[str]:
    chunks = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = len(_ENCODING.encode(para))

        # Split paragraphs that are individually too large
        if para_tokens > CHUNK_TOKENS:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_tokens = 0
            for part in _split_long_paragraph(para):
                chunks.append(part)
            continue

        if current and current_tokens + para_tokens > CHUNK_TOKENS:
            chunks.append("\n\n".join(current))
            current = []
            current_tokens = 0
        current.append(para)
        current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))
    return chunks
