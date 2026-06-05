import re
import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")
CHUNK_TOKENS = 3000
OVERLAP_TOKENS = 200

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?…])\s+')

# Hard-wrapped source text (PDF/EPUB column wrapping) puts newlines mid-sentence.
# Treat a newline as a real break only when the line ends a sentence (terminal
# punctuation, optionally a closing quote/bracket) or is a blank-line paragraph break;
# collapse every other mid-sentence newline to a space so sentences flow.
_SENTENCE_END_CHARS = '.!?…”’)]' + '"' + "'"
_UNWRAP_RE = re.compile(
    r'([^\n' + re.escape(_SENTENCE_END_CHARS) + r'])[ \t]*\n(?![ \t]*\n)[ \t]*'
)


def unwrap_hard_breaks(text: str) -> str:
    """Join hard-wrapped lines into flowing text.

    Replaces a mid-sentence newline (the line does not end with sentence-ending
    punctuation) with a single space, while preserving newlines after sentence-ending
    punctuation and blank-line paragraph breaks.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _UNWRAP_RE.sub(r"\1 ", text)


def chunk_text(text: str) -> list[dict]:
    """Split text into chunks. Returns list of {"content": str, "context": str}."""
    text = unwrap_hard_breaks(text)
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
        sentence_token_ids = _ENCODING.encode(sentence)
        s_tokens = len(sentence_token_ids)
        # A single sentence larger than CHUNK_TOKENS cannot fit in any chunk;
        # flush the pending buffer and hard-split it into token-sized windows.
        if s_tokens > CHUNK_TOKENS:
            if current:
                parts.append(current.strip())
                current = ""
                current_tokens = 0
            for start in range(0, s_tokens, CHUNK_TOKENS):
                window = sentence_token_ids[start:start + CHUNK_TOKENS]
                parts.append(_ENCODING.decode(window).strip())
            continue
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
