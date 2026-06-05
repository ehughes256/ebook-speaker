import os
import re
import tempfile
from pathlib import Path
from django.conf import settings as django_settings


_LINE_RE = re.compile(r"^\[([^\]]+)\]\s*(.*)")
_SPEAKER_TAG_RE = re.compile(r"^([^|]+?)(?:\s*\|\s*(?:mood=)?(.+))?$")


def chapter_dir_path(out_dir: Path, chapter_index: int, total_chapters: int) -> Path:
    """Return the path to a chapter directory, zero-padded consistently with run_book_pipeline."""
    pad = max(2, len(str(total_chapters)))
    return out_dir / "chapters" / str(chapter_index).zfill(pad)


def ensure_output_dir(content_hash: str) -> Path:
    out_dir = Path(django_settings.OUTPUTS_DIR) / content_hash
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


_ESCAPE_MAP = {"\\": "\\\\", "|": "\\p", ",": "\\c", "\n": "\\n"}
_UNESCAPE_MAP = {"\\": "\\", "p": "|", "c": ",", "n": "\n"}


def _escape(value: str) -> str:
    """Backslash-escape delimiter and structural characters for speakers.txt."""
    out = []
    for ch in value:
        out.append(_ESCAPE_MAP.get(ch, ch))
    return "".join(out)


def _unescape(value: str) -> str:
    """Reverse _escape, decoding backslash sequences back to literal characters."""
    out = []
    i = 0
    n = len(value)
    while i < n:
        ch = value[i]
        if ch == "\\" and i + 1 < n:
            nxt = value[i + 1]
            out.append(_UNESCAPE_MAP.get(nxt, nxt))
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _split_unescaped(value: str, delim: str) -> list[str]:
    """Split on unescaped occurrences of a single-character delimiter."""
    parts = []
    current = []
    i = 0
    n = len(value)
    while i < n:
        ch = value[i]
        if ch == "\\" and i + 1 < n:
            current.append(ch)
            current.append(value[i + 1])
            i += 2
            continue
        if ch == delim:
            parts.append("".join(current))
            current = []
            i += 1
            continue
        current.append(ch)
        i += 1
    parts.append("".join(current))
    return parts


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text to path atomically via a temp file in the same directory."""
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _speaker_to_line(s: dict) -> str:
    """Serialize a speaker dict to a speakers.txt line."""
    aliases = s.get("aliases", [])
    name_field = ",".join(_escape(p) for p in [s["name"]] + [a for a in aliases if a])
    parts = [
        name_field,
        f"sex={_escape(str(s.get('sex', 'unknown')))}",
        f"age={_escape(str(s.get('age', 'unknown')))}",
    ]
    if s.get("nationality"):
        parts.append(f"nationality={_escape(str(s['nationality']))}")
    if s.get("traits"):
        parts.append(f"traits={_escape(str(s['traits']))}")
    return " | ".join(parts)


def write_speakers(speakers: list[dict], out_dir: Path) -> None:
    narrator = next(
        (s for s in speakers if s["name"] == "NARRATOR"),
        {"name": "NARRATOR", "sex": "unknown", "age": "unknown"},
    )
    others = [s for s in speakers if s["name"] != "NARRATOR"]
    lines = [_speaker_to_line(s) for s in [narrator] + others]
    _atomic_write_text(out_dir / "speakers.txt", "\n".join(lines))


def normalize_speaker_names(annotated_chunks: list[str], speakers: list[dict]) -> list[str]:
    """Rewrite speaker tags so names match speakers.txt canonical name (aliases included)."""
    lookup = {"narrator": "NARRATOR"}
    for s in speakers:
        lookup[s["name"].lower()] = s["name"]
        for alias in s.get("aliases", []):
            lookup[alias.lower()] = s["name"]

    _TAG_RE = re.compile(r"\[([^\]|]+?)((?:\s*\|[^\]]*)?)\]")

    def _replace(m: re.Match) -> str:
        raw_name = m.group(1).strip()
        rest = m.group(2)
        canonical = lookup.get(raw_name.lower(), raw_name)
        return f"[{canonical}{rest}]"

    return [_TAG_RE.sub(_replace, chunk) for chunk in annotated_chunks]


def write_annotated(chunks: list[str], out_dir: Path) -> None:
    full_text = "\n".join(chunks)
    _atomic_write_text(out_dir / "annotated.txt", full_text)


def update_speaker_attrs(out_dir: Path, slug: str, sex: str, age: str, nationality: str, traits: str, aliases: list[str] | None = None) -> bool:
    from reader.tts import slugify_name
    speakers = read_speakers(out_dir)
    found = False
    for s in speakers:
        if slugify_name(s["name"]) == slug:
            s["sex"] = sex or "unknown"
            s["age"] = age or "unknown"
            s["nationality"] = nationality
            s["traits"] = traits
            if aliases is not None:
                s["aliases"] = aliases
            found = True
            break
    if not found:
        return False
    lines = [_speaker_to_line(s) for s in speakers]
    _atomic_write_text(out_dir / "speakers.txt", "\n".join(lines))
    return True


def read_speakers(out_dir: Path) -> list[dict]:
    text = (out_dir / "speakers.txt").read_text(encoding="utf-8")
    result = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in _split_unescaped(line, "|")]
        if not parts or not parts[0].strip():
            continue
        name_parts = [_unescape(n.strip()) for n in _split_unescaped(parts[0], ",") if n.strip()]
        if not name_parts:
            continue
        entry: dict = {"name": name_parts[0], "aliases": name_parts[1:]}
        for part in parts[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                entry[k.strip()] = _unescape(v.strip())
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
