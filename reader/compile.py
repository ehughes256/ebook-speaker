import logging
import re
import shutil
import subprocess
from pathlib import Path

from django.conf import settings

from reader.llm import split_segment
from reader.output import chapter_dir_path, read_speakers
from reader.tts import get_chatterbox_model, slugify_name, synthesize_line

_MOOD_EXAGGERATION = {
    "whisper": 0.2,
    "quiet": 0.3,
    "sad": 0.3,
    "melancholy": 0.3,
    "calm": 0.4,
    "formal": 0.4,
    "neutral": 0.5,
    "curious": 0.55,
    "amused": 0.6,
    "happy": 0.65,
    "surprised": 0.65,
    "excited": 0.75,
    "urgent": 0.7,
    "angry": 0.75,
    "passionate": 0.75,
    "nervous": 0.6,
    "fearful": 0.65,
    "afraid": 0.65,
    "anxious": 0.6,
    "sarcastic": 0.6,
    "mocking": 0.6,
    "commanding": 0.7,
    "authoritative": 0.7,
    "stern": 0.6,
    "tense": 0.65,
    "cold": 0.3,
    "warm": 0.55,
    "serious": 0.45,
    "playful": 0.65,
    "confident": 0.6,
    "hesitant": 0.4,
    "gentle": 0.4,
    "cheerful": 0.65,
    "somber": 0.3,
    "desperate": 0.75,
    "relieved": 0.5,
    "suspicious": 0.55,
    "tender": 0.4,
    "frustrated": 0.7,
    "disgusted": 0.6,
    "hopeful": 0.55,
}

logger = logging.getLogger(__name__)

CONVERT_BATCH_SIZE = 10

# Matches each [TAG] text block in annotated.txt as an independent segment.
# Captures: (1) tag content, (2) text until the next [ or end of string.
_SEGMENT_RE = re.compile(r"\[([^\]]+)\]\s*([^\[]+)", re.DOTALL)

# Matches double-quoted dialogue — handles ASCII (") and typographic ("...") quotes.
_OPEN_QUOTE = r'["\u201c]'
_CLOSE_QUOTE = r'["\u201d]'
_DIALOGUE_QUOTE_RE = re.compile(_OPEN_QUOTE + r'[^"\u201c\u201d]*' + _CLOSE_QUOTE)


def _has_mixed_content(speaker: str, text: str) -> bool:
    """True when a non-narrator segment has substantial unquoted text alongside dialogue."""
    if speaker.upper() == "NARRATOR":
        return False
    remainder = _DIALOGUE_QUOTE_RE.sub("", text).strip()
    return len(remainder) > 20


def _split_mixed_segment(speaker: str, mood: str, text: str) -> list[dict]:
    """Split a speaker segment containing mixed dialogue and narration.

    Uses regex for simple cases (just dialogue, no attribution). Falls back to
    the LLM for complex interleaving that regex cannot reliably parse.
    """
    if speaker.upper() == "NARRATOR" or not re.search(r'["\u201c]', text):
        return [{"speaker": speaker, "mood": mood, "text": text}]

    if _has_mixed_content(speaker, text):
        try:
            llm_parts = split_segment(speaker, text)
            return [
                {"speaker": p["speaker"], "mood": mood if p["speaker"] == speaker else "", "text": p["text"]}
                for p in llm_parts if p.get("text", "").strip()
            ]
        except Exception:
            logger.exception("LLM segment split failed for speaker %r, falling back to regex", speaker)

    # Fast path: regex split for simple cases
    parts = []
    last_end = 0
    for m in _DIALOGUE_QUOTE_RE.finditer(text):
        before = text[last_end:m.start()].strip()
        if before:
            parts.append({"speaker": "NARRATOR", "mood": "", "text": before})
        parts.append({"speaker": speaker, "mood": mood, "text": m.group()})
        last_end = m.end()

    after = text[last_end:].strip()
    if after:
        parts.append({"speaker": "NARRATOR", "mood": "", "text": after})

    return parts or [{"speaker": speaker, "mood": mood, "text": text}]


def _parse_segments(annotated_text: str) -> list[dict]:
    """Return a flat list of {speaker, mood, text} dicts from annotated.txt content."""
    segments = []
    for m in _SEGMENT_RE.finditer(annotated_text):
        tag = m.group(1).strip()
        text = m.group(2).strip()
        if not text:
            continue
        parts = tag.split("|", 1)
        speaker = parts[0].strip()
        mood = parts[1].strip().removeprefix("mood=").strip() if len(parts) > 1 else ""
        segments.extend(_split_mixed_segment(speaker, mood, text))
    return segments


def _convert_to_mp3(wav_path: Path) -> Path:
    mp3_path = wav_path.with_suffix(".mp3")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_path), "-q:a", "2", str(mp3_path)],
        check=True,
        capture_output=True,
    )
    wav_path.unlink()
    return mp3_path


def _concat_list_entry(path: Path) -> str:
    """Format a single ffmpeg concat demuxer entry, escaping single quotes.

    The concat demuxer wraps the path in single quotes; a literal quote in the
    path must be written as the sequence '\\'' (close-quote, escaped quote,
    reopen-quote) or the demuxer will mis-parse the line.
    """
    escaped = str(path.absolute()).replace("'", "'\\''")
    return f"file '{escaped}'"


def _concatenate_mp3s(compiled_dir: Path) -> Path:
    mp3_files = sorted(f for f in compiled_dir.glob("*.mp3") if f.name != "full.mp3")
    if not mp3_files:
        raise RuntimeError("No MP3 files to concatenate")
    list_path = compiled_dir / "filelist.txt"
    list_path.write_text(
        "\n".join(_concat_list_entry(p) for p in mp3_files),
        encoding="utf-8",
    )
    out_path = compiled_dir / "full.mp3"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path),
             "-c", "copy", str(out_path)],
            check=True,
            capture_output=True,
        )
    finally:
        list_path.unlink(missing_ok=True)
    for f in mp3_files:
        f.unlink(missing_ok=True)
    return out_path


def _convert_batch(wav_paths: list[Path]) -> list[Path]:
    """Convert each WAV to MP3, returning the paths that failed to convert."""
    failed = []
    for wav_path in wav_paths:
        try:
            _convert_to_mp3(wav_path)
        except Exception:
            logger.exception("Failed to convert %s to MP3", wav_path.name)
            failed.append(wav_path)
    return failed


def run_compile(content_hash: str, chapter: int | None = None):
    """Generator that synthesizes each annotated segment with the speaker's cloned voice and yields SSE strings."""
    if shutil.which("ffmpeg") is None:
        yield "data: error ffmpeg not found on PATH.\n\n"
        return
    try:
        import json as _json
        import soundfile as sf

        out_dir = Path(settings.OUTPUTS_DIR) / content_hash

        if chapter is not None:
            chapters_path = out_dir / "chapters.json"
            if not chapters_path.exists():
                yield "data: error No chapters.json found for this book.\n\n"
                return
            chapters_meta = _json.loads(chapters_path.read_text(encoding="utf-8"))
            if chapter < 1 or chapter > len(chapters_meta):
                yield "data: error Chapter index out of range.\n\n"
                return
            work_dir = chapter_dir_path(out_dir, chapter, len(chapters_meta))
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
        # Build case-insensitive lookup including aliases so both "Uncle Vernon"
        # and "Vernon Dursley" resolve to uncle_vernon.wav
        speaker_wavs_ci = {}
        for speaker in speakers:
            name = speaker["name"]
            if name in speaker_wavs:
                wav = speaker_wavs[name]
                speaker_wavs_ci[name.lower()] = wav
                for alias in speaker.get("aliases", []):
                    speaker_wavs_ci[alias.lower()] = wav

        if narrator_wav is None:
            yield "data: error Narrator voice file not found. Generate voices first.\n\n"
            return

        # Warn upfront about any speakers whose voice files are missing
        all_speakers = read_speakers(out_dir)
        missing_voices = [
            s["name"] for s in all_speakers
            if s["name"] != "NARRATOR" and s["name"].lower() not in speaker_wavs_ci
        ]
        if missing_voices:
            yield f"data: compile_warning Missing voice files for: {', '.join(missing_voices)}. Using narrator voice as fallback. Regenerate their voices on the results page first.\n\n"

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
                for p in _convert_batch(pending_wavs):
                    yield f"data: compile_warning Could not convert {p.name} to MP3\n\n"
                pending_wavs = []

        if pending_wavs:
            yield f"data: compile_converting {len(pending_wavs)}\n\n"
            for p in _convert_batch(pending_wavs):
                yield f"data: compile_warning Could not convert {p.name} to MP3\n\n"

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
