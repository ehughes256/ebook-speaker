import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def _normalize_traits(traits) -> str:
    """Normalize traits to a plain comma-separated string regardless of how the LLM returned it."""
    if isinstance(traits, list):
        return ", ".join(str(t).strip() for t in traits)
    if isinstance(traits, str) and traits.startswith("[") and traits.endswith("]"):
        import ast
        try:
            parsed = ast.literal_eval(traits)
            if isinstance(parsed, list):
                return ", ".join(str(t).strip() for t in parsed)
        except (ValueError, SyntaxError):
            pass
    return traits or ""


def _normalize_age(age: str) -> str:
    """Convert a bare numeric age to a descriptive phrase. Never uses the word 'young'."""
    if not re.fullmatch(r"\d+", age.strip()):
        # Also sanitise non-numeric strings the LLM might return with "young" in them
        return _sanitise_age(age)
    n = int(age.strip())
    if n <= 4:
        return "small child"
    if n <= 12:
        return "child"
    if n <= 17:
        return "adolescent"
    if n <= 25:
        return "early 20s adult"
    if n <= 40:
        return "adult"
    if n <= 60:
        return "middle-aged"
    if n <= 75:
        return "older adult"
    return "elderly"


def _sanitise_age(age: str) -> str:
    """Remove or replace words rejected by voice generation APIs."""
    replacements = {
        "young child": "small child",
        "young adult": "early 20s adult",
        "young": "early",
    }
    result = age
    for bad, good in replacements.items():
        result = re.sub(rf"\b{bad}\b", good, result, flags=re.IGNORECASE)
    return result.strip()


_model = None
_chatterbox_model = None

SAMPLE_TEXT = (
    "That quick beige fox jumped in the air over each thin dog. Look out, I shout, for he's foiled you again, creating chaos."
)


def slugify_name(name: str) -> str:
    s = name.lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def build_instruct(speaker: dict) -> str:
    sex = speaker.get("sex", "unknown")
    age = speaker.get("age", "unknown")
    if age and age != "unknown":
        age = _normalize_age(age) if re.fullmatch(r"\d+", age.strip()) else _sanitise_age(age)
    traits = _normalize_traits(speaker.get("traits", ""))
    nationality = speaker.get("nationality", "")

    has_attrs = any([
        sex and sex != "unknown",
        age and age != "unknown",
        nationality,
        traits,
    ])

    if not has_attrs:
        if speaker.get("name") == "NARRATOR":
            return "A neutral, clear, authoritative narrator. Measured pace, calm and composed delivery."
        return "A neutral voice with clear and measured tone."

    # Build a natural language voice description
    desc_parts = []
    if age and age != "unknown" and sex and sex != "unknown":
        desc_parts.append(f"A {age} {sex}")
    elif age and age != "unknown":
        desc_parts.append(f"A {age} person")
    elif sex and sex != "unknown":
        desc_parts.append(f"A {sex}")
    else:
        desc_parts.append("A person")

    if nationality:
        desc_parts.append(f"with a {nationality} accent")

    if traits:
        desc_parts.append(f"characterized by a {traits} demeanor")

    return ", ".join(desc_parts) + ". Speak naturally and conversationally."


def get_tts_model():
    global _model
    if _model is None:
        import torch
        from qwen_tts import Qwen3TTSModel
        _model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
    return _model


_MAX_CHARS = 250


def _split_text(text: str) -> list[str]:
    """Split text into chunks under _MAX_CHARS at sentence boundaries."""
    if len(text) <= _MAX_CHARS:
        return [text]

    chunks = []
    current = ""
    for sentence in re.split(r'(?<=[.!?…])\s+', text):
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= _MAX_CHARS:
            current += " " + sentence
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)

    return chunks or [text]


def get_chatterbox_model():
    global _chatterbox_model
    if _chatterbox_model is None:
        from chatterbox.tts import ChatterboxTTS
        _chatterbox_model = ChatterboxTTS.from_pretrained(device="cuda")
    return _chatterbox_model


def synthesize_line(text: str, ref_wav_path: Path, exaggeration: float = 0.5) -> tuple:
    import numpy as np
    model = get_chatterbox_model()
    chunks = _split_text(text)
    wavs = []
    for chunk in chunks:
        wav = model.generate(
            chunk,
            audio_prompt_path=str(ref_wav_path),
            exaggeration=exaggeration,
        )
        if wav is None or wav.numel() == 0:
            raise RuntimeError("Chatterbox returned empty audio")
        wavs.append(wav.squeeze().cpu().numpy())
    return np.concatenate(wavs), model.sr


def build_elevenlabs_description(speaker: dict) -> str:
    """Build a voice description optimised for ElevenLabs Voice Design API."""
    if speaker.get("name") == "NARRATOR":
        return (
            "A neutral, clear, authoritative narrator. "
            "Measured pace, calm and composed. Clear diction, professional tone."
        )
    sex = speaker.get("sex", "unknown")
    age = speaker.get("age", "unknown")
    if age and age != "unknown":
        age = _normalize_age(age) if re.fullmatch(r"\d+", age.strip()) else _sanitise_age(age)
    nationality = speaker.get("nationality", "")
    traits = _normalize_traits(speaker.get("traits", ""))

    if age != "unknown" and sex != "unknown":
        desc = f"A {age} {sex} voice"
    elif age != "unknown":
        desc = f"A {age} voice"
    elif sex != "unknown":
        desc = f"A {sex} voice"
    else:
        desc = "A voice"

    if nationality:
        desc += f" with a {nationality} accent"
    if traits:
        desc += f", {traits} in character"

    from reader.llm import generate_delivery_style
    delivery = generate_delivery_style(speaker)
    return f"{desc}. {delivery}."


def _generate_voice_elevenlabs(speaker: dict, voices_dir: Path) -> Path:
    import base64
    import os
    import subprocess
    import tempfile
    from django.conf import settings as django_settings
    from elevenlabs.client import ElevenLabs

    client = ElevenLabs(api_key=django_settings.ELEVENLABS_API_KEY)
    description = build_elevenlabs_description(speaker)
    slug = slugify_name(speaker["name"])
    logger.info("ElevenLabs voice design | slug=%s | description=%r", slug, description)

    response = client.text_to_voice.create_previews(
        voice_description=description,
        text=SAMPLE_TEXT,
        output_format="mp3_44100_128",
    )
    preview = response.previews[0]
    mp3_bytes = base64.b64decode(preview.audio_base_64)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(mp3_bytes)
        tmp_mp3 = f.name

    out_path = voices_dir / f"{slug}.wav"
    try:
        subprocess.run(
            ["ffmpeg", "-i", tmp_mp3, str(out_path), "-y"],
            check=True,
            capture_output=True,
        )
    finally:
        os.unlink(tmp_mp3)

    logger.info("ElevenLabs voice written | path=%s", out_path)
    return out_path


def generate_voice_sample(speaker: dict, voices_dir: Path) -> Path:
    from django.conf import settings as django_settings
    if getattr(django_settings, "ELEVENLABS_API_KEY", ""):
        try:
            return _generate_voice_elevenlabs(speaker, voices_dir)
        except ImportError:
            logger.warning("elevenlabs package not installed, falling back to Qwen")
        except Exception as exc:
            logger.warning("ElevenLabs failed for %r (%s), falling back to Qwen", speaker.get("name"), exc)

    import soundfile as sf
    model = get_tts_model()
    instruct = build_instruct(speaker)
    slug = slugify_name(speaker["name"])
    logger.debug("generate_voice_design | slug=%s | instruct=%r", slug, instruct)
    wavs, sr = model.generate_voice_design(
        text=SAMPLE_TEXT,
        language="English",
        instruct=instruct,
    )
    logger.debug("generate_voice_design returned | slug=%s | wavs_len=%d | sr=%s", slug, len(wavs), sr)
    out_path = voices_dir / f"{slug}.wav"
    sf.write(str(out_path), wavs[0], sr)
    logger.debug("Wrote WAV | path=%s", out_path)
    return out_path
