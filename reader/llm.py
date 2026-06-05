import json
from django.conf import settings
from openai import OpenAI

_client = None

_SYSTEM_MESSAGE = (
    "You are a literary annotation assistant processing published fiction for audiobook narration. "
    "Your task is to faithfully annotate all passages regardless of theme or content, "
    "as this is published literary work being prepared for audio production."
)


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


def _model() -> str:
    return settings.OPENAI_MODEL


_SPEAKER_PROMPT = """\
Identify every character who speaks in the following passage.
For each character provide:
- name: their full, consistent name as used in the text (e.g. "Elizabeth Bennet" not just "Elizabeth")
- sex: "male", "female", or "unknown"
- age: approximate age description using phrases only — never exact numbers, never the word "young" (e.g. "small child", "adolescent", "early 20s", "middle-aged", "elderly", "unknown")
- nationality: likely nationality or regional accent inferred from the text, setting, or your knowledge of this character (e.g. "Russian", "American Southern", "British RP", "Irish", "unknown")
- traits: 2-4 personality traits inferred from their speech and actions or anything you know about this character

Return ONLY a JSON object in this exact shape, no commentary:
{{"speakers": [{{"name": "...", "sex": "...", "age": "...", "nationality": "...", "traits": "..."}}]}}

If no characters speak, return: {{"speakers": []}}
{known_block}{context_block}
PASSAGE:
{text}"""

_ANNOTATE_PROMPT = """\
Annotate the following passage for narration. Apply a prefix tag to every segment.

Rules:
- A character tag [CHARACTER NAME | mood=X] must be followed by ONLY the exact words that \
character speaks — nothing else.
- Every piece of text that is not the character's literal spoken words must be tagged [NARRATOR]. \
This includes: action, description, attribution ("said the captain", "she replied"), and any \
text between or around dialogue. Ignore blank lines.
- Each tag starts a new segment. Never place attribution or narration under a character tag.
- Use ONLY names from the character list below, spelled exactly as listed.
- mood must be one or two descriptive words (e.g. happy, angry, sad, nervous, cold, teasing, formal).
- Return the annotated passage only, no commentary, no explanation.

WRONG — attribution bundled with dialogue:
[Captain Markof | curious] "Is it you?" said the captain, bending his head back. "What is it?"

CORRECT — each piece separately tagged:
[Captain Markof | curious] "Is it you?"
[NARRATOR] said the captain, bending his head back.
[Captain Markof | curious] "What is it?"

Known characters:
{speaker_list}

PASSAGE:
{text}"""


_SPLIT_PROMPT = """\
The following text is from a story, tagged as spoken by "{speaker}".
It contains a mix of the character's actual spoken dialogue and narrator text \
(action, description, attribution like "he said", "she replied").

Split it into sub-segments. Return ONLY a JSON object in this exact shape:
{{"segments": [{{"speaker": "...", "text": "..."}}]}}

Rules:
- Use "{speaker}" for actual spoken dialogue (text in quotation marks)
- Use "NARRATOR" for all other text (attribution, action, description)
- Keep original text exactly, including quotation marks
- Do not include empty segments

TEXT: {text}"""


def split_segment(speaker: str, text: str) -> list[dict]:
    prompt = _SPLIT_PROMPT.format(speaker=speaker, text=text)
    response = _get_client().chat.completions.create(
        model=_model(),
        messages=[{"role": "system", "content": _SYSTEM_MESSAGE}, {"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    data = json.loads(response.choices[0].message.content)
    return data.get("segments", [{"speaker": speaker, "text": text}])


def extract_speakers(chunk: dict, known_speakers: list[dict] | None = None) -> list[dict]:
    context_block = (
        f"\n[Prior context — for reference only]\n{chunk['context']}\n"
        if chunk["context"]
        else ""
    )
    known_block = ""
    if known_speakers:
        lines = []
        for s in known_speakers:
            line = f"- {s['name']} ({s.get('sex', '?')}, {s.get('age', '?')})"
            if s.get("aliases"):
                line += f" — also known as: {', '.join(s['aliases'])}"
            lines.append(line)
        known_block = (
            f"\nKnown characters already identified (use exact name spellings if you see them):\n"
            + "\n".join(lines) + "\n"
        )
    prompt = _SPEAKER_PROMPT.format(
        known_block=known_block,
        context_block=context_block,
        text=chunk["content"],
    )
    response = _get_client().chat.completions.create(
        model=_model(),
        messages=[{"role": "system", "content": _SYSTEM_MESSAGE}, {"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
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


import logging as _logging
_ann_log = _logging.getLogger(__name__)


def _annotate_text(text: str, speaker_list: str, label: str = "") -> tuple[str, bool]:
    """Annotate a single text string. Returns (annotated_text, was_truncated)."""
    in_chars = len(text)
    prompt = _ANNOTATE_PROMPT.format(speaker_list=speaker_list, text=text)
    prompt_chars = len(prompt)
    _ann_log.debug("annotate_text START%s | in=%d chars | prompt=%d chars",
                   f" [{label}]" if label else "", in_chars, prompt_chars)

    response = _get_client().chat.completions.create(
        model=_model(),
        messages=[{"role": "system", "content": _SYSTEM_MESSAGE}, {"role": "user", "content": prompt}],
    )
    choice = response.choices[0]
    out_chars = len(choice.message.content)
    truncated = choice.finish_reason in ("length", "content_filter")

    _ann_log.info(
        "annotate_text DONE%s | in=%d chars | out=%d chars | ratio=%.2f | finish=%s%s",
        f" [{label}]" if label else "",
        in_chars, out_chars,
        out_chars / in_chars if in_chars else 0,
        choice.finish_reason,
        " *** TRUNCATED ***" if truncated else "",
    )
    if out_chars < in_chars * 0.5:
        _ann_log.warning(
            "annotate_text%s output is less than 50%% of input size — content likely missing",
            f" [{label}]" if label else "",
        )

    return choice.message.content, truncated


def generate_delivery_style(speaker: dict) -> str:
    """Ask the LLM to write a short delivery-style sentence for a character."""
    name = speaker.get("name", "")
    traits = speaker.get("traits", "")
    if isinstance(traits, list):
        traits = ", ".join(str(t).strip() for t in traits)
    sex = speaker.get("sex", "")
    age = speaker.get("age", "")
    nationality = speaker.get("nationality", "")

    parts = []
    if name:
        parts.append(f"Name: {name}")
    if sex and sex not in ("unknown", ""):
        parts.append(f"Sex: {sex}")
    if age and age not in ("unknown", ""):
        parts.append(f"Age: {age}")
    if nationality and nationality not in ("unknown", ""):
        parts.append(f"Nationality: {nationality}")
    if traits:
        parts.append(f"Traits: {traits}")

    char_info = "\n".join(parts) if parts else "Unknown character"

    prompt = (
        "Write one sentence (10–15 words) describing this character's vocal delivery style "
        "for a voice synthesis prompt. Focus on cadence, tone, and emotional texture. "
        "Do not name the character. Return only the sentence, no commentary.\n\n"
        f"{char_info}"
    )
    response = _get_client().chat.completions.create(
        model=_model(),
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=60,
    )
    return response.choices[0].message.content.strip().rstrip(".")


def annotate_chunk(chunk: dict, speakers: list[dict], chunk_index: int = 0) -> str:
    label = f"chunk {chunk_index}"
    lines = []
    for s in speakers:
        line = f"- {s['name']}"
        if s.get("aliases"):
            line += f" (also known as: {', '.join(s['aliases'])})"
        lines.append(line)
    speaker_list = "\n".join(lines)

    text = chunk["content"]
    _ann_log.info("annotate_chunk %s | %d chars | first 80: %r",
                  label, len(text), text[:80])

    annotated, truncated = _annotate_text(text, speaker_list, label)

    if truncated:
        _ann_log.warning("annotate_chunk %s stopped early (length or content_filter) — splitting in half and retrying", label)
        mid = len(text) // 2
        split_at = text.rfind("\n\n", 0, mid)
        if split_at <= 0:
            split_at = text.rfind(". ", 0, mid)
        if split_at <= 0:
            split_at = mid
        part_a, part_b = text[:split_at].strip(), text[split_at:].strip()
        _ann_log.info("annotate_chunk %s split at %d | part_a=%d chars | part_b=%d chars",
                      label, split_at, len(part_a), len(part_b))
        annotated_a, trunc_a = _annotate_text(part_a, speaker_list, label + "a")
        annotated_b, trunc_b = _annotate_text(part_b, speaker_list, label + "b")
        if trunc_a or trunc_b:
            _ann_log.warning("annotate_chunk %s still truncated after split — output may be incomplete", label)
        annotated = annotated_a + "\n" + annotated_b

    _ann_log.info("annotate_chunk %s final output | %d chars | last 80: %r",
                  label, len(annotated), annotated[-80:])
    return annotated
