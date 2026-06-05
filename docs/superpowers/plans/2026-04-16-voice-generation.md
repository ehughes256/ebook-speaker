# Voice Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After annotation completes, automatically generate a WAV voice sample for every character in `speakers.txt` using the Qwen VoiceDesign TTS model, saving files to `outputs/<hash>/voices/`.

**Architecture:** New module `reader/tts.py` owns a lazy-loaded model singleton, instruct-string construction, and WAV generation. `pipeline.py` calls it as Pass 3 inside the existing try/except, after `write_annotated`. `progress.html` handles three new SSE events. `qwen-tts` and `torch` are imported lazily inside function bodies so the module is importable without a GPU; model load failures are caught and emitted as `voice_warning` events without failing the pipeline.

**Tech Stack:** `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`, `qwen-tts`, `torch` (bfloat16, CUDA), `soundfile`, `numpy`

---

## File Map

```
reader/
  tts.py              # NEW: slugify_name, build_instruct, get_tts_model, generate_voice_sample
  pipeline.py         # MODIFY: add Pass 3 + import from tts
  templates/reader/
    progress.html     # MODIFY: handle voices_start, voice_progress, voice_warning

requirements.txt      # MODIFY: add soundfile

tests/
  test_tts.py         # NEW: unit tests with mocked model
  test_pipeline.py    # MODIFY: add voice-pass tests + patch tts in existing tests
```

---

## Task 1: Add soundfile to requirements

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add soundfile**

`requirements.txt` should read:

```
django>=5.2,<6.0
pypdf>=4.3
ebooklib>=0.18
beautifulsoup4>=4.12
openai>=1.30
tiktoken>=0.7
python-decouple>=3.8
pytest>=8.0
pytest-django>=4.8
soundfile>=0.12
```

- [ ] **Step 2: Install it**

```bash
cd /Users/ehughes/code/claude/reader && pip install soundfile
```

Expected: installs soundfile and numpy if not already present.

- [ ] **Step 3: Verify existing tests still pass**

```bash
cd /Users/ehughes/code/claude/reader && python -m pytest -v --tb=short
```

Expected: all existing tests pass.

---

## Task 2: tts.py — slugify and instruct builder

**Files:**
- Create: `reader/tts.py`
- Create: `tests/test_tts.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tts.py`:

```python
import pytest
from reader.tts import slugify_name, build_instruct


def test_slugify_spaces():
    assert slugify_name("Elizabeth Bennet") == "elizabeth_bennet"


def test_slugify_period_and_space():
    assert slugify_name("Mr. Darcy") == "mr_darcy"


def test_slugify_narrator():
    assert slugify_name("NARRATOR") == "narrator"


def test_slugify_hyphen():
    assert slugify_name("Jean-Luc") == "jean_luc"


def test_slugify_consecutive_underscores():
    assert slugify_name("A  B") == "a_b"


def test_build_instruct_narrator():
    result = build_instruct({"name": "NARRATOR", "sex": "unknown", "age": "unknown", "traits": ""})
    assert "narrator" in result.lower() or "neutral" in result.lower()
    assert "authoritative" in result.lower() or "clear" in result.lower()


def test_build_instruct_full_attributes():
    speaker = {"name": "Alice", "sex": "female", "age": "early 20s", "traits": "witty, independent"}
    result = build_instruct(speaker)
    assert "Female" in result
    assert "early 20s" in result
    assert "witty" in result


def test_build_instruct_male():
    speaker = {"name": "Bob", "sex": "male", "age": "late 40s", "traits": "gruff, loyal"}
    result = build_instruct(speaker)
    assert "Male" in result
    assert "late 40s" in result


def test_build_instruct_unknown_sex_and_age():
    speaker = {"name": "Ghost", "sex": "unknown", "age": "unknown", "traits": ""}
    result = build_instruct(speaker)
    assert len(result) > 0
    assert "Neutral" in result or "neutral" in result


def test_build_instruct_no_traits():
    speaker = {"name": "Guard", "sex": "male", "age": "30s", "traits": ""}
    result = build_instruct(speaker)
    assert "Male" in result
    assert "30s" in result
```

- [ ] **Step 2: Run to confirm fail**

```bash
cd /Users/ehughes/code/claude/reader && python -m pytest tests/test_tts.py -v
```

Expected: `FAILED` — `ModuleNotFoundError: No module named 'reader.tts'`

- [ ] **Step 3: Create reader/tts.py with slugify and instruct builder**

```python
import re
from pathlib import Path

_model = None

SAMPLE_TEXT = (
    "The sun rose slowly over the horizon, "
    "casting long shadows across the quiet road ahead."
)


def slugify_name(name: str) -> str:
    s = name.lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def build_instruct(speaker: dict) -> str:
    if speaker.get("name") == "NARRATOR":
        return "Neutral, clear, authoritative narrator voice. Measured pace."

    sex = speaker.get("sex", "unknown")
    age = speaker.get("age", "unknown")
    traits = speaker.get("traits", "")

    parts = []
    if sex and sex != "unknown":
        parts.append(sex.capitalize())
    if age and age != "unknown":
        parts.append(age)
    if traits:
        parts.append(traits)

    if not parts:
        return "Neutral voice. Clear and measured tone."

    return ", ".join(parts) + ". Natural conversational delivery."


def get_tts_model():
    raise NotImplementedError("Model loading implemented in Task 3")


def generate_voice_sample(speaker: dict, voices_dir: Path) -> Path:
    raise NotImplementedError("Voice sample generation implemented in Task 3")
```

- [ ] **Step 4: Run slugify and instruct tests**

```bash
cd /Users/ehughes/code/claude/reader && python -m pytest tests/test_tts.py -v -k "slugify or instruct"
```

Expected: All 10 PASSED.

---

## Task 3: tts.py — model singleton and voice sample generation

**Files:**
- Modify: `reader/tts.py`
- Modify: `tests/test_tts.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_tts.py`:

```python
import numpy as np
from unittest.mock import patch, MagicMock, call
from reader.tts import generate_voice_sample, get_tts_model


def _make_mock_model():
    mock_model = MagicMock()
    mock_model.generate_voice_design.return_value = (
        [np.zeros(12000, dtype=np.float32)],
        12000,
    )
    return mock_model


def test_generate_voice_sample_writes_wav(tmp_path):
    speaker = {"name": "Alice", "sex": "female", "age": "30s", "traits": "brave"}
    with patch("reader.tts.get_tts_model", return_value=_make_mock_model()):
        result = generate_voice_sample(speaker, tmp_path)
    assert result == tmp_path / "alice.wav"
    assert result.exists()


def test_generate_voice_sample_slugifies_filename(tmp_path):
    speaker = {"name": "Mr. Darcy", "sex": "male", "age": "late 20s", "traits": "proud"}
    with patch("reader.tts.get_tts_model", return_value=_make_mock_model()):
        result = generate_voice_sample(speaker, tmp_path)
    assert result.name == "mr_darcy.wav"


def test_generate_voice_sample_passes_instruct_to_model(tmp_path):
    speaker = {"name": "Alice", "sex": "female", "age": "30s", "traits": "brave, kind"}
    mock_model = _make_mock_model()
    with patch("reader.tts.get_tts_model", return_value=mock_model):
        generate_voice_sample(speaker, tmp_path)
    call_kwargs = mock_model.generate_voice_design.call_args[1]
    assert "Female" in call_kwargs["instruct"]
    assert "brave" in call_kwargs["instruct"]


def test_generate_voice_sample_uses_sample_text(tmp_path):
    from reader.tts import SAMPLE_TEXT
    speaker = {"name": "Alice", "sex": "female", "age": "30s", "traits": "brave"}
    mock_model = _make_mock_model()
    with patch("reader.tts.get_tts_model", return_value=mock_model):
        generate_voice_sample(speaker, tmp_path)
    call_kwargs = mock_model.generate_voice_design.call_args[1]
    assert call_kwargs["text"] == SAMPLE_TEXT


def test_generate_voice_sample_narrator(tmp_path):
    speaker = {"name": "NARRATOR", "sex": "unknown", "age": "unknown", "traits": ""}
    mock_model = _make_mock_model()
    with patch("reader.tts.get_tts_model", return_value=mock_model):
        result = generate_voice_sample(speaker, tmp_path)
    assert result.name == "narrator.wav"
    call_kwargs = mock_model.generate_voice_design.call_args[1]
    assert "narrator" in call_kwargs["instruct"].lower() or "neutral" in call_kwargs["instruct"].lower()


def test_get_tts_model_caches_instance():
    import reader.tts as tts_module
    mock_instance = MagicMock()
    original = tts_module._model
    tts_module._model = None
    try:
        with patch.dict("sys.modules", {
            "torch": MagicMock(),
            "qwen_tts": MagicMock(**{
                "Qwen3TTSModel.from_pretrained.return_value": mock_instance
            }),
        }):
            first = tts_module.get_tts_model()
            second = tts_module.get_tts_model()
        assert first is second
        assert first is mock_instance
    finally:
        tts_module._model = original
```

- [ ] **Step 2: Run to confirm fail**

```bash
cd /Users/ehughes/code/claude/reader && python -m pytest tests/test_tts.py -v -k "generate or get_tts"
```

Expected: `FAILED` — `NotImplementedError`

- [ ] **Step 3: Replace stubs in reader/tts.py with full implementations**

Replace the entire file:

```python
import re
from pathlib import Path

_model = None

SAMPLE_TEXT = (
    "The sun rose slowly over the horizon, "
    "casting long shadows across the quiet road ahead."
)


def slugify_name(name: str) -> str:
    s = name.lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def build_instruct(speaker: dict) -> str:
    if speaker.get("name") == "NARRATOR":
        return "Neutral, clear, authoritative narrator voice. Measured pace."

    sex = speaker.get("sex", "unknown")
    age = speaker.get("age", "unknown")
    traits = speaker.get("traits", "")

    parts = []
    if sex and sex != "unknown":
        parts.append(sex.capitalize())
    if age and age != "unknown":
        parts.append(age)
    if traits:
        parts.append(traits)

    if not parts:
        return "Neutral voice. Clear and measured tone."

    return ", ".join(parts) + ". Natural conversational delivery."


def get_tts_model():
    global _model
    if _model is None:
        import torch
        from qwen_tts import Qwen3TTSModel
        _model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    return _model


def generate_voice_sample(speaker: dict, voices_dir: Path) -> Path:
    import soundfile as sf
    model = get_tts_model()
    instruct = build_instruct(speaker)
    wavs, sr = model.generate_voice_design(
        text=SAMPLE_TEXT,
        language="English",
        instruct=instruct,
    )
    slug = slugify_name(speaker["name"])
    out_path = voices_dir / f"{slug}.wav"
    sf.write(str(out_path), wavs[0], sr)
    return out_path
```

- [ ] **Step 4: Run all tts tests**

```bash
cd /Users/ehughes/code/claude/reader && python -m pytest tests/test_tts.py -v
```

Expected: All tests PASSED. (The `test_get_tts_model_caches_instance` test patches sys.modules to simulate qwen_tts being available.)

- [ ] **Step 5: Verify full suite still passes**

```bash
cd /Users/ehughes/code/claude/reader && python -m pytest -v --tb=short
```

Expected: All tests pass.

---

## Task 4: pipeline.py — Pass 3 voice generation

**Files:**
- Modify: `reader/pipeline.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Add to the END of `tests/test_pipeline.py`:

```python
from unittest.mock import MagicMock


@patch("reader.pipeline.generate_voice_sample")
@patch("reader.pipeline.get_tts_model")
@patch("reader.pipeline.write_annotated")
@patch("reader.pipeline.write_speakers")
@patch("reader.pipeline.annotate_chunk")
@patch("reader.pipeline.extract_speakers")
def test_pipeline_yields_voices_start(
    mock_extract, mock_annotate, mock_write_s, mock_write_a,
    mock_get_model, mock_gen_voice, tmp_path, settings
):
    settings.OUTPUTS_DIR = tmp_path
    mock_extract.return_value = MOCK_SPEAKERS
    mock_annotate.return_value = MOCK_ANNOTATED
    mock_get_model.return_value = MagicMock()

    events = list(run_pipeline("abc123", SHORT_TEXT, "Test Book"))
    assert any("voices_start" in e for e in events)


@patch("reader.pipeline.generate_voice_sample")
@patch("reader.pipeline.get_tts_model")
@patch("reader.pipeline.write_annotated")
@patch("reader.pipeline.write_speakers")
@patch("reader.pipeline.annotate_chunk")
@patch("reader.pipeline.extract_speakers")
def test_pipeline_yields_voice_progress(
    mock_extract, mock_annotate, mock_write_s, mock_write_a,
    mock_get_model, mock_gen_voice, tmp_path, settings
):
    settings.OUTPUTS_DIR = tmp_path
    mock_extract.return_value = MOCK_SPEAKERS
    mock_annotate.return_value = MOCK_ANNOTATED
    mock_get_model.return_value = MagicMock()

    events = list(run_pipeline("abc123", SHORT_TEXT, "Test Book"))
    assert any("voice_progress" in e for e in events)


@patch("reader.pipeline.generate_voice_sample")
@patch("reader.pipeline.get_tts_model")
@patch("reader.pipeline.write_annotated")
@patch("reader.pipeline.write_speakers")
@patch("reader.pipeline.annotate_chunk")
@patch("reader.pipeline.extract_speakers")
def test_pipeline_emits_warning_on_gpu_unavailable(
    mock_extract, mock_annotate, mock_write_s, mock_write_a,
    mock_get_model, mock_gen_voice, tmp_path, settings
):
    settings.OUTPUTS_DIR = tmp_path
    mock_extract.return_value = MOCK_SPEAKERS
    mock_annotate.return_value = MOCK_ANNOTATED
    mock_get_model.side_effect = RuntimeError("No CUDA GPU available")

    events = list(run_pipeline("abc123", SHORT_TEXT, "Test Book"))
    assert any("voice_warning" in e for e in events)
    assert any("done" in e for e in events)


@patch("reader.pipeline.generate_voice_sample")
@patch("reader.pipeline.get_tts_model")
@patch("reader.pipeline.write_annotated")
@patch("reader.pipeline.write_speakers")
@patch("reader.pipeline.annotate_chunk")
@patch("reader.pipeline.extract_speakers")
def test_pipeline_continues_on_individual_voice_failure(
    mock_extract, mock_annotate, mock_write_s, mock_write_a,
    mock_get_model, mock_gen_voice, tmp_path, settings
):
    settings.OUTPUTS_DIR = tmp_path
    mock_extract.return_value = MOCK_SPEAKERS
    mock_annotate.return_value = MOCK_ANNOTATED
    mock_get_model.return_value = MagicMock()
    mock_gen_voice.side_effect = RuntimeError("synthesis failed")

    events = list(run_pipeline("abc123", SHORT_TEXT, "Test Book"))
    assert any("voice_warning" in e for e in events)
    assert any("done" in e for e in events)


@patch("reader.pipeline.generate_voice_sample")
@patch("reader.pipeline.get_tts_model")
@patch("reader.pipeline.write_annotated")
@patch("reader.pipeline.write_speakers")
@patch("reader.pipeline.annotate_chunk")
@patch("reader.pipeline.extract_speakers")
def test_pipeline_generates_voice_for_narrator_and_characters(
    mock_extract, mock_annotate, mock_write_s, mock_write_a,
    mock_get_model, mock_gen_voice, tmp_path, settings
):
    settings.OUTPUTS_DIR = tmp_path
    mock_extract.return_value = MOCK_SPEAKERS
    mock_annotate.return_value = MOCK_ANNOTATED
    mock_get_model.return_value = MagicMock()

    list(run_pipeline("abc123", SHORT_TEXT, "Test Book"))
    # NARRATOR + 1 character in MOCK_SPEAKERS = 2 calls
    assert mock_gen_voice.call_count == 2
    names_called = [c[0][0]["name"] for c in mock_gen_voice.call_args_list]
    assert "NARRATOR" in names_called
    assert "Alice" in names_called
```

- [ ] **Step 2: Run to confirm fail**

```bash
cd /Users/ehughes/code/claude/reader && python -m pytest tests/test_pipeline.py -v -k "voice"
```

Expected: `FAILED` — `ImportError` or `AssertionError` (Pass 3 not yet in pipeline)

- [ ] **Step 3: Update reader/pipeline.py**

Replace the entire file:

```python
from reader.chunker import chunk_text
from reader.llm import extract_speakers, merge_speakers, annotate_chunk
from reader.output import ensure_output_dir, normalize_speaker_names, write_speakers, write_annotated
from reader.tts import generate_voice_sample, get_tts_model


def run_pipeline(content_hash: str, text: str, title: str):
    """Generator that runs the three-pass pipeline and yields SSE event strings."""
    try:
        yield "data: parsing\n\n"

        chunks = chunk_text(text)
        total = len(chunks)

        # Pass 1: extract speakers from each chunk
        all_speakers = []
        for chunk in chunks:
            speakers = extract_speakers(chunk)
            all_speakers.append(speakers)

        merged_speakers = merge_speakers(all_speakers)

        # Pass 2: annotate each chunk
        annotated_chunks = []
        for i, chunk in enumerate(chunks, start=1):
            annotated = annotate_chunk(chunk, merged_speakers)
            annotated_chunks.append(annotated)
            yield f"data: chunk_progress {i} {total}\n\n"

        # Normalize tag names to match speakers.txt exactly
        annotated_chunks = normalize_speaker_names(annotated_chunks, merged_speakers)

        # Write annotation output
        out_dir = ensure_output_dir(content_hash)
        write_speakers(merged_speakers, out_dir)
        write_annotated(annotated_chunks, out_dir)

        # Pass 3: generate voice samples
        try:
            get_tts_model()
            yield "data: voices_start\n\n"
            voices_dir = out_dir / "voices"
            voices_dir.mkdir(exist_ok=True)
            speakers_for_tts = [
                {"name": "NARRATOR", "sex": "unknown", "age": "unknown", "traits": ""}
            ] + merged_speakers
            total_voices = len(speakers_for_tts)
            for i, speaker in enumerate(speakers_for_tts, start=1):
                try:
                    generate_voice_sample(speaker, voices_dir)
                except Exception as exc:
                    yield f"data: voice_warning Failed voice for {speaker['name']}: {exc}\n\n"
                yield f"data: voice_progress {i} {total_voices}\n\n"
        except Exception as exc:
            yield f"data: voice_warning Voice generation unavailable: {exc}\n\n"

        yield "data: done\n\n"

    except Exception as exc:
        yield f"data: error {exc}\n\n"
```

- [ ] **Step 4: Also patch get_tts_model in the existing pipeline tests**

The four existing tests (`test_pipeline_yields_done_event`, `test_pipeline_yields_chunk_progress`, `test_pipeline_calls_extract_for_each_chunk`, `test_pipeline_yields_error_on_exception`) do not patch `get_tts_model` or `generate_voice_sample`. When qwen-tts is not installed they will still pass (the GPU unavailable path emits a warning and then `done`), but add the patches for speed and clarity.

Replace the four existing tests in `tests/test_pipeline.py` with these updated versions that add the two extra patches:

```python
@patch("reader.pipeline.generate_voice_sample")
@patch("reader.pipeline.get_tts_model")
@patch("reader.pipeline.write_annotated")
@patch("reader.pipeline.write_speakers")
@patch("reader.pipeline.annotate_chunk")
@patch("reader.pipeline.extract_speakers")
def test_pipeline_yields_done_event(
    mock_extract, mock_annotate, mock_write_s, mock_write_a,
    mock_get_model, mock_gen_voice, tmp_path, settings
):
    settings.OUTPUTS_DIR = tmp_path
    mock_extract.return_value = MOCK_SPEAKERS
    mock_annotate.return_value = MOCK_ANNOTATED
    mock_get_model.return_value = MagicMock()

    events = list(run_pipeline("abc123", SHORT_TEXT, "Test Book"))
    assert any("done" in e for e in events)


@patch("reader.pipeline.generate_voice_sample")
@patch("reader.pipeline.get_tts_model")
@patch("reader.pipeline.write_annotated")
@patch("reader.pipeline.write_speakers")
@patch("reader.pipeline.annotate_chunk")
@patch("reader.pipeline.extract_speakers")
def test_pipeline_yields_chunk_progress(
    mock_extract, mock_annotate, mock_write_s, mock_write_a,
    mock_get_model, mock_gen_voice, tmp_path, settings
):
    settings.OUTPUTS_DIR = tmp_path
    mock_extract.return_value = MOCK_SPEAKERS
    mock_annotate.return_value = MOCK_ANNOTATED
    mock_get_model.return_value = MagicMock()

    events = list(run_pipeline("abc123", SHORT_TEXT, "Test Book"))
    progress_events = [e for e in events if "chunk_progress" in e]
    assert len(progress_events) >= 1


@patch("reader.pipeline.generate_voice_sample")
@patch("reader.pipeline.get_tts_model")
@patch("reader.pipeline.write_annotated")
@patch("reader.pipeline.write_speakers")
@patch("reader.pipeline.annotate_chunk")
@patch("reader.pipeline.extract_speakers")
def test_pipeline_calls_extract_for_each_chunk(
    mock_extract, mock_annotate, mock_write_s, mock_write_a,
    mock_get_model, mock_gen_voice, tmp_path, settings
):
    settings.OUTPUTS_DIR = tmp_path
    mock_extract.return_value = MOCK_SPEAKERS
    mock_annotate.return_value = MOCK_ANNOTATED
    mock_get_model.return_value = MagicMock()

    list(run_pipeline("abc123", SHORT_TEXT, "Test Book"))
    assert mock_extract.call_count >= 1
    assert mock_annotate.call_count >= 1


@patch("reader.pipeline.generate_voice_sample")
@patch("reader.pipeline.get_tts_model")
@patch("reader.pipeline.write_annotated")
@patch("reader.pipeline.write_speakers")
@patch("reader.pipeline.annotate_chunk")
@patch("reader.pipeline.extract_speakers")
def test_pipeline_yields_error_on_exception(
    mock_extract, mock_annotate, mock_write_s, mock_write_a,
    mock_get_model, mock_gen_voice, tmp_path, settings
):
    settings.OUTPUTS_DIR = tmp_path
    mock_get_model.return_value = MagicMock()
    mock_extract.side_effect = RuntimeError("API down")

    events = list(run_pipeline("abc123", SHORT_TEXT, "Test Book"))
    assert any("error" in e for e in events)
```

- [ ] **Step 5: Run all pipeline tests**

```bash
cd /Users/ehughes/code/claude/reader && python -m pytest tests/test_pipeline.py -v
```

Expected: All 9 tests PASSED.

- [ ] **Step 6: Run full suite**

```bash
cd /Users/ehughes/code/claude/reader && python -m pytest -v --tb=short
```

Expected: All tests pass.

---

## Task 5: progress.html — new SSE event handlers

**Files:**
- Modify: `reader/templates/reader/progress.html`

No unit tests — validated by running the server.

- [ ] **Step 1: Replace the `<script>` block in progress.html**

The full updated `progress.html` (replace the existing file entirely):

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Processing — {{ book.title }}</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 600px; margin: 80px auto; padding: 0 20px; text-align: center; }
  h1 { font-size: 1.4rem; }
  .status { color: #555; margin: 12px 0 32px; }
  .bar-wrap { background: #eee; border-radius: 8px; height: 12px; overflow: hidden; margin-bottom: 16px; }
  .bar { height: 100%; background: #1a1a1a; border-radius: 8px; width: 5%; transition: width 0.4s ease; }
  .log { font-size: 0.85rem; color: #777; min-height: 24px; }
  .warnings { font-size: 0.8rem; color: #a60; margin-top: 8px; text-align: left; }
  .error { color: #c00; font-weight: 600; }
</style>
</head>
<body>
<h1>{{ book.title }}</h1>
<p class="status" id="status-msg">Connecting...</p>
<div class="bar-wrap"><div class="bar" id="bar"></div></div>
<p class="log" id="log"></p>
<div class="warnings" id="warnings"></div>

<script>
const streamUrl = "{% url 'reader:stream' content_hash=book.content_hash %}";
const resultsUrl = "{% url 'reader:results' content_hash=book.content_hash %}";

const source = new EventSource(streamUrl);
const bar = document.getElementById('bar');
const statusMsg = document.getElementById('status-msg');
const log = document.getElementById('log');
const warnings = document.getElementById('warnings');

source.onmessage = function(e) {
  const data = e.data.trim();
  if (data === 'parsing') {
    statusMsg.textContent = 'Parsing document...';
    bar.style.width = '10%';
  } else if (data === 'done') {
    source.close();
    statusMsg.textContent = 'Done! Redirecting...';
    bar.style.width = '100%';
    setTimeout(() => window.location.href = resultsUrl, 600);
  } else if (data.startsWith('chunk_progress ')) {
    const parts = data.split(' ');
    const n = parseInt(parts[1]);
    const total = parseInt(parts[2]);
    const pct = Math.round(10 + (n / total) * 83);
    bar.style.width = pct + '%';
    statusMsg.textContent = `Annotating chunk ${n} of ${total}...`;
    log.textContent = '';
  } else if (data === 'voices_start') {
    statusMsg.textContent = 'Generating character voices...';
    bar.style.width = '95%';
  } else if (data.startsWith('voice_progress ')) {
    const parts = data.split(' ');
    const n = parseInt(parts[1]);
    const total = parseInt(parts[2]);
    const pct = Math.round(95 + (n / total) * 4);
    bar.style.width = pct + '%';
    statusMsg.textContent = `Generating voice ${n} of ${total}...`;
  } else if (data.startsWith('voice_warning ')) {
    const msg = data.slice('voice_warning '.length);
    const p = document.createElement('p');
    p.textContent = '⚠ ' + msg;
    warnings.appendChild(p);
  } else if (data.startsWith('error ')) {
    source.close();
    statusMsg.innerHTML = '<span class="error">Error: ' + data.slice(6) + '</span>';
  }
};

source.onerror = function() {
  source.close();
  statusMsg.innerHTML = '<span class="error">Connection lost. Please try again.</span>';
};
</script>
</body>
</html>
```

Note: `chunk_progress` bar range is now 10%–93% (was 10%–95%) to leave room for the voice pass at 95%–99%.

- [ ] **Step 2: Verify Django template check passes**

```bash
cd /Users/ehughes/code/claude/reader && python manage.py check
```

Expected: no issues.

- [ ] **Step 3: Run full test suite**

```bash
cd /Users/ehughes/code/claude/reader && python -m pytest -v --tb=short
```

Expected: All tests pass.

---

## Summary

| Task | Deliverable |
|---|---|
| 1 | `soundfile` added to requirements |
| 2 | `tts.py` slugify + instruct builder with tests |
| 3 | `tts.py` model singleton + voice sample generation with tests |
| 4 | `pipeline.py` Pass 3 with tests; existing pipeline tests updated |
| 5 | `progress.html` handles `voices_start`, `voice_progress`, `voice_warning` |

**GPU note:** `qwen-tts`, `torch`, and optionally `flash-attn` must be installed on the target machine separately:
```bash
pip install qwen-tts torch
pip install flash-attn --no-build-isolation  # optional but recommended for speed
```
They are not in `requirements.txt` because they require a CUDA GPU and are not needed for the test suite.
