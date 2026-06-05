# Compile Audio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compile feature that reads `annotated.txt` line-by-line and synthesizes each line into a sequentially-numbered WAV file using the cloned voice for the correct speaker.

**Architecture:** Three new TTS functions (`get_clone_model`, `build_voice_clone_prompt`, `synthesize_line`) extend `tts.py`. A new `compile.py` generator drives the compile pipeline and streams SSE progress. Two new views and a new template mirror the existing progress/stream pattern.

**Tech Stack:** Django SSE streaming, Qwen3-TTS-12Hz-1.7B-Base (voice clone model), soundfile, pytest with unittest.mock

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `reader/tts.py` | Add `_clone_model` singleton, `get_clone_model`, `build_voice_clone_prompt`, `synthesize_line` |
| Create | `reader/compile.py` | `run_compile` SSE generator |
| Modify | `reader/views.py` | Add `compile_view`, `compile_stream_view` |
| Modify | `reader/urls.py` | Add compile and compile_stream URL patterns |
| Create | `reader/templates/reader/compile.html` | Compile progress page |
| Modify | `reader/templates/reader/results.html` | Add "Compile audio" header link |
| Modify | `tests/test_tts.py` | Tests for new clone model functions |
| Create | `tests/test_compile.py` | Tests for `run_compile` |

---

## Task 1: TTS clone model functions

**Files:**
- Modify: `reader/tts.py`
- Modify: `tests/test_tts.py`

- [ ] **Step 1: Write failing tests for the three new tts functions**

Append to `tests/test_tts.py`:

```python
# --- Clone model functions ---

from reader.tts import build_voice_clone_prompt, synthesize_line, get_clone_model


def test_get_clone_model_caches_instance():
    import reader.tts as tts_module
    mock_instance = MagicMock()
    original = tts_module._clone_model
    tts_module._clone_model = None
    try:
        with patch.dict("sys.modules", {
            "torch": MagicMock(),
            "qwen_tts": MagicMock(**{
                "Qwen3TTSModel.from_pretrained.return_value": mock_instance
            }),
        }):
            first = tts_module.get_clone_model()
            second = tts_module.get_clone_model()
        assert first is second
        assert first is mock_instance
    finally:
        tts_module._clone_model = original


def test_build_voice_clone_prompt_calls_create_clone_prompt(tmp_path):
    import soundfile as sf
    wav_file = tmp_path / "narrator.wav"
    sf.write(str(wav_file), np.zeros(12000, dtype=np.float32), 12000)

    mock_model = MagicMock()
    mock_prompt = MagicMock()
    mock_model.create_voice_clone_prompt.return_value = mock_prompt

    with patch("reader.tts.get_clone_model", return_value=mock_model):
        result = build_voice_clone_prompt(wav_file, "reference text")

    mock_model.create_voice_clone_prompt.assert_called_once_with(
        ref_audio=str(wav_file),
        ref_text="reference text",
    )
    assert result is mock_prompt


def test_synthesize_line_returns_wav_and_sr():
    mock_model = MagicMock()
    mock_wav = np.zeros(12000, dtype=np.float32)
    mock_model.generate_voice_clone.return_value = ([mock_wav], 12000)
    mock_prompt = MagicMock()

    with patch("reader.tts.get_clone_model", return_value=mock_model):
        wav, sr = synthesize_line("Hello world.", mock_prompt)

    mock_model.generate_voice_clone.assert_called_once_with(
        text="Hello world.",
        language="English",
        voice_clone_prompt=mock_prompt,
    )
    assert sr == 12000
    assert len(wav) == 12000
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_tts.py::test_get_clone_model_caches_instance tests/test_tts.py::test_build_voice_clone_prompt_calls_create_clone_prompt tests/test_tts.py::test_synthesize_line_returns_wav_and_sr -v
```

Expected: FAIL with `ImportError: cannot import name 'build_voice_clone_prompt'`

- [ ] **Step 3: Implement the three functions in `reader/tts.py`**

Add after the existing `_model = None` line:

```python
_clone_model = None
```

Add after the existing `get_tts_model` function:

```python
def get_clone_model():
    global _clone_model
    if _clone_model is None:
        import torch
        from qwen_tts import Qwen3TTSModel
        _clone_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    return _clone_model


def build_voice_clone_prompt(wav_path: Path, ref_text: str):
    model = get_clone_model()
    return model.create_voice_clone_prompt(
        ref_audio=str(wav_path),
        ref_text=ref_text,
    )


def synthesize_line(text: str, voice_clone_prompt) -> tuple:
    model = get_clone_model()
    wavs, sr = model.generate_voice_clone(
        text=text,
        language="English",
        voice_clone_prompt=voice_clone_prompt,
    )
    return wavs[0], sr
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_tts.py::test_get_clone_model_caches_instance tests/test_tts.py::test_build_voice_clone_prompt_calls_create_clone_prompt tests/test_tts.py::test_synthesize_line_returns_wav_and_sr -v
```

Expected: PASS all three

- [ ] **Step 5: Run the full test suite to confirm no regressions**

```bash
python -m pytest -v
```

Expected: all existing tests still PASS

- [ ] **Step 6: Commit**

```bash
git add reader/tts.py tests/test_tts.py
git commit -m "feat: add clone model TTS functions (get_clone_model, build_voice_clone_prompt, synthesize_line)"
```

---

## Task 2: Compile pipeline

**Files:**
- Create: `reader/compile.py`
- Create: `tests/test_compile.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_compile.py`:

```python
import numpy as np
import pytest
import soundfile as sf
from unittest.mock import patch, MagicMock
from reader.compile import run_compile


CONTENT_HASH = "abc123"


def _setup_output_dir(tmp_path):
    out_dir = tmp_path / CONTENT_HASH
    out_dir.mkdir()
    voices_dir = out_dir / "voices"
    voices_dir.mkdir()

    (out_dir / "speakers.txt").write_text(
        "NARRATOR | sex=unknown | age=unknown\n"
        "ALICE | sex=female | age=30s | traits=brave",
        encoding="utf-8",
    )
    (out_dir / "annotated.txt").write_text(
        '[NARRATOR] It was quiet.\n[ALICE | mood=nervous] "Hello?"',
        encoding="utf-8",
    )

    fake_wav = np.zeros(12000, dtype=np.float32)
    sf.write(str(voices_dir / "narrator.wav"), fake_wav, 12000)
    sf.write(str(voices_dir / "alice.wav"), fake_wav, 12000)
    return out_dir


def _mock_synthesize(text, voice_clone_prompt):
    return np.zeros(12000, dtype=np.float32), 12000


def _mock_build_prompt(wav_path, ref_text):
    return MagicMock()


@patch("reader.compile.synthesize_line", side_effect=_mock_synthesize)
@patch("reader.compile.build_voice_clone_prompt", side_effect=_mock_build_prompt)
@patch("reader.compile.get_clone_model", return_value=MagicMock())
def test_compile_yields_done(mock_model, mock_build, mock_synth, tmp_path, settings):
    settings.OUTPUTS_DIR = tmp_path
    _setup_output_dir(tmp_path)
    events = list(run_compile(CONTENT_HASH))
    assert any("done" in e for e in events)


@patch("reader.compile.synthesize_line", side_effect=_mock_synthesize)
@patch("reader.compile.build_voice_clone_prompt", side_effect=_mock_build_prompt)
@patch("reader.compile.get_clone_model", return_value=MagicMock())
def test_compile_yields_progress_per_line(mock_model, mock_build, mock_synth, tmp_path, settings):
    settings.OUTPUTS_DIR = tmp_path
    _setup_output_dir(tmp_path)
    events = list(run_compile(CONTENT_HASH))
    progress = [e for e in events if "compile_progress" in e]
    assert len(progress) == 2  # two lines in annotated.txt


@patch("reader.compile.synthesize_line", side_effect=_mock_synthesize)
@patch("reader.compile.build_voice_clone_prompt", side_effect=_mock_build_prompt)
@patch("reader.compile.get_clone_model", return_value=MagicMock())
def test_compile_writes_numbered_wav_files(mock_model, mock_build, mock_synth, tmp_path, settings):
    settings.OUTPUTS_DIR = tmp_path
    _setup_output_dir(tmp_path)
    list(run_compile(CONTENT_HASH))
    compiled_dir = tmp_path / CONTENT_HASH / "compiled"
    files = sorted(f.name for f in compiled_dir.iterdir())
    assert files[0].startswith("1_narrator")
    assert files[1].startswith("2_alice")


@patch("reader.compile.synthesize_line", side_effect=_mock_synthesize)
@patch("reader.compile.build_voice_clone_prompt", side_effect=_mock_build_prompt)
@patch("reader.compile.get_clone_model", return_value=MagicMock())
def test_compile_calls_synthesize_for_each_line(mock_model, mock_build, mock_synth, tmp_path, settings):
    settings.OUTPUTS_DIR = tmp_path
    _setup_output_dir(tmp_path)
    list(run_compile(CONTENT_HASH))
    assert mock_synth.call_count == 2


@patch("reader.compile.synthesize_line", side_effect=_mock_synthesize)
@patch("reader.compile.build_voice_clone_prompt", side_effect=_mock_build_prompt)
@patch("reader.compile.get_clone_model", return_value=MagicMock())
def test_compile_falls_back_to_narrator_for_missing_speaker_wav(mock_model, mock_build, mock_synth, tmp_path, settings):
    settings.OUTPUTS_DIR = tmp_path
    out_dir = _setup_output_dir(tmp_path)
    (out_dir / "voices" / "alice.wav").unlink()  # remove alice's voice
    events = list(run_compile(CONTENT_HASH))
    # should still complete — narrator prompt used as fallback
    assert any("done" in e for e in events)


@patch("reader.compile.synthesize_line", side_effect=RuntimeError("synthesis failed"))
@patch("reader.compile.build_voice_clone_prompt", side_effect=_mock_build_prompt)
@patch("reader.compile.get_clone_model", return_value=MagicMock())
def test_compile_emits_warning_on_line_failure_and_continues(mock_model, mock_build, mock_synth, tmp_path, settings):
    settings.OUTPUTS_DIR = tmp_path
    _setup_output_dir(tmp_path)
    events = list(run_compile(CONTENT_HASH))
    assert any("compile_warning" in e for e in events)
    assert any("done" in e for e in events)


def test_compile_yields_error_when_narrator_wav_missing(tmp_path, settings):
    settings.OUTPUTS_DIR = tmp_path
    out_dir = _setup_output_dir(tmp_path)
    (out_dir / "voices" / "narrator.wav").unlink()
    with patch("reader.compile.get_clone_model", return_value=MagicMock()):
        events = list(run_compile(CONTENT_HASH))
    assert any("error" in e for e in events)
    assert not any("done" in e for e in events)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_compile.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'reader.compile'`

- [ ] **Step 3: Implement `reader/compile.py`**

Create `reader/compile.py`:

```python
import logging
import soundfile as sf
from pathlib import Path

from django.conf import settings

from reader.output import read_annotated, read_speakers
from reader.tts import SAMPLE_TEXT, build_voice_clone_prompt, get_clone_model, slugify_name, synthesize_line

logger = logging.getLogger(__name__)


def run_compile(content_hash: str):
    try:
        out_dir = Path(settings.OUTPUTS_DIR) / content_hash
        speakers = read_speakers(out_dir)
        annotated_lines = read_annotated(out_dir)
        total = len(annotated_lines)

        if total == 0:
            yield "data: done\n\n"
            return

        pad = len(str(total))
        voices_dir = out_dir / "voices"
        compiled_dir = out_dir / "compiled"
        compiled_dir.mkdir(exist_ok=True)

        get_clone_model()

        clone_prompts = {}
        narrator_prompt = None
        for speaker in speakers:
            name = speaker["name"]
            wav_path = voices_dir / f"{slugify_name(name)}.wav"
            if wav_path.exists():
                prompt = build_voice_clone_prompt(wav_path, SAMPLE_TEXT)
                clone_prompts[name] = prompt
                if name == "NARRATOR":
                    narrator_prompt = prompt

        if narrator_prompt is None:
            yield "data: error Narrator voice file not found. Generate voices first.\n\n"
            return

        for i, line in enumerate(annotated_lines, start=1):
            try:
                if line["type"] == "dialogue":
                    prompt = clone_prompts.get(line["speaker"], narrator_prompt)
                    slug = slugify_name(line["speaker"])
                else:
                    prompt = narrator_prompt
                    slug = "narrator"

                wav, sr = synthesize_line(line["text"], prompt)
                filename = f"{str(i).zfill(pad)}_{slug}.wav"
                sf.write(str(compiled_dir / filename), wav, sr)
            except Exception as exc:
                logger.exception("Failed to synthesize line %d", i)
                yield f"data: compile_warning Line {i} failed: {exc}\n\n"

            yield f"data: compile_progress {i} {total}\n\n"

        yield "data: done\n\n"

    except Exception as exc:
        logger.exception("Compile pipeline failed")
        yield f"data: error {exc}\n\n"
```

- [ ] **Step 4: Run compile tests**

```bash
python -m pytest tests/test_compile.py -v
```

Expected: all PASS

- [ ] **Step 5: Run full suite**

```bash
python -m pytest -v
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add reader/compile.py tests/test_compile.py
git commit -m "feat: add compile pipeline (run_compile SSE generator)"
```

---

## Task 3: Views and URLs

**Files:**
- Modify: `reader/views.py`
- Modify: `reader/urls.py`

- [ ] **Step 1: Add compile views to `reader/views.py`**

Add after the `delete_view` function:

```python
def compile_view(request, content_hash):
    book = get_object_or_404(ProcessedBook, content_hash=content_hash, status="done")
    return render(request, "reader/compile.html", {"book": book})


def compile_stream_view(request, content_hash):
    get_object_or_404(ProcessedBook, content_hash=content_hash, status="done")
    from reader.compile import run_compile

    resp = StreamingHttpResponse(run_compile(content_hash), content_type="text/event-stream")
    resp["Cache-Control"] = "no-cache"
    resp["X-Accel-Buffering"] = "no"
    return resp
```

- [ ] **Step 2: Add URLs to `reader/urls.py`**

Add after the `regenerate_voice` pattern:

```python
path("compile/<str:content_hash>/", views.compile_view, name="compile"),
path("compile/<str:content_hash>/stream/", views.compile_stream_view, name="compile_stream"),
```

- [ ] **Step 3: Run Django system check**

```bash
python manage.py check
```

Expected: `System check identified no issues (0 silenced).`

- [ ] **Step 4: Commit**

```bash
git add reader/views.py reader/urls.py
git commit -m "feat: add compile_view and compile_stream_view"
```

---

## Task 4: Templates

**Files:**
- Create: `reader/templates/reader/compile.html`
- Modify: `reader/templates/reader/results.html`

- [ ] **Step 1: Create `reader/templates/reader/compile.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Compiling — {{ book.title }}</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 600px; margin: 80px auto; padding: 0 20px; text-align: center; }
  h1 { font-size: 1.4rem; }
  .status { color: #555; margin: 12px 0 32px; }
  .bar-wrap { background: #eee; border-radius: 8px; height: 12px; overflow: hidden; margin-bottom: 16px; }
  .bar { height: 100%; background: #1a1a1a; border-radius: 8px; width: 0%; transition: width 0.3s ease; }
  .log { font-size: 0.85rem; color: #777; min-height: 24px; }
  .warnings { font-size: 0.8rem; color: #a60; margin-top: 8px; text-align: left; }
  .error { color: #c00; font-weight: 600; }
  .done-msg { display: none; margin-top: 24px; }
  .done-msg a { color: #1a1a1a; font-size: 0.9rem; }
</style>
</head>
<body>
<h1>{{ book.title }}</h1>
<p class="status" id="status-msg">Starting...</p>
<div class="bar-wrap"><div class="bar" id="bar"></div></div>
<p class="log" id="log"></p>
<div class="warnings" id="warnings"></div>
<div class="done-msg" id="done-msg">
  <p>Done. Files written to <code>compiled/</code>.</p>
  <a href="{% url 'reader:results' content_hash=book.content_hash %}">← Back to results</a>
</div>

<script>
const streamUrl = "{% url 'reader:compile_stream' content_hash=book.content_hash %}";

const source = new EventSource(streamUrl);
const bar = document.getElementById('bar');
const statusMsg = document.getElementById('status-msg');
const log = document.getElementById('log');
const warnings = document.getElementById('warnings');
const doneMsg = document.getElementById('done-msg');

source.onmessage = function(e) {
  const data = e.data.trim();
  if (data.startsWith('compile_progress ')) {
    const parts = data.split(' ');
    const n = parseInt(parts[1]);
    const total = parseInt(parts[2]);
    const pct = Math.round((n / total) * 100);
    bar.style.width = pct + '%';
    statusMsg.textContent = `Synthesizing line ${n} of ${total}...`;
  } else if (data.startsWith('compile_warning ')) {
    const msg = data.slice('compile_warning '.length);
    const p = document.createElement('p');
    p.textContent = '⚠ ' + msg;
    warnings.appendChild(p);
  } else if (data === 'done') {
    source.close();
    bar.style.width = '100%';
    statusMsg.textContent = 'Complete!';
    doneMsg.style.display = 'block';
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

- [ ] **Step 2: Add "Compile audio" link to results header**

In `reader/templates/reader/results.html`, find:

```html
  <a href="{% url 'reader:upload' %}">← New book</a>
```

Replace with:

```html
  <a href="{% url 'reader:compile' content_hash=book.content_hash %}">Compile audio</a>
  <a href="{% url 'reader:upload' %}">← New book</a>
```

- [ ] **Step 3: Manually verify the flow**

Start the dev server:
```bash
python manage.py runserver
```

1. Open a processed book's results page — confirm "Compile audio" appears in the header
2. Click "Compile audio" — confirm the compile page loads with title and progress bar at 0%
3. Confirm the EventSource connects and `Synthesizing line N of total...` updates appear
4. Confirm the bar fills to 100% and "Done. Files written to `compiled/`." appears
5. Confirm `outputs/<hash>/compiled/` contains numbered WAV files with speaker slugs in the names

- [ ] **Step 4: Run full test suite**

```bash
python -m pytest -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add reader/templates/reader/compile.html reader/templates/reader/results.html
git commit -m "feat: add compile progress template and Compile audio link on results page"
```
