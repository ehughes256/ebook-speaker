# Voice Generation — Design Spec

**Date:** 2026-04-16  
**Status:** Approved

---

## Overview

Phase 2 of the ebook narrator app. After annotation completes, a new Pass 3 automatically generates a custom voice sample for every character in `speakers.txt` using the Qwen VoiceDesign TTS model. Each sample is a short neutral audio clip rendered in a voice synthesized from the character's attributes. Samples are saved as WAV files under `outputs/<hash>/voices/`.

---

## Architecture

One new module: `reader/tts.py`. It owns:
- The lazy-loaded model singleton
- Instruct string construction from character attributes
- WAV file generation and saving

`pipeline.py` calls `tts.py` as Pass 3, after annotation output is written. No other existing files change except `requirements.txt` and `progress.html`.

---

## Model

**Model:** `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`  
**Loading:** Lazy singleton — loaded on first call to `get_tts_model()`, cached in a module-level variable for the process lifetime.  
**Hardware:** CUDA GPU required (`device_map="cuda:0"`, `dtype=torch.bfloat16`).

```python
_model = None

def get_tts_model():
    global _model
    if _model is None:
        from qwen_tts import Qwen3TTSModel
        import torch
        _model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    return _model
```

---

## Voice Sample Generation

**Neutral phrase** (same for every character):
> "The sun rose slowly over the horizon, casting long shadows across the quiet road ahead."

**Instruct string** built from `speakers.txt` attributes:

| Character | Example instruct |
|---|---|
| NARRATOR | `"Neutral, clear, authoritative narrator voice. Measured pace."` |
| Female, early 20s, witty | `"Female, early 20s, witty and independent. Light, confident tone."` |
| Male, late 20s, proud | `"Male, late 20s, proud and reserved. Deep, formal tone."` |
| Unknown sex/age | `"Neutral voice. Clear and measured tone."` |

Construction rules:
- Start with `"{sex}, {age},"` — omit fields that are `unknown`
- Append `"{traits}."` if traits are present
- End with a derived tone descriptor (map from traits, or `"Clear and measured tone."` as fallback)

**File naming:** Speaker name slugified: lowercase, spaces and hyphens replaced with underscores, all non-alphanumeric characters (including periods) stripped, consecutive underscores collapsed to one. Examples: `Elizabeth Bennet` → `elizabeth_bennet.wav`, `Mr. Darcy` → `mr_darcy.wav`, `NARRATOR` → `narrator.wav`.

**Output directory:** `outputs/<hash>/voices/` — created if it does not exist.

---

## Pipeline Integration

Pass 3 is added to `run_pipeline()` in `pipeline.py`, after `write_annotated()`:

```python
# Pass 3: generate voice samples
yield "data: voices_start\n\n"
speakers_for_tts = [{"name": "NARRATOR", "sex": "unknown", "age": "unknown", "traits": ""}] + merged_speakers
total_voices = len(speakers_for_tts)
voices_dir = out_dir / "voices"
voices_dir.mkdir(exist_ok=True)
for i, speaker in enumerate(speakers_for_tts, start=1):
    try:
        generate_voice_sample(speaker, voices_dir)
    except Exception as exc:
        yield f"data: voice_warning Failed voice for {speaker['name']}: {exc}\n\n"
    yield f"data: voice_progress {i} {total_voices}\n\n"
```

### SSE Events

| Event | Meaning |
|---|---|
| `voices_start` | Model loaded, Pass 3 beginning |
| `voice_progress N M` | Voice N of M generated (or skipped) |
| `voice_warning <msg>` | Non-fatal: individual voice or GPU failure |
| `done` | All passes complete |

---

## Error Handling

| Scenario | Behavior |
|---|---|
| GPU unavailable (RuntimeError on model load) | Emit `voice_warning`, skip Pass 3 entirely, emit `done` |
| Model download fails | Same as GPU unavailable |
| Individual character voice fails | Emit `voice_warning`, skip that character, continue |
| Voices dir creation fails | Propagate as pipeline error |

Pass 3 failures never prevent `done` from being emitted — annotation output is always preserved.

---

## UI Changes

`progress.html` handles two new SSE events:

- `voices_start` → updates status text to "Generating character voices…", sets bar to 95%
- `voice_progress N M` → updates status to "Generating voice N of M…", animates bar from 95% to 99%
- `voice_warning` → displays a non-blocking warning below the progress bar (does not stop progress)

---

## File Map Changes

```
reader/
  tts.py          # NEW: lazy model singleton, instruct builder, generate_voice_sample()
  pipeline.py     # MODIFY: add Pass 3 after write_annotated()
  templates/reader/
    progress.html # MODIFY: handle voices_start, voice_progress, voice_warning events

requirements.txt  # MODIFY: add qwen-tts, soundfile

tests/
  test_tts.py     # NEW: unit tests with mocked model
```

---

## Dependencies

```
qwen-tts
soundfile
flash-attn  # optional but recommended; omit if build fails on target machine
```

---

## Out of Scope (this phase)

- Full narration audio generation (using voice samples as clone prompts)
- Displaying voice playback in the results UI
- Non-English character voices
