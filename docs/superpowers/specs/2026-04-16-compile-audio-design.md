# Compile Audio Design

**Date:** 2026-04-16

## Overview

A compile feature that reads `annotated.txt` line by line and synthesizes each line into a numbered WAV file using the matching character's cloned voice. Output is saved to `outputs/<hash>/compiled/`.

## TTS Layer (`tts.py`)

Three new functions added alongside existing ones:

- **`get_clone_model()`** — lazy singleton loading `Qwen/Qwen3-TTS-12Hz-1.7B-Base`. Separate from the VoiceDesign singleton.
- **`build_voice_clone_prompt(wav_path, ref_text)`** — calls `clone_model.create_voice_clone_prompt(ref_audio=wav_path, ref_text=ref_text)` using the speaker's existing WAV and `SAMPLE_TEXT` as the reference transcript. Returns a reusable voice clone prompt.
- **`synthesize_line(text, voice_clone_prompt)`** — calls `clone_model.generate_voice_clone(text=text, language="English", voice_clone_prompt=voice_clone_prompt)`. Returns `(wav, sr)`.

If a speaker's WAV file is missing, that speaker falls back to the narrator voice clone prompt.

## Compile Pipeline (`reader/compile.py`)

New file. `run_compile(content_hash)` is a generator yielding SSE strings, mirroring `run_pipeline`.

**Steps:**
1. Read `speakers.txt` and `annotated.txt`
2. Build a `voice_clone_prompt` per speaker upfront, keyed by speaker name. Narrator is always built first. Each prompt is loaded from `voices/<slug>.wav` + `SAMPLE_TEXT`.
3. Create `outputs/<hash>/compiled/`
4. Iterate every annotated line:
   - `dialogue` → use that speaker's clone prompt (fallback to narrator if missing)
   - `narrator` or `raw` → use narrator clone prompt
   - Synthesize the line text, save as `001_narrator.wav`, `002_lord_henry_wotton.wav`, etc. Zero-padded to the digit width of the total line count.
   - Yield `data: compile_progress i total\n\n`
5. Yield `data: done\n\n` on success, `data: error <msg>\n\n` on fatal failure

## Views and URLs

**Two new views in `views.py`:**
- `compile_view(request, content_hash)` — renders `reader/compile.html`
- `compile_stream_view(request, content_hash)` — `StreamingHttpResponse` wrapping `run_compile`

**Two new URLs in `urls.py`:**
- `compile/<hash>/` → `compile_view` (name: `compile`)
- `compile/<hash>/stream/` → `compile_stream_view` (name: `compile_stream`)

## Templates

**`reader/compile.html`** — mirrors `progress.html`:
- Opens `EventSource` to the stream URL on load
- Progress bar fills as `compile_progress i total` events arrive
- On `done`: shows "Done — N files written to `compiled/`" with a link back to results
- On `error`: shows the error message

**`reader/results.html`** — "Compile audio" link added to the header next to "← New book", pointing to `/compile/<hash>/`.

## File Naming

Output files are zero-padded based on total line count digit width:
- 1–9 lines: `1_narrator.wav`
- 10–99 lines: `01_narrator.wav`
- 100+ lines: `001_narrator.wav`

## Error Handling

- Missing speaker WAV → fall back to narrator clone prompt, no crash
- Fatal error in synthesis → yield `data: error <msg>\n\n`, stop generator
