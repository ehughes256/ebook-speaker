import numpy as np
import pytest
import soundfile as sf
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from reader.compile import run_compile, _convert_batch, _convert_to_mp3, _parse_segments, _split_mixed_segment, _has_mixed_content


CONTENT_HASH = "abc123"


def _setup_output_dir(tmp_path, n_extra_lines=0):
    out_dir = tmp_path / CONTENT_HASH
    out_dir.mkdir(exist_ok=True)
    voices_dir = out_dir / "voices"
    voices_dir.mkdir(exist_ok=True)

    (out_dir / "speakers.txt").write_text(
        "NARRATOR | sex=unknown | age=unknown\n"
        "ALICE | sex=female | age=30s | traits=brave",
        encoding="utf-8",
    )
    base_lines = '[NARRATOR] It was quiet.\n[ALICE | mood=nervous] "Hello?"'
    extra = "".join(f"\n[NARRATOR] Line {i}." for i in range(n_extra_lines))
    (out_dir / "annotated.txt").write_text(base_lines + extra, encoding="utf-8")

    fake_wav = np.zeros(12000, dtype=np.float32)
    sf.write(str(voices_dir / "narrator.wav"), fake_wav, 12000)
    sf.write(str(voices_dir / "alice.wav"), fake_wav, 12000)
    return out_dir


def _mock_synthesize(text, ref_wav_path, exaggeration=0.5):
    return np.zeros(12000, dtype=np.float32), 12000


# --- Pipeline tests (all mock _convert_batch to avoid needing ffmpeg) ---

@patch("reader.compile._convert_batch")
@patch("reader.compile.synthesize_line", side_effect=_mock_synthesize)
@patch("reader.compile.get_chatterbox_model", return_value=MagicMock())
def test_compile_yields_done(mock_model, mock_synth, mock_convert, tmp_path, settings):
    settings.OUTPUTS_DIR = tmp_path
    _setup_output_dir(tmp_path)
    events = list(run_compile(CONTENT_HASH))
    assert any("done" in e for e in events)


@patch("reader.compile._convert_batch")
@patch("reader.compile.synthesize_line", side_effect=_mock_synthesize)
@patch("reader.compile.get_chatterbox_model", return_value=MagicMock())
def test_compile_yields_progress_per_line(mock_model, mock_synth, mock_convert, tmp_path, settings):
    settings.OUTPUTS_DIR = tmp_path
    _setup_output_dir(tmp_path)
    events = list(run_compile(CONTENT_HASH))
    progress = [e for e in events if "compile_progress" in e]
    assert len(progress) == 2


@patch("reader.compile._convert_batch")
@patch("reader.compile.synthesize_line", side_effect=_mock_synthesize)
@patch("reader.compile.get_chatterbox_model", return_value=MagicMock())
def test_compile_writes_numbered_wav_files(mock_model, mock_synth, mock_convert, tmp_path, settings):
    settings.OUTPUTS_DIR = tmp_path
    _setup_output_dir(tmp_path)
    list(run_compile(CONTENT_HASH))
    compiled_dir = tmp_path / CONTENT_HASH / "compiled"
    files = sorted(f.name for f in compiled_dir.iterdir())
    assert files[0] == "1_narrator.wav"
    assert files[1] == "2_alice.wav"


@patch("reader.compile._convert_batch")
@patch("reader.compile.synthesize_line", side_effect=_mock_synthesize)
@patch("reader.compile.get_chatterbox_model", return_value=MagicMock())
def test_compile_calls_synthesize_for_each_line(mock_model, mock_synth, mock_convert, tmp_path, settings):
    settings.OUTPUTS_DIR = tmp_path
    _setup_output_dir(tmp_path)
    list(run_compile(CONTENT_HASH))
    assert mock_synth.call_count == 2


@patch("reader.compile._convert_batch")
@patch("reader.compile.synthesize_line", side_effect=_mock_synthesize)
@patch("reader.compile.get_chatterbox_model", return_value=MagicMock())
def test_compile_falls_back_to_narrator_for_missing_speaker_wav(mock_model, mock_synth, mock_convert, tmp_path, settings):
    settings.OUTPUTS_DIR = tmp_path
    out_dir = _setup_output_dir(tmp_path)
    (out_dir / "voices" / "alice.wav").unlink()
    events = list(run_compile(CONTENT_HASH))
    assert any("done" in e for e in events)


@patch("reader.compile.get_chatterbox_model", return_value=MagicMock())
@patch("reader.compile.synthesize_line", side_effect=RuntimeError("synthesis failed"))
def test_compile_emits_warning_on_line_failure_and_continues(mock_synth, mock_model, tmp_path, settings):
    settings.OUTPUTS_DIR = tmp_path
    _setup_output_dir(tmp_path)
    events = list(run_compile(CONTENT_HASH))
    assert any("compile_warning" in e for e in events)
    assert any("done" in e for e in events)


def test_compile_yields_error_when_chatterbox_model_unavailable(tmp_path, settings):
    settings.OUTPUTS_DIR = tmp_path
    _setup_output_dir(tmp_path)
    with patch("reader.compile.get_chatterbox_model", side_effect=RuntimeError("No CUDA GPU available")):
        events = list(run_compile(CONTENT_HASH))
    assert any("error" in e for e in events)
    assert not any("done" in e for e in events)


def test_compile_yields_error_when_narrator_wav_missing(tmp_path, settings):
    settings.OUTPUTS_DIR = tmp_path
    out_dir = _setup_output_dir(tmp_path)
    (out_dir / "voices" / "narrator.wav").unlink()
    with patch("reader.compile.get_chatterbox_model", return_value=MagicMock()):
        events = list(run_compile(CONTENT_HASH))
    assert any("error" in e for e in events)
    assert not any("done" in e for e in events)


# --- MP3 conversion tests ---

@patch("reader.compile._convert_batch")
@patch("reader.compile.synthesize_line", side_effect=_mock_synthesize)
@patch("reader.compile.get_chatterbox_model", return_value=MagicMock())
def test_convert_batch_called_once_for_remainder_under_10(mock_model, mock_synth, mock_convert, tmp_path, settings):
    settings.OUTPUTS_DIR = tmp_path
    _setup_output_dir(tmp_path)  # 2 lines
    list(run_compile(CONTENT_HASH))
    assert mock_convert.call_count == 1
    assert len(mock_convert.call_args[0][0]) == 2


@patch("reader.compile._convert_batch")
@patch("reader.compile.synthesize_line", side_effect=_mock_synthesize)
@patch("reader.compile.get_chatterbox_model", return_value=MagicMock())
def test_convert_batch_triggered_mid_loop_at_10(mock_model, mock_synth, mock_convert, tmp_path, settings):
    settings.OUTPUTS_DIR = tmp_path
    _setup_output_dir(tmp_path, n_extra_lines=8)  # 2 + 8 = 10 lines exactly
    list(run_compile(CONTENT_HASH))
    assert mock_convert.call_count == 1
    assert len(mock_convert.call_args[0][0]) == 10


@patch("reader.compile._convert_batch")
@patch("reader.compile.synthesize_line", side_effect=_mock_synthesize)
@patch("reader.compile.get_chatterbox_model", return_value=MagicMock())
def test_convert_batch_triggered_twice_for_12_lines(mock_model, mock_synth, mock_convert, tmp_path, settings):
    settings.OUTPUTS_DIR = tmp_path
    _setup_output_dir(tmp_path, n_extra_lines=10)  # 2 + 10 = 12 lines
    list(run_compile(CONTENT_HASH))
    assert mock_convert.call_count == 2
    assert len(mock_convert.call_args_list[0][0][0]) == 10
    assert len(mock_convert.call_args_list[1][0][0]) == 2


@patch("reader.compile._convert_batch")
@patch("reader.compile.synthesize_line", side_effect=_mock_synthesize)
@patch("reader.compile.get_chatterbox_model", return_value=MagicMock())
def test_compile_yields_converting_event_before_batch(mock_model, mock_synth, mock_convert, tmp_path, settings):
    settings.OUTPUTS_DIR = tmp_path
    _setup_output_dir(tmp_path)
    events = list(run_compile(CONTENT_HASH))
    assert any("compile_converting" in e for e in events)


def test_has_mixed_content_true_when_substantial_unquoted():
    assert _has_mixed_content("CAPTAIN MARKOF",
                              '"Is that all?" asked the captain impatiently, without turning his head.')


def test_has_mixed_content_false_for_narrator():
    assert not _has_mixed_content("NARRATOR", "He walked slowly down the road.")


def test_has_mixed_content_false_for_pure_dialogue():
    assert not _has_mixed_content("ALICE", '"Hello there."')


def test_split_mixed_segment_uses_llm_for_complex_case():
    llm_result = [
        {"speaker": "CAPTAIN MARKOF", "text": '"Is that all?"'},
        {"speaker": "NARRATOR", "text": "asked the captain impatiently, without turning his head."},
    ]
    with patch("reader.compile.split_segment", return_value=llm_result):
        parts = _split_mixed_segment("CAPTAIN MARKOF", "impatient",
                                     '"Is that all?" asked the captain impatiently, without turning his head.')
    assert len(parts) == 2
    assert parts[0] == {"speaker": "CAPTAIN MARKOF", "mood": "impatient", "text": '"Is that all?"'}
    assert parts[1] == {"speaker": "NARRATOR", "mood": "", "text": "asked the captain impatiently, without turning his head."}


def test_split_mixed_segment_curly_quotes():
    parts = _split_mixed_segment("CAPTAIN MARKOF", "curious",
                                 '\u201cIs it you?\u201d said the captain, bending his head back. \u201cWhat is it?\u201d')
    assert len(parts) == 3
    assert parts[0]["speaker"] == "CAPTAIN MARKOF"
    assert parts[0]["text"] == '\u201cIs it you?\u201d'
    assert parts[1]["speaker"] == "NARRATOR"
    assert parts[1]["text"] == "said the captain, bending his head back."
    assert parts[2]["speaker"] == "CAPTAIN MARKOF"
    assert parts[2]["text"] == '\u201cWhat is it?\u201d'


def test_split_mixed_segment_pure_dialogue_unchanged():
    parts = _split_mixed_segment("ALICE", "happy", '"I\'ll be there soon."')
    assert len(parts) == 1
    assert parts[0]["speaker"] == "ALICE"


def test_split_mixed_segment_narrator_unchanged():
    parts = _split_mixed_segment("NARRATOR", "", "It was a quiet evening.")
    assert len(parts) == 1
    assert parts[0]["speaker"] == "NARRATOR"


def test_split_mixed_segment_llm_fallback_on_error():
    with patch("reader.compile.split_segment", side_effect=RuntimeError("API down")):
        parts = _split_mixed_segment("BROWN", "neutral",
                                     '"Hello," he said nervously, "how are you?"')
    # Falls back to regex
    assert any(p["speaker"] == "NARRATOR" for p in parts)


def test_parse_segments_splits_multi_speaker_line():
    text = '[NARRATOR] It was quiet.\n[BROWN | neutral] "Hello," [NARRATOR] he said. [BROWN | formal] "Goodbye."'
    segments = _parse_segments(text)
    assert len(segments) == 4
    assert segments[0] == {"speaker": "NARRATOR", "mood": "", "text": "It was quiet."}
    assert segments[1] == {"speaker": "BROWN", "mood": "neutral", "text": '"Hello,"'}
    assert segments[2] == {"speaker": "NARRATOR", "mood": "", "text": "he said."}
    assert segments[3] == {"speaker": "BROWN", "mood": "formal", "text": '"Goodbye."'}


def test_parse_segments_handles_mood_prefix_and_bare_mood():
    text = '[ALICE | mood=angry] "No!" [BOB | sad] "Yes."'
    segments = _parse_segments(text)
    assert segments[0]["speaker"] == "ALICE"
    assert segments[0]["mood"] == "angry"
    assert segments[1]["speaker"] == "BOB"
    assert segments[1]["mood"] == "sad"


def test_convert_to_mp3_calls_ffmpeg_and_deletes_wav(tmp_path):
    wav_path = tmp_path / "test.wav"
    wav_path.write_bytes(b"fake")
    with patch("reader.compile.subprocess.run") as mock_run:
        result = _convert_to_mp3(wav_path)
    assert result == wav_path.with_suffix(".mp3")
    assert not wav_path.exists()
    args = mock_run.call_args[0][0]
    assert args[0] == "ffmpeg"
    assert str(wav_path) in args
    assert str(result) in args


def test_convert_batch_continues_on_individual_failure(tmp_path):
    wav1 = tmp_path / "1_narrator.wav"
    wav2 = tmp_path / "2_alice.wav"
    wav1.write_bytes(b"fake")
    wav2.write_bytes(b"fake")
    with patch("reader.compile._convert_to_mp3", side_effect=RuntimeError("ffmpeg missing")):
        _convert_batch([wav1, wav2])  # must not raise
