import pytest
from unittest.mock import patch, MagicMock
from reader.pipeline import run_pipeline


SHORT_TEXT = (
    "It was a quiet evening.\n\n"
    '"Hello?" said Alice.\n\n'
    '"Stay back," Bob replied.'
)

MOCK_SPEAKERS = [{"name": "Alice", "sex": "female", "age": "30s", "traits": "brave"}]
MOCK_ANNOTATED = '[NARRATOR] It was quiet.\n[ALICE | mood=nervous] "Hello?"'


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
    mock_extract.return_value = MOCK_SPEAKERS
    mock_annotate.return_value = MOCK_ANNOTATED
    # Per-chunk LLM failures are now swallowed (FIX #4); to exercise the outer
    # error path, fail an unguarded step instead.
    mock_write_s.side_effect = RuntimeError("disk full")

    events = list(run_pipeline("abc123", SHORT_TEXT, "Test Book"))
    assert any(e.startswith("data: error") for e in events)


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


@patch("reader.pipeline.generate_voice_sample")
@patch("reader.pipeline.get_tts_model")
@patch("reader.pipeline.write_annotated")
@patch("reader.pipeline.write_speakers")
@patch("reader.pipeline.annotate_chunk")
@patch("reader.pipeline.extract_speakers")
def test_pipeline_completes_when_annotate_chunk_raises(
    mock_extract, mock_annotate, mock_write_s, mock_write_a,
    mock_get_model, mock_gen_voice, tmp_path, settings
):
    settings.OUTPUTS_DIR = tmp_path
    mock_extract.return_value = MOCK_SPEAKERS
    mock_annotate.side_effect = RuntimeError("annotation API down")
    mock_get_model.return_value = MagicMock()

    events = list(run_pipeline("abc123", SHORT_TEXT, "Test Book"))
    # One chunk failing to annotate must not abort the pipeline.
    assert any("done" in e for e in events)
    assert not any(e.startswith("data: error") for e in events)
    assert any("warning" in e for e in events)


@patch("reader.pipeline.generate_voice_sample")
@patch("reader.pipeline.get_tts_model")
@patch("reader.pipeline.write_annotated")
@patch("reader.pipeline.write_speakers")
@patch("reader.pipeline.annotate_chunk")
@patch("reader.pipeline.extract_speakers")
def test_pipeline_completes_when_extract_speakers_raises_for_chunk(
    mock_extract, mock_annotate, mock_write_s, mock_write_a,
    mock_get_model, mock_gen_voice, tmp_path, settings
):
    settings.OUTPUTS_DIR = tmp_path
    mock_extract.side_effect = RuntimeError("extract API hiccup")
    mock_annotate.return_value = MOCK_ANNOTATED
    mock_get_model.return_value = MagicMock()

    events = list(run_pipeline("abc123", SHORT_TEXT, "Test Book"))
    # A per-chunk extract failure is swallowed (treated as no speakers); pipeline finishes.
    assert any("done" in e for e in events)
    assert not any(e.startswith("data: error") for e in events)
