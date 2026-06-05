import pytest
from unittest.mock import patch, MagicMock
from reader.pipeline import run_book_pipeline

CHAPTERS = [
    {"index": 1, "title": "Chapter 1", "text": '"Hello," said Alice. It was quiet.'},
    {"index": 2, "title": "Chapter 2", "text": '"Goodbye," said Bob. He left.'},
]

MOCK_SPEAKERS = [{"name": "Alice", "sex": "female", "age": "30s", "traits": "brave", "nationality": "British"}]
MOCK_ANNOTATED = '[NARRATOR] It was quiet.\n[ALICE | mood=happy] "Hello."'


@patch("reader.pipeline.generate_voice_sample")
@patch("reader.pipeline.get_tts_model")
@patch("reader.pipeline.write_annotated")
@patch("reader.pipeline.write_speakers")
@patch("reader.pipeline.annotate_chunk")
@patch("reader.pipeline.extract_speakers")
def test_book_pipeline_yields_done(
    mock_extract, mock_annotate, mock_write_s, mock_write_a,
    mock_get_model, mock_gen_voice, tmp_path, settings
):
    settings.OUTPUTS_DIR = tmp_path
    mock_extract.return_value = MOCK_SPEAKERS
    mock_annotate.return_value = MOCK_ANNOTATED
    mock_get_model.return_value = MagicMock()
    events = list(run_book_pipeline("abc123", CHAPTERS, "Test Book"))
    assert any("done" in e for e in events)


@patch("reader.pipeline.generate_voice_sample")
@patch("reader.pipeline.get_tts_model")
@patch("reader.pipeline.write_annotated")
@patch("reader.pipeline.write_speakers")
@patch("reader.pipeline.annotate_chunk")
@patch("reader.pipeline.extract_speakers")
def test_book_pipeline_yields_chapter_start_and_done_events(
    mock_extract, mock_annotate, mock_write_s, mock_write_a,
    mock_get_model, mock_gen_voice, tmp_path, settings
):
    settings.OUTPUTS_DIR = tmp_path
    mock_extract.return_value = MOCK_SPEAKERS
    mock_annotate.return_value = MOCK_ANNOTATED
    mock_get_model.return_value = MagicMock()
    events = list(run_book_pipeline("abc123", CHAPTERS, "Test Book"))
    starts = [e for e in events if "chapter_start" in e]
    dones = [e for e in events if "chapter_done" in e]
    assert len(starts) == 2
    assert len(dones) == 2


@patch("reader.pipeline.generate_voice_sample")
@patch("reader.pipeline.get_tts_model")
@patch("reader.pipeline.write_annotated")
@patch("reader.pipeline.write_speakers")
@patch("reader.pipeline.annotate_chunk")
@patch("reader.pipeline.extract_speakers")
def test_book_pipeline_writes_per_chapter_annotated(
    mock_extract, mock_annotate, mock_write_s, mock_write_a,
    mock_get_model, mock_gen_voice, tmp_path, settings
):
    settings.OUTPUTS_DIR = tmp_path
    mock_extract.return_value = MOCK_SPEAKERS
    mock_annotate.return_value = MOCK_ANNOTATED
    mock_get_model.return_value = MagicMock()
    list(run_book_pipeline("abc123", CHAPTERS, "Test Book"))
    assert mock_write_a.call_count == 2
    dirs_used = [call[0][1] for call in mock_write_a.call_args_list]
    assert any("01" in str(d) for d in dirs_used)
    assert any("02" in str(d) for d in dirs_used)


@patch("reader.pipeline.generate_voice_sample")
@patch("reader.pipeline.get_tts_model")
@patch("reader.pipeline.write_annotated")
@patch("reader.pipeline.write_speakers")
@patch("reader.pipeline.annotate_chunk")
@patch("reader.pipeline.extract_speakers")
def test_book_pipeline_passes_known_speakers_to_extract(
    mock_extract, mock_annotate, mock_write_s, mock_write_a,
    mock_get_model, mock_gen_voice, tmp_path, settings
):
    settings.OUTPUTS_DIR = tmp_path
    mock_extract.return_value = MOCK_SPEAKERS
    mock_annotate.return_value = MOCK_ANNOTATED
    mock_get_model.return_value = MagicMock()
    list(run_book_pipeline("abc123", CHAPTERS, "Test Book"))
    # By the time chapter 2 is processed, Alice should be in known_speakers
    # find any call where known_speakers kwarg contains Alice
    calls_with_known = [
        c for c in mock_extract.call_args_list
        if c[1].get("known_speakers") and
        any(s["name"] == "Alice" for s in c[1]["known_speakers"])
    ]
    assert len(calls_with_known) > 0


@patch("reader.pipeline.generate_voice_sample")
@patch("reader.pipeline.get_tts_model")
@patch("reader.pipeline.write_annotated")
@patch("reader.pipeline.write_speakers")
@patch("reader.pipeline.annotate_chunk")
@patch("reader.pipeline.extract_speakers")
def test_book_pipeline_skips_voice_for_existing_speakers(
    mock_extract, mock_annotate, mock_write_s, mock_write_a,
    mock_get_model, mock_gen_voice, tmp_path, settings
):
    settings.OUTPUTS_DIR = tmp_path
    mock_extract.return_value = MOCK_SPEAKERS
    mock_annotate.return_value = MOCK_ANNOTATED
    mock_get_model.return_value = MagicMock()
    # Pre-create alice.wav so her voice should not be regenerated
    voices_dir = tmp_path / "abc123" / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    (voices_dir / "alice.wav").write_bytes(b"fake")
    list(run_book_pipeline("abc123", CHAPTERS, "Test Book"))
    generated_names = [c[0][0]["name"] for c in mock_gen_voice.call_args_list]
    assert "Alice" not in generated_names
