import pytest
from reader.tts import slugify_name, build_instruct, _split_text, _MAX_CHARS


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
    assert "female" in result
    assert "early 20s" in result
    assert "witty" in result


def test_build_instruct_male():
    speaker = {"name": "Bob", "sex": "male", "age": "late 40s", "traits": "gruff, loyal"}
    result = build_instruct(speaker)
    assert "male" in result
    assert "late 40s" in result


def test_build_instruct_unknown_sex_and_age():
    speaker = {"name": "Ghost", "sex": "unknown", "age": "unknown", "traits": ""}
    result = build_instruct(speaker)
    assert len(result) > 0
    assert "Neutral" in result or "neutral" in result


def test_build_instruct_no_traits():
    speaker = {"name": "Guard", "sex": "male", "age": "30s", "traits": ""}
    result = build_instruct(speaker)
    assert "male" in result
    assert "30s" in result


import numpy as np
from unittest.mock import patch, MagicMock
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
    assert "female" in call_kwargs["instruct"]
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


# --- Clone model functions ---

from reader.tts import synthesize_line, get_chatterbox_model


def test_split_text_short_text_unchanged():
    text = "Hello there."
    assert _split_text(text) == [text]


def test_split_text_splits_at_sentence_boundary():
    sentence = "A" * 200 + "."
    text = sentence + " " + sentence
    chunks = _split_text(text)
    assert len(chunks) == 2
    assert all(len(c) <= _MAX_CHARS for c in chunks)


def test_split_text_preserves_all_content():
    text = "First sentence. Second sentence. Third sentence."
    chunks = _split_text(text)
    assert " ".join(chunks) == text


def test_split_text_single_long_sentence_returned_as_is():
    long = "A" * 400
    assert _split_text(long) == [long]


def test_get_chatterbox_model_caches_instance():
    import reader.tts as tts_module
    mock_instance = MagicMock()
    original = tts_module._chatterbox_model
    tts_module._chatterbox_model = None
    try:
        with patch.dict("sys.modules", {
            "chatterbox": MagicMock(),
            "chatterbox.tts": MagicMock(**{
                "ChatterboxTTS.from_pretrained.return_value": mock_instance
            }),
        }):
            first = tts_module.get_chatterbox_model()
            second = tts_module.get_chatterbox_model()
        assert first is second
        assert first is mock_instance
    finally:
        tts_module._chatterbox_model = original


def test_synthesize_line_calls_chatterbox_generate(tmp_path):
    torch = pytest.importorskip("torch", reason="torch not installed")
    import soundfile as sf
    wav_file = tmp_path / "narrator.wav"
    sf.write(str(wav_file), np.zeros(12000, dtype=np.float32), 12000)

    mock_model = MagicMock()
    mock_model.sr = 24000
    mock_model.generate.return_value = torch.zeros(1, 24000)

    with patch("reader.tts.get_chatterbox_model", return_value=mock_model):
        wav, sr = synthesize_line("Hello world.", wav_file, exaggeration=0.6)

    mock_model.generate.assert_called_once_with(
        "Hello world.",
        audio_prompt_path=str(wav_file),
        exaggeration=0.6,
    )
    assert sr == 24000
    assert len(wav) == 24000


def test_synthesize_line_raises_on_empty_audio(tmp_path):
    import soundfile as sf
    wav_file = tmp_path / "narrator.wav"
    sf.write(str(wav_file), np.zeros(12000, dtype=np.float32), 12000)

    mock_model = MagicMock()
    mock_model.sr = 24000
    empty = MagicMock()
    empty.numel.return_value = 0
    mock_model.generate.return_value = empty

    with patch("reader.tts.get_chatterbox_model", return_value=mock_model):
        with pytest.raises(RuntimeError, match="empty audio"):
            synthesize_line("Hello.", wav_file)
