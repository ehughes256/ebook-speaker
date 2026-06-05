import json
import pytest
from unittest.mock import patch, MagicMock
from reader.llm import extract_speakers, merge_speakers


def _mock_openai_response(content: str):
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


SPEAKER_JSON = json.dumps({
    "speakers": [
        {"name": "Alice", "sex": "female", "age": "30s", "traits": "brave, kind"},
        {"name": "Bob", "sex": "male", "age": "40s", "traits": "gruff, loyal"},
    ]
})


def test_extract_speakers_parses_llm_response():
    chunk = {"content": "\"Hello,\" said Alice. \"Indeed,\" said Bob.", "context": ""}
    with patch("reader.llm._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_openai_response(SPEAKER_JSON)
        speakers = extract_speakers(chunk)
    assert len(speakers) == 2
    assert speakers[0]["name"] == "Alice"
    assert speakers[1]["sex"] == "male"


def test_extract_speakers_includes_context_in_prompt():
    chunk = {"content": "\"Yes,\" she said.", "context": "Prior scene text."}
    with patch("reader.llm._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_openai_response('{"speakers": []}')
        extract_speakers(chunk)
        call_args = mock_client.chat.completions.create.call_args
        prompt = call_args[1]["messages"][1]["content"]
    assert "Prior scene text." in prompt


def test_extract_speakers_returns_empty_on_no_speakers():
    chunk = {"content": "It was a stormy night.", "context": ""}
    with patch("reader.llm._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_openai_response('{"speakers": []}')
        result = extract_speakers(chunk)
    assert result == []


def test_merge_speakers_deduplicates_by_name_case_insensitive():
    all_speakers = [
        [{"name": "Alice", "sex": "female", "age": "30s", "traits": "brave"}],
        [{"name": "alice", "sex": "female", "age": "30s", "traits": "brave"}],
        [{"name": "Bob", "sex": "male", "age": "40s", "traits": "gruff"}],
    ]
    merged = merge_speakers(all_speakers)
    names = [s["name"] for s in merged]
    assert len(merged) == 2
    assert "Alice" in names
    assert "Bob" in names


def test_merge_speakers_first_occurrence_wins():
    all_speakers = [
        [{"name": "Alice", "sex": "female", "age": "20s", "traits": "shy"}],
        [{"name": "ALICE", "sex": "female", "age": "30s", "traits": "bold"}],
    ]
    merged = merge_speakers(all_speakers)
    assert merged[0]["age"] == "20s"


from reader.llm import annotate_chunk

SPEAKERS = [
    {"name": "Alice", "sex": "female", "age": "30s", "traits": "brave"},
    {"name": "Bob", "sex": "male", "age": "40s", "traits": "gruff"},
]

ANNOTATED_RESPONSE = (
    '[NARRATOR] It was a quiet evening.\n'
    '[ALICE | mood=nervous] "Are you sure about this?"\n'
    '[BOB | mood=gruff] "Absolutely," he said.'
)


def test_annotate_chunk_returns_llm_response():
    chunk = {"content": 'It was a quiet evening. "Are you sure?" "Absolutely."', "context": ""}
    with patch("reader.llm._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_openai_response(ANNOTATED_RESPONSE)
        result = annotate_chunk(chunk, SPEAKERS)
    assert result == ANNOTATED_RESPONSE


def test_annotate_chunk_includes_speaker_names_in_prompt():
    chunk = {"content": '"Hello," she said.', "context": ""}
    with patch("reader.llm._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_openai_response("[NARRATOR] text")
        annotate_chunk(chunk, SPEAKERS)
        call_args = mock_client.chat.completions.create.call_args
        prompt = call_args[1]["messages"][1]["content"]
    assert "Alice" in prompt
    assert "Bob" in prompt


def test_annotate_chunk_passes_model_and_messages():
    chunk = {"content": "text", "context": ""}
    with patch("reader.llm._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_openai_response("[NARRATOR] text")
        annotate_chunk(chunk, SPEAKERS)
        call_args = mock_client.chat.completions.create.call_args
    assert "temperature" not in call_args[1]
    assert call_args[1]["model"] is not None


def test_extract_speakers_includes_known_speakers_in_prompt():
    chunk = {"content": '"Hello," said Alice.', "context": ""}
    known = [{"name": "Alice", "sex": "female", "age": "30s"}]
    with patch("reader.llm._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_openai_response('{"speakers": []}')
        extract_speakers(chunk, known_speakers=known)
        prompt = mock_client.chat.completions.create.call_args[1]["messages"][1]["content"]
    assert "Alice" in prompt
    assert "female" in prompt


def test_extract_speakers_no_known_speakers_unchanged():
    chunk = {"content": '"Hi," said Bob.', "context": ""}
    with patch("reader.llm._client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_openai_response('{"speakers": []}')
        extract_speakers(chunk, known_speakers=None)
        prompt = mock_client.chat.completions.create.call_args[1]["messages"][0]["content"]
    assert "Known characters" not in prompt
