import pytest
from pathlib import Path
from reader.output import (
    ensure_output_dir,
    normalize_speaker_names,
    write_speakers,
    write_annotated,
    read_speakers,
    read_annotated,
    parse_annotated_line,
)

SPEAKERS = [
    {"name": "Alice", "sex": "female", "age": "30s", "traits": "brave, kind"},
    {"name": "Bob", "sex": "male", "age": "40s", "traits": "gruff"},
]

ANNOTATED_CHUNKS = [
    "[NARRATOR] It was quiet.\n[ALICE | mood=nervous] \"Hello?\"",
    "[BOB | mood=cold] \"Stay back.\"\n[NARRATOR] He stepped forward.",
]


def test_ensure_output_dir_creates_directory(tmp_path, settings):
    settings.OUTPUTS_DIR = tmp_path
    out_dir = ensure_output_dir("abc123")
    assert out_dir.is_dir()
    assert out_dir.name == "abc123"


def test_write_and_read_speakers(tmp_path, settings):
    settings.OUTPUTS_DIR = tmp_path
    out_dir = ensure_output_dir("abc123")
    write_speakers(SPEAKERS, out_dir)
    result = read_speakers(out_dir)
    assert len(result) == 3  # NARRATOR + 2 speakers
    assert result[0]["name"] == "NARRATOR"
    assert result[1]["name"] == "Alice"
    assert result[1]["sex"] == "female"
    assert result[2]["name"] == "Bob"


def test_speakers_file_includes_narrator(tmp_path, settings):
    settings.OUTPUTS_DIR = tmp_path
    out_dir = ensure_output_dir("abc123")
    write_speakers(SPEAKERS, out_dir)
    text = (out_dir / "speakers.txt").read_text()
    assert "NARRATOR" in text


def test_write_and_read_annotated(tmp_path, settings):
    settings.OUTPUTS_DIR = tmp_path
    out_dir = ensure_output_dir("abc123")
    write_annotated(ANNOTATED_CHUNKS, out_dir)
    lines = read_annotated(out_dir)
    assert len(lines) > 0
    assert any(line["type"] == "narrator" for line in lines)
    assert any(line["type"] == "dialogue" for line in lines)


def test_parse_annotated_line_narrator():
    line = "[NARRATOR] It was a dark night."
    result = parse_annotated_line(line)
    assert result["type"] == "narrator"
    assert result["text"] == "It was a dark night."
    assert result["speaker"] is None
    assert result["mood"] is None


def test_parse_annotated_line_dialogue():
    line = '[ALICE | mood=nervous] "Hello?"'
    result = parse_annotated_line(line)
    assert result["type"] == "dialogue"
    assert result["speaker"] == "ALICE"
    assert result["mood"] == "nervous"
    assert result["text"] == '"Hello?"'


def test_parse_annotated_line_unrecognized_falls_back_to_raw():
    line = "Some plain text with no tag."
    result = parse_annotated_line(line)
    assert result["type"] == "raw"
    assert result["text"] == line


def test_normalize_speaker_names_fixes_casing():
    speakers = [{"name": "Alice Bennet"}, {"name": "Mr. Darcy"}]
    chunks = ['[ALICE BENNET | mood=happy] "Hello."\n[MR. DARCY | mood=cold] "Indeed."']
    result = normalize_speaker_names(chunks, speakers)
    assert "[Alice Bennet | mood=happy]" in result[0]
    assert "[Mr. Darcy | mood=cold]" in result[0]


def test_normalize_speaker_names_preserves_narrator():
    speakers = [{"name": "Alice"}]
    chunks = ["[NARRATOR] It was quiet.\n[ALICE | mood=sad] \"Oh.\""]
    result = normalize_speaker_names(chunks, speakers)
    assert "[NARRATOR]" in result[0]
    assert "[Alice | mood=sad]" in result[0]


def test_normalize_speaker_names_leaves_unknown_names_unchanged():
    speakers = [{"name": "Alice"}]
    chunks = ["[UNKNOWN PERSON | mood=angry] \"Stop!\""]
    result = normalize_speaker_names(chunks, speakers)
    assert "[UNKNOWN PERSON | mood=angry]" in result[0]


def test_write_speakers_preserves_narrator_attributes(tmp_path, settings):
    settings.OUTPUTS_DIR = tmp_path
    out_dir = ensure_output_dir("abc123")
    custom_narrator = {"name": "NARRATOR", "sex": "female", "age": "elderly", "traits": "wise"}
    write_speakers([custom_narrator, {"name": "Alice", "sex": "female", "age": "30s"}], out_dir)
    result = read_speakers(out_dir)
    narrator = next(s for s in result if s["name"] == "NARRATOR")
    assert narrator["sex"] == "female"
    assert narrator["age"] == "elderly"
    assert narrator.get("traits") == "wise"
