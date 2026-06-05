import logging

from reader.chunker import chunk_text
from reader.llm import extract_speakers, merge_speakers, annotate_chunk
from reader.output import ensure_output_dir, normalize_speaker_names, write_speakers, write_annotated, read_speakers
from reader.tts import generate_voice_sample, get_tts_model, slugify_name

logger = logging.getLogger(__name__)


def run_pipeline(content_hash: str, text: str, title: str):
    """Generator that runs the three-pass pipeline and yields SSE event strings."""
    try:
        yield "data: parsing\n\n"

        out_dir = ensure_output_dir(content_hash)
        voices_dir = out_dir / "voices"
        voices_dir.mkdir(exist_ok=True)

        annotated_path = out_dir / "annotated.txt"
        speakers_path = out_dir / "speakers.txt"

        if annotated_path.exists() and speakers_path.exists():
            # Resume: skip Pass 1 + 2, reload speakers from existing file
            logger.info("Skipping annotation for %s — already exists", content_hash)
            all_existing = read_speakers(out_dir)
            merged_speakers = [s for s in all_existing if s["name"] != "NARRATOR"]
            narrator_entry = next(
                (s for s in all_existing if s["name"] == "NARRATOR"),
                {"name": "NARRATOR", "sex": "unknown", "age": "unknown", "traits": ""},
            )
            yield "data: chunk_progress 1 1\n\n"
        else:
            narrator_entry = {"name": "NARRATOR", "sex": "unknown", "age": "unknown", "traits": ""}
            chunks = chunk_text(text)
            total = len(chunks)

            # Pass 1: extract speakers from each chunk
            per_chunk_speakers = []
            for chunk in chunks:
                speakers = extract_speakers(chunk)
                per_chunk_speakers.append(speakers)
            merged_speakers = merge_speakers(per_chunk_speakers)

            # Pass 2: annotate each chunk
            annotated_chunks = []
            for i, chunk in enumerate(chunks, start=1):
                annotated = annotate_chunk(chunk, merged_speakers, chunk_index=i)
                annotated_chunks.append(annotated)
                yield f"data: chunk_progress {i} {total}\n\n"

            annotated_chunks = normalize_speaker_names(annotated_chunks, merged_speakers)
            write_speakers(merged_speakers, out_dir)
            write_annotated(annotated_chunks, out_dir)

        # Pass 3: generate voices, skipping any that already exist
        speakers_for_tts = []
        if not (voices_dir / "narrator.wav").exists():
            speakers_for_tts.append(narrator_entry)
        for speaker in merged_speakers:
            slug = slugify_name(speaker["name"])
            if not (voices_dir / f"{slug}.wav").exists():
                speakers_for_tts.append(speaker)

        if speakers_for_tts:
            try:
                get_tts_model()
                yield "data: voices_start\n\n"
                total_voices = len(speakers_for_tts)
                for i, speaker in enumerate(speakers_for_tts, start=1):
                    try:
                        logger.info("Generating voice for %s | %s", speaker["name"], speaker)
                        generate_voice_sample(speaker, voices_dir)
                        logger.info("Voice generated successfully for %s", speaker["name"])
                    except Exception as exc:
                        logger.exception("Failed to generate voice for %s", speaker["name"])
                        yield f"data: voice_warning Failed voice for {speaker['name']}: {exc}\n\n"
                    yield f"data: voice_progress {i} {total_voices}\n\n"
            except Exception as exc:
                logger.exception("Voice generation unavailable")
                yield f"data: voice_warning Voice generation unavailable: {exc}\n\n"
        else:
            logger.info("All voices already exist, skipping Pass 3")

        yield "data: done\n\n"

    except Exception as exc:
        yield f"data: error {exc}\n\n"


def run_book_pipeline(content_hash: str, chapters: list[dict], title: str):
    """Generator that processes a multi-chapter book and yields SSE event strings."""
    try:
        logger.info("run_book_pipeline starting for %s (%d chapters)", content_hash, len(chapters))
        out_dir = ensure_output_dir(content_hash)
        voices_dir = out_dir / "voices"
        voices_dir.mkdir(exist_ok=True)
        total_chapters = len(chapters)
        pad = max(2, len(str(total_chapters)))

        # Load existing speakers (preserving any custom NARRATOR attributes)
        logger.info("run_book_pipeline loading existing speakers for %s", content_hash)
        if (out_dir / "speakers.txt").exists():
            all_existing = read_speakers(out_dir)
            narrator_entry = next(
                (s for s in all_existing if s["name"] == "NARRATOR"),
                {"name": "NARRATOR", "sex": "unknown", "age": "unknown", "traits": ""},
            )
            known_speakers = [s for s in all_existing if s["name"] != "NARRATOR"]
        else:
            narrator_entry = {"name": "NARRATOR", "sex": "unknown", "age": "unknown", "traits": ""}
            known_speakers = []

        for i, chapter in enumerate(chapters, start=1):
            chapter_title = chapter["title"]
            chapter_dir = out_dir / "chapters" / str(i).zfill(pad)
            chapter_dir.mkdir(parents=True, exist_ok=True)

            logger.info("run_book_pipeline chapter %d/%d starting: %r", i, total_chapters, chapter_title)
            # Read chapter text from its raw.txt (written by process_view at upload time)
            raw_path = chapter_dir / "raw.txt"
            if not raw_path.exists():
                # Fallback: chapter dict may still contain text (old format)
                chapter_text = chapter.get("text", "")
                if chapter_text:
                    raw_path.write_text(chapter_text, encoding="utf-8")
                else:
                    logger.error("Chapter %d has no raw.txt and no text in dict — skipping", i)
                    continue
            else:
                chapter_text = raw_path.read_text(encoding="utf-8")

            safe_title = chapter_title.replace("\n", " ").replace("\r", "")
            yield f"data: chapter_start {i} {total_chapters} {safe_title}\n\n"

            logger.info("run_book_pipeline chapter %d text loaded: %d chars", i, len(chapter_text))
            annotated_path = chapter_dir / "annotated.txt"
            if annotated_path.exists():
                # Resume: skip Pass 1 + 2 for this chapter, reload cumulative speakers
                logger.info("Skipping annotation for chapter %d — already exists", i)
                if (out_dir / "speakers.txt").exists():
                    current = read_speakers(out_dir)
                    narrator_entry = next(
                        (s for s in current if s["name"] == "NARRATOR"), narrator_entry
                    )
                    known_speakers = [s for s in current if s["name"] != "NARRATOR"]
                yield "data: chunk_progress 1 1\n\n"
            else:
                # Pass 1: extract speakers, merge into cumulative list
                logger.info("run_book_pipeline chapter %d Pass 1: chunking text", i)
                chunks = chunk_text(chapter_text)
                chapter_total = len(chunks)
                logger.info("run_book_pipeline chapter %d Pass 1: %d chunks, calling LLM", i, chapter_total)
                per_chunk_speakers = []
                for ci, chunk in enumerate(chunks, start=1):
                    logger.info("run_book_pipeline chapter %d Pass 1: extract_speakers chunk %d/%d", i, ci, chapter_total)
                    speakers = extract_speakers(chunk, known_speakers=known_speakers)
                    logger.info("run_book_pipeline chapter %d Pass 1: chunk %d/%d done, found %d speakers", i, ci, chapter_total, len(speakers))
                    per_chunk_speakers.append(speakers)

                merged_chapter = merge_speakers(per_chunk_speakers)
                known_names_lower = {s["name"].lower() for s in known_speakers}
                for s in known_speakers:
                    for alias in s.get("aliases", []):
                        known_names_lower.add(alias.lower())
                truly_new = [s for s in merged_chapter if s["name"].lower() not in known_names_lower]
                known_speakers = known_speakers + truly_new
                write_speakers([narrator_entry] + known_speakers, out_dir)

                # Pass 2: annotate chapter with full cumulative speaker list
                annotated_chunks = []
                for j, chunk in enumerate(chunks, start=1):
                    annotated = annotate_chunk(chunk, known_speakers, chunk_index=j)
                    annotated_chunks.append(annotated)
                    yield f"data: chunk_progress {j} {chapter_total}\n\n"

                annotated_chunks = normalize_speaker_names(annotated_chunks, known_speakers)
                write_annotated(annotated_chunks, chapter_dir)
                ann_path = chapter_dir / "annotated.txt"
                if ann_path.exists():
                    raw_bytes = len(chapter_text.encode())
                    ann_bytes = ann_path.stat().st_size
                    if ann_bytes < raw_bytes:
                        logger.warning(
                            "Chapter %d annotated.txt (%d bytes) is smaller than raw.txt (%d bytes) "
                            "— annotation may be incomplete",
                            i, ann_bytes, raw_bytes,
                        )

            # Pass 3: generate voices only for speakers without an existing WAV
            speakers_for_tts = []
            if not (voices_dir / "narrator.wav").exists():
                speakers_for_tts.append(narrator_entry)
            for speaker in known_speakers:
                slug = slugify_name(speaker["name"])
                if not (voices_dir / f"{slug}.wav").exists():
                    speakers_for_tts.append(speaker)

            if speakers_for_tts:
                try:
                    get_tts_model()
                    yield "data: voices_start\n\n"
                    total_voices = len(speakers_for_tts)
                    for k, speaker in enumerate(speakers_for_tts, start=1):
                        try:
                            logger.info("Generating voice for %s", speaker["name"])
                            generate_voice_sample(speaker, voices_dir)
                        except Exception as exc:
                            logger.exception("Failed to generate voice for %s", speaker["name"])
                            yield f"data: voice_warning Failed voice for {speaker['name']}: {exc}\n\n"
                        yield f"data: voice_progress {k} {total_voices}\n\n"
                except Exception as exc:
                    logger.exception("Voice generation unavailable")
                    yield f"data: voice_warning Voice generation unavailable: {exc}\n\n"
            else:
                logger.info("Chapter %d: all voices already exist, skipping Pass 3", i)

            yield f"data: chapter_done {i} {total_chapters}\n\n"

        yield "data: done\n\n"

    except Exception as exc:
        logger.exception("Book pipeline failed")
        yield f"data: error {exc}\n\n"
