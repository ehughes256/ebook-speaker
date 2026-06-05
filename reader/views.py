import json
import logging
import re
import shutil
from pathlib import Path

from django.conf import settings
from django.http import FileResponse, Http404, HttpResponseBadRequest, JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render

logger = logging.getLogger(__name__)

_CONTENT_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SLUG_RE = re.compile(r"^[a-z0-9_]+$")


def validate_content_hash(content_hash):
    """Reject anything that is not a 64-char lowercase hex SHA-256 (blocks path traversal)."""
    if not _CONTENT_HASH_RE.match(content_hash):
        raise Http404


def validate_slug(slug):
    """Reject anything outside the charset reader.tts.slugify_name produces (blocks path traversal)."""
    if not _SLUG_RE.match(slug):
        raise Http404


def _assert_within_outputs(path):
    """Defense-in-depth: ensure the resolved path stays inside OUTPUTS_DIR."""
    base = Path(settings.OUTPUTS_DIR).resolve()
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(base):
        raise Http404


def book_output_dir(content_hash):
    return Path(settings.OUTPUTS_DIR) / content_hash


def load_chapters(out_dir):
    """Return the parsed chapters.json list, or None if absent."""
    chapters_path = out_dir / "chapters.json"
    if not chapters_path.exists():
        return None
    return json.loads(chapters_path.read_text(encoding="utf-8"))

from reader.ingestion import (
    compute_hash, normalize_input, split_text_chapters, split_epub_chapters,
    extract_epub_cover, fetch_openlibrary_cover, save_cover,
)
from reader.models import ProcessedBook
from django.urls import reverse
from reader.output import chapter_dir_path, ensure_output_dir, read_annotated, read_speakers, update_speaker_attrs
from reader.pipeline import run_pipeline, run_book_pipeline
from reader.tts import generate_voice_sample, slugify_name


def upload_view(request):
    recent_books = ProcessedBook.objects.filter(status="done").order_by("-created_at")[:10]
    return render(request, "reader/upload.html", {"recent_books": recent_books})


def cover_view(request, content_hash):
    validate_content_hash(content_hash)
    cover_path = book_output_dir(content_hash) / "cover"
    _assert_within_outputs(cover_path)
    if not cover_path.exists():
        raise Http404
    header = cover_path.read_bytes()[:4]
    content_type = "image/png" if header[:4] == b"\x89PNG" else "image/jpeg"
    return FileResponse(cover_path.open("rb"), content_type=content_type)


def listen_list_view(request):
    books = ProcessedBook.objects.filter(status="done").order_by("-created_at")
    available = []
    for book in books:
        out_dir = book_output_dir(book.content_hash)
        chapters_meta = load_chapters(out_dir)
        if chapters_meta is not None:
            n = len(chapters_meta)
            audio_count = sum(
                1 for i in range(1, n + 1)
                if (chapter_dir_path(out_dir, i, n) / "compiled" / "full.mp3").exists()
            )
            if audio_count > 0:
                available.append({"book": book, "chapter_count": audio_count, "is_multi": True, "has_cover": (out_dir / "cover").exists()})
        else:
            if (out_dir / "compiled" / "full.mp3").exists():
                available.append({"book": book, "chapter_count": 1, "is_multi": False, "has_cover": (out_dir / "cover").exists()})
    return render(request, "reader/listen_list.html", {"available": available})


def listen_book_view(request, content_hash):
    validate_content_hash(content_hash)
    book = get_object_or_404(ProcessedBook, content_hash=content_hash, status="done")
    out_dir = book_output_dir(content_hash)
    chapters_meta = load_chapters(out_dir)

    if chapters_meta is not None:
        n = len(chapters_meta)
        chapters = []
        for ch in chapters_meta:
            idx = ch["index"]
            if (chapter_dir_path(out_dir, idx, n) / "compiled" / "full.mp3").exists():
                chapters.append({
                    "index": idx,
                    "title": ch["title"],
                    "audio_url": reverse("reader:full_audio_chapter", args=[content_hash, idx]),
                })
    else:
        if not (out_dir / "compiled" / "full.mp3").exists():
            raise Http404
        chapters = [{
            "index": 1,
            "title": book.title,
            "audio_url": reverse("reader:full_audio", args=[content_hash]),
        }]

    if not chapters:
        raise Http404

    has_cover = (out_dir / "cover").exists()
    return render(request, "reader/listen_book.html", {"book": book, "chapters": chapters, "has_cover": has_cover})


def process_view(request):
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")

    input_text = request.POST.get("input_text", "").strip()
    input_file = request.FILES.get("input_file")

    if not input_text and not input_file:
        return HttpResponseBadRequest("Provide text or a file")

    if input_text and input_file:
        return HttpResponseBadRequest("Provide text or a file, not both")

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if input_file and input_file.size > max_bytes:
        return HttpResponseBadRequest(f"File too large (max {settings.MAX_UPLOAD_MB} MB)")
    if input_text and len(input_text) > max_bytes:
        return HttpResponseBadRequest(f"Text too large (max {settings.MAX_UPLOAD_MB} MB)")

    try:
        if input_file:
            file_bytes = input_file.read()
            text, title = normalize_input(file_bytes=file_bytes, filename=input_file.name, text=None)
            ext = Path(input_file.name).suffix.lower()
            chapters = split_epub_chapters(file_bytes) if ext == ".epub" else split_text_chapters(text)
        else:
            text, title = normalize_input(file_bytes=None, filename=None, text=input_text)
            chapters = split_text_chapters(text)
    except ValueError as exc:
        return HttpResponseBadRequest(str(exc))

    content_hash = compute_hash(text)
    book, created = ProcessedBook.objects.get_or_create(
        content_hash=content_hash,
        defaults={
            "title": title,
            "status": "pending",
            "output_path": f"outputs/{content_hash}/",
        },
    )

    out_dir = ensure_output_dir(content_hash)
    if created:
        (out_dir / "raw.txt").write_text(text, encoding="utf-8")
        # Try to find and save cover art
        cover_path = out_dir / "cover"
        if not cover_path.exists():
            cover_bytes = None
            if input_file and Path(input_file.name).suffix.lower() == ".epub":
                cover_bytes = extract_epub_cover(file_bytes)
            if not cover_bytes:
                cover_bytes = fetch_openlibrary_cover(title)
            if cover_bytes:
                save_cover(out_dir, cover_bytes)

    chapters_path = out_dir / "chapters.json"
    if chapters is not None and len(chapters) > 1 and not chapters_path.exists() and book.status != "done":
        # Write metadata only — texts go to per-chapter raw.txt files to keep chapters.json small
        chapters_meta = [{"index": i + 1, "title": c["title"]} for i, c in enumerate(chapters)]
        chapters_path.write_text(json.dumps(chapters_meta, ensure_ascii=False), encoding="utf-8")
        # Pre-write chapter raw.txt files so stream_view can read them without the chapter texts
        pad = max(2, len(str(len(chapters))))
        for i, c in enumerate(chapters):
            chapter_dir = out_dir / "chapters" / str(i + 1).zfill(pad)
            chapter_dir.mkdir(parents=True, exist_ok=True)
            raw_path = chapter_dir / "raw.txt"
            if not raw_path.exists():
                raw_path.write_text(c["text"], encoding="utf-8")

    if book.status == "done":
        if _has_incomplete_chapters(out_dir):
            book.status = "pending"
            book.save(update_fields=["status", "updated_at"])
        else:
            return redirect("reader:results", content_hash=content_hash)

    return redirect("reader:progress", content_hash=content_hash)


def _has_incomplete_chapters(out_dir: Path) -> bool:
    """Return True if a chapter book is missing any annotated.txt files."""
    chapters_meta = load_chapters(out_dir)
    if chapters_meta is None:
        return not (out_dir / "annotated.txt").exists()
    n = len(chapters_meta)
    pad = max(2, len(str(n)))
    return any(
        not (out_dir / "chapters" / str(i).zfill(pad) / "annotated.txt").exists()
        for i in range(1, n + 1)
    )


def progress_view(request, content_hash):
    validate_content_hash(content_hash)
    book = get_object_or_404(ProcessedBook, content_hash=content_hash)
    return render(request, "reader/progress.html", {"book": book})


def stream_view(request, content_hash):
    validate_content_hash(content_hash)
    logger.debug("stream_view called for %s", content_hash)

    book = get_object_or_404(ProcessedBook, content_hash=content_hash)
    logger.debug("stream_view book status=%s for %s", book.status, content_hash)

    if book.status == "done":
        def _done():
            yield "data: done\n\n"
        resp = StreamingHttpResponse(_done(), content_type="text/event-stream")
        resp["Cache-Control"] = "no-cache"
        resp["X-Accel-Buffering"] = "no"
        return resp

    # Atomically claim the book so two concurrent requests can't both run the pipeline.
    claimed = ProcessedBook.objects.filter(
        content_hash=content_hash, status__in=["pending", "failed"]
    ).update(status="processing")
    if claimed == 0:
        logger.debug("stream_view could not claim %s (already processing)", content_hash)

        def _busy():
            yield "data: error Another process is already handling this book.\n\n"
        resp = StreamingHttpResponse(_busy(), content_type="text/event-stream")
        resp["Cache-Control"] = "no-cache"
        resp["X-Accel-Buffering"] = "no"
        return resp
    logger.debug("stream_view status saved for %s", content_hash)

    out_dir = book_output_dir(content_hash)
    raw_path = out_dir / "raw.txt"
    text = raw_path.read_text(encoding="utf-8") if raw_path.exists() else ""
    chapters_meta = load_chapters(out_dir)

    if chapters_meta is not None:
        logger.debug("stream_view routing to run_book_pipeline, %d chapters for %s", len(chapters_meta), content_hash)
        pipeline_gen = run_book_pipeline(content_hash, chapters_meta, book.title)
    else:
        logger.debug("stream_view routing to run_pipeline for %s", content_hash)
        pipeline_gen = run_pipeline(content_hash, text, book.title)

    def _event_stream():
        logger.debug("stream_view _event_stream generator started for %s", content_hash)
        from django.db import close_old_connections
        close_old_connections()
        logger.debug("stream_view connections closed, beginning pipeline for %s", content_hash)
        success = True
        error_message = ""
        try:
            for event in pipeline_gen:
                yield event
                if event.startswith("data: error"):
                    success = False
                    error_message = event[len("data: error"):].strip()
        except Exception as exc:
            success = False
            error_message = str(exc)
            logger.exception("stream_view pipeline raised for %s", content_hash)
            yield f"data: error {exc}\n\n"
        update_fields = {"status": "done" if success else "failed"}
        if not success:
            # error_message is a TextField; guard against absurdly long values.
            update_fields["error_message"] = error_message[:5000]
        ProcessedBook.objects.filter(content_hash=content_hash).update(**update_fields)

    resp = StreamingHttpResponse(_event_stream(), content_type="text/event-stream")
    resp["Cache-Control"] = "no-cache"
    resp["X-Accel-Buffering"] = "no"
    return resp


def delete_view(request, content_hash):
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")
    validate_content_hash(content_hash)
    book = get_object_or_404(ProcessedBook, content_hash=content_hash)
    book.delete()
    out_dir = book_output_dir(content_hash)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    return redirect("reader:upload")


def update_speaker_view(request, content_hash, slug):
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")
    validate_content_hash(content_hash)
    validate_slug(slug)
    get_object_or_404(ProcessedBook, content_hash=content_hash, status="done")
    out_dir = book_output_dir(content_hash)
    sex = request.POST.get("sex", "").strip()
    age = request.POST.get("age", "").strip()
    nationality = request.POST.get("nationality", "").strip()
    traits = request.POST.get("traits", "").strip()
    aliases_raw = request.POST.get("aliases", "").strip()
    aliases = [a.strip() for a in aliases_raw.split(",") if a.strip()] if aliases_raw else []
    if not update_speaker_attrs(out_dir, slug, sex, age, nationality, traits, aliases=aliases):
        raise Http404
    return JsonResponse({"ok": True})


def full_audio_view(request, content_hash, chapter=None):
    validate_content_hash(content_hash)
    get_object_or_404(ProcessedBook, content_hash=content_hash, status="done")
    out_dir = book_output_dir(content_hash)
    if chapter is not None:
        chapters_meta = load_chapters(out_dir)
        if chapters_meta is None:
            raise Http404
        mp3_path = chapter_dir_path(out_dir, chapter, len(chapters_meta)) / "compiled" / "full.mp3"
    else:
        mp3_path = out_dir / "compiled" / "full.mp3"
    _assert_within_outputs(mp3_path)
    if not mp3_path.exists():
        raise Http404
    return FileResponse(mp3_path.open("rb"), content_type="audio/mpeg")


def chapter_content_view(request, content_hash, chapter):
    validate_content_hash(content_hash)
    get_object_or_404(ProcessedBook, content_hash=content_hash, status="done")
    out_dir = book_output_dir(content_hash)
    chapters_meta = load_chapters(out_dir)
    if chapters_meta is None:
        raise Http404
    if chapter < 1 or chapter > len(chapters_meta):
        raise Http404
    chapter_dir = chapter_dir_path(out_dir, chapter, len(chapters_meta))
    annotated_path = chapter_dir / "annotated.txt"
    annotated_lines = read_annotated(chapter_dir) if annotated_path.exists() else []
    has_full_audio = (chapter_dir / "compiled" / "full.mp3").exists()
    lines_data = [
        {"type": line["type"], "speaker": line["speaker"], "mood": line["mood"], "text": line["text"]}
        for line in annotated_lines
    ]
    return JsonResponse({"lines": lines_data, "has_full_audio": has_full_audio})


def compile_all_view(request, content_hash):
    validate_content_hash(content_hash)
    book = get_object_or_404(ProcessedBook, content_hash=content_hash, status="done")
    out_dir = book_output_dir(content_hash)
    chapters_meta = load_chapters(out_dir)

    if chapters_meta is not None:
        n = len(chapters_meta)
        missing = []
        for ch in chapters_meta:
            idx = ch["index"]
            if not (chapter_dir_path(out_dir, idx, n) / "compiled" / "full.mp3").exists():
                missing.append({
                    "index": idx,
                    "title": ch["title"],
                    "stream_url": reverse("reader:compile_stream_chapter", args=[content_hash, idx]),
                })
    else:
        if not (out_dir / "compiled" / "full.mp3").exists():
            missing = [{"index": None, "title": book.title,
                        "stream_url": reverse("reader:compile_stream", args=[content_hash])}]
        else:
            missing = []

    return render(request, "reader/compile_all.html", {
        "book": book,
        "missing": missing,
    })


def compile_view(request, content_hash, chapter=None):
    validate_content_hash(content_hash)
    book = get_object_or_404(ProcessedBook, content_hash=content_hash, status="done")
    chapter_title = None
    if chapter is not None:
        out_dir = book_output_dir(content_hash)
        if not (out_dir / "chapters.json").exists():
            raise Http404
        try:
            meta = load_chapters(out_dir)
            match = next((c for c in meta if c["index"] == chapter), None)
            if match:
                chapter_title = match["title"]
        except Exception:
            pass
    return render(request, "reader/compile.html", {"book": book, "chapter": chapter, "chapter_title": chapter_title})


def compile_stream_view(request, content_hash, chapter=None):
    validate_content_hash(content_hash)
    get_object_or_404(ProcessedBook, content_hash=content_hash, status="done")
    from reader.compile import run_compile

    resp = StreamingHttpResponse(
        run_compile(content_hash, chapter=chapter), content_type="text/event-stream"
    )
    resp["Cache-Control"] = "no-cache"
    resp["X-Accel-Buffering"] = "no"
    return resp


def voice_view(request, content_hash, slug):
    validate_content_hash(content_hash)
    validate_slug(slug)
    out_dir = book_output_dir(content_hash)
    wav_path = out_dir / "voices" / f"{slug}.wav"
    _assert_within_outputs(wav_path)
    if not wav_path.exists():
        raise Http404
    return FileResponse(wav_path.open("rb"), content_type="audio/wav")


def regenerate_voice_view(request, content_hash, slug):
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")
    validate_content_hash(content_hash)
    validate_slug(slug)
    out_dir = book_output_dir(content_hash)
    speakers = read_speakers(out_dir)
    speaker = next((s for s in speakers if slugify_name(s["name"]) == slug), None)
    if speaker is None:
        raise Http404
    try:
        voices_dir = out_dir / "voices"
        voices_dir.mkdir(exist_ok=True)
        generate_voice_sample(speaker, voices_dir)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)
    return JsonResponse({"ok": True})


def results_view(request, content_hash):
    validate_content_hash(content_hash)
    book = get_object_or_404(ProcessedBook, content_hash=content_hash, status="done")
    out_dir = book_output_dir(content_hash)
    speakers = read_speakers(out_dir)
    for speaker in speakers:
        slug = slugify_name(speaker["name"])
        speaker["slug"] = slug
        speaker["has_voice"] = (out_dir / "voices" / f"{slug}.wav").exists()

    chapters_meta = load_chapters(out_dir)
    if chapters_meta is not None:
        n = len(chapters_meta)
        first_chapter_dir = chapter_dir_path(out_dir, 1, n)
        annotated_path = first_chapter_dir / "annotated.txt"
        annotated_lines = read_annotated(first_chapter_dir) if annotated_path.exists() else []
        has_full_audio = (first_chapter_dir / "compiled" / "full.mp3").exists()
        has_missing_audio = any(
            not (chapter_dir_path(out_dir, c["index"], n) / "compiled" / "full.mp3").exists()
            for c in chapters_meta
        )
        chapters_data = [{"index": c["index"], "title": c["title"]} for c in chapters_meta]
        current_chapter = 1
    else:
        chapters_data = None
        current_chapter = None
        annotated_lines = read_annotated(out_dir)
        has_full_audio = (out_dir / "compiled" / "full.mp3").exists()
        has_missing_audio = not has_full_audio

    return render(request, "reader/results.html", {
        "book": book,
        "speakers": speakers,
        "annotated_lines": annotated_lines,
        "has_full_audio": has_full_audio,
        "has_missing_audio": has_missing_audio,
        "chapters": chapters_data,
        "current_chapter": current_chapter,
    })
