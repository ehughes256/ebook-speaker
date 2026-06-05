import pytest
from django.test import TestCase, Client
from django.urls import reverse
from reader.models import ProcessedBook
from reader.ingestion import compute_hash


class UploadViewTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_get_upload_returns_200(self):
        response = self.client.get(reverse("reader:upload"))
        self.assertEqual(response.status_code, 200)

    def test_get_upload_contains_form(self):
        response = self.client.get(reverse("reader:upload"))
        self.assertContains(response, "<form")


class ProcessViewTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_post_text_creates_processedbook(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.settings(OUTPUTS_DIR=tmp_dir):
                response = self.client.post(
                    reverse("reader:process"),
                    {"input_text": "Hello world. \"Hi,\" she said."},
                )
        self.assertEqual(ProcessedBook.objects.count(), 1)

    def test_post_text_redirects_to_progress(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.settings(OUTPUTS_DIR=tmp_dir):
                response = self.client.post(
                    reverse("reader:process"),
                    {"input_text": "Hello world."},
                )
        book = ProcessedBook.objects.first()
        self.assertRedirects(
            response,
            reverse("reader:progress", args=[book.content_hash]),
            fetch_redirect_response=False,
        )

    def test_post_same_text_twice_reuses_record(self):
        import tempfile
        text = "Identical content."
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.settings(OUTPUTS_DIR=tmp_dir):
                self.client.post(reverse("reader:process"), {"input_text": text})
                self.client.post(reverse("reader:process"), {"input_text": text})
        self.assertEqual(ProcessedBook.objects.count(), 1)

    def test_post_done_book_redirects_to_results(self):
        import tempfile
        text = "Already done."
        content_hash = compute_hash(text)
        ProcessedBook.objects.create(
            content_hash=content_hash,
            title="Done",
            status="done",
            output_path=f"outputs/{content_hash}/",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.settings(OUTPUTS_DIR=tmp_dir):
                # A genuinely-complete book has annotated.txt; otherwise process_view
                # treats it as incomplete and re-runs (redirecting to progress).
                from pathlib import Path
                out = Path(tmp_dir) / content_hash
                out.mkdir(parents=True, exist_ok=True)
                (out / "annotated.txt").write_text("[NARRATOR] Already done.")
                response = self.client.post(reverse("reader:process"), {"input_text": text})
        self.assertRedirects(
            response,
            reverse("reader:results", args=[content_hash]),
            fetch_redirect_response=False,
        )

    def test_post_empty_input_returns_400(self):
        response = self.client.post(reverse("reader:process"), {})
        self.assertEqual(response.status_code, 400)

    def test_post_both_inputs_returns_400(self):
        from django.core.files.uploadedfile import SimpleUploadedFile
        response = self.client.post(
            reverse("reader:process"),
            {
                "input_text": "some text",
                "input_file": SimpleUploadedFile("test.txt", b"file content", content_type="text/plain"),
            },
        )
        self.assertEqual(response.status_code, 400)


class ProcessViewChapterTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_post_chaptered_text_writes_chapters_json(self):
        import tempfile, json
        text = "Chapter 1\n\nFirst chapter.\n\nChapter 2\n\nSecond chapter."
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.settings(OUTPUTS_DIR=tmp_dir):
                self.client.post(reverse("reader:process"), {"input_text": text})
            book = ProcessedBook.objects.first()
            from pathlib import Path
            chapters_path = Path(tmp_dir) / book.content_hash / "chapters.json"
            assert chapters_path.exists()
            data = json.loads(chapters_path.read_text())
            assert len(data) == 2
            assert data[0]["title"] == "Chapter 1"

    def test_post_plain_text_does_not_write_chapters_json(self):
        import tempfile
        text = "Just some plain text without chapters."
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.settings(OUTPUTS_DIR=tmp_dir):
                self.client.post(reverse("reader:process"), {"input_text": text})
            book = ProcessedBook.objects.first()
            from pathlib import Path
            chapters_path = Path(tmp_dir) / book.content_hash / "chapters.json"
            assert not chapters_path.exists()


class ListenListViewTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_listen_list_returns_200(self):
        response = self.client.get(reverse("reader:listen"))
        self.assertEqual(response.status_code, 200)

    def test_listen_list_only_shows_books_with_audio(self):
        import tempfile, json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.settings(OUTPUTS_DIR=tmp_dir):
                b1 = ProcessedBook.objects.create(
                    content_hash="aaa111",
                    title="Has Audio",
                    status="done",
                    output_path="outputs/aaa111/",
                )
                p1 = Path(tmp_dir) / "aaa111" / "compiled"
                p1.mkdir(parents=True)
                (p1 / "full.mp3").write_bytes(b"fake")

                ProcessedBook.objects.create(
                    content_hash="bbb222",
                    title="No Audio",
                    status="done",
                    output_path="outputs/bbb222/",
                )
                Path(tmp_dir, "bbb222").mkdir(parents=True)

                response = self.client.get(reverse("reader:listen"))
                available = response.context["available"]
        self.assertEqual(len(available), 1)
        self.assertEqual(available[0]["book"].content_hash, "aaa111")

    def test_listen_list_multi_chapter_counts_audio_chapters(self):
        import tempfile, json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.settings(OUTPUTS_DIR=tmp_dir):
                ProcessedBook.objects.create(
                    content_hash="ccc333",
                    title="Multi",
                    status="done",
                    output_path="outputs/ccc333/",
                )
                out = Path(tmp_dir) / "ccc333"
                out.mkdir()
                chapters_meta = [{"index": 1, "title": "Ch 1"}, {"index": 2, "title": "Ch 2"}]
                (out / "chapters.json").write_text(json.dumps(chapters_meta))
                ch1 = out / "chapters" / "01" / "compiled"
                ch1.mkdir(parents=True)
                (ch1 / "full.mp3").write_bytes(b"fake")

                response = self.client.get(reverse("reader:listen"))
                available = response.context["available"]
        self.assertEqual(len(available), 1)
        self.assertEqual(available[0]["chapter_count"], 1)


class ListenBookViewTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_listen_book_single_chapter(self):
        import tempfile
        from pathlib import Path
        content_hash = "d" * 64
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.settings(OUTPUTS_DIR=tmp_dir):
                ProcessedBook.objects.create(
                    content_hash=content_hash,
                    title="Single",
                    status="done",
                    output_path=f"outputs/{content_hash}/",
                )
                p = Path(tmp_dir) / content_hash / "compiled"
                p.mkdir(parents=True)
                (p / "full.mp3").write_bytes(b"fake")

                response = self.client.get(reverse("reader:listen_book", args=[content_hash]))
        self.assertEqual(response.status_code, 200)
        chapters = response.context["chapters"]
        self.assertEqual(len(chapters), 1)
        self.assertIn(f"/audio/{content_hash}/", chapters[0]["audio_url"])

    def test_listen_book_multi_chapter(self):
        import tempfile, json
        from pathlib import Path
        content_hash = "e" * 64
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.settings(OUTPUTS_DIR=tmp_dir):
                ProcessedBook.objects.create(
                    content_hash=content_hash,
                    title="Multi",
                    status="done",
                    output_path=f"outputs/{content_hash}/",
                )
                out = Path(tmp_dir) / content_hash
                out.mkdir()
                meta = [{"index": 1, "title": "Ch 1"}, {"index": 2, "title": "Ch 2"}]
                (out / "chapters.json").write_text(json.dumps(meta))
                for i in (1, 2):
                    p = out / "chapters" / f"0{i}" / "compiled"
                    p.mkdir(parents=True)
                    (p / "full.mp3").write_bytes(b"fake")

                response = self.client.get(reverse("reader:listen_book", args=[content_hash]))
        self.assertEqual(response.status_code, 200)
        chapters = response.context["chapters"]
        self.assertEqual(len(chapters), 2)
        self.assertIn(f"/audio/{content_hash}/1/", chapters[0]["audio_url"])

    def test_listen_book_404_when_no_audio(self):
        import tempfile
        from pathlib import Path
        content_hash = "f" * 64
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.settings(OUTPUTS_DIR=tmp_dir):
                ProcessedBook.objects.create(
                    content_hash=content_hash,
                    title="No Audio",
                    status="done",
                    output_path=f"outputs/{content_hash}/",
                )
                Path(tmp_dir, content_hash).mkdir(parents=True)
                response = self.client.get(reverse("reader:listen_book", args=[content_hash]))
        self.assertEqual(response.status_code, 404)


class PathTraversalValidationTest(TestCase):
    """FIX #1 — content_hash / slug params must be rejected unless they match the
    expected charset, blocking path traversal into the filesystem."""

    def setUp(self):
        self.client = Client()

    def _valid_hash(self):
        return "a" * 64

    def test_results_rejects_non_hex_content_hash(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.settings(OUTPUTS_DIR=tmp_dir):
                # "abc" is not 64 hex chars — must 404 before any DB/filesystem hit.
                response = self.client.get(reverse("reader:results", args=["abc"]))
        self.assertEqual(response.status_code, 404)

    def test_cover_rejects_traversal_content_hash(self):
        import tempfile
        from urllib.parse import quote
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.settings(OUTPUTS_DIR=tmp_dir):
                # Encoded ".." traversal attempt resolves to a non-hex segment -> 404.
                response = self.client.get("/cover/" + quote("..%2f..", safe="") + "/")
        self.assertEqual(response.status_code, 404)

    def test_results_accepts_valid_hash_but_404_when_absent(self):
        # A well-formed hash passes validation; absence of the book still 404s
        # (proves the helper isn't blanket-rejecting valid hashes).
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.settings(OUTPUTS_DIR=tmp_dir):
                response = self.client.get(reverse("reader:results", args=[self._valid_hash()]))
        self.assertEqual(response.status_code, 404)

    def test_voice_rejects_bad_slug(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.settings(OUTPUTS_DIR=tmp_dir):
                ProcessedBook.objects.create(
                    content_hash=self._valid_hash(),
                    title="Book",
                    status="done",
                    output_path=f"outputs/{self._valid_hash()}/",
                )
                # Uppercase / slash not in [a-z0-9_] -> 404 before touching the fs.
                response = self.client.get(
                    "/voice/" + self._valid_hash() + "/Bad%2FSlug/"
                )
        self.assertEqual(response.status_code, 404)

    def test_voice_valid_slug_404_when_file_absent(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.settings(OUTPUTS_DIR=tmp_dir):
                ProcessedBook.objects.create(
                    content_hash=self._valid_hash(),
                    title="Book",
                    status="done",
                    output_path=f"outputs/{self._valid_hash()}/",
                )
                response = self.client.get(
                    reverse("reader:voice", args=[self._valid_hash(), "narrator"])
                )
        self.assertEqual(response.status_code, 404)


class UploadSizeCapTest(TestCase):
    """FIX #15 — uploads and pasted text over MAX_UPLOAD_MB are rejected with 400."""

    def setUp(self):
        self.client = Client()

    def test_oversized_file_returns_400(self):
        import tempfile
        from django.core.files.uploadedfile import SimpleUploadedFile
        # 1 MB cap, send 2 MB.
        payload = b"x" * (2 * 1024 * 1024)
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.settings(
                OUTPUTS_DIR=tmp_dir,
                MAX_UPLOAD_MB=1,
                DATA_UPLOAD_MAX_MEMORY_SIZE=10 * 1024 * 1024,
            ):
                response = self.client.post(
                    reverse("reader:process"),
                    {"input_file": SimpleUploadedFile(
                        "big.txt", payload, content_type="text/plain")},
                )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(ProcessedBook.objects.count(), 0)

    def test_oversized_pasted_text_returns_400(self):
        import tempfile
        big_text = "a" * (2 * 1024 * 1024)
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.settings(
                OUTPUTS_DIR=tmp_dir,
                MAX_UPLOAD_MB=1,
                DATA_UPLOAD_MAX_MEMORY_SIZE=10 * 1024 * 1024,
            ):
                response = self.client.post(
                    reverse("reader:process"), {"input_text": big_text}
                )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(ProcessedBook.objects.count(), 0)

    def test_small_valid_text_still_processes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.settings(OUTPUTS_DIR=tmp_dir, MAX_UPLOAD_MB=50):
                response = self.client.post(
                    reverse("reader:process"),
                    {"input_text": "A short valid book."},
                )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(ProcessedBook.objects.count(), 1)
