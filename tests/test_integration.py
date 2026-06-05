import tempfile
from unittest.mock import patch
from django.test import TestCase
from django.urls import reverse
from reader.models import ProcessedBook


MOCK_SPEAKERS = [{"name": "Alice", "sex": "female", "age": "30s", "traits": "brave"}]
MOCK_ANNOTATED = '[NARRATOR] Once upon a time.\n[ALICE | mood=happy] "Hello!"'


class FullFlowTest(TestCase):
    @patch("reader.pipeline.annotate_chunk", return_value=MOCK_ANNOTATED)
    @patch("reader.pipeline.extract_speakers", return_value=MOCK_SPEAKERS)
    def test_upload_to_results_full_flow(self, mock_extract, mock_annotate):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.settings(OUTPUTS_DIR=tmp_dir):
                # Upload
                response = self.client.post(
                    reverse("reader:process"),
                    {"input_text": 'Once upon a time.\n\n"Hello!" said Alice.'},
                )
                self.assertIn(response.status_code, [301, 302])
                book = ProcessedBook.objects.get()
                self.assertEqual(book.status, "pending")

                # Consume stream (forces pipeline to run)
                stream_response = self.client.get(
                    reverse("reader:stream", args=[book.content_hash])
                )
                content = b"".join(stream_response.streaming_content).decode()
                self.assertIn("done", content)

                # Results page
                book.refresh_from_db()
                self.assertEqual(book.status, "done")
                results_response = self.client.get(
                    reverse("reader:results", args=[book.content_hash])
                )
                self.assertEqual(results_response.status_code, 200)
                self.assertContains(results_response, "Alice")
