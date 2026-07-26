import json
import os
import tempfile
import unittest

from ingestion.state import load_ingestion_state, save_ingestion_state, should_skip_file


class IngestionStateTests(unittest.TestCase):
    def test_should_skip_file_when_hash_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "ingestion_state.json")
            pdf_path = os.path.join(tmpdir, "doc.pdf")
            with open(pdf_path, "wb") as fh:
                fh.write(b"sample pdf")

            state = load_ingestion_state(state_path)
            self.assertEqual(state, {})

            save_ingestion_state(state_path, pdf_path, "abc123", {"doc_number": "doc.pdf", "year": 2024})
            self.assertTrue(should_skip_file(pdf_path, "abc123", state_path))

    def test_should_not_skip_file_when_hash_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "ingestion_state.json")
            pdf_path = os.path.join(tmpdir, "doc.pdf")
            with open(pdf_path, "wb") as fh:
                fh.write(b"sample pdf")

            save_ingestion_state(state_path, pdf_path, "abc123", {"doc_number": "doc.pdf", "year": 2024})
            self.assertFalse(should_skip_file(pdf_path, "def456", state_path))

    def test_load_ingestion_state_handles_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_path = os.path.join(tmpdir, "missing.json")
            self.assertEqual(load_ingestion_state(missing_path), {})
