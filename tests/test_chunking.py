import unittest
from unittest.mock import MagicMock, patch

from ingestion.chunking import translate_marathi_batch_gcp


class ChunkingTests(unittest.TestCase):
    def test_translate_marathi_batch_gcp_handles_empty(self):
        result = translate_marathi_batch_gcp([])
        self.assertEqual(result, [])

    @patch("ingestion.chunking.translate")
    def test_translate_marathi_batch_gcp_sub_batches_large_requests(self, mock_translate):
        # Create 13 chunks of 2,000 chars each (total 26,000 chars > 25,000 limit)
        chunks = ["मराठी " * 333 for _ in range(13)]

        mock_client = MagicMock()
        mock_response1 = MagicMock()
        mock_response1.translations = [MagicMock(translated_text=f"Trans {i}") for i in range(1, 13)]
        mock_response2 = MagicMock()
        mock_response2.translations = [MagicMock(translated_text="Trans 13")]

        mock_client.translate_text.side_effect = [mock_response1, mock_response2]
        mock_translate.TranslationServiceClient.return_value = mock_client

        translations = translate_marathi_batch_gcp(chunks)

        expected = [f"Trans {i}" for i in range(1, 14)]
        self.assertEqual(translations, expected)
        self.assertEqual(mock_client.translate_text.call_count, 2)


if __name__ == "__main__":
    unittest.main()
