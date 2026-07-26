import unittest

from ingestion.metadata import extract_document_metadata


class IngestionMetadataTests(unittest.TestCase):
    def test_extract_document_metadata_finds_title_year_authority_and_number(self):
        markdown = """
        # Notification for Digital Services
        Department of Administrative Reforms
        Document No.: 2024-07
        Issued by: Ministry of Finance
        This circular provides guidance for public service delivery.
        """

        metadata = extract_document_metadata(markdown, "/tmp/sample.pdf", fallback_year=2025)

        self.assertEqual(metadata["document_title"], "Notification for Digital Services")
        self.assertEqual(metadata["year"], 2024)
        self.assertEqual(metadata["doc_number"], "2024-07")
        self.assertEqual(metadata["issuing_authority"], "Ministry of Finance")
        self.assertEqual(metadata["document_category"], "Notification")
