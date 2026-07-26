import unittest

from ingestion.parsers import format_plain_text_to_markdown, format_plain_text_with_llm


class ParserTests(unittest.TestCase):
    def test_format_plain_text_to_markdown_adds_headers(self):
        raw_text = """Circular No. 247/04/2025-GST
Government of India
Ministry of Finance
Subject: Clarification regarding GST rates
1. Clarification regarding classification
1.1 Details about pepper supply
Section 23(1) of CGST Act applies here.
        """
        formatted = format_plain_text_to_markdown(raw_text)

        self.assertIn("# Circular No. 247/04/2025-GST", formatted)
        self.assertIn("# Subject: Clarification regarding GST rates", formatted)
        self.assertIn("## 1. Clarification regarding classification", formatted)
        self.assertIn("### 1.1", formatted)

    def test_format_plain_text_to_markdown_fallback_sections(self):
        # Monolithic unformatted plain text > 1500 chars with no headers
        paragraphs = [f"Paragraph {i}: " + ("Sample plain text content for testing section fallback splitting. " * 5) for i in range(15)]
        raw_text = "\n\n".join(paragraphs)

        formatted = format_plain_text_to_markdown(raw_text)

        self.assertIn("## Section 1", formatted)
        self.assertIn("## Section 2", formatted)

    def test_format_plain_text_to_markdown_preserves_existing_markdown(self):
        md_input = "# Header 1\n\n## Subheader\n\nSome text."
        self.assertEqual(format_plain_text_to_markdown(md_input), md_input)


if __name__ == "__main__":
    unittest.main()
