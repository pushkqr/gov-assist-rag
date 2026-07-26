import unittest
from types import SimpleNamespace

from retrieval.pipeline import should_use_fast_path
from retrieval.query import build_generation_prompt, contextualize_query, extract_query_filter
from retrieval.support import build_context_text, extract_response_text


class RetrievalPerformanceTests(unittest.TestCase):
    def test_simple_queries_use_fast_path(self):
        self.assertTrue(should_use_fast_path("What is this policy about?"))

    def test_generic_greetings_use_fast_path(self):
        self.assertTrue(should_use_fast_path("Hello there"))

    def test_complex_queries_do_not_use_fast_path(self):
        self.assertFalse(
            should_use_fast_path(
                "What eligibility conditions apply for the 2024 circular on rural infrastructure funding and the appendix section?"
            )
        )


class RetrievalRefactorTests(unittest.TestCase):
    def test_build_context_text_deduplicates_parent_sections(self):
        results = [
            SimpleNamespace(payload={"parent_id": "p1", "parent_context": "First parent context"}),
            SimpleNamespace(payload={"parent_id": "p1", "parent_context": "First parent context"}),
            SimpleNamespace(payload={"parent_id": "p2", "parent_context": "Second parent context"}),
        ]

        self.assertEqual(
            build_context_text(results),
            "First parent context\n\n---\n\nSecond parent context"
        )

    def test_contextualize_query_returns_original_when_already_standalone(self):
        class StubClient:
            pass

        standalone_query, history_text = contextualize_query(StubClient(), "What is the eligibility threshold?", [])
        self.assertEqual(standalone_query, "What is the eligibility threshold?")
        self.assertEqual(history_text, "")

    def test_build_generation_prompt_contains_source_requirements(self):
        prompt = build_generation_prompt(
            query="What is the threshold?",
            history_text="",
            context_text="A policy excerpt"
        )

        self.assertIn("User Question:", prompt)
        self.assertIn("What is the threshold?", prompt)
        self.assertIn("A policy excerpt", prompt)

    def test_extract_response_text_handles_plain_strings(self):
        self.assertEqual(extract_response_text("[1, 2, 3]"), "[1, 2, 3]")

    def test_extract_query_filter_returns_none_when_no_metadata_found(self):
        class StubClient:
            pass

        self.assertIsNone(extract_query_filter(StubClient(), "Tell me about the policy"))
