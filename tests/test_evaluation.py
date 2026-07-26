import unittest

from benchmark.evaluation import evaluate_response, run_evaluation


class EvaluationTests(unittest.TestCase):
    def test_evaluate_response_scores_expected_terms(self):
        response = "The circular mentions eligibility for pensioners and students."
        evidence = [{"quote": "Eligibility for pensioners and students is covered."}]
        report = evaluate_response(
            query="What eligibility rules apply?",
            response_text=response,
            evidence=evidence,
            expected_terms=["eligibility", "pensioners", "students"],
        )
        self.assertEqual(report["matched_terms"], 3)
        self.assertEqual(report["total_terms"], 3)
        self.assertEqual(report["score"], 1.0)

    def test_run_evaluation_returns_summary(self):
        cases = [
            {
                "query": "What is the eligibility rule?",
                "expected_terms": ["eligibility", "rule"],
                "response_text": "This rule covers eligibility.",
                "evidence": [{"quote": "The rule covers eligibility."}],
            }
        ]

        report = run_evaluation(cases)
        self.assertEqual(report["case_count"], 1)
        self.assertEqual(report["average_score"], 1.0)
        self.assertEqual(report["passed_cases"], 1)
