import unittest
from benchmark import load_benchmark_cases, run_benchmark


class BenchmarkTests(unittest.TestCase):
    def test_load_benchmark_cases(self):
        cases = load_benchmark_cases()
        self.assertIsInstance(cases, list)

    def test_run_benchmark_handles_empty_cases(self):
        report = run_benchmark(gemini_client=None, qdrant_client=None, cases=[])
        self.assertEqual(report["case_count"], 0)
        self.assertIn("error", report)


if __name__ == "__main__":
    unittest.main()
