import json
import os
from pathlib import Path
from typing import Any, Dict, List

from google.genai import types
from core.utils import generate_content_safe
from benchmark.evaluation import evaluate_response
from core.log_config import get_logger
from retrieval import run_retrieval

logger = get_logger(__name__)


BENCHMARK_FILE = Path("benchmark/benchmark.json")


def load_benchmark_cases() -> List[Dict[str, Any]]:
    """Load benchmark cases from benchmark.json."""
    if not BENCHMARK_FILE.exists():
        logger.warning(f"{BENCHMARK_FILE} not found.")
        return []
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        cases = json.load(f)
    logger.info(f"Loaded {len(cases)} cases from {BENCHMARK_FILE}")
    return cases


def llm_judge_answer(query: str, response_text: str, ideal_answer: str, gemini_client) -> Dict[str, Any]:
    """Use the generation model as an LLM judge to score a candidate answer."""

    prompt = f"""You are a fair but thorough evaluator for a government-document Q&A system.

Compare the candidate answer against the ideal answer for the given query.

Scoring Guidelines (0-5 scale):
- 5: Covers all key facts from the ideal answer with correct details.
- 4: Covers most key facts. Minor omissions or slight paraphrasing is acceptable.
- 3: Covers the core point correctly but misses some supporting details.
- 2: Partially correct — gets the topic right but misses critical facts.
- 1: Mentions the right topic area but the answer is largely wrong or vague.
- 0: Completely wrong, irrelevant, or a refusal to answer when information was available.

Important:
- The candidate does NOT need to match the ideal answer word-for-word.
- Paraphrased or restructured answers that convey the same facts should score 4-5.
- If the candidate provides additional correct context beyond the ideal answer, do not penalize.
- Only penalize for factual errors, missing critical information, or hallucinated content.

Query:
{query}

Candidate Answer:
{response_text}

Ideal Answer:
{ideal_answer}

Return ONLY valid JSON with fields: "score" (integer 0-5) and "justification" (one sentence).
"""

    try:
        result = generate_content_safe(
            gemini_client,
            model=os.getenv("GEN_MODEL_NAME", "gemini-2.5-flash"),
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json"),
        )
        payload = json.loads(result.text)
        return {
            "score": float(payload.get("score", 0)),
            "justification": payload.get("justification", ""),
        }
    except Exception as exc:
        return {"score": 0.0, "justification": f"judge failed: {exc}"}


def run_benchmark(gemini_client, qdrant_client, cases: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """Run the benchmark suite and return structured results."""
    if cases is None:
        cases = load_benchmark_cases()

    if not cases:
        return {"case_count": 0, "error": "No benchmark cases found."}

    results = []
    for i, case in enumerate(cases, 1):
        logger.info(f"({i}/{len(cases)}) {case['query'][:80]}...")

        retrieval_result = run_retrieval(
            gemini_client=gemini_client,
            qdrant_client=qdrant_client,
            query=case["query"],
            collection_name="gov_docs",
            chat_history=[],
        )

        # Extract the response text
        response_text = retrieval_result.get("response_text") or ""
        if retrieval_result.get("status") == "success" and retrieval_result.get("answer_stream"):
            # Consume the stream to get full text
            stream = retrieval_result["answer_stream"]
            for _ in stream:
                pass
            response_text = stream.full_text

        # Term-match evaluation
        term_score = evaluate_response(
            query=case["query"],
            response_text=response_text,
            evidence=retrieval_result.get("evidence", []),
            expected_terms=case.get("expected_terms", []),
        )

        # LLM judge evaluation
        judge_result = llm_judge_answer(
            query=case["query"],
            response_text=response_text,
            ideal_answer=case.get("expected_answer", ""),
            gemini_client=gemini_client,
        )

        results.append(
            {
                "query": case["query"],
                "term_score": term_score,
                "judge_score": judge_result,
                "response_text": response_text,
                "evidence": retrieval_result.get("evidence", [])[:3],
            }
        )

    average_judge_score = sum(item["judge_score"]["score"] for item in results) / len(results) if results else 0.0
    average_term_score = sum(item["term_score"]["score"] for item in results) / len(results) if results else 0.0

    return {
        "case_count": len(results),
        "average_term_score": round(average_term_score, 3),
        "average_judge_score": round(average_judge_score, 3),
        "results": results,
    }


def print_benchmark_report(report: Dict[str, Any]) -> None:
    """Pretty-print benchmark results as a readable report."""

    if report.get("error"):
        print(f"\n[Benchmark] Error: {report['error']}")
        return

    n = report["case_count"]
    avg_term = report["average_term_score"]
    avg_judge = report["average_judge_score"]

    print("\n")
    print("=" * 72)
    print("  MIMIR BENCHMARK REPORT")
    print("=" * 72)
    print(f"  Total Cases : {n}")
    print(f"  Avg Judge   : {avg_judge:.1f} / 5.0")
    print("=" * 72)

    # Per-case breakdown
    passed = 0
    failed_cases = []

    for i, r in enumerate(report["results"], 1):
        t_score = r["term_score"]["score"]
        t_matched = r["term_score"]["matched_terms"]
        t_total = r["term_score"]["total_terms"]
        j_score = r["judge_score"]["score"]
        j_reason = r["judge_score"]["justification"]
        query_short = r["query"][:65]

        status = "PASS" if j_score >= 3.0 or (j_score >= 2.0 and t_score >= 0.5) else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            failed_cases.append(i)

        print(f"\n  [{i:02d}] {status}  {query_short}{'...' if len(r['query']) > 65 else ''}")
        print(f"       Judge Score: {j_score:.0f}/5")
        print(f"       Reason: {j_reason}")

    # Summary
    pass_rate = (passed / n * 100) if n else 0
    print("\n" + "-" * 72)
    print(f"  PASS RATE: {passed}/{n} ({pass_rate:.0f}%)")

    if failed_cases:
        print(f"  Failed case numbers: {', '.join(str(c) for c in failed_cases)}")

    # Grade
    if (avg_judge >= 4.0 and avg_term >= 0.5) or (avg_judge >= 4.2):
        grade = "A"
    elif (avg_judge >= 3.0 and avg_term >= 0.3) or (avg_judge >= 3.2):
        grade = "B"
    elif avg_judge >= 2.0:
        grade = "C"
    else:
        grade = "D"

    print(f"\n  OVERALL GRADE: {grade}")
    print("=" * 72)

    # Save to file
    output_path = Path("temp/benchmark_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Full results saved to: {output_path}")
