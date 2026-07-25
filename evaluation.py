from typing import Any, Dict, List


def evaluate_response(query: str, response_text: str, evidence: List[Dict[str, Any]], expected_terms: List[str]) -> Dict[str, Any]:
    response_lower = (response_text or "").lower()
    evidence_lower = " ".join((item.get("quote") or "").lower() for item in evidence or [])
    combined_text = f"{query.lower()} {response_lower} {evidence_lower}"

    matched_terms = 0
    for term in expected_terms:
        if term.lower() in combined_text:
            matched_terms += 1

    total_terms = len(expected_terms)
    score = matched_terms / total_terms if total_terms else 0.0

    return {
        "query": query,
        "matched_terms": matched_terms,
        "total_terms": total_terms,
        "score": round(score, 3),
        "passed": score >= 0.75,
    }


def run_evaluation(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    results = [evaluate_response(**case) for case in cases]
    average_score = sum(item["score"] for item in results) / len(results) if results else 0.0
    passed_cases = sum(1 for item in results if item["passed"])
    return {
        "case_count": len(results),
        "average_score": round(average_score, 3),
        "passed_cases": passed_cases,
        "results": results,
    }
