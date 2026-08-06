"""Regression suite: the four demo queries, asserted on content not just latency.

This is the canary for the build plan in scratch/build-plan.md. Every phase that touches
retrieval, ingestion or generation should pass this before being considered done. Run it
directly against the pipeline (not over HTTP) so it works without a running server:

    python -m scratch.regress
    python -m scratch.regress --only 3
    python -m scratch.regress --verbose

Exit code 0 if all pass, 1 otherwise, so it can gate a phase without eyeballing output.

Respects the Cerebras 5-requests-per-minute-per-model limit (build-plan.md Guardrail 6):
queries are spaced 12s apart by default. Do not lower this for a faster run; a rate-limit
fallback to Gemini changes both the latency and, occasionally, the exact wording, and would
make failures harder to interpret, not easier.
"""

import argparse
import os
import re
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from core.utils import get_genai_client, get_weaviate_client, get_cerebras_client  # noqa: E402
from retrieval.pipeline import run_retrieval  # noqa: E402
from core.schema import CORPUS_COLLECTION  # noqa: E402

# Match app.py's ACTIVE_COLLECTION resolution exactly - otherwise this suite silently
# queries the wrong collection whenever CORPUS_COLLECTION overrides the default (e.g.
# after a bulk ingest into GovDocsV2), and every result looks like an empty corpus.
COLLECTION = os.environ.get("CORPUS_COLLECTION", CORPUS_COLLECTION).strip() or CORPUS_COLLECTION
QUERY_SPACING_S = 12

# The model (temperature 0.0, but this varies across runs/models regardless) sometimes
# renders GR numbers with typographic hyphen/dash variants instead of ASCII '-' - cosmetically
# identical, but an ASCII-only regex silently misses them. Found when check_q3 failed on a
# genuinely correct answer that cited "DEMO‑2019" (U+2011 NON-BREAKING HYPHEN) rather
# than "DEMO-2019". Every literal '-' check below runs against normalized text instead.
_HYPHEN_VARIANTS = "‐‑‒–—―−"  # hyphen, nb-hyphen, figure
# dash, en dash, em dash, horizontal bar, minus sign


def _normalize_hyphens(text: str) -> str:
    return text.translate({ord(c): "-" for c in _HYPHEN_VARIANTS})


def _is_mostly_devanagari(text: str, threshold: float = 0.25) -> bool:
    if not text:
        return False
    devanagari = sum(1 for c in text if "ऀ" <= c <= "ॿ")
    letters = sum(1 for c in text if c.isalpha())
    return letters > 0 and (devanagari / letters) >= threshold


def _no_currency_amount(text: str) -> bool:
    """A refusal should not state a rupee figure. Loose match: digits near Rs/rupee/INR."""
    return not re.search(r"(rs\.?|rupee|inr|₹)\s*[\d,]+", text, re.IGNORECASE)


def check_q1(text: str) -> tuple:
    ok = (
        re.search(r"RECOGNITION", text, re.IGNORECASE) is not None
        and "2019" in text
        and re.search(r"51\s*/\s*19", text) is not None
        and re.search(r"TANSHI", text, re.IGNORECASE) is not None
    )
    return ok, "GR number RECOGNITION-2019/(51/19)/TANSHI-5 present" if ok else "GR number not found in answer"


def check_q2(text: str) -> tuple:
    if not text.strip():
        return False, "empty answer"
    if not _is_mostly_devanagari(text):
        return False, "answer is not predominantly Devanagari"
    has_date_signal = bool(re.search(r"\d{4}", text)) or bool(re.search(r"[०-९]", text))
    if not has_date_signal:
        return False, "no date-like content found (neither Latin nor Devanagari digits)"
    return True, "answered in Marathi with date content"


def check_q3(text: str) -> tuple:
    if "[!WARNING]" not in text:
        return False, "no [!WARNING] callout in answer"
    if "35" not in text or "36" not in text:
        return False, "both conflicting values (35, 36) not present"
    if not re.search(r"DEMO-2019", text) or not re.search(r"DEMO-2022", text):
        return False, "both DEMO documents not cited"
    return True, "warning block present, both values and both documents cited"


def check_q4(text: str) -> tuple:
    if not text.strip():
        return False, "empty answer"
    # Matched as a pattern rather than a fixed phrase list. The original list held four exact
    # strings and reported a correct refusal as a regression: the self-hosted model answered
    # "the context does not provide any information regarding the pension amount", which is
    # precisely the wanted behaviour and matched none of them. What is being asserted is
    # "declines and names nothing", so any negated contain/provide/include/mention/specify
    # counts, and the currency check below still catches an invented figure.
    has_refusal_language = re.search(
        r"(do(es)?\s+not\s+(contain|provide|include|mention|specify|state)"
        r"|not\s+(available|provided|contained|specified|mentioned)"
        r"|no\s+information)",
        text, re.IGNORECASE,
    ) is not None
    if not has_refusal_language:
        return False, "no refusal language detected"
    if not _no_currency_amount(text):
        return False, "answer states a currency amount despite refusing"
    return True, "clean refusal, no invented figure"


CASES = [
    (1, "Q1 English", "What is the Government Decision number for the continuation of 3527 temporary posts in the Directorate of Technical Education?", check_q1),
    (2, "Q2 Marathi", "तंत्र शिक्षण संचालनालयातील तात्पुरत्या पदांना मान्यता देणारा शासन निर्णय कोणत्या तारखेपासून कोणत्या तारखेपर्यंत लागू आहे?", check_q2),
    (3, "Q3 Conflict", "What is the maximum age limit for direct recruitment to the post of Assistant Professor?", check_q3),
    (4, "Q4 Refusal", "What is the pension amount for retired professors in Maharashtra?", check_q4),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=int, help="Run a single case by number (1-4)")
    parser.add_argument("--verbose", action="store_true", help="Print full answer text")
    parser.add_argument("--spacing", type=float, default=QUERY_SPACING_S)
    args = parser.parse_args()

    cases = [c for c in CASES if args.only is None or c[0] == args.only]
    if not cases:
        print(f"No case numbered {args.only}")
        sys.exit(2)

    gemini = get_genai_client()
    weaviate_client = get_weaviate_client()
    cerebras = get_cerebras_client()

    results = []
    try:
        for i, (num, label, query, checker) in enumerate(cases):
            if i > 0:
                time.sleep(args.spacing)

            started = time.time()
            try:
                result = run_retrieval(
                    gemini, cerebras, weaviate_client=weaviate_client,
                    query=query, collection_name=COLLECTION, fast_mode=True,
                )
                text = _normalize_hyphens("".join(result["answer_stream"]))
                elapsed = time.time() - started
                passed, detail = checker(text)
            except Exception as exc:
                elapsed = time.time() - started
                passed, detail, text = False, f"exception: {exc}", ""

            results.append((num, label, passed, detail, elapsed))
            if args.verbose:
                print(f"\n--- {label} ({elapsed:.2f}s) ---")
                print(text)
    finally:
        weaviate_client.close()

    print()
    print(f"{'#':>3}  {'case':<14} {'result':<6} {'time':>7}  detail")
    print("-" * 78)
    all_passed = True
    for num, label, passed, detail, elapsed in results:
        status = "PASS" if passed else "FAIL"
        all_passed = all_passed and passed
        print(f"{num:>3}  {label:<14} {status:<6} {elapsed:>6.2f}s  {detail}")

    print()
    if all_passed:
        print(f"All {len(results)} checks passed.")
    else:
        failed = sum(1 for r in results if not r[2])
        print(f"{failed} of {len(results)} checks FAILED. Do not proceed to the next phase.")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
