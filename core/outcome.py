"""Classify a generated answer as a refusal or a real answer.

Phase 4 of the build plan. When the system refuses a question, that is a signal about what
the corpus is missing — aggregated across officers it becomes a ranked list of what to
ingest next, generated from real usage instead of guesswork. This module is the detector
that feeds that aggregation; see db.py's query_log table and app.py's /api/admin/gaps.

Detection is text-pattern matching against the exact refusal phrasing the system prompt in
retrieval/pipeline.py instructs ("state plainly that the information is not available in the
retrieved documents"), widened to catch paraphrasing, plus a weak-evidence fallback: an
answer built from zero retrieved evidence is treated as a refusal even if it doesn't use the
expected wording, since that combination should not happen for a genuine answer.
"""

import re

_REFUSAL_PATTERNS = [
    r"not available in the retrieved documents",
    r"not available in the (?:provided |retrieved )?context",
    r"does not contain (?:the|any|information)",
    r"do not contain (?:the|any|information)",
    r"not contain any information",
    r"no information (?:is )?available",
    r"could not (?:be )?find",
    r"cannot (?:be )?find",
    r"not (?:be )?found in the (?:retrieved |provided )?(?:documents|context|corpus)",
    r"context does not (?:answer|address|cover)",
    r"documents do not (?:contain|mention|address)",
]

_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def classify_outcome(answer_text: str, evidence_count: int) -> str:
    """Returns 'refused' or 'answered'. Deliberately binary: the clarifying-question case in
    the system prompt (ambiguous query, asks the user to narrow down) is closer to "answered"
    than "refused" for gap-analysis purposes — it means the corpus did have relevant material,
    just too much of it — and folding it in as a third bucket was more machinery than the
    stated use case needed."""
    text = (answer_text or "").strip()
    if not text:
        return "refused"
    if _REFUSAL_RE.search(text):
        return "refused"
    if evidence_count == 0:
        return "refused"
    return "answered"


def normalize_query(query: str) -> str:
    """Reduce a query to a grouping key: lowercase, collapsed whitespace, no trailing
    punctuation. Deliberately simple (exact/near-exact matching) rather than embedding-based
    clustering — the plan calls for starting simple and only reaching for clustering once
    volume justifies it, and this already satisfies the stated case of the same question
    asked verbatim more than once."""
    normalized = re.sub(r"\s+", " ", (query or "").strip().lower())
    return normalized.rstrip("?!.,;: ")
