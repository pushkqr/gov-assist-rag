"""DEPLOYMENT_MODE cascade: sets defaults for the individual provider switches.

Phase 6 of the build plan. GEN_PROVIDER already existed and did the hard half of this: moving
answer generation off Cerebras. This generalises the same idea to the three ingestion-time
switches that also call a third party - PDF OCR fallback, ingest-time structuring, and
ingest-time Marathi translation - so DEPLOYMENT_MODE=sovereign is one variable instead of
four.

Each switch is resolved fresh at the read site (not applied once at startup), which is what
makes this safe to use from every entry point that touches ingestion: app.py's admin upload
path, main.py's CLI ingestion, and scratch/ingest_department.py all call these functions
directly rather than depending on some initialization step having already run in that process.
An explicit value for the individual switch always wins over DEPLOYMENT_MODE - that's the
"individually overridable" half of the plan's design.
"""

import os

# name -> (hybrid default, sovereign default)
_SWITCHES = {
    "GEN_PROVIDER": ("cerebras", "local"),
    "PDF_PARSE_TIER2_PROVIDER": ("docai", "docling"),
    "INGEST_STRUCTURE_PROVIDER": ("gemini", "local"),
    "INGEST_TRANSLATE_PROVIDER": ("gcp", "indictrans2"),
}


def current_mode() -> str:
    mode = os.environ.get("DEPLOYMENT_MODE", "hybrid").strip().lower()
    return mode if mode in ("hybrid", "sovereign") else "hybrid"


def _resolve(var: str) -> str:
    explicit = os.environ.get(var, "").strip()
    if explicit:
        return explicit
    hybrid_value, sovereign_value = _SWITCHES[var]
    return sovereign_value if current_mode() == "sovereign" else hybrid_value


def gen_provider() -> str:
    return _resolve("GEN_PROVIDER")


def pdf_parse_tier2_provider() -> str:
    return _resolve("PDF_PARSE_TIER2_PROVIDER")


def ingest_structure_provider() -> str:
    return _resolve("INGEST_STRUCTURE_PROVIDER")


def ingest_translate_provider() -> str:
    return _resolve("INGEST_TRANSLATE_PROVIDER")


def summary() -> dict:
    """Read-only snapshot for the admin panel and `deploy.py status`."""
    return {
        "mode": current_mode(),
        "generation": gen_provider(),
        "pdf_parse_tier2": pdf_parse_tier2_provider(),
        "ingest_structure": ingest_structure_provider(),
        "ingest_translate": ingest_translate_provider(),
    }
