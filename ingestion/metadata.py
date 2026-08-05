import os
import re
from typing import Any, Dict


# Matches a Maharashtra GR-style document number, e.g. "NGC-2010 / (193/10) / Mashi-4" or
# "Rusayo-2013/ P.S. No.418/ VC-3D". Runs from the department stem to a natural terminator
# (comma, a date word, or end of line) rather than to the first period, since these numbers
# are full of abbreviation periods.
_DOC_NUM_RE = re.compile(
    r"No\.?\s*([A-Za-zऀ-ॿ]{2,}[-–]\d{2,4}[^,\n]{0,60}?)"
    r"(?=\s*(?:,|\bdated\b|\bDy\.|\bDt\.|\bd\.|\bदिनांक\b|$))",
    re.IGNORECASE,
)


def _extract_ref_target(clause: str) -> str:
    """Reduce a citation clause to the referenced document number where one is present.

    The clause is free text ("Government Resolution No. NGC-2010/(193/10)/Mashi-4, dated
    30.10.2010"), so an abbreviation period would truncate a naive match at "No". Pull the
    structured number out when it is there and fall back to the trimmed clause when it is not.
    """
    clause = re.sub(r"\s+", " ", clause).strip(" ,;:-")
    match = _DOC_NUM_RE.search(clause)
    if match:
        return match.group(1).strip(" ,.;:-")
    return clause[:120].rstrip(" ,.;:-")


def extract_document_metadata(markdown_text: str, source_path: str, fallback_year: int = 2025) -> Dict[str, Any]:
    """Extract structured document metadata fields from markdown text."""
    normalized = (markdown_text or "").strip()
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]

    title = None
    for line in lines:
        if line.startswith("#"):
            title = line.lstrip("#").strip()
            break

    if not title:
        title = os.path.splitext(os.path.basename(source_path))[0].replace("_", " ").replace("-", " ").strip()

    doc_number = None
    patterns = [
        r"document\s*(?:no\.?|number)\s*[:#-]?\s*([A-Za-z0-9\-/\.]+)",
        r"\bno\.\s*([A-Za-z0-9\-/\.]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            doc_number = match.group(1).strip()
            break

    if not doc_number:
        doc_number = os.path.splitext(os.path.basename(source_path))[0]

    year = fallback_year
    year_match = re.search(r"\b(19|20)\d{2}\b", normalized)
    if year_match:
        year = int(year_match.group(0))

    issuing_authority = "Government"
    authority_patterns = [r"issued\s+by\s*[\:\-]\s*(.+)", r"authority\s*[\:\-]\s*(.+)"]
    for pattern in authority_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            issuing_authority = re.sub(r"\s+", " ", match.group(1)).strip()
            break

    document_category = "Document"
    for label, category in [
        ("notification", "Notification"),
        ("circular", "Circular"),
        ("order", "Order"),
        ("rule", "Rule"),
        ("guideline", "Guideline"),
        ("directive", "Directive"),
    ]:
        if re.search(rf"\b{label}\b", (title or ""), flags=re.IGNORECASE):
            document_category = category
            break

    supersedes = None
    supersede_match = re.search(r"in\s+supersession\s+of\s+([^\n]{3,200})", normalized, flags=re.IGNORECASE)
    if supersede_match:
        supersedes = _extract_ref_target(supersede_match.group(1))

    references = None
    ref_match = re.search(r"reference\s*[:-]\s*([^\n]+)", normalized, flags=re.IGNORECASE)
    if ref_match:
        references = ref_match.group(1).strip()

    return {
        "document_title": title,
        "year": year,
        "doc_number": doc_number,
        "issuing_authority": issuing_authority,
        "document_category": document_category,
        "source_filename": os.path.basename(source_path),
        "supersedes": supersedes,
        "references": references,
    }
