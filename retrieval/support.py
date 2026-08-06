import os
from typing import Any, List

import core.deployment as deployment


def extract_response_text(response: Any) -> str:
    """Safely extract text from either a model response object or a plain string."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text
    return ""


def _is_mostly_indic(text: str) -> bool:
    """Return True if more than 30% of characters are Devanagari (Marathi/Hindi source)."""
    if not text:
        return False
    indic = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    return indic / max(len(text), 1) > 0.3


def _block_char_cap() -> int:
    """Per-block context cap, applied only when generation runs on local CPU.

    Prompt processing on the self-hosted node runs at 40-53 tok/s and the rate decays as the
    window grows (measured: 1024 tokens in 19s, 4971 in 135s), so context length is the
    dominant term in sovereign latency, not thread count or the serving stack.

    The cap is per block rather than on the joined string because trimming the tail would
    drop whole documents, and a contradiction between two documents is only detectable when
    both are still present - which is the behaviour the conflict callout exists to show.

    Unset means unlimited, so the default is byte-for-byte the current output and the hosted
    path never reads this at all.
    """
    if deployment.gen_provider() != "local":
        return 0
    raw = os.getenv("SOVEREIGN_CONTEXT_BLOCK_CHARS", "").strip()
    return int(raw) if raw.isdigit() else 0


def build_context_text(top_results: List[Any]) -> str:
    """Stitch unique parent contexts and child passages into a unified context block.

    Each block is prefixed with a [Document: ... | Section: ...] header so the LLM
    can cite sources precisely. For Marathi-source chunks, English translations are
    preferred over raw Devanagari to reduce LLM reasoning overhead on English queries.
    """
    parent_blocks = {}
    standalone_chunks = []
    char_cap = _block_char_cap()

    for result in top_results:
        payload = getattr(result, "payload", None) or {}
        parent_id = payload.get("parent_id")
        parent_ctx = (payload.get("parent_context") or "").strip()
        child_txt = (payload.get("child_text") or "").strip()
        translated = (payload.get("translated_text") or "").strip()

        if parent_id:
            if parent_id not in parent_blocks:
                parent_blocks[parent_id] = {
                    "doc_number": payload.get("doc_number", "Document"),
                    "document_title": payload.get("document_title", ""),
                    "section_title": payload.get("section_title", ""),
                    "context": parent_ctx if parent_ctx else child_txt,
                    "children": [child_txt] if child_txt else [],
                    "translated_texts": [translated] if translated else [],
                    "supersedes": payload.get("supersedes"),
                    "references": payload.get("references"),
                }
            else:
                if child_txt and child_txt not in parent_blocks[parent_id]["children"]:
                    parent_blocks[parent_id]["children"].append(child_txt)
                if translated and translated not in parent_blocks[parent_id]["translated_texts"]:
                    parent_blocks[parent_id]["translated_texts"].append(translated)
        elif child_txt:
            standalone_chunks.append({
                "doc_number": payload.get("doc_number", "Document"),
                "document_title": payload.get("document_title", ""),
                "section_title": payload.get("section_title", ""),
                "text": child_txt,
                "translated": translated,
                "supersedes": payload.get("supersedes"),
                "references": payload.get("references"),
            })

    def _make_header(doc_number: str, document_title: str, section_title: str) -> str:
        doc_label = (
            f"{document_title} ({doc_number})"
            if document_title and document_title != doc_number
            else doc_number
        )
        header = f"[Document: {doc_label}"
        if section_title:
            header += f" | Section: {section_title}"
        return header + "]"

    formatted_sections = []

    for pid, block in parent_blocks.items():
        ctx = block["context"]
        if not ctx:
            continue

        # For Marathi-source blocks, prefer joined English translations to reduce
        # LLM reasoning overhead when answering English queries.
        translated_texts = block.get("translated_texts", [])
        if translated_texts and _is_mostly_indic(ctx):
            ctx = "\n\n".join(t for t in translated_texts if t)

        if not ctx:
            continue

        header = _make_header(
            block["doc_number"],
            block.get("document_title", ""),
            block.get("section_title", ""),
        )
        meta = header + "\n"
        if block.get("supersedes"):
            meta += f"[Supersedes: {block['supersedes']}]\n"
        if block.get("references"):
            meta += f"[References: {block['references']}]\n"
        formatted_sections.append(meta + (ctx[:char_cap] if char_cap else ctx))

    seen_texts = {block["context"] for block in parent_blocks.values() if block.get("context")}
    for chunk in standalone_chunks:
        txt = chunk["text"]
        translated = chunk.get("translated", "")
        # Prefer English translation for Marathi-source standalone chunks
        display_txt = translated if translated and _is_mostly_indic(txt) else txt
        if not display_txt or display_txt in seen_texts:
            continue
        header = _make_header(
            chunk["doc_number"],
            chunk.get("document_title", ""),
            chunk.get("section_title", ""),
        )
        meta = header + "\n"
        if chunk.get("supersedes"):
            meta += f"[Supersedes: {chunk['supersedes']}]\n"
        if chunk.get("references"):
            meta += f"[References: {chunk['references']}]\n"
        formatted_sections.append(meta + (display_txt[:char_cap] if char_cap else display_txt))
        seen_texts.add(display_txt)

    return "\n\n---\n\n".join(s for s in formatted_sections if s.strip())
