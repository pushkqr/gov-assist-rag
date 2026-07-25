from typing import Any, List


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


def build_context_text(top_results: List[Any]) -> str:
    """Stitch unique parent contexts and child passages cleanly into a unified context block."""
    parent_blocks = {}
    standalone_chunks = []

    for result in top_results:
        payload = getattr(result, "payload", None) or {}
        parent_id = payload.get("parent_id")
        parent_ctx = (payload.get("parent_context") or "").strip()
        child_txt = (payload.get("child_text") or "").strip()

        if parent_id:
            if parent_id not in parent_blocks:
                parent_blocks[parent_id] = {
                    "doc_number": payload.get("doc_number", "Document"),
                    "section_title": payload.get("section_title", "Section"),
                    "context": parent_ctx if parent_ctx else child_txt,
                    "children": [child_txt] if child_txt else [],
                }
            elif child_txt and child_txt not in parent_blocks[parent_id]["children"]:
                parent_blocks[parent_id]["children"].append(child_txt)
        elif child_txt:
            standalone_chunks.append(child_txt)

    formatted_sections = []
    for pid, block in parent_blocks.items():
        ctx = block["context"]
        if ctx:
            formatted_sections.append(ctx)

    for chunk in standalone_chunks:
        if chunk not in formatted_sections:
            formatted_sections.append(chunk)

    return "\n\n---\n\n".join(s for s in formatted_sections if s.strip())
