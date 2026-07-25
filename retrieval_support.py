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
    """Collect unique parent contexts from the top results for grounded generation."""
    unique_parents = {}
    for result in top_results:
        payload = getattr(result, "payload", None) or {}
        parent_id = payload.get("parent_id")
        if parent_id and parent_id not in unique_parents:
            unique_parents[parent_id] = payload.get("parent_context", "")

    return "\n\n---\n\n".join(unique_parents.values())
