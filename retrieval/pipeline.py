import os
import re
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
from qdrant_client import QdrantClient, models

from core.log_config import get_logger
from retrieval.query import build_generation_prompt, contextualize_query, extract_query_filter, generate_query_variations
from retrieval.support import build_context_text
from retrieval.search import run_hybrid_search
from core.utils import embed_content_safe, generate_content_stream_safe

logger = get_logger(__name__)

_QUERY_CACHE: Dict[str, Dict[str, Any]] = {}


def should_use_fast_path(query: str) -> bool:
    """Check if query is suitable for fast lightweight retrieval."""
    if not query or not query.strip():
        return True

    text = query.strip().lower()
    if len(text.split()) <= 5:
        return True

    if re.search(r"\b(hello|hi|thanks|thank you|help|what can you do|who are you)\b", text):
        return True

    if re.search(r"\b(what|who|when|where|why|how)\b", text) and text.count(" ") <= 12:
        return True

    return False


def should_escalate_to_deep(search_results: List[Any], evidence: List[Dict[str, Any]], use_fast_path: bool) -> bool:
    """Escalate to deeper retrieval when fast path results are sparse."""
    if not use_fast_path or not search_results:
        return not search_results if use_fast_path else False

    if len(search_results) < 2 or len(evidence) < 2:
        return True

    return sum(1 for item in evidence if (item.get("quote") or "").strip()) < 2


class StreamingResponse:
    """Wraps answer stream to support multiple UI iteration replays."""

    def __init__(self, answer_stream):
        self._answer_stream = answer_stream
        self.captured_parts: List[str] = []
        self._exhausted = False

    def __iter__(self):
        if self._exhausted:
            for part in self.captured_parts:
                yield part
            return

        for chunk in self._answer_stream:
            text = getattr(chunk, "text", None)
            if text:
                self.captured_parts.append(text)
                yield text
        self._exhausted = True

    @property
    def full_text(self) -> str:
        return "".join(self.captured_parts)


def run_retrieval(
    gemini_client: genai.Client,
    qdrant_client: QdrantClient,
    query: str,
    collection_name: str = "gov_docs",
    chat_history: Optional[List[Dict[str, str]]] = None,
    fast_mode: bool = False,
) -> Dict[str, Any]:
    """Execute hybrid search retrieval and stream grounded generation."""
    if chat_history is None:
        chat_history = []

    cache_key = f"{collection_name}:{query.strip().lower()}"
    if cache_key in _QUERY_CACHE:
        cached_result = _QUERY_CACHE[cache_key]
        if cached_result.get("status") == "success":
            return cached_result
        del _QUERY_CACHE[cache_key]

    standalone_query = query
    history_text = ""
    if chat_history:
        standalone_query, history_text = contextualize_query(gemini_client, query, chat_history)

    use_fast_path = fast_mode and should_use_fast_path(query)

    # Generate multi-query variations for deep retrieval mode
    query_variations = [standalone_query]
    if not use_fast_path:
        query_variations = generate_query_variations(gemini_client, standalone_query)

    config = types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=1536)
    model_name = os.getenv("EMBED_MODEL_NAME", "gemini-embedding-001")

    query_vectors = []
    for q_var in query_variations:
        try:
            resp = embed_content_safe(gemini_client, model=model_name, contents=q_var, config=config)
            query_vectors.append(resp.embeddings[0].values)
        except Exception as exc:
            if not query_vectors:
                return {
                    "status": "error",
                    "response_text": f"Embedding failed: {exc}",
                    "answer_stream": None,
                    "evidence": [],
                }

    # Extract metadata filters
    query_filter = None
    try:
        extracted_filters = extract_query_filter(gemini_client, standalone_query)
        must_conditions = []
        if extracted_filters and extracted_filters.get("year"):
            raw_year = str(extracted_filters["year"])
            year_match = re.search(r"\b(19|20)\d{2}\b", raw_year)
            if year_match:
                year_val = int(year_match.group(0))
                must_conditions.append(models.FieldCondition(key="year", match=models.MatchValue(value=year_val)))

        if extracted_filters and extracted_filters.get("section_title"):
            must_conditions.append(models.FieldCondition(key="section_title", match=models.MatchValue(value=str(extracted_filters["section_title"]).strip())))

        if must_conditions:
            query_filter = models.Filter(must=must_conditions)
    except Exception as e:
        logger.warning(f"Could not extract filters: {e}")

    # Execute hybrid search
    search_results, top_results, evidence = run_hybrid_search(
        gemini_client, qdrant_client, query_vectors, query_variations,
        standalone_query, collection_name, query_filter, use_fast_path,
    )

    if not search_results:
        empty_result = {
            "status": "empty",
            "response_text": "Sorry, I could not find an exact answer in the indexed government documents.",
            "answer_stream": None,
            "evidence": [],
        }
        # Escalate from fast to deep if needed
        if should_escalate_to_deep(search_results, evidence, use_fast_path):
            search_results, top_results, evidence = run_hybrid_search(
                gemini_client, qdrant_client, query_vectors, query_variations,
                standalone_query, collection_name, query_filter, False,
            )
            if not search_results:
                return empty_result
        else:
            return empty_result

    if search_results and should_escalate_to_deep(search_results, evidence, use_fast_path):
        search_results, top_results, evidence = run_hybrid_search(
            gemini_client, qdrant_client, query_vectors, query_variations,
            standalone_query, collection_name, query_filter, False,
        )
        if not search_results:
            return {
                "status": "empty",
                "response_text": "Sorry, I could not find an exact answer in the indexed government documents.",
                "answer_stream": None,
                "evidence": [],
            }

    context_text = build_context_text(top_results)
    prompt = build_generation_prompt(query=query, history_text=history_text, context_text=context_text)

    try:
        answer_stream = generate_content_stream_safe(
            gemini_client,
            model=os.getenv("GEN_MODEL_NAME", "gemma-4-31b-it"),
            contents=prompt,
        )
    except Exception as exc:
        return {
            "status": "error",
            "response_text": f"Generation failed: {exc}",
            "answer_stream": None,
            "evidence": evidence,
        }

    result = {
        "status": "success",
        "response_text": None,
        "answer_stream": StreamingResponse(answer_stream),
        "evidence": evidence,
    }
    _QUERY_CACHE[cache_key] = result
    return result
