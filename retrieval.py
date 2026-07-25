import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from fastembed import SparseTextEmbedding
from google import genai
from google.genai import types
from qdrant_client import QdrantClient, models

from retrieval_pipeline import build_generation_prompt, contextualize_query, extract_query_filter
from retrieval_support import build_context_text, extract_response_text
from utils import embed_content_safe, generate_content_safe, generate_content_stream_safe

_QUERY_CACHE: Dict[str, Dict[str, Any]] = {}
_SPARSE_MODEL: Optional[SparseTextEmbedding] = None


def get_sparse_model() -> SparseTextEmbedding:
    """Lazy singleton for BM25 sparse embedding model."""
    global _SPARSE_MODEL
    if _SPARSE_MODEL is None:
        _SPARSE_MODEL = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _SPARSE_MODEL


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


def _build_evidence(search_results: List[Any]) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    for result in search_results:
        payload = result.payload or {}
        child_text = (payload.get("child_text") or "").strip()
        parent_context = (payload.get("parent_context") or "").strip()
        quote = child_text[:400] if child_text else parent_context[:400]
        section_parts = [payload.get(k) for k in ["Document_Part", "Header_1", "Header_2", "Header_3"] if payload.get(k)]
        evidence.append(
            {
                "document": payload.get("doc_number", "Unknown document"),
                "year": payload.get("year"),
                "section": " > ".join(str(p) for p in section_parts) if section_parts else "Section not available",
                "quote": quote,
            }
        )
    return evidence


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

    config = types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=1536)
    model_name = os.getenv("EMBED_MODEL_NAME", "gemini-embedding-001")

    try:
        response = embed_content_safe(gemini_client, model=model_name, contents=standalone_query, config=config)
        query_vector = response.embeddings[0].values
    except Exception as exc:
        return {
            "status": "error",
            "response_text": f"Embedding failed: {exc}",
            "answer_stream": None,
            "evidence": [],
        }

    query_filter = None
    try:
        extracted_filters = extract_query_filter(gemini_client, standalone_query)
        must_conditions = []
        if extracted_filters and extracted_filters.get("year"):
            must_conditions.append(models.FieldCondition(key="year", match=models.MatchValue(value=int(extracted_filters["year"]))))

        if extracted_filters and extracted_filters.get("section_title"):
            must_conditions.append(models.FieldCondition(key="section_title", match=models.MatchValue(value=str(extracted_filters["section_title"]).strip())))

        if must_conditions:
            query_filter = models.Filter(must=must_conditions)
    except Exception as e:
        print(f"[Filter] Could not extract filters: {e}")

    use_fast_path = fast_mode or should_use_fast_path(query)

    def run_search(current_fast_path: bool) -> Tuple[Optional[List[Any]], List[Any], List[Dict[str, Any]]]:
        limit = 3 if current_fast_path else 8
        rerank_limit = 2 if current_fast_path else 5

        sparse_model = get_sparse_model()
        sparse_embedding = list(sparse_model.embed([standalone_query]))[0]
        sparse_query = models.SparseVector(
            indices=sparse_embedding.indices.tolist(),
            values=sparse_embedding.values.tolist(),
        )

        prefetch = [
            models.Prefetch(query=query_vector, using="dense", limit=limit, filter=query_filter),
            models.Prefetch(query=sparse_query, using="bm25", limit=limit, filter=query_filter),
        ]

        search_response = qdrant_client.query_points(
            collection_name=collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
        )
        current_results = search_response.points
        if not current_results:
            return None, [], []

        if current_fast_path:
            top_results = current_results[:rerank_limit]
        else:
            ranking_prompt = f"""Rank the passages below from most relevant to least relevant for answering the question.
Return ONLY a JSON array of indices in descending order of relevance, e.g. [2, 0, 1].

Question: {standalone_query}

Passages:
"""
            for idx, res in enumerate(current_results):
                payload = res.payload or {}
                passage_text = payload.get("child_text") or payload.get("parent_context") or ""
                ranking_prompt += f"{idx}: {passage_text[:800]}\n\n"

            try:
                ranking_response = generate_content_safe(
                    gemini_client,
                    model=os.getenv("GEN_MODEL_NAME", "gemma-4-31b-it"),
                    contents=ranking_prompt,
                    config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json"),
                )
                ranked_indices = json.loads(extract_response_text(ranking_response))
                if isinstance(ranked_indices, list):
                    top_results = [current_results[idx] for idx in ranked_indices if 0 <= idx < len(current_results)][:rerank_limit]
                else:
                    top_results = current_results[:rerank_limit]
            except Exception:
                top_results = current_results[:rerank_limit]

        return current_results, top_results, _build_evidence(top_results)

    search_results, top_results, evidence = run_search(use_fast_path)

    if not search_results:
        return {
            "status": "empty",
            "response_text": "Sorry, I could not find an exact answer in the indexed government documents.",
            "answer_stream": None,
            "evidence": [],
        }

    if should_escalate_to_deep(search_results, evidence, use_fast_path):
        search_results, top_results, evidence = run_search(False)
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
