import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types
from qdrant_client import QdrantClient, models

from core.embedding import get_sparse_model
from retrieval.support import extract_response_text
from core.utils import embed_content_safe, generate_content_safe


def build_evidence(search_results: List[Any]) -> List[Dict[str, Any]]:
    """Build evidence list from search results for citations."""
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


def rerank_results(
    gemini_client: genai.Client,
    results: List[Any],
    standalone_query: str,
    rerank_limit: int,
) -> List[Any]:
    """Rerank search results using LLM judge and return top results."""
    ranking_prompt = f"""Rank the passages below from most relevant to least relevant for answering the question.
Return ONLY a JSON array of indices in descending order of relevance, e.g. [2, 0, 1].

Question: {standalone_query}

Passages:
"""
    for idx, res in enumerate(results):
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
            return [results[idx] for idx in ranked_indices if 0 <= idx < len(results)][:rerank_limit]
    except Exception:
        pass
    return results[:rerank_limit]


def run_hybrid_search(
    gemini_client: genai.Client,
    qdrant_client: QdrantClient,
    query_vectors: List[List[float]],
    query_variations: List[str],
    standalone_query: str,
    collection_name: str,
    query_filter: Optional[models.Filter],
    fast_path: bool,
) -> Tuple[List[Any], List[Any], List[Dict[str, Any]]]:
    """Execute hybrid dense+BM25 search with RRF fusion, optional reranking, and evidence extraction."""
    limit = 4 if fast_path else 25
    rerank_limit = 3 if fast_path else 12

    sparse_model = get_sparse_model()
    query_vector = query_vectors[0]

    # Main query sparse vector
    main_sparse_emb = list(sparse_model.embed([standalone_query]))[0]
    main_sparse_vec = models.SparseVector(
        indices=main_sparse_emb.indices.tolist(),
        values=main_sparse_emb.values.tolist(),
    )

    has_keywords = bool(re.search(r"\b\d+[-/]?\d*\b|\b(section|circular|notification|act|rule|form|gst|hsn)\b", standalone_query, re.IGNORECASE))
    dense_limit = limit if has_keywords else int(limit * 1.5)
    bm25_limit = int(limit * 2.5) if has_keywords else limit

    # Build prefetches across all query variations (Dense + BM25)
    prefetch = []
    for idx, q_var in enumerate(query_variations):
        q_vec = query_vectors[idx] if idx < len(query_vectors) else query_vectors[0]
        sparse_emb = list(sparse_model.embed([q_var]))[0]
        sparse_vec = models.SparseVector(
            indices=sparse_emb.indices.tolist(),
            values=sparse_emb.values.tolist(),
        )
        prefetch.append(models.Prefetch(query=q_vec, using="dense", limit=dense_limit, filter=query_filter))
        prefetch.append(models.Prefetch(query=sparse_vec, using="bm25", limit=bm25_limit, filter=query_filter))

    if has_keywords:
        prefetch.append(models.Prefetch(query=main_sparse_vec, using="bm25", limit=bm25_limit, filter=query_filter))

    search_response = qdrant_client.query_points(
        collection_name=collection_name,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
    )
    current_results = search_response.points

    # Fallback: if strict metadata filter yielded no results, retry without filter
    if not current_results and query_filter is not None:
        prefetch_no_filter = [
            models.Prefetch(query=query_vector, using="dense", limit=limit),
            models.Prefetch(query=main_sparse_vec, using="bm25", limit=limit),
        ]
        search_response = qdrant_client.query_points(
            collection_name=collection_name,
            prefetch=prefetch_no_filter,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
        )
        current_results = search_response.points

    if not current_results:
        return [], [], []

    if fast_path:
        top_results = current_results[:rerank_limit]
    else:
        top_results = rerank_results(gemini_client, current_results, standalone_query, rerank_limit)

    return current_results, top_results, build_evidence(top_results)
