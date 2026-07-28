import json
import os
import re
import concurrent.futures
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types
import weaviate
import weaviate.classes as wvc
from retrieval.query import generate_query_variations
from retrieval.support import extract_response_text, build_context_text
from core.utils import embed_content_safe, generate_content_safe
from core.log_config import get_logger

logger = get_logger(__name__)


search_policy_docs_tool = {
    "type": "function",
    "function": {
        "name": "search_policy_docs",
        "description": "Search indexed government policy documents, circulars, acts, and rules for factual answers.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query or keywords to look up in government documents.",
                },
                "year": {
                    "type": "integer",
                    "description": "Optional publication year filter (e.g. 2025, 2020, 2019, 1961).",
                },
                "fast_mode": {
                    "type": "boolean",
                    "description": "Set true for fast search, false for deep analytical multi-query search.",
                },
            },
            "required": ["query"],
        },
    }
}


def build_evidence(search_results: List[Any]) -> List[Dict[str, Any]]:
    """Build evidence list from search results for citations."""
    evidence: List[Dict[str, Any]] = []
    for result in search_results:
        payload = result.payload or {}
        child_text = (payload.get("child_text") or "").strip()
        parent_context = (payload.get("parent_context") or "").strip()
        quote = child_text[:400] if child_text else parent_context[:400]
        section_parts = [payload.get(k) for k in ["Document_Part", "Header_1", "Header_2", "Header_3"] if payload.get(k)]
        raw_section = " > ".join(str(p) for p in section_parts) if section_parts else "Section not available"
        clean_sec = raw_section.split(" > ")[-1] if " > " in raw_section else raw_section
        clean_sec = "".join(c for c in clean_sec if ord(c) < 128).strip()
        if not clean_sec:
            clean_sec = "Section not available"

        doc_number = payload.get("doc_number", "Unknown document")
        title = payload.get("document_title", "")
        doc_label = f"{title} ({doc_number})" if title else doc_number
        
        evidence.append(
            {
                "document": doc_label,
                "year": payload.get("year"),
                "section": clean_sec,
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
            model=os.getenv("GEN_MODEL_NAME", "gemini-2.5-flash"),
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
    weaviate_client: Any,
    query_vectors: List[List[float]],
    query_variations: List[str],
    standalone_query: str,
    collection_name: str,
    year_filter: Optional[int],
    fast_mode: bool,
) -> Tuple[List[Any], List[Any], List[Dict[str, Any]]]:
    """Execute hybrid dense+BM25 search with RRF fusion, optional reranking, and evidence extraction."""
    limit = 15 if fast_mode else 25
    rerank_limit = 15 if fast_mode else 12

    query_vector = query_vectors[0] if query_vectors else []

    has_keywords = bool(re.search(r"\b\d+[-/]?\d*\b|\b(section|circular|notification|act|rule|form|gst|hsn)\b", standalone_query, re.IGNORECASE))
    
    class DummyResult:
        def __init__(self, props):
            self.payload = props
    
    weaviate_collection = weaviate_client.collections.get("GovDocs")
    
    # Determine weight for alpha fusion (0 = pure keyword BM25, 1 = pure semantic dense vector)
    alpha = 0.5 if has_keywords else 0.75
    
    weaviate_filters = None
    if year_filter:
        weaviate_filters = wvc.query.Filter.by_property("year").equal(year_filter)
    
    search_res = weaviate_collection.query.hybrid(
        query=standalone_query,
        vector=query_vector,
        alpha=alpha,
        limit=limit,
        filters=weaviate_filters
    )
    current_results = [DummyResult(obj.properties) for obj in search_res.objects]

    if not current_results:
        return [], [], []

    if fast_mode:
        top_results = current_results[:rerank_limit]
    else:
        top_results = rerank_results(gemini_client, current_results, standalone_query, rerank_limit)

    return current_results, top_results, build_evidence(top_results)


def execute_search_tool(
    gemini_client: genai.Client,
    weaviate_client: Any,
    collection_name: str,
    query: str,
    year: Optional[int] = None,
    fast_mode: bool = False,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Execute the search tool and format results for the LLM."""
    query_variations = [query]
    logger.info("Generating query variations...")
    query_variations = [query]
    if not fast_mode:
        query_variations = generate_query_variations(gemini_client, query)

    config = types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=1536)
    model_name = os.getenv("EMBED_MODEL_NAME", "gemini-embedding-001")
    
    query_vectors = []
    
    def embed_single_variation(q_var):
        try:
            logger.info(f"Embedding variation: {q_var}")
            resp = embed_content_safe(gemini_client, model=model_name, contents=q_var, config=config)
            return resp.embeddings[0].values
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return None
            
    logger.info("Starting threadpool for embeddings...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(embed_single_variation, query_variations))
        
    logger.info("Finished threadpool for embeddings.")
    for res in results:
        if res is not None:
            query_vectors.append(res)
            
    if not query_vectors:
        return json.dumps({"error": "Failed to generate embeddings."}), []

    search_results, top_results, evidence = run_hybrid_search(
        gemini_client, weaviate_client, query_vectors, query_variations,
        query, collection_name, year, fast_mode
    )

    if not evidence:
        return json.dumps({"results": "No relevant documents found. Try modifying the keywords or changing fast_mode to false."}), []

    context_text = build_context_text(top_results)
    
    result_dict = {
        "status": "success",
        "fast_mode_used": fast_mode,
        "results_count": len(evidence),
        "context": context_text
    }
    return json.dumps(result_dict), evidence
