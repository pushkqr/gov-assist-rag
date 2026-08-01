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
        score = getattr(result.metadata, "score", 0.0) if hasattr(result, "metadata") and result.metadata else 0.0
        
        evidence.append(
            {
                "document": doc_label,
                "year": payload.get("year"),
                "section": clean_sec,
                "quote": quote,
                "score": score,
                "filename": payload.get("source_filename"),
                "supersedes": payload.get("supersedes")
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
        passage_text = payload.get("translated_text") or payload.get("child_text") or payload.get("parent_context") or ""
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
    limit = 35 if fast_mode else 150
    rerank_limit = 35 if fast_mode else 12

    query_vector = query_vectors[0] if query_vectors else []

    # Weight alpha: 
    # - 0.25 (BM25-heavy) for explicit GR code patterns
    # - 0.50 (Balanced) for all other general queries
    gr_code_pattern = r"[A-Z]{2,}[-/]\d+|P\.?No\.?\s*\d+|No\.\s+\d+/"
    if re.search(gr_code_pattern, standalone_query):
        alpha = 0.25
    else:
        alpha = 0.50
    
    bm25_query = " ".join(query_variations) if query_variations else standalone_query
    
    class DummyResult:
        def __init__(self, props, metadata=None):
            self.payload = props
            self.metadata = metadata
            
    weaviate_collection = weaviate_client.collections.get(collection_name)
    
    weaviate_filters = None
    if year_filter:
        weaviate_filters = wvc.query.Filter.by_property("year").equal(year_filter)
    
    search_res = weaviate_collection.query.hybrid(
        query=bm25_query,
        query_properties=["translated_text", "parent_context", "section_title", "child_text", "doc_number"],
        vector=query_vector,
        alpha=alpha,
        limit=limit,
        filters=weaviate_filters,
        return_metadata=wvc.query.MetadataQuery(score=True)
    )
    current_results = [DummyResult(obj.properties, obj.metadata) for obj in search_res.objects]

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
    fast_mode: bool = True,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, float], List[Dict[str, Any]]]:
    """Execute the search tool and format results for the LLM."""
    import time
    import requests
    t_start = time.time()
    
    def is_indic(text: str) -> bool:
        return any('\u0900' <= c <= '\u097f' for c in text)

    def translate_marathi_batch_local(text: str) -> str:
        try:
            import os
            url = os.environ.get("TRANSLATION_SERVICE_URL", "http://localhost:8000/translate")
            response = requests.post(
                url,
                json={"text": text, "src_lang": "mar_Deva", "tgt_lang": "eng_Latn"},
                timeout=2.0
            )
            response.raise_for_status()
            return response.json().get("translated_text", text)
        except Exception as e:
            logger.error(f"Local translation failed: {e}")
            return text

    if is_indic(query):
        logger.info("Indic query detected. Translating query to English via local microservice...")
        search_query = translate_marathi_batch_local(query)
    else:
        logger.info("English query detected. Bypassing translation API.")
        search_query = query
        
    
    t_translate = time.time()
    logger.info(f"[PROFILING] Translation took: {t_translate - t_start:.3f}s")
    
    query_variations = [search_query]
    logger.info("Generating query variations...")
    if not fast_mode:
        query_variations = generate_query_variations(gemini_client, search_query)
        
    t_variations = time.time()
    if not fast_mode:
        logger.info(f"[PROFILING] Query expansion took: {t_variations - t_translate:.3f}s")

    config = types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=1536)
    model_name = os.getenv("EMBED_MODEL_NAME", "text-embedding-004")
    
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
    t_embed = time.time()
    logger.info(f"[PROFILING] Embeddings took: {t_embed - t_variations:.3f}s")
    
    for res in results:
        if res is not None:
            query_vectors.append(res)
            
    if not query_vectors:
        return json.dumps({"error": "Failed to generate embeddings."}), []

    t_weaviate_start = time.time()
    search_results, top_results, evidence = run_hybrid_search(
        gemini_client, weaviate_client, query_vectors, query_variations,
        search_query, collection_name, year, fast_mode
    )
    t_weaviate = time.time()
    logger.info(f"[PROFILING] Hybrid Search took: {t_weaviate - t_weaviate_start:.3f}s")

    t_translate_time = t_translate - t_start
    t_expansion_time = t_variations - t_translate if not fast_mode else 0.0
    t_embed_time = t_embed - t_variations
    t_weaviate_time = t_weaviate - t_weaviate_start

    profiling = {
        "translation_s": round(t_translate_time, 3),
        "expansion_s": round(t_expansion_time, 3),
        "embedding_s": round(t_embed_time, 3),
        "weaviate_s": round(t_weaviate_time, 3),
    }

    if not evidence:
        return json.dumps({"results": "No relevant documents found. Try modifying the keywords or changing fast_mode to false."}), [], profiling, []

    context_text = build_context_text(top_results)
    
    used_titles = {e.get("document") for e in evidence if e.get("document")}
    recommendations = []
    seen_recs = set(used_titles)
    for res in search_results:
        payload = res.payload or {}
        doc_num = payload.get("doc_number", "Unknown document")
        title = payload.get("document_title", "")
        doc_label = f"{title} ({doc_num})" if title and title != doc_num else doc_num
        
        if doc_label and doc_label not in seen_recs:
            seen_recs.add(doc_label)
            recommendations.append({
                "document": doc_label,
                "year": payload.get("year"),
                "category": payload.get("document_category", "Document"),
                "source_filename": payload.get("source_filename")
            })
            if len(recommendations) >= 5:
                break
    
    result_dict = {
        "status": "success",
        "fast_mode_used": fast_mode,
        "results_count": len(evidence),
        "context": context_text
    }
    return json.dumps(result_dict), evidence, profiling, recommendations
