import os
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from utils import generate_content_safe, embed_content_safe, generate_content_stream_safe

def run_retrieval(gemini_client: genai.Client, qdrant_client: QdrantClient, query: str, collection_name: str = "gov_docs", chat_history: list = None):
    """
    Main entry point for the retrieval module.
    """
    if chat_history is None:
        chat_history = []
        
    print(f"Processing Query: '{query}'")
    
    # 0. Contextualize query with Memory
    standalone_query = query
    history_text = ""
    if chat_history:
        print("\n[Memory] Contextualizing query using conversation history...")
        history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['text']}" for msg in chat_history[-4:]]) # Keep last 2 turns
        
        ctx_prompt = (
            "Given the following conversation history and the user's latest question, rephrase the latest question "
            "to be a standalone question that can be understood without the context of the conversation. "
            "CRITICAL INSTRUCTION: If the latest question introduces a completely new topic or is clearly unrelated to the "
            "conversation history, do NOT merge or carry over constraints (like years, document names, or specific rules) "
            "from the history. Treat it as a completely new query. If it is already standalone, return it exactly as is.\n\n"
            f"Conversation History:\n{history_text}\n\n"
            f"Latest Question: {query}\n\nStandalone Question:"
        )
        try:
            ctx_response = generate_content_safe(
                gemini_client,
                model=os.getenv("GEN_MODEL_NAME", "gemma-4-31b-it"),
                contents=ctx_prompt,
                config=types.GenerateContentConfig(temperature=0.0)
            )
            if ctx_response.text:
                standalone_query = ctx_response.text.strip()
                print(f"[Memory] Rewrote query to: '{standalone_query}'")
        except Exception as e:
            print(f"[Memory] Failed to contextualize query: {e}")

    # 1. Embed the standalone query
    config = types.EmbedContentConfig(
        task_type="RETRIEVAL_QUERY", # Note the different task type for queries!
        output_dimensionality=1536
    )
    model_name = os.getenv("EMBED_MODEL_NAME", "gemini-embedding-001")
    
    print("Generating query embedding...")
    response = embed_content_safe(
        gemini_client,
        model=model_name,
        contents=standalone_query,
        config=config
    )
    
    query_vector = response.embeddings[0].values
    
    # 1.5 Extract Metadata Filters (Dynamic)
    print("\n[Filter] Checking query for metadata filters...")
    filter_prompt = f"""Extract any filtering criteria from the following question. 
If the user explicitly mentions a specific year, extract it. Otherwise return empty.
Question: {standalone_query}
Output ONLY a valid JSON object, e.g., {{"year": 2025}} or {{}}"""
    
    query_filter = None
    from qdrant_client import models
    try:
        filter_response = generate_content_safe(
            gemini_client,
            model=os.getenv("GEN_MODEL_NAME", "gemma-4-31b-it"),
            contents=filter_prompt,
            config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json")
        )
        import json
        extracted_filters = json.loads(filter_response.text)
        
        must_conditions = []
        if "year" in extracted_filters and extracted_filters["year"]:
            year_val = int(extracted_filters["year"])
            must_conditions.append(models.FieldCondition(key="year", match=models.MatchValue(value=year_val)))
            print(f"[Filter] Applied Year Filter: {year_val}")
            
        if must_conditions:
            query_filter = models.Filter(must=must_conditions)
        else:
            print("[Filter] No specific metadata filters detected.")
    except Exception as e:
        print(f"[Filter] Failed to extract filters: {e}")
    
    # 2. Qdrant Vector Search (Hybrid)
    print(f"\n[Search] Running hybrid (dense + sparse) search in Qdrant collection '{collection_name}'...")
    
    # Generate Sparse Vector for the query
    from fastembed import SparseTextEmbedding
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    sparse_embedding = list(sparse_model.embed([standalone_query]))[0]
    sparse_query = models.SparseVector(
        indices=sparse_embedding.indices.tolist(),
        values=sparse_embedding.values.tolist()
    )
    
    prefetch = [
        models.Prefetch(
            query=query_vector,
            using="dense",
            limit=5,
            filter=query_filter
        ),
        models.Prefetch(
            query=sparse_query,
            using="bm25",
            limit=5,
            filter=query_filter
        )
    ]
    
    search_response = qdrant_client.query_points(
        collection_name=collection_name,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=5 # Fetch top 5 closest chunks after fusion
    )
    search_results = search_response.points
    
    if not search_results:
        return None
        
    print(f"[Search] Found {len(search_results)} relevant child chunks before reranking.")
    
    # 2.5 Cross-Encoder Re-Ranking
    print("[Rerank] Re-ranking chunks with Cross-Encoder...")
    from sentence_transformers import CrossEncoder
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Pair the query with each chunk's text for scoring
    pairs = [[standalone_query, result.payload.get("child_text", "")] for result in search_results]
    scores = cross_encoder.predict(pairs)
    
    # Sort results by score descending
    scored_results = sorted(zip(scores, search_results), key=lambda x: x[0], reverse=True)
    # Take the top 3 most relevant chunks
    top_results = [result for score, result in scored_results[:3]]
    
    print(f"[Rerank] Kept top {len(top_results)} most relevant chunks.")
    
    # 3. Deduplication Logic (Small-to-Big)
    print("[Retrieval] Deduplicating retrieved child chunks to extract unique parent context...")
    unique_parents = {}
    for result in top_results:
        parent_id = result.payload.get("parent_id")
        if parent_id and parent_id not in unique_parents:
            unique_parents[parent_id] = result.payload.get("parent_context", "")
            
    context_text = "\n\n---\n\n".join(unique_parents.values())
    print(f"[Retrieval] Extracted {len(unique_parents)} unique parent sections for LLM context.")
    
    # 4. Generate Answer using LLM
    print("\n[Generation] Generating grounded answer with Gemini...")
    
    prompt = f"""You are GovAssist, an AI Question Answering Assistant for Government Documents.

Your task is to answer the user's question ONLY using the retrieved search results.

Guidelines:
1. Use ONLY the information present in the retrieved search results.
2. Do NOT use outside knowledge, assumptions, or prior training.
3. If the retrieved results do not contain enough information to answer the question, respond exactly:
   "Sorry, I could not find an exact answer in the indexed government documents."
4. Do not fabricate laws, dates, section numbers, definitions, or procedures.
5. If multiple retrieved passages contain complementary information, combine them into a single coherent answer.
6. Prefer the most recent notification if multiple documents appear to conflict.

Response Format:
### Answer
Provide a concise, well-structured answer in natural language.

### Explanation
Briefly explain how the retrieved information answers the user's question.

### Source(s) & Citations
Mention:
- Document title
- Relevant chapter/section (if available)
- CRITICAL: Provide the exact, verbatim quote from the retrieved text that justifies your answer.

Formatting:
- CRITICAL: You MUST write your entire response (Answer and Explanation) in the exact same language as the User Question. If the document is in Marathi and the user asks in English, you MUST translate your answer into English.
- Use bullet points where appropriate.
- Keep answers factual and professional.
- Avoid repeating the same information.
- Do not mention internal retrieval or search processes.
- Do not say "According to the search results."

Recent Conversation History:
{history_text if history_text else "No previous history."}

Search Results:
{context_text}

User Question:
{query}
"""
    
    answer_stream = generate_content_stream_safe(
        gemini_client,
        model=os.getenv("GEN_MODEL_NAME", "gemma-4-31b-it"),
        contents=prompt
    )
    
    return answer_stream
