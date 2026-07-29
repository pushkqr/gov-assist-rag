import json
import os
from typing import Any, Dict, List, Optional, Tuple

from google.genai import types

from retrieval.support import extract_response_text
from core.utils import generate_content_safe
from core.log_config import get_logger

logger = get_logger(__name__)


def contextualize_query(gemini_client: Any, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Tuple[str, str]:
    """Rewrite follow-up question into standalone query using conversation history."""
    if chat_history is None:
        chat_history = []

    standalone_query = query
    history_text = ""
    if chat_history:
        history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['text']}" for msg in chat_history[-4:]])
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
                config=types.GenerateContentConfig(temperature=0.0),
            )
            ctx_text = extract_response_text(ctx_response)
            if ctx_text:
                standalone_query = ctx_text.strip()
        except Exception as exc:
            logger.warning(f"Could not contextualize query: {exc}")

    return standalone_query, history_text


def generate_query_variations(gemini_client: Any, standalone_query: str) -> List[str]:
    """Generate multi-query variations including Marathi Devanagari transliterations and formal HSN/legal terms."""
    if not standalone_query or len(standalone_query.strip().split()) <= 3:
        return [standalone_query]

    prompt = f"""You are a query expansion specialist for Indian and Maharashtra state government policy documents.

Task: Given the user question, generate 2 complementary search queries in English to maximize document retrieval recall:

1. Keyword-Dense Search String:
   - Extract core entities, numbers, dates, HSN codes, and legal section numbers.
   - If the query mentions GST/tax products, include official HSN/tariff classifications.

2. Broad Concept Variation:
   - Rephrase the query using broader policy synonyms (e.g., if asking about 'scholarship', include 'financial aid' or 'educational scheme').

User Question: {standalone_query}

Return ONLY a valid JSON array of 2 strings: ["Keyword-Dense Search String", "Broad Concept Variation"]."""

    try:
        response = generate_content_safe(
            gemini_client,
            model=os.getenv("GEN_MODEL_NAME", "gemma-4-31b-it"),
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json"),
        )
        parsed = json.loads(extract_response_text(response))
        if isinstance(parsed, list):
            variations = [standalone_query] + [str(v).strip() for v in parsed if v and str(v).strip()]
            return variations[:3]
    except Exception as exc:
        logger.warning(f"Failed to generate variations: {exc}")

    return [standalone_query]


def extract_query_filter(gemini_client: Any, standalone_query: str) -> Optional[Dict[str, Any]]:
    """Extract metadata filters (year, section title) from query using the generation model."""
    if not standalone_query or not standalone_query.strip():
        return None

    filter_prompt = f"""Extract any filtering criteria from the following question. 
If the user explicitly mentions a specific year, extract it. Otherwise return empty.
If the user explicitly mentions a document or section name, extract it as a string. Otherwise return empty.
Question: {standalone_query}
Output ONLY a valid JSON object, e.g., {{"year": 2025, "section_title": "Eligibility"}} or {{}}"""

    try:
        filter_response = generate_content_safe(
            gemini_client,
            model=os.getenv("GEN_MODEL_NAME", "gemma-4-31b-it"),
            contents=filter_prompt,
            config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json"),
        )
        extracted_filters = extract_response_text(filter_response)
        if isinstance(extracted_filters, str):
            extracted_filters = extracted_filters.strip()
            if not extracted_filters:
                return None
            extracted_filters = json.loads(extracted_filters)
        return extracted_filters if isinstance(extracted_filters, dict) else None
    except Exception as exc:
        logger.warning(f"Could not extract filters: {exc}")
        return None


def build_generation_prompt(query: str, history_text: str, context_text: str) -> str:
    """Build grounded generation prompt for the final answer."""
    return f"""You are Mimir, an AI Question Answering Assistant for Government Documents.

Your task is to answer the user's question ONLY using the retrieved search results.

Guidelines:
1. Use ONLY the information present in the retrieved search results.
2. Do NOT use outside knowledge, assumptions, or prior training.
3. Answer the question as completely and accurately as possible using the retrieved search results. Synthesize information across the search results. Only if the search results are completely unrelated or contain zero relevant information, respond exactly:
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
{history_text if history_text else 'No previous history.'}

Search Results:
{context_text}

User Question:
{query}
"""
