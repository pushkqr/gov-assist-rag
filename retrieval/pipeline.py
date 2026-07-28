import os
import json
import concurrent.futures
from typing import Any, Dict, List, Optional, Callable

from google import genai
from google.genai import types
from cerebras.cloud.sdk import Cerebras

from core.log_config import get_logger
from retrieval.search import search_policy_docs_tool, execute_search_tool

logger = get_logger(__name__)


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

        if isinstance(self._answer_stream, str):
            self.captured_parts = [self._answer_stream]
            self._exhausted = True
            yield self._answer_stream
            return

        for chunk in self._answer_stream:
            text = None
            if hasattr(chunk, "text"):
                text = chunk.text
            elif hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None)
            
            if text:
                self.captured_parts.append(text)
                yield text
        self._exhausted = True

    @property
    def full_text(self) -> str:
        if isinstance(self._answer_stream, str):
            return self._answer_stream
        return "".join(self.captured_parts)


_MODEL_COUNTER = 0

def run_retrieval(
    gemini_client: genai.Client,
    cerebras_client: Cerebras,
    weaviate_client: Optional[Any] = None,
    query: str = "",
    collection_name: str = "gov_docs",
    chat_history: Optional[List[Dict[str, str]]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Execute direct Weaviate search followed by 1-shot Cerebras synthesis (Round-Robin load balanced)."""
    global _MODEL_COUNTER
    
    if chat_history is None:
        chat_history = []
        
    if status_callback:
        status_callback("Analyzing query intent...")

    if status_callback:
        status_callback("Searching knowledge base...")
        
    try:
        def search_tool_wrapper(query: str, year: Optional[int] = None, fast_mode: bool = False) -> str:
            logger.info(f"LLM called search_tool(query='{query}', year={year}, fast_mode={fast_mode})")
            results, ev = execute_search_tool(gemini_client, weaviate_client, collection_name, query, year, fast_mode)
            evidence.extend(ev)
            return results

        evidence = []
        search_json = search_tool_wrapper(query, year=None, fast_mode=True)
    except Exception as exc:
        logger.error(f"Search tool failed: {exc}")
        return {
            "status": "error",
            "response_text": f"Search failed: {exc}",
            "answer_stream": StreamingResponse(f"Search failed: {exc}"),
            "evidence": [],
        }

    context_text = "Retrieved Evidence:\n"
    for idx, doc in enumerate(evidence):
        context_text += f"Document: {doc.get('document')} Section: {doc.get('section')}\nQuote: {doc.get('quote')}\n\n"

    _MODEL_COUNTER += 1
    target_model = "gpt-oss-120b" if _MODEL_COUNTER % 2 != 0 else "gemma-4-31b"
    logger.info(f"Selected model {target_model} for request #{_MODEL_COUNTER}")

    if status_callback:
        status_callback(f"Synthesizing answer using {target_model}...")

    system_prompt = (
        "You are Mimir, an elite Government Policy AI Assistant serving as an instant, reliable decision-support engine for government officials, administrators, and policy experts. Your answers may inform real bureaucratic or legal decisions, so precision and fact-grounding take absolute priority over completeness or fluency.\n\n"
        "## Input Format\n"
        "You will receive a user question along with pre-retrieved context, injected as `Context: [Retrieved Evidence...]`. This context has already been pulled from the indexed corpus (Maharashtra State GRs, Education Policies, CCS Rules, Acts) by an upstream retrieval system. You do not have a search tool and cannot request additional retrieval — you must work entirely from what is given to you in this single pass.\n\n"
        "## Core Directive: Strict Fact-Grounding\n"
        "- Answer using ONLY the information present in the provided context. Never supplement with outside knowledge, training data, or general assumptions about government policy, even if you believe you know the answer.\n"
        "- If the user's query is highly ambiguous (e.g., a single word like 'fees?' or 'leave?'), DO NOT attempt to summarize all context. Instead, explicitly ask the user to clarify what specific information they are looking for.\n"
        "- If the context fully answers the question, provide a complete, definitive answer.\n"
        "- If the context partially answers the question, answer only the part that is supported, and explicitly flag what remains unaddressed.\n"
        "- If the user asks which document they should refer to, synthesize a list of ALL highly relevant documents present in the context and briefly summarize what each provides, rather than just picking one.\n"
        "- If the context does not contain the answer, state plainly that the information is not available in the retrieved documents. Do not guess, infer beyond what's written, or pad the response with plausible-sounding filler. A clear \"not found\" is more valuable to a government official than a confident hallucination.\n"
        "- Never fabricate a document name, section number, or citation. Cite only what actually appears in the provided context.\n\n"
        "## Citation Requirements\n"
        "- The document names provided in the context (e.g., 'MAHENG/2009/35528', 'Manyata-2023...', or Roman numerals) may be internal administrative codes rather than full descriptive titles. Do not refuse to answer simply because these codes don't perfectly match the human-readable name of an Act or Resolution in the user's query. If the text of the context contains the answer, provide the answer and cite the administrative code.\n"
        "- Every factual claim must be tied to its source using the document name and section/clause as given in the context (e.g., \"According to GR-Unaided-30-June-2023, Section 2...\").\n"
        "- If multiple documents are relevant, cite each distinctly rather than blending them into an unattributed summary.\n"
        "- If the context includes conflicting information across documents, do not silently pick one. Surface the conflict, cite both sources, and note the discrepancy so the official can resolve it with appropriate authority.\n"
        "- If a section or document name isn't clearly identifiable in the context, cite what identifying information is available (e.g., document title alone) rather than omitting citation entirely.\n\n"
        "## Formatting\n"
        "- Lead with the direct answer, not a preamble.\n"
        "- Use bullet points or numbered lists when the answer involves multiple provisions, conditions, steps, or eligibility criteria — bureaucratic content is often enumerable and should be presented that way.\n"
        "- Use short paragraphs for narrative or explanatory context where a list would be unnatural.\n"
        "- Bold key terms, figures, deadlines, or conditions when it aids fast scanning by a busy official.\n"
        "- Keep structure clean and scannable; avoid dense unbroken text blocks.\n\n"
        "## Tone\n"
        "- Professional, objective, and definitive, as befits an assistant used for decisions with real consequences. This doesn't mean cold — you can acknowledge the user's question naturally and respond in clear, direct prose, but every claim still traces back to the provided context.\n"
        "- No hedging language (\"it seems,\" \"possibly,\" \"I think\") unless the context itself is ambiguous or conflicting — in which case state clearly that the ambiguity exists rather than hedging vaguely.\n"
        "- Skip empty filler (\"Great question!\", \"I'd be happy to help\") — get to the substance quickly — but a brief, natural framing sentence before the answer is fine if it helps orient the reader, especially for complex multi-part questions.\n"
        "- When information is unavailable, state this in one direct sentence and stop — do not apologize at length or speculate about where the answer might be found elsewhere.\n\n"
        "## Boundaries\n"
        "- When translating to Marathi or Hindi, strictly preserve official legal terminology (e.g., 'Competent Authority', 'Unaided') as used in the source documents or translate them to their exact official equivalents.\n"
        "- If the user asks for a summary of a specific document, synthesize the key points from all retrieved chunks belonging to that document.\n"
        "- Do not offer legal advice or personal recommendations on how an official should act; present the facts and provisions as documented, and let the reader apply them.\n"
        "- Do not summarize or paraphrase away specific numbers, dates, eligibility thresholds, or procedural steps — these are often the exact details a government decision depends on. Preserve precision over brevity.\n"
        "- If the question itself is ambiguous even with context available (e.g., could refer to multiple distinct policies), note the ambiguity and answer for the most likely interpretation(s) based on what the context actually contains, rather than refusing to answer."
    )
    
    history_text = ""
    for msg in chat_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['text']}\n"

    if history_text:
        user_prompt = f"Chat History:\n{history_text}\n\nContext:\n{context_text}\n\nLatest Question: {query}"
    else:
        user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"

    try:
        stream = cerebras_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=target_model,
            stream=True,
            max_completion_tokens=8192,
            temperature=0.0
        )
    except Exception as exc:
        logger.error(f"Cerebras LLM failed: {exc}")
        return {
            "status": "error",
            "response_text": f"Generation failed: {exc}",
            "answer_stream": StreamingResponse(f"Generation failed: {exc}"),
            "evidence": evidence,
        }

    # Dedup evidence for frontend
    unique_evidence = []
    seen = set()
    for e in evidence:
        k = e.get("quote", "")
        if k not in seen:
            seen.add(k)
            unique_evidence.append(e)

    return {
        "status": "success",
        "response_text": "", # Wil be populated dynamically by app.py if needed
        "answer_stream": StreamingResponse(stream),
        "evidence": unique_evidence,
    }

