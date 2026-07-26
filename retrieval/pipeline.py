import os
import json
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
from qdrant_client import QdrantClient

from core.log_config import get_logger
from retrieval.search import search_policy_docs_tool, execute_search_tool

logger = get_logger(__name__)

_QUERY_CACHE: Dict[str, Dict[str, Any]] = {}


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
            text = getattr(chunk, "text", None) or str(chunk)
            if text:
                self.captured_parts.append(text)
                yield text
        self._exhausted = True

    @property
    def full_text(self) -> str:
        if isinstance(self._answer_stream, str):
            return self._answer_stream
        return "".join(self.captured_parts)


def run_retrieval(
    gemini_client: genai.Client,
    qdrant_client: QdrantClient,
    query: str,
    collection_name: str = "gov_docs",
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Execute true agentic retrieval using Gemini Function Calling."""
    if chat_history is None:
        chat_history = []

    cache_key = f"{collection_name}:{query.strip().lower()}"
    if cache_key in _QUERY_CACHE and not chat_history:
        cached_result = _QUERY_CACHE[cache_key]
        if cached_result.get("status") == "success":
            return cached_result
        del _QUERY_CACHE[cache_key]

    system_instruction = (
        "You are an expert Government Policy Assistant. "
        "You must answer user questions accurately based ONLY on the provided government documents. "
        "1. If the user's query is highly ambiguous (e.g. just a single word like 'leave?' or 'fees?'), DO NOT call the search tool. Directly ask the user to clarify which document or topic they are interested in. "
        "2. Otherwise, you must use the `search_policy_docs` tool to retrieve factual evidence before answering. "
        "3. You can set `fast_mode=True` for simple lookups, or `fast_mode=False` for deep analytical searches. If fast mode fails to find relevant information, call the tool again with `fast_mode=False` or different keywords. "
        "4. DO NOT loop endlessly. If you have searched multiple times and cannot find the exact answer, STOP searching and state that the specific information is not available in the indexed documents. "
        "5. Always synthesize your final answer clearly, citing the sections and documents returned by the search tool."
    )

    history_contents = []
    for msg in chat_history:
        role = "user" if msg["role"] == "user" else "model"
        history_contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["text"])]))

    chat = gemini_client.chats.create(
        model=os.getenv("SPEC_MODEL_NAME", "gemini-2.5-flash"),
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[search_policy_docs_tool],
            temperature=0.0,
        ),
        history=history_contents if history_contents else None,
    )

    evidence_collected = []
    
    # 1. Send the initial query to the Agent
    try:
        response = chat.send_message(query)
    except Exception as exc:
        return {
            "status": "error",
            "response_text": f"Agent initiation failed: {exc}",
            "answer_stream": None,
            "evidence": [],
        }

    # 2. Agent Execution Loop
    MAX_ITERATIONS = 3
    for iteration in range(MAX_ITERATIONS):
        if response.function_calls:
            for function_call in response.function_calls:
                if function_call.name == "search_policy_docs":
                    args = function_call.args or {}
                    t_query = args.get("query", query)
                    t_year = args.get("year")
                    t_fast = args.get("fast_mode", False)
                    
                    logger.info(f"Agent executing search_policy_docs(query={t_query}, year={t_year}, fast_mode={t_fast})")
                    tool_result_json, tool_evidence = execute_search_tool(
                        gemini_client, qdrant_client, collection_name,
                        query=t_query, year=t_year, fast_mode=t_fast
                    )
                    
                    evidence_collected.extend(tool_evidence)
                    
                    try:
                        # Send the tool result back to the Agent
                        response = chat.send_message(
                            [types.Part.from_function_response(
                                name="search_policy_docs", 
                                response={"result": json.loads(tool_result_json)}
                            )]
                        )
                    except Exception as exc:
                        return {
                            "status": "error",
                            "response_text": f"Tool feedback failed: {exc}",
                            "answer_stream": None,
                            "evidence": evidence_collected,
                        }
        else:
            # Model generated a final text response, break out of tool loop
            break

    # If the model still returned a function call after MAX_ITERATIONS (infinite loop guard),
    # force it to generate a final text answer based on context so far.
    if response.function_calls:
        try:
            response = chat.send_message(
                "SYSTEM: Maximum search iterations reached. Please synthesize a final answer based on the context retrieved so far. If you cannot find the answer, state that the specific information is not found in the documents. Do NOT call the search tool again."
            )
        except Exception:
            pass

    # Final fallback if it's completely stuck
    final_text = response.text
    if not final_text:
        final_text = "I apologize, but I could not find a definitive answer in the indexed documents after multiple searches."
        status_flag = "not_found"
    else:
        status_flag = "success"

    # Dedup evidence
    unique_evidence = []
    seen = set()
    for e in evidence_collected:
        k = e.get("quote", "")
        if k not in seen:
            seen.add(k)
            unique_evidence.append(e)

    result = {
        "status": status_flag,
        "response_text": final_text,
        "answer_stream": StreamingResponse(final_text),
        "evidence": unique_evidence,
    }
    
    if not chat_history:
        _QUERY_CACHE[cache_key] = result
        
    return result
