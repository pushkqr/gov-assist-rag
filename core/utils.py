import os
import time
import json
import requests
from functools import wraps

from google import genai
from cerebras.cloud.sdk import Cerebras
from core.log_config import get_logger

logger = get_logger(__name__)

_aistudio_client = None


def get_genai_client() -> genai.Client:
    """Initialize genai.Client configured for GCP Vertex AI using ADC or AI Studio."""
    use_vertex = os.getenv("USE_VERTEX_AI", "True").strip().lower() in ("true", "1", "yes")
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "asia-south1")

    if use_vertex:
        logger.info(f"Initializing GenAI Client via Vertex AI (Project: {project}, Location: {location})...")
        return genai.Client(
            vertexai=True,
            project=project,
            location=location,
        )

    api_key = os.getenv("GEMINI_API_KEY")
    return genai.Client(api_key=api_key) if api_key else genai.Client()


def get_cerebras_client() -> Cerebras:
    """Initialize Cerebras client."""
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        logger.warning("CEREBRAS_API_KEY not found in environment")
    return Cerebras(api_key=api_key)


def get_weaviate_client():
    """Initialize Weaviate client, connecting to a remote droplet if configured."""
    import weaviate
    from weaviate.classes.init import Auth
    url = os.getenv("WEAVIATE_URL")
    if url:
        grpc_port = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
        host = url.replace("http://", "").replace("https://", "").split(":")[0].strip("/")
        api_key = os.getenv("WEAVIATE_API_KEY")
        auth = Auth.api_key(api_key) if api_key else None
        
        logger.info(f"Connecting to remote Weaviate at {host}")
        return weaviate.connect_to_custom(
            http_host=host,
            http_port=8080,
            http_secure=False,
            grpc_host=host,
            grpc_port=grpc_port,
            grpc_secure=False,
            auth_credentials=auth
        )
    else:
        logger.info("Connecting to local Weaviate")
        return weaviate.connect_to_local()


def get_aistudio_client() -> genai.Client:
    """Initialize dedicated AI Studio genai.Client using GEMINI_API_KEY for embedding calls."""
    global _aistudio_client
    if _aistudio_client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            logger.info("Initializing dedicated AI Studio Client for Embeddings (via GEMINI_API_KEY)...")
            _aistudio_client = genai.Client(api_key=api_key)
        else:
            _aistudio_client = get_genai_client()
    return _aistudio_client


def with_retry_and_throttle(constant_delay_env=None, default_delay=0, max_retries=5, initial_backoff=5, backoff_factor=2):
    """
    Decorator that applies a configurable delay before execution, and
    implements exponential backoff if a rate limit or quota error occurs.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay_val = default_delay
            if constant_delay_env:
                delay_val = float(os.getenv(constant_delay_env, str(default_delay)))
            if delay_val > 0:
                time.sleep(delay_val)

            backoff_delay = initial_backoff
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_msg = str(e).lower()
                    is_transient = any(
                        err in error_msg
                        for err in [
                            "429",
                            "500",
                            "502",
                            "503",
                            "504",
                            "quota",
                            "exhausted",
                            "remoteprotocolerror",
                            "connecterror",
                            "connection",
                            "disconnect",
                            "unreachable",
                            "socket",
                            "winerror",
                            "host",
                            "reset",
                            "timeout",
                        ]
                    )
                    if is_transient:
                        if attempt == max_retries - 1:
                            logger.warning(f"[Retry] Max retries ({max_retries}) reached. Failing.")
                            raise e
                        logger.warning(f"[Retry] Transient network/rate-limit error ({e}). Retrying in {backoff_delay} seconds...")
                        time.sleep(backoff_delay)
                        backoff_delay *= backoff_factor
                    else:
                        raise e
        return wrapper
    return decorator


@with_retry_and_throttle(constant_delay_env="GEN_API_DELAY", default_delay=0)
def generate_content_safe(client, *args, **kwargs):
    """Wrapper for client.models.generate_content with rate limiting."""
    return client.models.generate_content(*args, **kwargs)


class MockEmbedding:
    def __init__(self, values):
        self.values = values

class MockEmbedResponse:
    def __init__(self, values):
        self.embeddings = [MockEmbedding(values)]


@with_retry_and_throttle(constant_delay_env="EMBED_API_DELAY", default_delay=0)
def embed_content_safe(client, *args, **kwargs):
    """Wrapper for client.models.embed_content routed to AI Studio, or a local server if LOCAL_EMBED_URL is set."""
    local_url = os.getenv("LOCAL_EMBED_URL")
    if local_url:
        api_key = os.getenv("LOCAL_EMBED_API_KEY", "")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        text = kwargs.get("contents")
        model_name = os.getenv("LOCAL_EMBED_MODEL_NAME") or kwargs.get("model") or "BAAI/bge-m3"
        if model_name.startswith(("gemini-", "text-embedding")):
            model_name = "BAAI/bge-m3"
        payload = {
            "input": text,
            "model": model_name,
        }
        timeout = float(os.getenv("LOCAL_EMBED_TIMEOUT_S", "10"))
        resp = requests.post(local_url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        vector = data["data"][0]["embedding"]
        return MockEmbedResponse(vector)

    use_aistudio_embed = os.getenv("USE_AISTUDIO_FOR_EMBEDDINGS", "False").strip().lower() in ("true", "1", "yes")
    if use_aistudio_embed and os.getenv("GEMINI_API_KEY"):
        embed_client = get_aistudio_client()
        return embed_client.models.embed_content(*args, **kwargs)
    return client.models.embed_content(*args, **kwargs)


def local_rerank_safe(query: str, texts: list[str], top_n: int = 35) -> list[int]:
    """Hits the local Infinity reranker on the Droplet and returns sorted indices."""
    local_url = os.getenv("LOCAL_RERANK_URL")
    if not local_url:
        # Fallback to returning original indices if not configured
        return list(range(min(top_n, len(texts))))
        
    api_key = os.getenv("LOCAL_RERANK_API_KEY", "")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        
    payload = {
        "query": query,
        "documents": texts,
        "model": os.getenv("LOCAL_RERANK_MODEL_NAME", "BAAI/bge-reranker-v2-m3"),
        "top_n": top_n,
        "return_documents": False
    }
    
    try:
        timeout = float(os.getenv("LOCAL_RERANK_TIMEOUT_S", "3"))
        resp = requests.post(local_url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        
        # Infinity /rerank returns {"results": [{"index": 5, "relevance_score": 0.9}, ...]}
        results = data.get("results", [])
        return [item["index"] for item in results]
    except Exception as e:
        logger.error(f"Local Reranking failed: {e}")
        return list(range(min(top_n, len(texts))))


class _TextChunk:
    """Minimal stand-in exposing .text, the shape StreamingResponse already unwraps."""

    __slots__ = ("text",)

    def __init__(self, text: str):
        self.text = text


def local_generate_stream(system_prompt: str, user_prompt: str, timeout: float = None):
    """Stream a completion from a self-hosted, OpenAI-compatible inference server.

    Deliberately speaks the /v1/chat/completions dialect over plain requests rather than
    Ollama's native API: the same code then runs against Ollama, vLLM, llama.cpp or LM
    Studio, so an on-premise deployment is a base URL change and not a code change.

    Yields chunks lazily. The caller sees the first token as soon as the server emits it,
    which matters because a CPU-only department server is slow enough that waiting for a
    complete response would look like a hang.
    """
    base_url = os.getenv("LOCAL_GEN_URL", "http://localhost:11434/v1").rstrip("/")
    model = os.getenv("LOCAL_GEN_MODEL", "qwen3:4b")
    api_key = os.getenv("LOCAL_GEN_API_KEY", "")
    timeout = timeout if timeout is not None else float(os.getenv("LOCAL_GEN_TIMEOUT_S", "120"))

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": True,
        "temperature": 0.0,
    }

    logger.info(f"Generating via self-hosted model {model} at {base_url}")
    response = requests.post(
        f"{base_url}/chat/completions", json=payload, headers=headers,
        stream=True, timeout=timeout,
    )
    response.raise_for_status()

    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        if raw_line.startswith("data:"):
            raw_line = raw_line[5:].strip()
        if not raw_line or raw_line == "[DONE]":
            continue
        try:
            chunk = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices") or []
        if not choices:
            continue
        text = (choices[0].get("delta") or {}).get("content")
        if text:
            yield _TextChunk(text)


@with_retry_and_throttle(constant_delay_env="CEREBRAS_API_DELAY", default_delay=0, initial_backoff=10)
def cerebras_chat_completions_create_safe(client, *args, **kwargs):
    """Wrapper for cerebras_client.chat.completions.create with rate limiting for TPM limits."""
    return client.chat.completions.create(*args, **kwargs)


@with_retry_and_throttle(constant_delay_env="GEN_API_DELAY", default_delay=0)
def generate_content_stream_safe(client, *args, **kwargs):
    """Wrapper for client.models.generate_content_stream with rate limiting."""
    return client.models.generate_content_stream(*args, **kwargs)
