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
        payload = {
            "input": text,
            "model": kwargs.get("model", "BAAI/bge-m3")
        }
        resp = requests.post(local_url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        vector = data["data"][0]["embedding"]
        return MockEmbedResponse(vector)

    use_aistudio_embed = os.getenv("USE_AISTUDIO_FOR_EMBEDDINGS", "False").strip().lower() in ("true", "1", "yes")
    if use_aistudio_embed and os.getenv("GEMINI_API_KEY"):
        embed_client = get_aistudio_client()
        return embed_client.models.embed_content(*args, **kwargs)
    return client.models.embed_content(*args, **kwargs)


@with_retry_and_throttle(constant_delay_env="CEREBRAS_API_DELAY", default_delay=0, initial_backoff=10)
def cerebras_chat_completions_create_safe(client, *args, **kwargs):
    """Wrapper for cerebras_client.chat.completions.create with rate limiting for TPM limits."""
    return client.chat.completions.create(*args, **kwargs)


@with_retry_and_throttle(constant_delay_env="GEN_API_DELAY", default_delay=0)
def generate_content_stream_safe(client, *args, **kwargs):
    """Wrapper for client.models.generate_content_stream with rate limiting."""
    return client.models.generate_content_stream(*args, **kwargs)
