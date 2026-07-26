import os
import time
from functools import wraps

from google import genai
from core.log_config import get_logger

logger = get_logger(__name__)

_aistudio_client = None


def get_genai_client() -> genai.Client:
    """Initialize genai.Client configured for GCP Vertex AI using ADC or AI Studio."""
    use_vertex = os.getenv("USE_VERTEX_AI", "True").strip().lower() in ("true", "1", "yes")
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "project-a69d4df0-3f7f-4e30-a8a")
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
                    if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
                        if attempt == max_retries - 1:
                            logger.warning(f"[RateLimit] Max retries ({max_retries}) reached. Failing.")
                            raise e
                        logger.warning(f"[RateLimit] Hit API quota/rate-limit (Attempt {attempt+1}/{max_retries}). Retrying in {backoff_delay} seconds...")
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


@with_retry_and_throttle(constant_delay_env="EMBED_API_DELAY", default_delay=0)
def embed_content_safe(client, *args, **kwargs):
    """Wrapper for client.models.embed_content routed to AI Studio if GEMINI_API_KEY is available."""
    use_aistudio_embed = os.getenv("USE_AISTUDIO_FOR_EMBEDDINGS", "True").strip().lower() in ("true", "1", "yes")
    if use_aistudio_embed and os.getenv("GEMINI_API_KEY"):
        embed_client = get_aistudio_client()
        return embed_client.models.embed_content(*args, **kwargs)
    return client.models.embed_content(*args, **kwargs)


@with_retry_and_throttle(constant_delay_env="GEN_API_DELAY", default_delay=0)
def generate_content_stream_safe(client, *args, **kwargs):
    """Wrapper for client.models.generate_content_stream with rate limiting."""
    return client.models.generate_content_stream(*args, **kwargs)
