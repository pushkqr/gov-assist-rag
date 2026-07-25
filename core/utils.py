import os
import time
from functools import wraps

from core.log_config import get_logger

logger = get_logger(__name__)


def with_retry_and_throttle(constant_delay_env=None, default_delay=0, max_retries=5, initial_backoff=5, backoff_factor=2):
    """
    Decorator that applies a configurable delay before execution, and
    implements exponential backoff if a rate limit or quota error occurs.

    The delay is read from an environment variable at call time. If the env
    var is not set, `default_delay` is used. Set to 0 for no delay.
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
    """Wrapper for client.models.embed_content with rate limiting."""
    return client.models.embed_content(*args, **kwargs)

@with_retry_and_throttle(constant_delay_env="GEN_API_DELAY", default_delay=0)
def generate_content_stream_safe(client, *args, **kwargs):
    """Wrapper for client.models.generate_content_stream with rate limiting."""
    return client.models.generate_content_stream(*args, **kwargs)
