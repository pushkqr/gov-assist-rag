import time
from functools import wraps

def with_retry_and_throttle(constant_delay=1.0, max_retries=5, initial_backoff=5, backoff_factor=2):
    """
    Decorator that applies a constant delay before execution, and 
    implements exponential backoff if a rate limit or quota error occurs.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Apply constant baseline delay
            if constant_delay > 0:
                time.sleep(constant_delay)
                
            delay = initial_backoff
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_msg = str(e).lower()
                    if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
                        if attempt == max_retries - 1:
                            print(f"[RateLimit] Max retries ({max_retries}) reached. Failing.")
                            raise e
                        print(f"[RateLimit] Hit API quota/rate-limit (Attempt {attempt+1}/{max_retries}). Retrying in {delay} seconds...")
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        # If it's not a rate limit error, raise it immediately
                        raise e
        return wrapper
    return decorator

@with_retry_and_throttle(constant_delay=2.0)
def generate_content_safe(client, *args, **kwargs):
    """Wrapper for client.models.generate_content with rate limiting."""
    return client.models.generate_content(*args, **kwargs)

@with_retry_and_throttle(constant_delay=0.5)
def embed_content_safe(client, *args, **kwargs):
    """Wrapper for client.models.embed_content with rate limiting."""
    return client.models.embed_content(*args, **kwargs)

@with_retry_and_throttle(constant_delay=2.0)
def generate_content_stream_safe(client, *args, **kwargs):
    """Wrapper for client.models.generate_content_stream with rate limiting."""
    return client.models.generate_content_stream(*args, **kwargs)
