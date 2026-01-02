import time
import random
import openai


def retry_openai(func, *args, max_retries=5, base_delay=1.0, max_delay=30.0, **kwargs):
    """Call `func(*args, **kwargs)` with exponential backoff on common OpenAI transient errors.

    Returns the call result or raises the last exception after retries.
    """
    attempt = 0
    while True:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # handle common transient errors; treat all as retryable here
            attempt += 1
            if attempt > max_retries:
                raise
            # exponential backoff with jitter
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            jitter = random.uniform(0, delay * 0.1)
            sleep_time = delay + jitter
            time.sleep(sleep_time)