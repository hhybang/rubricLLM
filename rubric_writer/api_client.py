"""Anthropic API wrapper with retries."""
from rubric_writer.imports import anthropic, random, time
from rubric_writer.config import client

def _api_call_with_retry(max_retries=3, base_delay=2, **kwargs):
    """Wrapper around client.messages.create with exponential backoff retry on overloaded/rate-limit errors."""
    for attempt in range(max_retries + 1):
        try:
            return client.messages.create(**kwargs)
        except anthropic.APIStatusError as e:
            if e.status_code in (429, 529) and attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
                continue
            raise
        except anthropic.APIConnectionError:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
                continue
            raise
