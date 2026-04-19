"""Shared Anthropic calls: backoff, JSON extraction, token usage."""

from __future__ import annotations

import json
import random
import re
import time
from typing import Any

from anthropic import Anthropic, APIStatusError, RateLimitError

DEFAULT_MAX_RETRIES = 8


def backoff_sleep(attempt: int, base: float = 2.0, cap: float = 120.0) -> None:
    delay = min(cap, base * (2**attempt) + random.uniform(0, 1))
    time.sleep(delay)


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def call_anthropic(
    client: Anthropic,
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float = 0.0,
    thinking: dict[str, Any] | None = None,
) -> tuple[str, int | None, int | None]:
    """
    Returns (text, input_tokens, output_tokens).
    Retries on rate limits / overloaded with exponential backoff.
    """
    last_err: Exception | None = None
    for attempt in range(DEFAULT_MAX_RETRIES):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            if thinking:
                kwargs["thinking"] = thinking
                # Extended-thinking runs can exceed the non-streaming client timeout (~10 min).
                with client.messages.stream(**kwargs) as stream:
                    message = stream.get_final_message()
                parts = [
                    block.text
                    for block in message.content
                    if getattr(block, "type", None) == "text" and hasattr(block, "text")
                ]
                text = "".join(parts)
                in_tok = getattr(message.usage, "input_tokens", None) if message.usage else None
                out_tok = getattr(message.usage, "output_tokens", None) if message.usage else None
                return text, in_tok, out_tok
            try:
                resp = client.messages.create(**kwargs)
            except ValueError as ve:
                if "streaming" in str(ve).lower() and "10 minute" in str(ve).lower():
                    with client.messages.stream(**kwargs) as stream:
                        message = stream.get_final_message()
                    parts = [
                        block.text
                        for block in message.content
                        if getattr(block, "type", None) == "text" and hasattr(block, "text")
                    ]
                    text = "".join(parts)
                    in_tok = getattr(message.usage, "input_tokens", None) if message.usage else None
                    out_tok = getattr(message.usage, "output_tokens", None) if message.usage else None
                    return text, in_tok, out_tok
                raise
            parts = [
                block.text
                for block in resp.content
                if getattr(block, "type", None) == "text" and hasattr(block, "text")
            ]
            text = "".join(parts)
            in_tok = getattr(resp.usage, "input_tokens", None) if resp.usage else None
            out_tok = getattr(resp.usage, "output_tokens", None) if resp.usage else None
            return text, in_tok, out_tok
        except (RateLimitError, APIStatusError) as e:
            last_err = e
            code = getattr(e, "status_code", None)
            if code == 429 or (isinstance(e, APIStatusError) and code and code >= 500):
                backoff_sleep(attempt)
                continue
            raise
        except Exception as e:
            err_s = str(e).lower()
            if "overloaded" in err_s or "rate_limit" in err_s or "429" in err_s:
                last_err = e
                backoff_sleep(attempt)
                continue
            raise
    assert last_err is not None
    raise last_err
