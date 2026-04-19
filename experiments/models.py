"""Anthropic API wrapper with retries and token logging."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import anthropic

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import RETRY_ATTEMPTS, RETRY_DELAY_SECONDS

_log = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def _cost_log_path() -> Path:
    return _ROOT / "results" / "cost_log.jsonl"


def log_cost_line(entry: dict[str, Any]) -> None:
    path = _cost_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def complete(
    *,
    system: str,
    messages: list[dict[str, Any]],
    model: str,
    temperature: float,
    max_tokens: int,
    label: str = "",
) -> tuple[str, dict[str, int]]:
    """
    Returns (assistant_text, {"input_tokens": int, "output_tokens": int}).
    """
    client = get_client()
    last_err: Exception | None = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=messages,
            )
            text = "".join(
                b.text for b in resp.content if getattr(b, "type", None) == "text"
            )
            usage = resp.usage
            meta = {
                "input_tokens": getattr(usage, "input_tokens", 0) or 0,
                "output_tokens": getattr(usage, "output_tokens", 0) or 0,
            }
            log_cost_line(
                {
                    "label": label,
                    "model": model,
                    **meta,
                }
            )
            return text, meta
        except Exception as e:
            last_err = e
            _log.warning("complete attempt %s failed: %s", attempt, e)
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY_SECONDS)
    raise last_err  # type: ignore[misc]
