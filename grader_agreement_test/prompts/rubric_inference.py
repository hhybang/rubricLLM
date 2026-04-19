"""
Rubric inference prompts — sourced from the main RubricLLM repo ``prompts.py``.

Adds repo root to ``sys.path`` so ``collect_test_data.py`` can run from this folder.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from prompts import (  # noqa: E402
    RUBRIC_INFER_ONLY_SYSTEM_PROMPT,
    RUBRIC_infer_only_user_prompt,
)

__all__ = [
    "RUBRIC_INFER_ONLY_SYSTEM_PROMPT",
    "RUBRIC_infer_only_user_prompt",
]
