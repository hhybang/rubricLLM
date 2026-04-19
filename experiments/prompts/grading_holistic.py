"""Holistic 1–10 + paragraph grading."""

HOLISTIC_SYSTEM = """You are an expert writing evaluator. Read the rubric and the draft below.
Provide:
1. A single overall score from 1-10
2. A paragraph of feedback (3-5 sentences) explaining what works and what doesn't

Do NOT break your feedback into per-criterion scores. Give one holistic assessment.

Respond in this exact JSON format (no markdown fences):
{
  "score": <int 1-10>,
  "feedback": "<paragraph>"
}
"""


def holistic_user_prompt(formatted_rubric: str, draft: str) -> str:
    return f"""RUBRIC:
{formatted_rubric}

DRAFT TO EVALUATE:
{draft}
"""
