"""
Per-criterion grading (1–10 + justification), aligned with RubricLLM dimension-based ideas:
check dimensions as MET/NOT_MET mentally, then assign one 1–10 score per criterion.
"""

CRITERION_SYSTEM = """You are grading a writing draft against a personalized rubric. The rubric
represents one user's preferences — not generic standards of "good writing."

For EACH criterion in the rubric:
1. Read the criterion description and every dimension (checkable item).
2. For each dimension, decide if the draft clearly satisfies it (yes) or not (no), using the same
   strictness as RubricLLM: only "yes" if the draft unambiguously satisfies the dimension; if
   partial or ambiguous, count as no.
3. Assign ONE integer score from 1-10 for that criterion reflecting how well the draft meets that
   criterion overall (dimensions met, priority of failures, and severity).
4. Write a short justification citing concrete aspects of the draft.

Respond with ONLY valid JSON — no markdown fences, no preamble:
{
  "criteria": [
    {
      "criterion_name": "<exact criterion title — the short name on the ### line only, no priority/category suffix>",
      "score": <int 1-10>,
      "justification": "<2-4 sentences with specific evidence>"
    }
  ]
}

Include ALL criteria from the rubric, ordered by priority (1 first).
"""


def criterion_user_prompt(formatted_rubric: str, draft: str) -> str:
    return f"""RUBRIC:
{formatted_rubric}

DRAFT TO EVALUATE:
{draft}
"""
