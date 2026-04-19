"""Simulated human correction after rounds 1–2."""

CORRECTION_SYSTEM_TEMPLATE = """You are {persona_name}, {persona_role}. Below are your TRUE writing preferences
that you have not fully shared with anyone:

TRUE PREFERENCES:
{hidden_preferences}

DEALBREAKERS:
{dealbreakers}

You have been shown a rubric that was inferred from your writing interactions, and you've
seen two rounds of drafts graded against this rubric. Your task: identify where the GRADER
is misinterpreting the rubric criteria — cases where the score doesn't match what YOU
actually care about.

CURRENT RUBRIC:
{formatted_rubric}

ROUND 1 DRAFT AND SCORES:
{round_1_block}

ROUND 2 DRAFT AND SCORES:
{round_2_block}

For each criterion where you see a misinterpretation, provide a correction.
Respond in this exact JSON format (no markdown fences):
{{
  "corrections": [
    {{
      "criterion_name": "<name>",
      "issue": "<what the grader is getting wrong>",
      "negative_examples": ["<example of what this criterion does NOT mean>", "..."],
      "clarification": "<what this criterion ACTUALLY means to you>",
      "priority_adjustment": null
    }}
  ]
}}

Use null for priority_adjustment if unchanged, or an integer 1-6 if you want a new priority rank.
If no corrections are needed, return {{"corrections": []}}.
"""


def build_correction_system(
    persona_name: str,
    persona_role: str,
    hidden_preferences: str,
    dealbreakers: str,
    formatted_rubric: str,
    round_1_block: str,
    round_2_block: str,
) -> str:
    return CORRECTION_SYSTEM_TEMPLATE.format(
        persona_name=persona_name,
        persona_role=persona_role,
        hidden_preferences=hidden_preferences,
        dealbreakers=dealbreakers,
        formatted_rubric=formatted_rubric,
        round_1_block=round_1_block,
        round_2_block=round_2_block,
    )
