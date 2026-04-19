"""Ground-truth alignment vs hidden preferences (Opus)."""

GROUND_TRUTH_SYSTEM = """You are evaluating whether a piece of writing matches a specific person's
preferences. You have access to their TRUE preferences (including things they wouldn't
say upfront). Score how well the draft satisfies these preferences.

Respond in JSON only (no markdown fences):
{
  "overall_alignment": <int 1-10>,
  "dealbreaker_violations": ["<violation>", ...],
  "hidden_pref_satisfaction": [
    {
      "preference": "<short summary of one hidden preference theme>",
      "satisfied": "<yes|partially|no>",
      "evidence": "<quote or explanation>"
    }
  ]
}

Produce one hidden_pref_satisfaction entry per major theme in TRUE PREFERENCES (roughly 3-8 entries).
"""


def ground_truth_user_prompt(
    hidden_preferences: str,
    core_preferences: str,
    dealbreakers: str,
    draft: str,
) -> str:
    return f"""TRUE PREFERENCES:
{hidden_preferences}

CORE PREFERENCES (what they say upfront):
{core_preferences}

DEALBREAKERS:
{dealbreakers}

DRAFT TO EVALUATE:
{draft}
"""
