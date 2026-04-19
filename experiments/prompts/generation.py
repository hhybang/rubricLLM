"""Draft generation system / user prompts."""

from __future__ import annotations


def generation_system_prompt(formatted_rubric: str, core_preferences: str) -> str:
    cp = (core_preferences or "").strip()
    pref_block = ""
    if cp:
        pref_block = (
            f'The user has stated the following preferences for their writing:\n"{cp}"\n\n'
        )
    return f"""You are a writing assistant. You will write a draft according to the user's request,
following the rubric below as closely as possible.

{pref_block}RUBRIC:
{formatted_rubric}

Write naturally — do not reference the rubric explicitly in your draft.
Do not include meta-commentary about your writing process.
Just write the email/document."""


def generation_user_prompt(task_description: str) -> str:
    return task_description.strip()


def revision_user_prompt(feedback_block: str) -> str:
    return f"""Here is feedback on your draft:

{feedback_block}

Please revise the draft to address this feedback. Write the complete revised draft.
Do not explain your changes — just write the revised version."""
