"""Draft/rubric LLM helpers (edit analysis, regeneration)."""
from rubric_writer.imports import *
from rubric_writer.config import MODEL_PRIMARY
from rubric_writer.api_client import _api_call_with_retry
from rubric_writer.persistence import get_active_rubric

def _rubric_list_for_json(rubric_list: list):
    """Return a deep copy of the rubric list with _diff removed so it is JSON-serializable (no sets)."""
    cleaned = copy.deepcopy(rubric_list)
    for criterion in cleaned:
        if isinstance(criterion, dict) and "_diff" in criterion:
            del criterion["_diff"]
    return cleaned
def regenerate_selected_text(full_draft: str, selected_sentences: list, instruction: str, rubric_list: list = None) -> dict:
    """
    Call the LLM to regenerate selected sentences in a draft.
    selected_sentences: list of sentence strings to rephrase.
    Returns dict with 'replacements' (list of {original, replacement}) and 'explanation', or 'error'.
    """
    from prompts import INLINE_REGEN_SYSTEM_PROMPT, INLINE_regen_user_prompt

    try:
        user_prompt = INLINE_regen_user_prompt(full_draft, selected_sentences, instruction, rubric_list)
        response = _api_call_with_retry(
            max_tokens=8000,
            system=INLINE_REGEN_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            model=MODEL_PRIMARY,
            thinking={"type": "adaptive"}
        )
        response_text = ""
        for block in response.content:
            if block.type == "text":
                response_text = block.text

        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            result = json.loads(json_match.group())
            if "replacements" in result and isinstance(result["replacements"], list):
                return result
            # Backwards compat: single replacement_text → wrap as list
            if "replacement_text" in result and len(selected_sentences) == 1:
                return {
                    "replacements": [{"original": selected_sentences[0], "replacement": result["replacement_text"]}],
                    "explanation": result.get("explanation", ""),
                }
            return {"error": "Response missing 'replacements' array."}
        return {"error": "Could not parse response from model."}
    except Exception as e:
        return {"error": str(e)}


def generate_edit_feedback_reply(previous_draft: str, edited_draft: str, rubric_list=None, prior_scorecard=None) -> str:
    """Produce a short conversational reply acknowledging a direct user edit and
    asking 1-2 grounded questions. Returns plain prose, or "" on failure."""
    from prompts import DRAFT_EDIT_FEEDBACK_SYSTEM_PROMPT, DRAFT_edit_feedback_prompt

    try:
        user_prompt = DRAFT_edit_feedback_prompt(
            previous_draft or "",
            edited_draft or "",
            _rubric_list_for_json(rubric_list or []),
            prior_scorecard,
        )
        response = _api_call_with_retry(
            max_tokens=600,
            system=DRAFT_EDIT_FEEDBACK_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
            model=MODEL_PRIMARY,
        )
        text = ""
        for block in response.content:
            if block.type == "text":
                text = block.text
        return (text or "").strip()
    except Exception:
        return ""


