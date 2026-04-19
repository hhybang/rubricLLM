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
def update_rubric_from_draft_edit(original_draft: str, edited_draft: str):
    """
    Call the LLM to analyze draft edits and suggest rubric updates.
    Shows analysis in a chat-like format for better readability.
    """
    # Get the active rubric
    active_rubric_dict, _, _ = get_active_rubric()

    if not active_rubric_dict:
        st.warning("No active rubric to update. Please create or select a rubric first.")
        return

    active_rubric_list = active_rubric_dict.get("rubric", [])

    if not active_rubric_list:
        st.warning("Active rubric has no criteria to update.")
        return

    with st.spinner("Analyzing your edits to suggest rubric updates..."):
        try:
            client = anthropic.Anthropic()

            # Build the prompt
            user_prompt = DRAFT_revise_after_rubric_change_prompt(
                active_rubric_list,
                original_draft,
                edited_draft
            )

            # Make API call
            response = _api_call_with_retry(
                max_tokens=16000,
                system=DRAFT_REVISE_AFTER_RUBRIC_CHANGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                model=MODEL_PRIMARY,
                thinking={"type": "adaptive"}
            )

            # Extract thinking and text from response
            thinking_text = ""
            response_text = ""
            for block in response.content:
                if block.type == "thinking":
                    thinking_text = block.thinking
                elif block.type == "text":
                    response_text = block.text

            # Parse the JSON response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
                # Include thinking in the result
                result['thinking'] = thinking_text

                # Store result in session state to display outside the button callback
                st.session_state.rubric_update_result = result
                st.rerun()

            else:
                st.error("Could not parse rubric update suggestions. Please try again.")

        except json.JSONDecodeError as e:
            st.error(f"Error parsing response: {str(e)}")
        except Exception as e:
            st.error(f"Error analyzing draft edits: {str(e)}")
def regenerate_draft_from_rubric_changes(original_rubric: list, updated_rubric: list, current_draft: str, conversation_history: str = None, rubric_suggestion_text: str = None, user_edit_feedback: str = None):
    """
    Call the LLM to regenerate the draft based on rubric changes.
    Returns dict with revised_draft, etc. on success, or dict with only 'error' key on failure.
    """
    from prompts import DRAFT_REGENERATE_SYSTEM_PROMPT, DRAFT_regenerate_prompt

    with st.spinner("Regenerating draft based on edit feedback..."):
        try:
            client = anthropic.Anthropic()
            original_clean = _rubric_list_for_json(original_rubric)
            updated_clean = _rubric_list_for_json(updated_rubric)
            user_prompt = DRAFT_regenerate_prompt(original_clean, updated_clean, current_draft, conversation_history, rubric_suggestion_text, user_edit_feedback)

            response = _api_call_with_retry(
                max_tokens=16000,
                system=DRAFT_REGENERATE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                model=MODEL_PRIMARY,
                thinking={"type": "adaptive"}
            )

            thinking_text = ""
            response_text = ""
            for block in response.content:
                if block.type == "thinking":
                    thinking_text = block.thinking
                elif block.type == "text":
                    response_text = block.text

            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
                result['thinking'] = thinking_text
                return result
            return {"error": "Could not parse regenerated draft from the model response. Please try again."}

        except json.JSONDecodeError as e:
            return {"error": f"Error parsing response: {str(e)}"}
        except Exception as e:
            return {"error": str(e)}


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


def get_last_draft_from_messages():
    """
    Find the last message with a <draft></draft> block and return the draft content.
    Checks both content and display_content. Returns (draft_content, message_index) or (None, None).
    """
    pattern = r'<draft>(.*?)</draft>'

    for idx in range(len(st.session_state.messages) - 1, -1, -1):
        msg = st.session_state.messages[idx]
        if msg.get('role') != 'assistant':
            continue
        for field in ('content', 'display_content'):
            text = msg.get(field, '') or ''
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip(), idx
    return None, None
