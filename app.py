import streamlit as st
import os
import anthropic
import html as html_module
from textwrap import dedent
import json
import re
import time
import uuid
import copy
from datetime import datetime
from pathlib import Path
from auth_supabase import (
    get_supabase_client, init_auth_state, register_user, login_user,
    logout_user, get_current_user, is_authenticated,
    get_user_projects, create_project as db_create_project, delete_project,
    save_conversation, load_conversations as db_load_conversations,
    load_conversation_by_id, save_rubric_history as db_save_rubric_history,
    load_rubric_history as db_load_rubric_history, save_project_data,
    load_project_data, get_schema_sql, delete_rubric_version
)
from prompts import (
    RUBRIC_COMPARE_DRAFTS_PROMPT,
    CHAT_build_system_prompt,
    DRAFT_REVISE_AFTER_RUBRIC_CHANGE_SYSTEM_PROMPT,
    DRAFT_revise_after_rubric_change_prompt,
    RUBRIC_compare_to_coldstart_prompt,
    GRADING_generate_draft_from_rubric_prompt,
    GRADING_judge_per_dimension_prompt,
    GRADING_generate_degraded_draft_prompt,
    GRADING_rubric_judge_prompt,
    GRADING_generic_judge_prompt,
    RUBRIC_suggest_changes_from_feedback_prompt,
    RUBRIC_apply_suggestion_prompt,
    GRADING_generate_writing_task_prompt,
    GRADING_generate_draft_from_coldstart_prompt,
    GRADING_generate_draft_generic_prompt,
    GRADING_unified_eval_prompt,
    RUBRIC_INFER_ONLY_SYSTEM_PROMPT,
    RUBRIC_infer_only_user_prompt,
    RUBRIC_EXTRACT_DPS_SYSTEM_PROMPT,
    RUBRIC_extract_dps_user_prompt,
    RUBRIC_FINAL_INFER_SYSTEM_PROMPT,
    RUBRIC_final_infer_user_prompt,
    GRADING_pairwise_judge_prompt,
)
from scipy.stats import kendalltau
import random

# ── Model configuration ──────────────────────────────────────────────────────
# Change these to switch all API calls at once.
MODEL_PRIMARY = "claude-opus-4-6"    # Main model for chat, rubric inference, grading, etc.
MODEL_LIGHT   = "claude-sonnet-4-5"      # Lighter model for suggestions, small helper calls
AB_DRAFT_INTERVAL = 1                    # Every N-th draft triggers A/B comparison (set to 1 for every draft)

def display_rubric_criteria(rubric_data, container, comparison_rubric_data=None):
    """
    Display rubric criteria in a user-friendly format with headings, descriptions,
    priority icons, and expandable evidence sections. Criteria are grouped by category.

    If comparison_rubric_data is provided, highlights criteria that differ with ‼️ emoji
    and bolds the specific elements that are different (name, description, priority, dimensions).
    """
    if not rubric_data or 'rubric' not in rubric_data:
        container.warning("No rubric data available")
        return

    rubric_list = rubric_data.get('rubric', [])

    if not rubric_list:
        container.info("No criteria defined")
        return

    # Build comparison map for efficient lookup
    comparison_map = {}
    if comparison_rubric_data and 'rubric' in comparison_rubric_data:
        for c in comparison_rubric_data['rubric']:
            name = c.get('name', '').lower().strip()
            comparison_map[name] = c

    # Helper function to get dimension labels as a set for comparison
    def get_dimension_labels(dims):
        return set(d.get('label', '').strip() for d in dims if d.get('label', '').strip())

    # Group criteria by category
    from collections import defaultdict
    categories = defaultdict(list)
    for criterion in rubric_list:
        category = criterion.get('category', 'Uncategorized')

        # Track specific differences for comparison
        criterion['_diff'] = {
            'is_new_criterion': False,
            'name_changed': False,
            'description_changed': False,
            'priority_changed': False,
            'dimensions_changed': False,
            'added_dimensions': set(),
            'removed_dimensions': set()
        }

        if comparison_rubric_data:
            criterion_name = criterion.get('name', '').lower().strip()

            if criterion_name not in comparison_map:
                # Completely new criterion
                criterion['_diff']['is_new_criterion'] = True
            else:
                old_criterion = comparison_map[criterion_name]

                # Check name (case-sensitive comparison for display purposes)
                if criterion.get('name', '') != old_criterion.get('name', ''):
                    criterion['_diff']['name_changed'] = True

                # Check description
                if criterion.get('description', '') != old_criterion.get('description', ''):
                    criterion['_diff']['description_changed'] = True

                # Check priority
                old_priority = old_criterion.get('priority', old_criterion.get('weight', 0))
                new_priority = criterion.get('priority', criterion.get('weight', 0))
                if old_priority != new_priority:
                    criterion['_diff']['priority_changed'] = True

                # Check dimensions
                old_dims = get_dimension_labels(old_criterion.get('dimensions', []))
                new_dims = get_dimension_labels(criterion.get('dimensions', []))
                if old_dims != new_dims:
                    criterion['_diff']['dimensions_changed'] = True
                    criterion['_diff']['added_dimensions'] = new_dims - old_dims
                    criterion['_diff']['removed_dimensions'] = old_dims - new_dims

        categories[category].append(criterion)

    # Display each category group
    for category_name, criteria in categories.items():
        # Start the category box with HTML
        container.markdown(f"""
            <div style="border: 1px solid #2196F3; border-radius: 6px; padding: 8px; margin-bottom: 16px; background-color: rgba(33, 150, 243, 0.1);">
                <div style="font-size: 18px; font-weight: 600; margin-bottom: 8px; color: #2196F3; text-transform: capitalize; text-align: center;">{category_name}</div>
        """, unsafe_allow_html=True)

        # Display criteria within this category
        for criterion in criteria:
            diff = criterion.get('_diff', {})

            # Check if any difference exists
            has_any_diff = (
                diff.get('is_new_criterion') or
                diff.get('name_changed') or
                diff.get('description_changed') or
                diff.get('priority_changed') or
                diff.get('dimensions_changed')
            )

            # Build criterion label
            criterion_name = criterion.get('name', 'Unnamed')
            if has_any_diff:
                criterion_label = f"‼️  {criterion_name}"
            else:
                criterion_label = criterion_name

            with container.expander(criterion_label, expanded=False):
                # Description - bold if changed
                description = criterion.get('description', 'No description provided')
                if diff.get('description_changed'):
                    st.markdown(f"**{description}**")
                else:
                    st.markdown(description)

                # Priority display - bold if changed
                priority = criterion.get('priority', criterion.get('weight', 0))
                if priority > 0:
                    if diff.get('priority_changed'):
                        st.markdown(f"***Priority: #{priority}***")
                    else:
                        st.markdown(f"*Priority: #{priority}*")
                    st.markdown("")

                # Dimensions section (expandable)
                dimensions = criterion.get('dimensions', [])
                added_dims = diff.get('added_dimensions', set())

                if dimensions:
                    dims_label = f"📊 Dimensions ({len(dimensions)})"
                    if diff.get('dimensions_changed'):
                        dims_label = f"**📊 Dimensions ({len(dimensions)})**"

                    with st.expander(dims_label, expanded=False):
                        for dim in dimensions:
                            dim_label = dim.get('label', 'Unnamed dimension')
                            # Bold dimensions that are new/added
                            if dim_label.strip() in added_dims:
                                st.markdown(f"• **{dim_label}**")
                            else:
                                st.markdown(f"• {dim_label}")
                else:
                    st.caption("No dimensions defined")

        # Close the category box
        container.markdown("</div>", unsafe_allow_html=True)

def _md_diff_to_html(marked_text):
    """
    Convert diff formatting to HTML with colors, processing markdown inline:
    +text+ → green addition span
    ~text~ → red deletion span
    Converts markdown **bold**, *italic*, etc to HTML.
    """
    if not marked_text:
        return ""
    
    # Process paragraphs separately
    paragraphs = marked_text.split("\n\n")
    
    result = "<div class='diff-container'>"
    
    for para in paragraphs:
        if not para.strip():
            continue
        
        # First escape HTML special characters
        para = (para
               .replace("&", "&amp;")
               .replace("<", "&lt;")
               .replace(">", "&gt;"))
        
        # Process markdown formatting **bold**, *italic*
        para = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', para)
        para = re.sub(r'\*(.+?)\*', r'<em>\1</em>', para)
        
        # Then process diff markers
        # Process +text+ for additions (green)
        para = re.sub(r"\+(.+?)\+", r'<span class="diff-add">\1</span>', para)
        
        # Process ~text~ for deletions (red)  
        para = re.sub(r"~([^~]+)~", r'<span class="diff-del">\1</span>', para)
        
        result += f"<p>{para}</p>"
    
    result += "</div>"
    return result

def parse_draft_content(content: str):
    """
    Parse content to extract draft sections wrapped in <draft></draft> tags.
    Returns a list of tuples: (text_before, draft_content, text_after) for each draft found,
    or None if no drafts are found.
    """
    pattern = r'<draft>(.*?)</draft>'
    matches = list(re.finditer(pattern, content, re.DOTALL))

    if not matches:
        return None

    parts = []
    last_end = 0

    for match in matches:
        text_before = content[last_end:match.start()]
        draft_content = match.group(1).strip()
        parts.append({
            'before': text_before,
            'draft': draft_content,
            'start': match.start(),
            'end': match.end()
        })
        last_end = match.end()

    # Add any remaining text after the last draft
    if last_end < len(content):
        parts.append({
            'after': content[last_end:]
        })

    return parts

def strip_draft_tags_for_streaming(content: str) -> str:
    """
    Strip <draft> tags from content for display during streaming.
    Replaces <draft>content</draft> with just the content inside the tags.
    This ensures streaming display matches what's shown in editable text areas after rerun.
    """
    # Replace complete <draft>...</draft> with just the inner content
    pattern = r'<draft>(.*?)</draft>'
    result = re.sub(pattern, r'\1', content, flags=re.DOTALL)

    # Handle incomplete/partial draft tags during streaming
    # If there's an opening <draft> without closing, show content up to the tag
    if '<draft>' in result and '</draft>' not in result:
        # Keep partial draft content visible during streaming
        result = result.replace('<draft>', '')

    return result

def _safe_annotated_draft_html(annotated_text: str, annotated_changes: list, message_id: str = "") -> str:
    """
    Turn revised_draft_annotated into safe HTML: allow <ins>/<del>, replace [1],[2] with
    clickable markers (link to edit below). annotated_changes is used for [N] tooltips and anchor ids.
    """
    import html as html_lib
    if not annotated_text:
        return ""
    escaped = html_lib.escape(annotated_text)
    escaped = escaped.replace("&lt;ins&gt;", "<ins>").replace("&lt;/ins&gt;", "</ins>")
    escaped = escaped.replace("&lt;del&gt;", "<del>").replace("&lt;/del&gt;", "</del>")
    safe_msg_id = re.sub(r'[^a-zA-Z0-9_-]', '_', str(message_id))
    for i in range(1, len(annotated_changes) + 1):
        reason = annotated_changes[i - 1].get("reason", "") if i <= len(annotated_changes) else ""
        reason_esc = html_lib.escape(reason)
        marker = f"[{i}]"
        edit_anchor_id = f"rubric-edit-{safe_msg_id}-{i}"
        link = f'<a href="#{edit_anchor_id}" class="rubric-marker" title="{reason_esc}" style="background:#e3f2fd;color:#1565c0;padding:0 4px;border-radius:3px;font-size:0.85em;text-decoration:none;">[{i}]</a>'
        escaped = escaped.replace(marker, link)
    return (
        '<div class="annotated-draft" style="background:#fafafa;border:1px solid #e0e0e0;border-radius:6px;padding:12px;margin:8px 0;">'
        '<style>.annotated-draft ins { background:#c8e6c9; font-weight:bold; text-decoration:none; padding:0 2px; } '
        '.annotated-draft del { background:#ffcdd2; text-decoration:line-through; padding:0 2px; }</style>'
        f'{escaped}</div>'
    )

def render_message_with_draft(content: str, message_id: str, wrap_draft_in_expander: bool = False):
    """
    Render a message that may contain <draft> tags.
    Draft sections are rendered as editable text areas.
    If wrap_draft_in_expander is True, the draft (editable) section is shown inside a collapsed expander.
    Returns True if the message contained drafts and was rendered, False otherwise.
    """
    draft_parts = parse_draft_content(content)

    if not draft_parts:
        return False

    # Initialize session state for draft editing if needed
    draft_key = f"draft_edit_{message_id}"
    if draft_key not in st.session_state:
        st.session_state[draft_key] = {}

    # Store original drafts for comparison (for Update Rubric feature)
    original_key = f"draft_original_{message_id}"
    if original_key not in st.session_state:
        st.session_state[original_key] = {}

    # Track content hash to detect when source content has changed
    content_hash_key = f"draft_content_hash_{message_id}"
    current_content_hash = hash(content)

    # If content has changed since last render, reset the draft to match new content
    content_changed = (content_hash_key in st.session_state and
                       st.session_state[content_hash_key] != current_content_hash)
    st.session_state[content_hash_key] = current_content_hash

    draft_idx = 0
    for part in draft_parts:
        # Render text before the draft
        if 'before' in part and part['before'].strip():
            st.markdown(part['before'])

        # Render the draft as an editable text area
        if 'draft' in part:
            draft_content = part['draft']
            edit_key = f"{draft_key}_{draft_idx}"
            orig_key = f"{original_key}_{draft_idx}"
            reset_counter_key = f"reset_counter_{message_id}_{draft_idx}"

            # Initialize the draft content in session state if not already there
            # OR if the source content has changed (content_changed flag)
            if edit_key not in st.session_state[draft_key] or content_changed:
                st.session_state[draft_key][edit_key] = draft_content

            # Store the original draft for comparison (also update if content changed)
            if orig_key not in st.session_state[original_key] or content_changed:
                st.session_state[original_key][orig_key] = draft_content

            # Initialize reset counter (used to force new widget key on reset)
            if reset_counter_key not in st.session_state:
                st.session_state[reset_counter_key] = 0
            # Increment counter if content changed to force widget refresh
            elif content_changed:
                st.session_state[reset_counter_key] += 1

            # Create a container for the draft with visual styling (optionally in expander when message has rubric_revision)
            draft_container = st.expander("📝 **Draft** (editable)", expanded=not wrap_draft_in_expander) if wrap_draft_in_expander else st.container()
            with draft_container:
                if not wrap_draft_in_expander:
                    st.markdown("📝 **Draft** (editable)")

                # Get current value to display
                current_value = st.session_state[draft_key][edit_key]

                # Text area for editing - include reset counter in key to force refresh
                textarea_widget_key = f"textarea_{edit_key}_v{st.session_state[reset_counter_key]}"
                edited_draft = st.text_area(
                    label="Edit draft",
                    value=current_value,
                    key=textarea_widget_key,
                    height=300,
                    label_visibility="collapsed"
                )

                # Update session state with edited content
                if edited_draft != st.session_state[draft_key][edit_key]:
                    st.session_state[draft_key][edit_key] = edited_draft

                # Check if draft has been modified from original
                original_draft = st.session_state[original_key].get(orig_key, draft_content)
                has_changes = edited_draft != original_draft

                # Buttons row
                col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
                with col1:
                    # Only enable Save if there are changes
                    if st.button("💾 Save", key=f"save_{edit_key}", disabled=not has_changes):
                        if has_changes:
                            # Create a NEW message with the edited draft instead of updating
                            import uuid
                            new_message_id = f"assistant_{uuid.uuid4().hex[:8]}"
                            new_message_content = f"The user has made edits to the draft. Here's the edited draft:\n\n<draft>{edited_draft}</draft>"

                            new_message = {
                                "role": "assistant",
                                "content": new_message_content,
                                "display_content": new_message_content,
                                "id": new_message_id
                            }

                            st.session_state.messages.append(new_message)

                            # Reset the old message's text area back to the original draft
                            st.session_state[draft_key][edit_key] = original_draft
                            # Increment reset counter to force widget refresh
                            st.session_state[reset_counter_key] += 1

                            st.success("Draft saved as new message!")
                            st.rerun()

                with col2:
                    # Only enable Update Rubric if there are changes
                    if st.button("🔄 Update Rubric", key=f"update_rubric_{edit_key}", disabled=not has_changes):
                        if has_changes:
                            update_rubric_from_draft_edit(original_draft, edited_draft)

                with col3:
                    # Only enable Reset if there are changes
                    if st.button("↩️ Reset", key=f"reset_{edit_key}", disabled=not has_changes):
                        # Reset to original draft
                        st.session_state[draft_key][edit_key] = original_draft
                        # Increment reset counter to force a new widget key
                        st.session_state[reset_counter_key] += 1
                        st.rerun()

            draft_idx += 1

        # Render text after all drafts
        if 'after' in part and part['after'].strip():
            st.markdown(part['after'])

    return True

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
                thinking={
                    "type": "enabled",
                    "budget_tokens": 8000
                }
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


def _rubric_list_for_json(rubric_list: list):
    """Return a deep copy of the rubric list with _diff removed so it is JSON-serializable (no sets)."""
    cleaned = copy.deepcopy(rubric_list)
    for criterion in cleaned:
        if isinstance(criterion, dict) and "_diff" in criterion:
            del criterion["_diff"]
    return cleaned


def regenerate_draft_from_rubric_changes(original_rubric: list, updated_rubric: list, current_draft: str):
    """
    Call the LLM to regenerate the draft based on rubric changes.
    Returns dict with revised_draft, etc. on success, or dict with only 'error' key on failure.
    """
    from prompts import DRAFT_REGENERATE_SYSTEM_PROMPT, DRAFT_regenerate_prompt

    with st.spinner("Regenerating draft based on rubric changes..."):
        try:
            client = anthropic.Anthropic()
            original_clean = _rubric_list_for_json(original_rubric)
            updated_clean = _rubric_list_for_json(updated_rubric)
            user_prompt = DRAFT_regenerate_prompt(original_clean, updated_clean, current_draft)

            response = _api_call_with_retry(
                max_tokens=16000,
                system=DRAFT_REGENERATE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                model=MODEL_PRIMARY,
                thinking={"type": "enabled", "budget_tokens": 8000}
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


def _ab_commit_choice(ab_state, chosen_content, chosen_thinking, chosen_label):
    """Commit the user's A/B draft preference: add messages to history and log metadata."""
    import time as _time

    # Add the user message that triggered the A/B comparison
    st.session_state.messages.append(ab_state["user_message_data"])

    # Build assistant message with the chosen draft + A/B metadata
    message_data = {
        "role": "assistant",
        "content": chosen_content,
        "display_content": chosen_content,
        "message_id": ab_state["message_id"],
        "rubric_version": ab_state["rubric_version"],
        "rubric_assessment": None,
        "thinking": chosen_thinking,
        "ab_comparison": {
            "chosen": chosen_label,  # "rubric" or "conversation_only"
            "draft_rubric": ab_state["draft_a"],
            "draft_conversation_only": ab_state["draft_b"],
            "left_is_rubric": ab_state["left_is_rubric"],
            "timestamp": _time.time(),
        }
    }
    st.session_state.messages.append(message_data)

    # Restore analysis content and clear feedback
    st.session_state.current_analysis = ab_state.get("analysis_content", "")
    if ab_state.get("feedback_context"):
        st.session_state.assessment_feedback = {}

    # Run silent LLM-as-judge under 3 conditions (in background thread)
    import threading
    _ab_judge_args = {
        "draft_rubric": message_data.get("ab_comparison", {}).get("draft_rubric", ""),
        "draft_conv": message_data.get("ab_comparison", {}).get("draft_conversation_only", ""),
        "chosen_label": chosen_label,
        "rubric_version": message_data.get("rubric_version"),
        "conversation_id": st.session_state.get("selected_conversation", ""),
        "coldstart_text": st.session_state.get("infer_coldstart_text", "").strip(),
        "rubric_json": "",
        "supabase": st.session_state.get("supabase"),
        "project_id": st.session_state.get("current_project_id"),
    }
    _ab_rb_dict, _, _ = get_active_rubric()
    if _ab_rb_dict:
        _ab_judge_args["rubric_json"] = json.dumps(_rubric_to_json_serializable(_ab_rb_dict), indent=2)
    # Keep references for the background thread to write into
    _ab_judge_args["message_data_ref"] = message_data
    _ab_judge_args["results_list_ref"] = st.session_state.ab_judge_results

    _ab_done_event = threading.Event()

    def _ab_judge_background(args, done_event):
        try:
            _run_ab_judge_silent_bg(args)
        except Exception:
            pass
        finally:
            done_event.set()

    threading.Thread(target=_ab_judge_background, args=(_ab_judge_args, _ab_done_event), daemon=True).start()
    st.session_state._ab_judge_done_event = _ab_done_event

    # Clear pending state
    st.session_state.ab_comparison_pending = None


def _run_ab_judge_silent_bg(args):
    """Background-thread-safe version of A/B judge. Does NOT access st.session_state.
    All needed data is passed via the args dict. Writes results into shared list/dict refs."""
    draft_rubric = args.get("draft_rubric", "")
    draft_conv = args.get("draft_conv", "")
    chosen_label = args.get("chosen_label")
    rubric_json = args.get("rubric_json", "")
    coldstart_text = args.get("coldstart_text", "")

    if not draft_rubric or not draft_conv:
        return

    result = {
        "timestamp": time.time(),
        "user_chosen": chosen_label,
        "rubric_version": args.get("rubric_version"),
        "conversation_id": args.get("conversation_id", ""),
        "pairwise_details": [],
    }

    conditions = [("rubric", rubric_json), ("generic", "")]
    if coldstart_text:
        conditions.insert(1, ("coldstart", coldstart_text))

    for condition, context in conditions:
        try:
            prompt = GRADING_pairwise_judge_prompt(draft_rubric, draft_conv, condition, context)
            resp = _api_call_with_retry(
                model=MODEL_LIGHT, max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            text = "".join(b.text for b in resp.content if b.type == "text")
            js_match = re.search(r'\{[\s\S]*\}', text)
            if js_match:
                parsed = json.loads(js_match.group())
            else:
                parsed = {"preferred": "tie", "confidence": "low", "reasoning": "parse error"}
            pref = parsed.get("preferred", "tie")
            # Map A/B back to rubric/conversation_only (A = rubric draft, B = conv draft)
            if pref == "A":
                result[f"{condition}_judge"] = "rubric"
            elif pref == "B":
                result[f"{condition}_judge"] = "conversation_only"
            else:
                result[f"{condition}_judge"] = "tie"
            result["pairwise_details"].append({
                "condition": condition,
                "preferred": pref,
                "confidence": parsed.get("confidence", ""),
                "reasoning": parsed.get("reasoning", ""),
            })
        except Exception:
            result[f"{condition}_judge"] = "error"

    if not coldstart_text:
        result["coldstart_judge"] = "skipped"

    # Write into shared references (list append and dict update are thread-safe in CPython)
    results_list = args.get("results_list_ref")
    if results_list is not None:
        results_list.append(result)

    message_data_ref = args.get("message_data_ref")
    if message_data_ref and "ab_comparison" in message_data_ref:
        message_data_ref["ab_comparison"]["judge_results"] = result

    # Persist to database
    _save_sb = args.get("supabase")
    _save_pid = args.get("project_id")
    if _save_sb and _save_pid:
        try:
            save_project_data(_save_sb, _save_pid, "ab_judge_results", result)
        except Exception:
            pass


def _process_ranking_checkpoint(rcp, user_ranking):
    """Run LLM-as-judge under 3 conditions on 3 drafts, compute Borda + Kendall's tau."""
    drafts = rcp["drafts"]
    rubric_dict, _, _ = get_active_rubric()
    rubric_json = json.dumps(
        _rubric_to_json_serializable(rubric_dict), indent=2
    ) if rubric_dict else ""
    coldstart_text = st.session_state.get("infer_coldstart_text", "").strip()

    conditions = [("rubric", rubric_json), ("generic", "")]
    if coldstart_text:
        conditions.insert(1, ("coldstart", coldstart_text))

    source_keys = list(drafts.keys())  # ["rubric", "coldstart", "generic"]
    pairs = [(source_keys[i], source_keys[j])
             for i in range(len(source_keys)) for j in range(i + 1, len(source_keys))]

    llm_rankings = {}
    all_pairwise_details = []

    for condition, context in conditions:
        wins = {k: 0 for k in source_keys}
        for src_a, src_b in pairs:
            try:
                prompt = GRADING_pairwise_judge_prompt(
                    drafts[src_a], drafts[src_b], condition, context
                )
                resp = _api_call_with_retry(
                    model=MODEL_LIGHT, max_tokens=300,
                    messages=[{"role": "user", "content": prompt}]
                )
                text = "".join(b.text for b in resp.content if b.type == "text")
                js_match = re.search(r'\{[\s\S]*\}', text)
                parsed = json.loads(js_match.group()) if js_match else {}
                pref = parsed.get("preferred", "tie")
                if pref == "A":
                    wins[src_a] += 2
                elif pref == "B":
                    wins[src_b] += 2
                else:
                    wins[src_a] += 1
                    wins[src_b] += 1
                all_pairwise_details.append({
                    "condition": condition, "draft_a_src": src_a, "draft_b_src": src_b,
                    "preferred": pref, "confidence": parsed.get("confidence", ""),
                    "reasoning": parsed.get("reasoning", ""),
                })
            except Exception:
                wins[src_a] += 1
                wins[src_b] += 1

        sorted_sources = sorted(wins, key=lambda k: -wins[k])
        llm_rankings[condition] = sorted_sources

    # Compute Borda scores and Kendall's tau
    def borda_score(ranking):
        return {src: (len(ranking) - 1 - i) for i, src in enumerate(ranking)}

    user_borda = borda_score(user_ranking)
    borda_dist = {}
    kendall_results = {}

    for condition in [c[0] for c in conditions]:
        cond_borda = borda_score(llm_rankings[condition])
        borda_dist[condition] = sum(
            abs(user_borda.get(s, 0) - cond_borda.get(s, 0)) for s in source_keys
        )
        user_ranks = [user_ranking.index(s) + 1 if s in user_ranking else len(source_keys) for s in source_keys]
        cond_ranks = [llm_rankings[condition].index(s) + 1 for s in source_keys]
        try:
            tau, _ = kendalltau(user_ranks, cond_ranks)
        except Exception:
            tau = None
        kendall_results[condition] = tau

    result = {
        "timestamp": datetime.now().isoformat(),
        "writing_task": rcp.get("writing_task", ""),
        "drafts": drafts,
        "shuffle_order": rcp.get("shuffle_order", []),
        "user_ranking": user_ranking,
        "llm_rankings": llm_rankings,
        "borda_distance": borda_dist,
        "kendall_tau": kendall_results,
        "rubric_version": rubric_dict.get("version") if rubric_dict else None,
        "pairwise_details": all_pairwise_details,
    }

    st.session_state.ranking_checkpoint_results.append(result)
    st.session_state.ranking_checkpoint_pending = None

    # Persist to database
    _save_sb = st.session_state.get('supabase')
    _save_pid = st.session_state.get('current_project_id')
    if _save_sb and _save_pid:
        try:
            save_project_data(_save_sb, _save_pid, "ranking_checkpoint", result)
        except Exception:
            pass

    return result


def _word_level_diff(old_text: str, new_text: str) -> str:
    """
    Generate word-level diff HTML between old and new text.
    Unchanged words are shown normally, removed words have strikethrough in red,
    added words are shown in green.
    """
    import difflib

    if not old_text and not new_text:
        return '<span class="no-change-badge">Not specified</span>'
    if not old_text:
        return f'<span class="text-added">{new_text}</span>'
    if not new_text:
        return f'<span class="text-removed">{old_text}</span>'
    if old_text == new_text:
        return new_text

    # Split into words while preserving whitespace
    old_words = old_text.split()
    new_words = new_text.split()

    # Use SequenceMatcher to find differences
    matcher = difflib.SequenceMatcher(None, old_words, new_words)
    result = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Words are the same
            result.append(' '.join(old_words[i1:i2]))
        elif tag == 'replace':
            # Words were replaced
            if i1 < i2:
                result.append(f'<span class="text-removed">{" ".join(old_words[i1:i2])}</span>')
            if j1 < j2:
                result.append(f'<span class="text-added">{" ".join(new_words[j1:j2])}</span>')
        elif tag == 'delete':
            # Words were deleted
            result.append(f'<span class="text-removed">{" ".join(old_words[i1:i2])}</span>')
        elif tag == 'insert':
            # Words were inserted
            result.append(f'<span class="text-added">{" ".join(new_words[j1:j2])}</span>')

    return ' '.join(result)


def display_rubric_comparison(current_rubric: list, updated_rubric: list, apply_context: dict = None):
    """
    Display a comparison of current and updated rubric using collapsible sections.
    Each criterion is shown as an expander with status badge visible when collapsed.
    Word-level diffing highlights specific changes.
    If apply_context is provided, adds per-criterion "Apply this criterion" buttons and
    an "Apply all suggestions" button. apply_context = {"safe_msg_id", "message", "message_id"}.
    """
    # Build a map of current criteria by name for comparison
    current_map = {c.get('name', '').lower().strip(): c for c in current_rubric}
    updated_map = {c.get('name', '').lower().strip(): c for c in updated_rubric}
    safe_msg_id = (apply_context or {}).get("safe_msg_id", "")
    message = (apply_context or {}).get("message")
    message_id = (apply_context or {}).get("message_id", "")

    # CSS for highlighting
    st.markdown("""
    <style>
    .diff-field {
        margin: 8px 0;
        padding: 10px;
        background: #f8f9fa;
        border-radius: 6px;
        border-left: 3px solid #e0e0e0;
    }
    .diff-field-changed {
        border-left: 3px solid #ff9800;
        background: #fff8e1;
    }
    .diff-field-label {
        font-weight: 600;
        font-size: 12px;
        color: #555;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .diff-field-content {
        font-size: 14px;
        color: #333;
        line-height: 1.6;
    }
    .text-removed {
        text-decoration: line-through;
        color: #c62828;
        background-color: rgba(198, 40, 40, 0.1);
        padding: 1px 4px;
        border-radius: 3px;
    }
    .text-added {
        color: #2e7d32;
        background-color: rgba(46, 125, 50, 0.15);
        padding: 1px 4px;
        border-radius: 3px;
    }
    .weight-change {
        font-size: 13px;
        margin-bottom: 10px;
        padding: 6px 10px;
        background: #fff3e0;
        border-radius: 4px;
        display: inline-block;
    }
    .no-change-badge {
        font-size: 12px;
        color: #9e9e9e;
        font-style: italic;
    }
    .status-badge {
        font-size: 11px;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 500;
        margin-left: 8px;
    }
    .status-modified {
        background: #fff3e0;
        color: #e65100;
    }
    .status-new {
        background: #e8f5e9;
        color: #2e7d32;
    }
    .status-removed {
        background: #ffebee;
        color: #c62828;
    }
    .status-unchanged {
        background: #f5f5f5;
        color: #757575;
    }
    .dimension-block {
        margin: 6px 0;
        padding: 8px 12px;
        background: #f5f5f5;
        border-radius: 4px;
        border-left: 3px solid #e0e0e0;
        font-size: 14px;
    }
    .dimensions-container {
        margin-top: 6px;
    }
    .dimensions-section {
        margin-top: 10px;
    }
    .dimensions-section .dimensions-label {
        font-weight: 600;
        font-size: 12px;
        color: #555;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: block;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("#### 📋 Rubric Changes")

    # Show criteria in the updated rubric (maintains order), then removed ones
    for criterion in updated_rubric:
        name = criterion.get('name', 'Unnamed')
        name_key = name.lower().strip()
        priority = criterion.get('priority', criterion.get('weight', 0))

        if name_key not in current_map:
            # NEW criterion
            status_html = '<span class="status-badge status-new">NEW</span>'
            expander_label = f"✅ {name} (Priority #{priority})"

            with st.expander(expander_label, expanded=False):
                st.markdown(status_html, unsafe_allow_html=True)
                st.markdown(f"**Priority:** #{priority}")
                desc = criterion.get('description', '') or 'Not specified'
                st.markdown(f"""
                <div class="diff-field">
                    <div class="diff-field-label">Description</div>
                    <div class="diff-field-content"><span class="text-added">{html_module.escape(desc)}</span></div>
                </div>
                """, unsafe_allow_html=True)
                dims = criterion.get('dimensions', [])
                dim_labels = [d.get('label', '') for d in dims if d.get('label')]
                st.markdown('<div class="dimensions-section"><span class="dimensions-label">Dimensions</span><div class="dimensions-container">', unsafe_allow_html=True)
                if dim_labels:
                    for lbl in dim_labels:
                        st.markdown(f'<div class="dimension-block"><span class="text-added">{html_module.escape(lbl)}</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="dimension-block"><span class="text-added">None</span></div>', unsafe_allow_html=True)
                st.markdown('</div></div>', unsafe_allow_html=True)
                if apply_context:
                    name_safe = re.sub(r'[^a-zA-Z0-9_-]', '_', name)[:50]
                    _bcol, _ = st.columns([0.22, 0.78])
                    with _bcol:
                        if st.button("✓ Apply", key=f"apply_crit_{safe_msg_id}_new_{name_safe}", type="primary", use_container_width=True):
                            editing = list(st.session_state.editing_criteria or [])
                            editing.append(copy.deepcopy(criterion))
                            new_criteria = copy.deepcopy(editing)
                            st.session_state.editing_criteria = new_criteria
                            st.session_state.rubric = new_criteria
                            st.session_state.editing_criteria_ui_version = st.session_state.get("editing_criteria_ui_version", 0) + 1
                            st.toast(f"Added «{name}» to Rubric Configuration. Check the sidebar.")
                            st.rerun()

        elif _criterion_changed(current_map[name_key], criterion):
            # MODIFIED criterion
            old = current_map[name_key]
            old_priority = old.get('priority', old.get('weight', 0))

            status_html = '<span class="status-badge status-modified">MODIFIED</span>'
            expander_label = f"🔄 {name} (Priority #{priority})"

            with st.expander(expander_label, expanded=False):
                st.markdown(status_html, unsafe_allow_html=True)

                # Show priority change if applicable
                if old_priority != priority:
                    st.markdown(f"""
                    <div class="priority-change">
                        <strong>Priority:</strong> <span class="text-removed">#{old_priority}</span> → <span class="text-added">#{priority}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"**Priority:** #{priority}")

                # Description
                old_desc = old.get('description', '') or ''
                new_desc = criterion.get('description', '') or ''
                if old_desc != new_desc:
                    diff_html = _word_level_diff(old_desc, new_desc)
                    field_class = "diff-field diff-field-changed"
                else:
                    diff_html = new_desc if new_desc else '<span class="no-change-badge">Not specified</span>'
                    field_class = "diff-field"
                st.markdown(f"""
                <div class="{field_class}">
                    <div class="diff-field-label">Description</div>
                    <div class="diff-field-content">{diff_html}</div>
                </div>
                """, unsafe_allow_html=True)

                # Dimensions (one block per dimension)
                old_dims = [d.get('label', '') for d in old.get('dimensions', [])]
                new_dims = [d.get('label', '') for d in criterion.get('dimensions', [])]
                dims_changed = old_dims != new_dims
                st.markdown('<div class="dimensions-section"><span class="dimensions-label">Dimensions</span><div class="dimensions-container">', unsafe_allow_html=True)
                if new_dims:
                    for lbl in new_dims:
                        if dims_changed and lbl not in old_dims:
                            st.markdown(f'<div class="dimension-block"><span class="text-added">{html_module.escape(lbl)}</span></div>', unsafe_allow_html=True)
                        elif dims_changed and lbl in old_dims:
                            st.markdown(f'<div class="dimension-block">{html_module.escape(lbl)}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="dimension-block">{html_module.escape(lbl)}</div>', unsafe_allow_html=True)
                    if dims_changed:
                        for lbl in old_dims:
                            if lbl not in new_dims:
                                st.markdown(f'<div class="dimension-block"><span class="text-removed">{html_module.escape(lbl)}</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="dimension-block"><span class="no-change-badge">None</span></div>' if not dims_changed else '<div class="dimension-block"><span class="text-removed">None</span></div>', unsafe_allow_html=True)
                st.markdown('</div></div>', unsafe_allow_html=True)
                if apply_context:
                    name_safe = re.sub(r'[^a-zA-Z0-9_-]', '_', name)[:50]
                    _bcol, _ = st.columns([0.22, 0.78])
                    with _bcol:
                        if st.button("✓ Apply", key=f"apply_crit_{safe_msg_id}_mod_{name_safe}", type="primary", use_container_width=True):
                            editing = list(st.session_state.editing_criteria or [])
                            for i, c in enumerate(editing):
                                if (c.get('name') or '').lower().strip() == name_key:
                                    editing[i] = copy.deepcopy(criterion)
                                    break
                            new_criteria = copy.deepcopy(editing)
                            st.session_state.editing_criteria = new_criteria
                            st.session_state.rubric = new_criteria
                            st.session_state.editing_criteria_ui_version = st.session_state.get("editing_criteria_ui_version", 0) + 1
                            st.toast(f"Updated «{name}» in Rubric Configuration. Check the sidebar.")
                            st.rerun()

        else:
            # UNCHANGED criterion
            status_html = '<span class="status-badge status-unchanged">UNCHANGED</span>'
            expander_label = f"⚪ {name} (Priority #{priority})"

            with st.expander(expander_label, expanded=False):
                st.markdown(status_html, unsafe_allow_html=True)
                st.markdown(f"**Priority:** #{priority}")
                desc = criterion.get('description', '') or '<span class="no-change-badge">Not specified</span>'
                st.markdown(f"""
                <div class="diff-field">
                    <div class="diff-field-label">Description</div>
                    <div class="diff-field-content">{html_module.escape(desc) if isinstance(desc, str) and not desc.startswith('<') else desc}</div>
                </div>
                """, unsafe_allow_html=True)
                dims = criterion.get('dimensions', [])
                dim_labels = [d.get('label', '') for d in dims if d.get('label')]
                st.markdown('<div class="dimensions-section"><span class="dimensions-label">Dimensions</span><div class="dimensions-container">', unsafe_allow_html=True)
                if dim_labels:
                    for lbl in dim_labels:
                        st.markdown(f'<div class="dimension-block">{html_module.escape(lbl)}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="dimension-block"><span class="no-change-badge">None</span></div>', unsafe_allow_html=True)
                st.markdown('</div></div>', unsafe_allow_html=True)

    # Show removed criteria
    for criterion in current_rubric:
        name = criterion.get('name', 'Unnamed')
        name_key = name.lower().strip()

        if name_key not in updated_map:
            priority = criterion.get('priority', criterion.get('weight', 0))
            status_html = '<span class="status-badge status-removed">REMOVED</span>'
            expander_label = f"❌ {name} (Priority #{priority}) - REMOVED"

            with st.expander(expander_label, expanded=False):
                st.markdown(status_html, unsafe_allow_html=True)
                st.markdown(f"**Priority:** ~~#{priority}~~")
                desc = criterion.get('description', '') or 'Not specified'
                st.markdown(f"""
                <div class="diff-field">
                    <div class="diff-field-label">Description</div>
                    <div class="diff-field-content"><span class="text-removed">{html_module.escape(desc)}</span></div>
                </div>
                """, unsafe_allow_html=True)
                dims = criterion.get('dimensions', [])
                dim_labels = [d.get('label', '') for d in dims if d.get('label')]
                st.markdown('<div class="dimensions-section"><span class="dimensions-label">Dimensions</span><div class="dimensions-container">', unsafe_allow_html=True)
                if dim_labels:
                    for lbl in dim_labels:
                        st.markdown(f'<div class="dimension-block"><span class="text-removed">{html_module.escape(lbl)}</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="dimension-block"><span class="text-removed">None</span></div>', unsafe_allow_html=True)
                st.markdown('</div></div>', unsafe_allow_html=True)
                if apply_context:
                    name_safe = re.sub(r'[^a-zA-Z0-9_-]', '_', name)[:50]
                    _bcol, _ = st.columns([0.22, 0.78])
                    with _bcol:
                        if st.button("✓ Apply (remove)", key=f"apply_crit_{safe_msg_id}_rem_{name_safe}", use_container_width=True):
                            editing = [c for c in (st.session_state.editing_criteria or []) if (c.get('name') or '').lower().strip() != name_key]
                            new_criteria = copy.deepcopy(editing)
                            st.session_state.editing_criteria = new_criteria
                            st.session_state.rubric = new_criteria
                            st.session_state.editing_criteria_ui_version = st.session_state.get("editing_criteria_ui_version", 0) + 1
                            st.toast(f"Removed «{name}» from Rubric Configuration. Check the sidebar.")
                            st.rerun()

    # Apply all suggestions (when in apply_context)
    if apply_context and message is not None and updated_rubric:
        st.markdown("---")
        _acol, _ = st.columns([0.3, 0.7])
        with _acol:
            if st.button("✅ Apply all", key=f"apply_all_{safe_msg_id}", type="primary", use_container_width=True):
                hist = load_rubric_history()
                new_version = next_version_number()
                hist.append({"version": new_version, "rubric": copy.deepcopy(updated_rubric), "source": "edit_feedback"})
                save_rubric_history(hist)
                st.session_state.active_rubric_idx = len(hist) - 1
                st.session_state.rubric = copy.deepcopy(updated_rubric)
                st.session_state.editing_criteria = copy.deepcopy(updated_rubric)
                st.session_state.editing_criteria_ui_version = st.session_state.get("editing_criteria_ui_version", 0) + 1
                if message and "rubric_suggestion" in message:
                    message["rubric_suggestion"]["applied"] = True
                    message["rubric_suggestion"]["applied_version"] = new_version
                    # Update the rubric_version on the rubric revision message to reflect the new version
                    message["rubric_version"] = new_version
                st.toast(f"All suggestions applied and saved as v{new_version}. Check Rubric Configuration in the sidebar.")
                st.rerun()


def _criterion_changed(old: dict, new: dict) -> bool:
    """Check if a criterion has changed between old and new versions."""
    if old.get('description') != new.get('description'):
        return True
    if old.get('priority') != new.get('priority'):
        return True
    if old.get('category') != new.get('category'):
        return True
    old_dims = [d.get('label', '') for d in old.get('dimensions', [])]
    new_dims = [d.get('label', '') for d in new.get('dimensions', [])]
    if old_dims != new_dims:
        return True
    return False


def _build_rubric_version_changelog(old_rubric_list, new_rubric_list, old_version, new_version):
    """Build a concise text summary of changes between two rubric versions.

    Returns a string suitable for injecting into api_messages as context.
    """
    old_by_name = {c.get('name', '').strip().lower(): c for c in (old_rubric_list or [])}
    new_by_name = {c.get('name', '').strip().lower(): c for c in (new_rubric_list or [])}

    added = [new_by_name[n] for n in new_by_name if n not in old_by_name]
    removed = [old_by_name[n] for n in old_by_name if n not in new_by_name]
    modified = []
    for name_key in old_by_name:
        if name_key in new_by_name and _criterion_changed(old_by_name[name_key], new_by_name[name_key]):
            modified.append((old_by_name[name_key], new_by_name[name_key]))

    if not added and not removed and not modified:
        return ""

    parts = [f"[Rubric updated from v{old_version} to v{new_version}]", "", "Key changes:"]

    if added:
        names = ", ".join(c.get('name', '?') for c in added)
        parts.append(f"- Added criteria: {names}")

    if removed:
        names = ", ".join(c.get('name', '?') for c in removed)
        parts.append(f"- Removed criteria: {names}")

    for old_c, new_c in modified:
        name = new_c.get('name', '?')
        changes = []
        if old_c.get('description') != new_c.get('description'):
            changes.append("description updated")
        old_dims = set(d.get('label', '') for d in old_c.get('dimensions', []))
        new_dims = set(d.get('label', '') for d in new_c.get('dimensions', []))
        added_dims = new_dims - old_dims
        removed_dims = old_dims - new_dims
        if added_dims:
            changes.append(f"+dims: {', '.join(added_dims)}")
        if removed_dims:
            changes.append(f"-dims: {', '.join(removed_dims)}")
        if old_c.get('priority') != new_c.get('priority'):
            changes.append(f"priority {old_c.get('priority')} → {new_c.get('priority')}")
        if changes:
            parts.append(f"- Modified \"{name}\": {'; '.join(changes)}")

    return "\n".join(parts)


def display_rubric_update_result():
    """
    Display the rubric update result with side-by-side comparison.
    Called from the main app flow when there's a pending result.
    """
    if 'rubric_update_result' not in st.session_state or not st.session_state.rubric_update_result:
        return

    result = st.session_state.rubric_update_result
    rubric_updates = result.get("rubric_updates", {})
    thinking_text = result.get("thinking", "")

    # Display in a chat message style container
    with st.chat_message("assistant"):
        # Show thinking if available
        if thinking_text:
            with st.expander("🧠 Thinking", expanded=False):
                st.markdown(thinking_text)

        if rubric_updates.get("has_updates"):
            st.markdown("### ✨ Suggested Rubric Updates")
            st.markdown(f"**Rationale:** {rubric_updates.get('rationale', '')}")

            modified_rubric = rubric_updates.get("modified_rubric", [])

            if modified_rubric:
                # Get current rubric for comparison
                active_rubric_dict, _, _ = get_active_rubric()
                current_rubric = active_rubric_dict.get("rubric", []) if active_rubric_dict else []

                # Show side-by-side comparison
                display_rubric_comparison(current_rubric, modified_rubric)

                # Action buttons
                col1, col2, col3 = st.columns([1, 1, 3])
                with col1:
                    if st.button("✅ Apply Updates", key="apply_rubric_updates", type="primary"):
                        import copy
                        # Save as new rubric version
                        active_rubric_dict, _, _ = get_active_rubric()
                        hist = load_rubric_history()
                        new_version = next_version_number()

                        new_rubric_entry = {
                            "version": new_version,
                            "rubric": modified_rubric,
                            "writing_type": active_rubric_dict.get("writing_type", "Unknown") if active_rubric_dict else "Unknown",
                            "user_goals_summary": active_rubric_dict.get("user_goals_summary", "") if active_rubric_dict else "",
                            "weighting_rationale": f"Updated based on user draft edits (from v{active_rubric_dict.get('version', '?') if active_rubric_dict else '?'})",
                            "coaching_notes": rubric_updates.get('rationale', 'Rubric updated based on draft edit analysis')
                        }

                        hist.append(new_rubric_entry)
                        save_rubric_history(hist)

                        # Log edit to conversation (tagged so it only shows when this conversation is selected)
                        old_rubric_for_diff = active_rubric_dict.get("rubric", []) if active_rubric_dict else []
                        old_ver = active_rubric_dict.get("version", "?") if active_rubric_dict else 0
                        edit_class = classify_rubric_edits(old_rubric_for_diff, modified_rubric)
                        log_msg = format_edit_log_message(edit_class, old_ver, new_version, "draft_edit")
                        if st.session_state.selected_conversation is not None:
                            st.session_state.messages.append({
                                "role": "system",
                                "content": log_msg,
                                "conversation_id": st.session_state.selected_conversation,
                            })

                        # Update active rubric
                        st.session_state.active_rubric_idx = len(hist) - 1
                        st.session_state.rubric = modified_rubric
                        # Also update editing_criteria so Rubric Configuration shows the new version
                        st.session_state.editing_criteria = copy.deepcopy(modified_rubric)

                        # Clear the result
                        st.session_state.rubric_update_result = None

                        st.success(f"✓ Rubric updated to version {new_version}!")
                        st.rerun()

                with col2:
                    if st.button("❌ Dismiss", key="dismiss_rubric_updates"):
                        st.session_state.rubric_update_result = None
                        st.rerun()
        else:
            st.info(f"**No rubric updates needed.** {rubric_updates.get('rationale', 'Your edits are already well-captured by the current rubric.')}")

            if st.button("OK", key="dismiss_no_updates"):
                st.session_state.rubric_update_result = None
                st.rerun()


def _parse_compare_output(text: str):
    """
    Extract Base, Rubric A, Rubric B, Key Differences, Summary from the model's combined output.
    We look for the exact headings requested in the prompt.
    Returns dict with keys: base, a, b, key_diffs, summary (strings).
    """
    import re
    s = text.replace("\r\n", "\n")

    # Key differences (before Stage 1)
    key_pat = r"###\s*Key Rubric Differences\s*\n(.*?)(?=\n###\s*Stage 1|\Z)"
    # Stage 1 – Base
    base_pat = r"###\s*Stage 1\s*–\s*Base Draft\s*\n(.*?)(?:\n###\s*Stage 2|\Z)"
    # Stage 2 – Revisions: A and B
    a_pat    = r"####\s*Rubric A Revision[^\n]*\n(.*?)(?:\n####\s*Rubric B Revision|\Z)"
    b_pat    = r"####\s*Rubric B Revision[^\n]*\n(.*?)(?:\n###\s*Summary|\n###\s*Stage 3|\Z)"
    # Summary (at end)
    sum_pat  = r"###\s*Summary of Impact\s*\n(.*)\Z"

    key_m  = re.search(key_pat,  s, flags=re.S)
    base_m = re.search(base_pat, s, flags=re.S)
    a_m    = re.search(a_pat,    s, flags=re.S)
    b_m    = re.search(b_pat,    s, flags=re.S)
    sum_m  = re.search(sum_pat,  s, flags=re.S)

    key_diffs = (key_m.group(1).strip() if key_m else "")
    base      = (base_m.group(1).strip() if base_m else "")
    a_txt     = (a_m.group(1).strip() if a_m else "")
    b_txt     = (b_m.group(1).strip() if b_m else "")
    summary   = (sum_m.group(1).strip() if sum_m else "")

    return {"base": base, "a": a_txt, "b": b_txt, "key_diffs": key_diffs, "summary": summary}

def _md_diff_to_html_compare(marked_text: str) -> str:
    """
    Convert **bold** → green add span, ~~strike~~ → red del span.
    Keep paragraphs. Works on the revision texts or comparison cells.
    """
    if marked_text is None:
        return ""
    # Escape HTML special chars first
    esc = (marked_text
           .replace("&", "&amp;")
           .replace("<", "&lt;")
           .replace(">", "&gt;"))
    # Bold additions
    esc = re.sub(r"\*\*(.+?)\*\*", r"<span class='add'>\1</span>", esc)
    # Strikethrough deletions
    esc = re.sub(r"~~(.+?)~~", r"<span class='del'>\1</span>", esc)
    # Paragraphs
    parts = [f"<p>{p}</p>" for p in esc.split("\n\n") if p.strip()]
    return "<div class='diff-wrap'>" + "".join(parts) + "</div>"

def compare_rubrics(task, rubric_a, rubric_b):
    """Compare two rubrics using the RUBRIC_COMPARE_DRAFTS_PROMPT"""
    prompt = RUBRIC_COMPARE_DRAFTS_PROMPT.format(
        task=task,
        rubric_a=json.dumps(rubric_a, indent=2),
        rubric_b=json.dumps(rubric_b, indent=2)
    )

    # Call Claude API directly
    message = _api_call_with_retry(
        model=MODEL_PRIMARY,
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}],
        thinking={
            "type": "enabled",
            "budget_tokens": 8000
        }
    )

    # Extract thinking and text from response
    thinking_text = ""
    response = ""
    for block in message.content:
        if block.type == "thinking":
            thinking_text = block.thinking
        elif block.type == "text":
            response = block.text

    # Parse using the exact same logic as the notebook
    parsed = _parse_compare_output(response)

    return {
        "base_txt": parsed["base"],
        "a_txt": parsed["a"],
        "b_txt": parsed["b"],
        "key_diffs": parsed["key_diffs"],
        "summary": parsed["summary"],
        "thinking": thinking_text
    }

# Set the API key - check Streamlit secrets first, then environment variable
try:
    api_key = st.secrets["ANTHROPIC_API_KEY"]
except (KeyError, FileNotFoundError):
    api_key = os.getenv('ANTHROPIC_API_KEY')

if api_key:
    os.environ['ANTHROPIC_API_KEY'] = api_key

client = anthropic.Anthropic()


def _api_call_with_retry(max_retries=3, base_delay=2, **kwargs):
    """Wrapper around client.messages.create with exponential backoff retry on overloaded/rate-limit errors."""
    for attempt in range(max_retries + 1):
        try:
            return client.messages.create(**kwargs)
        except anthropic.APIStatusError as e:
            if e.status_code in (429, 529) and attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
                continue
            raise
        except anthropic.APIConnectionError:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
                continue
            raise


def parse_analysis_and_content(full_text):
    """Extract analysis content and main content from response, removing both analysis and rubric_assessment tags"""
    # Pattern to match <analysis>...</analysis> tags
    analysis_pattern = r'<analysis>(.*?)</analysis>'
    analysis_matches = re.findall(analysis_pattern, full_text, re.DOTALL)

    # Extract analysis content
    analysis_content = '\n\n'.join(analysis_matches)

    # Remove complete analysis tags
    main_content = re.sub(analysis_pattern, '', full_text, flags=re.DOTALL)

    # Remove complete rubric_assessment tags
    rubric_assessment_pattern = r'<rubric_assessment>(.*?)</rubric_assessment>'
    main_content = re.sub(rubric_assessment_pattern, '', main_content, flags=re.DOTALL)

    # Handle incomplete opening tags (partial tags during streaming)
    if '<analysis>' in main_content:
        main_content = main_content.split('<analysis>')[0].strip()

    if '<rubric_assessment>' in main_content:
        main_content = main_content.split('<rubric_assessment>')[0].strip()

    return analysis_content, main_content

def parse_rubric_assessment(full_text):
    """Extract rubric assessment from response text
    Returns: dict with evaluation details and JSON summary, or None if not found
    """
    result = {
        'evaluation_text': None,
        'json_summary': None,
        'criteria_details': {}
    }

    # First, try to extract <evaluation> tags content
    eval_pattern = r'<evaluation>(.*?)</evaluation>'
    eval_match = re.search(eval_pattern, full_text, re.DOTALL)

    if eval_match:
        result['evaluation_text'] = eval_match.group(1).strip()

        # Parse individual criterion sections from evaluation text
        # Find all ### headers and extract sections
        # Use finditer to get all matches with their positions
        criterion_pattern = r'### (.+?)(?=\n###|\Z)'
        matches = re.finditer(criterion_pattern, result['evaluation_text'], re.DOTALL)

        for match in matches:
            section = match.group(0)  # Get the full matched text including ###
            lines = section.split('\n')
            # Remove ### prefix from the first line
            criterion_name = lines[0].replace('###', '').strip()

            # Initialize criterion details
            criterion_data = {
                'priority': None,
                'achievement_level': None,
                'evidence': [],
                'rationale': None,
                'to_reach_next_level': None
            }

            current_field = None
            content_buffer = []

            for line in lines[1:]:
                line = line.strip()

                # Handle both old format (Weight) and new format (Priority)
                if line.startswith('**Weight**:'):
                    criterion_data['weight'] = line.replace('**Weight**:', '').strip()
                elif line.startswith('**Priority**:'):
                    priority_text = line.replace('**Priority**:', '').strip()
                    # Extract number from "#N" format
                    priority_match = re.search(r'#?(\d+)', priority_text)
                    if priority_match:
                        criterion_data['priority'] = int(priority_match.group(1))
                elif line.startswith('**Achievement Level**:'):
                    level_text = line.replace('**Achievement Level**:', '').strip()
                    criterion_data['achievement_level'] = level_text
                elif line.startswith('**Evidence from draft**:'):
                    current_field = 'evidence'
                    content_buffer = []
                elif line.startswith('**Rationale**:'):
                    if current_field == 'evidence':
                        criterion_data['evidence'] = content_buffer
                    current_field = 'rationale'
                    content_buffer = []
                elif line.startswith('**To reach next level**:'):
                    if current_field == 'rationale':
                        criterion_data['rationale'] = ' '.join(content_buffer)
                    current_field = 'to_reach_next_level'
                    content_buffer = []
                elif line.startswith('---'):
                    # End of criterion section
                    if current_field == 'to_reach_next_level':
                        criterion_data['to_reach_next_level'] = ' '.join(content_buffer)
                    break
                elif current_field:
                    if line.startswith('- '):
                        content_buffer.append(line[2:])  # Remove bullet
                    elif line:
                        content_buffer.append(line)

            # Save any remaining buffer
            if current_field == 'rationale':
                criterion_data['rationale'] = ' '.join(content_buffer)
            elif current_field == 'to_reach_next_level':
                criterion_data['to_reach_next_level'] = ' '.join(content_buffer)

            result['criteria_details'][criterion_name] = criterion_data

    # Extract JSON summary (look for it anywhere in the text, not just in tags)
    json_pattern = r'```json\s*(.*?)\s*```'
    json_match = re.search(json_pattern, full_text, re.DOTALL)

    if json_match:
        try:
            json_data = json.loads(json_match.group(1))
            result['json_summary'] = json_data
        except json.JSONDecodeError:
            pass

    # Return None if we found nothing useful
    if not result['evaluation_text'] and not result['json_summary']:
        return None

    return result

def display_rubric_assessment(assessment_data, message_id=None, draft_text=None):
    """Display rubric assessment in a card-based layout with dimension checklists and evidence highlights"""
    if not assessment_data:
        return

    st.markdown("---")

    # Initialize feedback storage in session state if needed
    if 'assessment_feedback' not in st.session_state:
        st.session_state.assessment_feedback = {}

    # Use message_id for unique key, fallback to id(assessment_data)
    assessment_key = f"assessment_{message_id}" if message_id else f"assessment_{id(assessment_data)}"
    if assessment_key not in st.session_state.assessment_feedback:
        st.session_state.assessment_feedback[assessment_key] = {}

    # Get assessment summary from JSON
    overall_assessment = None
    criteria_scores = []
    evidence_highlights = []

    if assessment_data.get('json_summary'):
        json_data = assessment_data['json_summary']
        overall_assessment = json_data.get('overall_assessment')
        criteria_scores = json_data.get('criteria_scores', [])
        evidence_highlights = json_data.get('evidence_highlights', [])

    # Helper function to get level info (star-based system)
    def get_level_info(achievement_level):
        if not achievement_level:
            return "#999", "❓"
        level_lower = achievement_level.lower()
        if 'excellent' in level_lower:
            return "#4CAF50", "⭐⭐⭐"
        elif 'good' in level_lower:
            return "#2196F3", "⭐⭐"
        elif 'fair' in level_lower:
            return "#FF9800", "⭐"
        elif 'needs work' in level_lower or 'needs_work' in level_lower:
            return "#E65100", "◇"
        else:  # weak
            return "#F44336", "☆"

    # Display assessment header
    with st.expander("📊 Rubric Assessment", expanded=False):
        # Show thinking if available
        if assessment_data.get('thinking'):
            with st.expander("🧠 Thinking", expanded=False):
                st.markdown(assessment_data['thinking'])

        # Sort criteria by priority
        sorted_criteria = sorted(criteria_scores, key=lambda x: x.get('priority', 99))

        # --- Summary dashboard ---
        if sorted_criteria:
            # Count levels
            level_counts = {"excellent": 0, "good": 0, "fair": 0, "needs work": 0, "weak": 0}
            total_dims_met = 0
            total_dims_all = 0
            for crit in sorted_criteria:
                lvl = crit.get('achievement_level', '').lower()
                for k in level_counts:
                    if k in lvl:
                        level_counts[k] += 1
                        break
                total_dims_met += crit.get('dimensions_met', 0)
                total_dims_all += crit.get('dimensions_total', 0)
            overall_pct = round(total_dims_met / total_dims_all * 100) if total_dims_all > 0 else 0

            # Overall score bar — compute 1-5 rating and label from percentage
            if overall_pct >= 90:
                overall_rating, overall_label, bar_color = 5, "Excellent", "#4CAF50"
            elif overall_pct >= 75:
                overall_rating, overall_label, bar_color = 4, "Good", "#2196F3"
            elif overall_pct >= 50:
                overall_rating, overall_label, bar_color = 3, "Fair", "#FF9800"
            elif overall_pct >= 25:
                overall_rating, overall_label, bar_color = 2, "Needs Work", "#E65100"
            else:
                overall_rating, overall_label, bar_color = 1, "Weak", "#F44336"
            summary_html = f'''<div style="background:linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);border:1px solid #e0e0e0;border-radius:12px;padding:20px;margin-bottom:16px;">
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
  <div>
    <span style="font-size:1.3em;font-weight:700;">Overall Score</span>
    <span style="display:inline-block;margin-left:10px;padding:3px 12px;background:{bar_color}20;color:{bar_color};border-radius:14px;font-size:0.95em;font-weight:700;">{overall_label} ({overall_rating})</span>
  </div>
  <span style="font-size:2em;font-weight:800;color:{bar_color};">{overall_pct}%</span>
</div>
<div style="background:#e9ecef;border-radius:8px;height:12px;overflow:hidden;margin-bottom:14px;">
  <div style="background:{bar_color};height:100%;width:{overall_pct}%;border-radius:8px;transition:width 0.5s;"></div>
</div>
<div style="font-size:0.85em;color:#666;margin-bottom:10px;">{total_dims_met} of {total_dims_all} dimensions met across {len(sorted_criteria)} criteria</div>
<div style="display:flex;gap:6px;flex-wrap:wrap;">'''
            level_configs = [
                ("excellent", "Excellent", "#E8F5E9", "#2E7D32"),
                ("good", "Good", "#E3F2FD", "#1565C0"),
                ("fair", "Fair", "#FFF3E0", "#E65100"),
                ("needs work", "Needs Work", "#FBE9E7", "#BF360C"),
                ("weak", "Weak", "#FFEBEE", "#C62828"),
            ]
            for key, label, bg, fg in level_configs:
                count = level_counts.get(key, 0)
                if count > 0:
                    summary_html += f'<span style="padding:4px 12px;background:{bg};color:{fg};border-radius:16px;font-size:0.85em;font-weight:600;">{count} {label}</span>'
            summary_html += '</div></div>'
            st.markdown(summary_html, unsafe_allow_html=True)

        # Overall assessment narrative
        if overall_assessment:
            st.markdown(f"*{overall_assessment}*")
            st.markdown("")

        # Build a lookup from (criterion, dimension_id) to evidence_highlights quotes
        highlight_lookup = {}
        for ev in evidence_highlights:
            key = (ev.get('criterion', ''), ev.get('dimension_id', ''))
            if key not in highlight_lookup:
                highlight_lookup[key] = []
            highlight_lookup[key].append({
                'quote': ev.get('quote', ''),
                'relevance': ev.get('relevance', '')
            })

        # --- Criteria cards ---
        for crit in sorted_criteria:
            crit_name = crit.get('name', 'Unknown')
            priority = crit.get('priority', 'N/A')
            achievement_level = crit.get('achievement_level', 'N/A')
            dimensions_detail = crit.get('dimensions_detail', [])
            dims_met = crit.get('dimensions_met', 0)
            dims_total = crit.get('dimensions_total', 0)
            improvement_explanation = crit.get('improvement_explanation', '')

            level_color, level_emoji = get_level_info(achievement_level)
            pct = round(dims_met / dims_total * 100) if dims_total > 0 else 0
            # Compute 1-5 rating for this criterion
            crit_pct_frac = dims_met / dims_total if dims_total > 0 else 0
            if crit_pct_frac >= 0.90: crit_rating = 5
            elif crit_pct_frac >= 0.75: crit_rating = 4
            elif crit_pct_frac >= 0.50: crit_rating = 3
            elif crit_pct_frac >= 0.25: crit_rating = 2
            else: crit_rating = 1

            # Criterion card header with progress bar
            card_header = f'''<div style="background:white;border:1px solid #e0e0e0;border-left:4px solid {level_color};border-radius:8px;padding:14px 16px;margin-bottom:4px;">
<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
  <div>
    <span style="font-size:0.75em;color:#888;text-transform:uppercase;letter-spacing:0.5px;">Priority #{priority}</span>
    <div style="font-size:1.05em;font-weight:600;margin-top:2px;">{level_emoji} {crit_name}</div>
  </div>
  <div style="text-align:right;">
    <span style="background:{level_color}20;color:{level_color};padding:3px 10px;border-radius:12px;font-size:0.8em;font-weight:600;">{achievement_level} ({crit_rating})</span>
    <div style="font-size:0.8em;color:#888;margin-top:4px;">{dims_met}/{dims_total} dimensions · {pct}%</div>
  </div>
</div>
<div style="background:#e9ecef;border-radius:6px;height:6px;overflow:hidden;margin-top:10px;">
  <div style="background:{level_color};height:100%;width:{pct}%;border-radius:6px;"></div>
</div>
</div>'''
            st.markdown(card_header, unsafe_allow_html=True)

            with st.expander(f"View details — {crit_name}", expanded=False):
                # Dimension checklist as styled items
                for dim in dimensions_detail:
                    dim_id = dim.get('id', '')
                    dim_label = dim.get('label', 'Unknown dimension')
                    dim_met = dim.get('met', False)
                    dim_evidence = dim.get('evidence', '')
                    linked_highlights = highlight_lookup.get((crit_name, dim_id), [])

                    if dim_met:
                        st.markdown(f'''<div style="display:flex;align-items:flex-start;gap:8px;padding:8px 12px;background:#f0faf0;border-radius:6px;margin-bottom:4px;border:1px solid #c8e6c9;">
<span style="color:#2E7D32;font-size:1.1em;flex-shrink:0;">✅</span>
<div><span style="font-weight:500;">{dim_label}</span></div>
</div>''', unsafe_allow_html=True)
                        evidence_text = ""
                        if linked_highlights:
                            for hl in linked_highlights:
                                quote = hl.get('quote', '')
                                if quote:
                                    evidence_text = quote
                                    break
                        if not evidence_text and dim_evidence:
                            evidence_text = dim_evidence
                        if evidence_text:
                            st.markdown(f'''<div style="margin-left:32px;padding:6px 12px;border-left:3px solid #a5d6a7;color:#555;font-size:0.88em;font-style:italic;margin-bottom:6px;">"{evidence_text}"</div>''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''<div style="display:flex;align-items:flex-start;gap:8px;padding:8px 12px;background:#fef0ef;border-radius:6px;margin-bottom:4px;border:1px solid #ffcdd2;">
<span style="color:#C62828;font-size:1.1em;flex-shrink:0;">❌</span>
<div><span style="font-weight:500;">{dim_label}</span></div>
</div>''', unsafe_allow_html=True)
                        evidence_text = ""
                        if linked_highlights:
                            for hl in linked_highlights:
                                quote = hl.get('quote', '')
                                if quote:
                                    evidence_text = quote
                                    break
                        if not evidence_text and dim_evidence:
                            evidence_text = dim_evidence
                        if evidence_text:
                            st.markdown(f'''<div style="margin-left:32px;padding:6px 12px;border-left:3px solid #ef9a9a;color:#555;font-size:0.88em;font-style:italic;margin-bottom:6px;">"{evidence_text}"</div>''', unsafe_allow_html=True)

                # Improvement explanation
                if improvement_explanation and 'excellent' not in achievement_level.lower():
                    st.markdown(f'''<div style="background:#FFF8E1;border:1px solid #FFE082;border-radius:8px;padding:12px 14px;margin-top:8px;">
<div style="font-weight:600;font-size:0.9em;color:#F57F17;margin-bottom:4px;">💡 To improve</div>
<div style="font-size:0.9em;color:#555;">{improvement_explanation}</div>
</div>''', unsafe_allow_html=True)

                # Feedback input for this criterion
                st.markdown("---")
                st.markdown("**Your feedback on this assessment:**")
                feedback_text = st.text_area(
                    "Share your thoughts (optional)",
                    value=st.session_state.assessment_feedback[assessment_key].get(crit_name, ""),
                    key=f"feedback_{assessment_key}_{crit_name}",
                    placeholder="Disagree with the score? Have additional context? Share it here...",
                    height=80,
                    label_visibility="collapsed"
                )

                # Store feedback in session state
                if feedback_text:
                    st.session_state.assessment_feedback[assessment_key][crit_name] = feedback_text
                elif crit_name in st.session_state.assessment_feedback[assessment_key]:
                    del st.session_state.assessment_feedback[assessment_key][crit_name]

        # Evidence Highlights Section
        if evidence_highlights and draft_text:
            st.markdown("---")
            st.markdown("### 📄 Draft with Evidence Highlights")
            st.markdown("*Hover over highlighted text to see the criterion and relevance explanation.*")

            # Build evidence map with actual positions
            all_evidence = []
            for ev in evidence_highlights:
                quote = ev.get("quote", "").strip()
                if quote:
                    # Find actual position by searching for the quote
                    start_idx = draft_text.find(quote)
                    if start_idx == -1:
                        # Try finding first 30 chars as a fallback
                        if len(quote) > 30:
                            start_idx = draft_text.find(quote[:30])

                    if start_idx != -1:
                        all_evidence.append({
                            "criterion": ev.get("criterion", ""),
                            "quote": quote,
                            "start_index": start_idx,
                            "end_index": start_idx + len(quote),
                            "relevance": ev.get("relevance", ""),
                            "dimension_id": ev.get("dimension_id", ""),
                            "dimension_met": ev.get("dimension_met", True)
                        })

            if all_evidence:
                # Sort by start_index
                sorted_evidence = sorted(all_evidence, key=lambda x: x.get("start_index", 0))

                # Remove overlapping highlights
                non_overlapping = []
                last_end = 0
                for ev in sorted_evidence:
                    if ev["start_index"] >= last_end:
                        non_overlapping.append(ev)
                        last_end = ev["end_index"]
                sorted_evidence = non_overlapping

                # Define colors for different criteria
                criterion_colors = {}
                colors = ["#ffeb3b", "#81d4fa", "#a5d6a7", "#ffcc80", "#ce93d8", "#ef9a9a", "#80cbc4"]
                for i, crit in enumerate(sorted_criteria):
                    criterion_colors[crit.get("name", f"Criterion {i+1}")] = colors[i % len(colors)]

                # Build dimension label lookup: (criterion_name, dimension_id) -> dimension_label
                dimension_labels = {}
                for crit in sorted_criteria:
                    crit_name = crit.get("name", "")
                    for dim in crit.get("dimensions_detail", []):
                        dim_id = dim.get("id", "")
                        dim_label = dim.get("label", "")
                        if dim_id:
                            dimension_labels[(crit_name, dim_id)] = dim_label

                # Legend showing criterion colors with colored boxes
                st.markdown("**Legend:**")
                legend_html = '<div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px;">'
                for crit in sorted_criteria:
                    crit_name = crit.get("name", "")
                    color = criterion_colors.get(crit_name, "#ffeb3b")
                    legend_html += f'<span style="background-color: {color}; padding: 2px 8px; border-radius: 3px; font-size: 0.9em;">{crit_name}</span>'
                legend_html += '</div>'
                st.markdown(legend_html, unsafe_allow_html=True)

                # Build inline highlighted draft with tooltips (same as Evaluate: Alignment tab)
                tooltip_css = """<style>
.evidence-highlight { position: relative; cursor: help; padding: 1px 2px; border-radius: 2px; }
.evidence-highlight .tooltip-text { visibility: hidden; background-color: #333; color: #fff; text-align: left; padding: 8px 12px; border-radius: 6px; position: absolute; z-index: 1000; bottom: 125%; left: 50%; transform: translateX(-50%); width: 280px; font-size: 0.85em; line-height: 1.4; box-shadow: 0 2px 8px rgba(0,0,0,0.3); }
.evidence-highlight .tooltip-text::after { content: ""; position: absolute; top: 100%; left: 50%; margin-left: -5px; border-width: 5px; border-style: solid; border-color: #333 transparent transparent transparent; }
.evidence-highlight:hover .tooltip-text { visibility: visible; }
.tooltip-criterion { font-weight: bold; color: #ffc107; margin-bottom: 5px; }
</style>"""

                highlighted_html = tooltip_css
                highlighted_html += '<div style="line-height: 1.8; padding: 10px; background-color: #f9f9f9; border-radius: 8px; border: 1px solid #ddd;">'

                last_end = 0
                for ev in sorted_evidence:
                    start = ev.get("start_index", 0)
                    end = ev.get("end_index", 0)
                    criterion = ev.get("criterion", "")
                    dimension_id = ev.get("dimension_id", "")
                    dimension_met = ev.get("dimension_met", True)
                    relevance = ev.get("relevance", "")
                    color = criterion_colors.get(criterion, "#ffeb3b")

                    # Look up dimension label
                    dimension_label = dimension_labels.get((criterion, dimension_id), dimension_id)

                    # Build status indicator
                    status_icon = "✅" if dimension_met else "❌"
                    status_color = "#4CAF50" if dimension_met else "#F44336"

                    if start >= last_end and end > start and start < len(draft_text):
                        # Add unhighlighted text before this highlight
                        highlighted_html += draft_text[last_end:start].replace("\n", "<br>")

                        # Add highlighted text with tooltip showing criterion + dimension + status
                        quote_text = draft_text[start:end].replace("\n", "<br>")
                        tooltip_content = f'<div class="tooltip-criterion">{criterion}</div>'
                        if dimension_label:
                            tooltip_content += f'<div style="color: {status_color}; margin-bottom: 5px;">{status_icon} {dimension_label}</div>'
                        if relevance:
                            tooltip_content += f'<div style="font-size: 0.9em; opacity: 0.9;">{relevance}</div>'

                        highlighted_html += f'<span class="evidence-highlight" style="background-color: {color};">{quote_text}<span class="tooltip-text">{tooltip_content}</span></span>'

                        last_end = end

                # Add remaining text
                if last_end < len(draft_text):
                    highlighted_html += draft_text[last_end:].replace("\n", "<br>")

                highlighted_html += '</div>'

                st.markdown(highlighted_html, unsafe_allow_html=True)
            else:
                # No evidence found in draft text - show without highlights
                st.info("No evidence quotes could be matched to the draft text.")
                st.markdown(draft_text)
        elif draft_text and not evidence_highlights:
            # Show draft without highlights if no evidence_highlights provided
            st.markdown("---")
            st.markdown("### 📄 Draft")
            st.markdown(draft_text)

        # Submit feedback button at the bottom
        st.markdown("---")
        if st.session_state.assessment_feedback[assessment_key]:
            st.markdown(f"**{len(st.session_state.assessment_feedback[assessment_key])} criterion/criteria** have feedback")

            if st.button("💬 Submit Feedback to Conversation", key=f"submit_feedback_{assessment_key}", use_container_width=True):
                # Format feedback into a structured message
                feedback_message = format_assessment_feedback(assessment_key)

                if feedback_message:
                    # Add the feedback as a user message
                    st.session_state.messages.append({
                        "role": "user",
                        "content": feedback_message
                    })

                    # Clear the feedback after submitting
                    st.session_state.assessment_feedback[assessment_key] = {}

                    st.success("✓ Feedback submitted to conversation!")
                    st.rerun()
        else:
            st.caption("💡 Add feedback to any criterion above, then click submit to continue the conversation")

def format_assessment_feedback(assessment_key):
    """Format assessment feedback into a structured message for the conversation"""
    if assessment_key not in st.session_state.assessment_feedback:
        return None

    feedback_dict = st.session_state.assessment_feedback[assessment_key]
    if not feedback_dict:
        return None

    # Build structured feedback message
    feedback_parts = ["I have some feedback on your rubric assessment:\n"]

    for criterion_name, feedback_text in feedback_dict.items():
        feedback_parts.append(f"**{criterion_name}:**")
        feedback_parts.append(f"{feedback_text}\n")

    return "\n".join(feedback_parts)

def format_feedback_for_context():
    """Format collected assessment feedback into a structured message for the model"""
    if 'assessment_feedback' not in st.session_state or not st.session_state.assessment_feedback:
        return None

    # Get the most recent assistant message ID
    most_recent_message_id = None
    for msg in reversed(st.session_state.messages):
        if msg['role'] == 'assistant' and msg.get('message_id'):
            most_recent_message_id = msg['message_id']
            break

    if not most_recent_message_id:
        return None

    # Only collect feedback for the most recent assistant message
    feedback_items = []

    for feedback_key, feedback_data in st.session_state.assessment_feedback.items():
        # Only include feedback from the most recent message
        if not feedback_key.startswith(most_recent_message_id):
            continue

        # Parse the key to extract criterion name
        # Key format: "{message_id}_{criterion_name}_{idx}"
        parts = feedback_key.split('_')
        if len(parts) >= 4:
            # Skip message_id parts (assistant_{timestamp}), get everything except the last index
            criterion_name = '_'.join(parts[2:-1])
        else:
            criterion_name = feedback_key

        rating = feedback_data.get('rating', '')
        comment = feedback_data.get('comment', '')

        if rating or comment:
            feedback_str = f"- **{criterion_name}**:"
            if rating:
                feedback_str += f" {rating} feedback"
            if comment:
                feedback_str += f"\n  Comment: {comment}"
            feedback_items.append(feedback_str)

    if not feedback_items:
        return None

    feedback_message = """**[RUBRIC ASSESSMENT FEEDBACK]**
The user has provided feedback on your previous rubric assessment. Please incorporate this feedback into your response:

""" + "\n".join(feedback_items) + "\n\n**[USER'S REQUEST]**\n"

    return feedback_message

def stream_without_analysis(stream, response_placeholder, message_id, thinking_placeholder=None):
    """Stream response while hiding analysis and rubric_assessment tags.

    Note: This function strips out any rubric_assessment tags the model may generate
    during normal chat. Assessments should only be displayed when explicitly requested
    via the 'Assess Draft' button.

    With extended thinking enabled, this also captures thinking content but does NOT
    stream it - thinking is buffered silently and only shown in a collapsed expander
    after the response is complete.
    """
    full_response = ""
    thinking_content = ""
    started_main_content = False

    # Show a simple thinking indicator while thinking (no streaming of thinking content)
    if thinking_placeholder:
        thinking_placeholder.markdown("🧠 *Thinking...*")

    # Handle streaming with extended thinking
    for event in stream:
        # Handle thinking events
        if hasattr(event, 'type'):
            if event.type == 'content_block_start':
                if hasattr(event, 'content_block') and event.content_block.type == 'text':
                    # Clear thinking indicator when main content starts
                    if thinking_placeholder and not started_main_content:
                        thinking_placeholder.empty()
                        started_main_content = True
            elif event.type == 'content_block_delta':
                if hasattr(event, 'delta'):
                    if event.delta.type == 'thinking_delta':
                        # Buffer thinking content silently - don't stream it
                        thinking_content += event.delta.thinking
                    elif event.delta.type == 'text_delta':
                        full_response += event.delta.text

                        # Stop streaming if we hit rubric_assessment tag (strip it out)
                        if '<rubric_assessment>' in full_response:
                            # Get content before rubric_assessment tag
                            content_before_assessment = full_response.split('<rubric_assessment>')[0]
                            _, main_content = parse_analysis_and_content(content_before_assessment)
                            # Strip draft tags for streaming display to match post-rerun appearance
                            streaming_content = strip_draft_tags_for_streaming(main_content)
                            response_placeholder.markdown(streaming_content)
                            continue

                        # Parse the current accumulated response to filter out analysis
                        _, main_content = parse_analysis_and_content(full_response)
                        # Strip draft tags for streaming display to match post-rerun appearance
                        streaming_content = strip_draft_tags_for_streaming(main_content)
                        response_placeholder.markdown(streaming_content + "▌")

    # Final parse to ensure clean output
    analysis_content, main_content = parse_analysis_and_content(full_response)

    # Strip any rubric_assessment content from main_content
    # (model should not be generating assessments during normal chat)
    if '<rubric_assessment>' in main_content:
        main_content = main_content.split('<rubric_assessment>')[0].strip()

    # Display the final content (without assessment)
    # Strip draft tags for streaming display to match post-rerun appearance
    streaming_content = strip_draft_tags_for_streaming(main_content)
    response_placeholder.markdown(streaming_content)

    # Return the main content (without analysis or assessment), analysis, None for assessment, and thinking
    return main_content, analysis_content, None, thinking_content

def save_message_log(messages, rubric, analysis=None):
    """Save conversation to Supabase database"""
    project_id = st.session_state.get('current_project_id')
    if not project_id:
        raise ValueError("No project selected. Please select a project first.")

    supabase = st.session_state.get('supabase')
    if not supabase:
        raise ValueError("Database connection not available.")

    # Save to Supabase
    conv_id = save_conversation(supabase, project_id, messages, rubric, analysis or "")
    # Invalidate conversations cache so the list refreshes
    cache_key = f"conversations_{project_id}"
    if cache_key in st.session_state:
        del st.session_state[cache_key]
    return conv_id

# ========================
# Rubric Management Functions
# ========================
def load_rubric_history(force_reload=False):
    """Load rubric history from Supabase database with caching"""
    project_id = st.session_state.get('current_project_id')
    if not project_id:
        return []

    supabase = st.session_state.get('supabase')
    if not supabase:
        return []

    # Use cached version if available and not forcing reload
    cache_key = f"rubric_history_{project_id}"
    if not force_reload and cache_key in st.session_state:
        return st.session_state[cache_key]

    # Load from database and cache
    history = db_load_rubric_history(supabase, project_id)
    st.session_state[cache_key] = history
    return history


def invalidate_rubric_cache():
    """Clear the rubric history cache to force reload on next access"""
    project_id = st.session_state.get('current_project_id')
    if project_id:
        cache_key = f"rubric_history_{project_id}"
        if cache_key in st.session_state:
            del st.session_state[cache_key]

def save_rubric_history(history):
    """Save rubric history to Supabase database (saves only the latest version)"""
    project_id = st.session_state.get('current_project_id')
    if not project_id:
        st.error("No project selected. Please select a project first.")
        return

    supabase = st.session_state.get('supabase')
    if not supabase:
        st.error("Database connection not available.")
        return

    # Save the latest rubric version
    if history:
        latest = history[-1]
        db_save_rubric_history(supabase, project_id, latest)
        # Invalidate cache after saving - force reload to get the database-generated ID
        invalidate_rubric_cache()
        # Reload from database to get the proper IDs
        load_rubric_history(force_reload=True)

def next_version_number():
    """Get the next version number for a new rubric"""
    hist = load_rubric_history()
    if not hist:
        return 1
    return max(r.get("version", 1) for r in hist) + 1

def load_general_rubrics():
    """Load general rubrics from the general_rubrics folder.
    Returns a dict mapping display names to rubric data.
    """
    general_rubrics_dir = Path("general_rubrics")
    rubrics = {}

    if general_rubrics_dir.exists():
        for rubric_file in general_rubrics_dir.glob("*.json"):
            try:
                with open(rubric_file, 'r', encoding='utf-8') as f:
                    rubric_data = json.load(f)
                    # Create a friendly display name from the filename
                    display_name = rubric_file.stem.replace('_', ' ').title()
                    rubrics[display_name] = rubric_data
            except Exception as e:
                st.warning(f"Could not load {rubric_file.name}: {e}")

    return rubrics

def _rubric_to_json_serializable(obj):
    """Return a deep copy of obj with sets converted to lists so it can be JSON-serialized."""
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, dict):
        return {k: _rubric_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rubric_to_json_serializable(v) for v in obj]
    return obj


def get_active_rubric():
    """Get the active rubric and its index
    Returns: (full_rubric_dict, active_idx, rubric_history)
    where full_rubric_dict contains both 'version' and 'rubric' keys
    """
    hist = load_rubric_history()
    if not hist:
        return None, None, []

    idx = st.session_state.get("active_rubric_idx")
    if idx is None:
        idx = len(hist) - 1
    if 0 <= idx < len(hist):
        return hist[idx], idx, hist
    return hist[-1] if hist else None, len(hist) - 1 if hist else 0, hist

def _build_conversation_text(messages):
    """Build numbered conversation text from messages for rubric/DP prompts."""
    conversation_text = ""
    msg_num = 1
    for msg in messages:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        # Synthetic changelog messages are context-only — include but don't number
        if msg.get('_synthetic_changelog'):
            conversation_text += f"\n\n[Rubric Version Change]\n{content}"
            continue
        if role == 'user':
            conversation_text += f"\n\n[Message #{msg_num}] USER:\n{content}"
            msg_num += 1
        elif role == 'assistant':
            conversation_text += f"\n\n[Message #{msg_num}] ASSISTANT:\n{content}"
            msg_num += 1
    return conversation_text


def infer_rubric_only(messages):
    """Infer a rubric from conversation WITHOUT extracting decision points.

    Step 1 of the 5-step flow. DPs are extracted separately in step 3.
    Returns:
        dict with rubric data (no 'inference_decision_points'), or None on failure.
    """
    conversation_text = _build_conversation_text(messages)

    # Get the current active rubric to build upon
    active_rubric_dict, _, _ = get_active_rubric()
    previous_rubric = active_rubric_dict.get("rubric", []) if active_rubric_dict else []
    previous_rubric_version = active_rubric_dict.get("version", 1) if active_rubric_dict else 1

    if previous_rubric:
        st.info(f"Using previous rubric v{previous_rubric_version}")
        previous_rubric_json = json.dumps(previous_rubric, ensure_ascii=False, indent=2)
    else:
        st.info("No previous rubric found - creating new rubric from scratch")
        previous_rubric_json = ""

    system_prompt = RUBRIC_INFER_ONLY_SYSTEM_PROMPT
    user_prompt = RUBRIC_infer_only_user_prompt(conversation_text, previous_rubric_json)

    max_retries = 3
    retry_delay = 5
    progress_placeholder = st.empty()

    for attempt in range(max_retries):
        try:
            thinking_text = ""
            response_text = ""

            if attempt > 0:
                progress_placeholder.info(f"Retry attempt {attempt + 1} of {max_retries}...")

            with client.messages.stream(
                model=MODEL_PRIMARY,
                max_tokens=32000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                thinking={"type": "enabled", "budget_tokens": 10000}
            ) as stream:
                for event in stream:
                    if event.type == "content_block_delta":
                        if hasattr(event.delta, 'thinking'):
                            thinking_text += event.delta.thinking
                        elif hasattr(event.delta, 'text'):
                            response_text += event.delta.text

            progress_placeholder.empty()
            response_text = response_text.strip()

            json_match = re.search(r'\{.*"rubric".*\}', response_text, re.DOTALL)
            if json_match:
                rubric_data = json.loads(json_match.group())
            else:
                rubric_data = json.loads(response_text)

            rubric_data["version"] = next_version_number()
            rubric_data["source"] = "inferred"

            # Save to rubric history and activate
            rubric_history = load_rubric_history()
            rubric_history.append(rubric_data)
            save_rubric_history(rubric_history)
            st.session_state.active_rubric_idx = len(rubric_history) - 1
            invalidate_rubric_cache()

            rubric_list = rubric_data.get("rubric", [])
            st.session_state.rubric = rubric_list
            st.session_state.editing_criteria = copy.deepcopy(rubric_list)

            return rubric_data

        except Exception as e:
            error_str = str(e)
            if 'overloaded' in error_str.lower():
                if attempt < max_retries - 1:
                    countdown_placeholder = st.empty()
                    for remaining in range(retry_delay, 0, -1):
                        countdown_placeholder.warning(
                            f"Claude's servers are currently experiencing high demand. "
                            f"Retrying in {remaining} seconds... (Attempt {attempt + 1} of {max_retries})"
                        )
                        time.sleep(1)
                    countdown_placeholder.empty()
                    continue
                else:
                    st.error("Claude's servers are currently overloaded. Please try again in a few minutes.")
                    return None
            else:
                st.error(f"Error inferring rubric: {error_str}")
                return None

    return None


def extract_decision_points(messages, rubric_json, classification_feedback):
    """Extract decision points from conversation with classification context.

    Step 3 of the 5-step flow. Called after classification is confirmed.
    Args:
        messages: Conversation messages (same as infer_dp_messages)
        rubric_json: JSON string of the current rubric criteria
        classification_feedback: dict with 'classifications' (criterion_name -> stated/real/hallucinated)
    Returns:
        dict with 'parsed_data' containing decision_points, or None on failure.
    """
    conversation_text = _build_conversation_text(messages)
    classification_feedback_json = json.dumps(classification_feedback, ensure_ascii=False, indent=2)

    system_prompt = RUBRIC_EXTRACT_DPS_SYSTEM_PROMPT
    user_prompt = RUBRIC_extract_dps_user_prompt(conversation_text, rubric_json, classification_feedback_json)

    max_retries = 3
    retry_delay = 5
    progress_placeholder = st.empty()

    for attempt in range(max_retries):
        try:
            thinking_text = ""
            response_text = ""

            if attempt > 0:
                progress_placeholder.info(f"Retry attempt {attempt + 1} of {max_retries}...")

            with client.messages.stream(
                model=MODEL_PRIMARY,
                max_tokens=16000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                thinking={"type": "enabled", "budget_tokens": 10000}
            ) as stream:
                for event in stream:
                    if event.type == "content_block_delta":
                        if hasattr(event.delta, 'thinking'):
                            thinking_text += event.delta.thinking
                        elif hasattr(event.delta, 'text'):
                            response_text += event.delta.text

            progress_placeholder.empty()
            response_text = response_text.strip()

            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                parsed = json.loads(json_match.group())
            else:
                parsed = json.loads(response_text)

            # Ensure parsed_data wrapper
            if "decision_points" in parsed:
                return {"parsed_data": parsed}
            elif "parsed_data" in parsed:
                return parsed
            else:
                return {"parsed_data": {"decision_points": []}}

        except Exception as e:
            error_str = str(e)
            if ('overloaded' in error_str.lower() or 'rate' in error_str.lower()) and attempt < max_retries - 1:
                countdown_placeholder = st.empty()
                for remaining in range(retry_delay, 0, -1):
                    countdown_placeholder.warning(f"Servers busy. Retrying in {remaining}s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(1)
                countdown_placeholder.empty()
                continue
            else:
                st.error(f"Error extracting decision points: {error_str}")
                return None

    return None


def infer_final_rubric(messages, rubric_json, classification_feedback_json, corrected_dps_json, coldstart_text=""):
    """Infer the final rubric incorporating ALL accumulated feedback.

    Step 5 of the 5-step flow. Called after DP confirmation when hallucinated
    criteria existed or DP corrections were made.
    Args:
        messages: Conversation messages (same as infer_dp_messages)
        rubric_json: JSON string of the current rubric criteria
        classification_feedback_json: JSON string of classification feedback
        corrected_dps_json: JSON string of corrected DPs
        coldstart_text: User's cold-start preference description (optional)
    Returns:
        dict with rubric data + _change_explanation + _refinement_summary, or None.
    """
    conversation_text = _build_conversation_text(messages)

    system_prompt = RUBRIC_FINAL_INFER_SYSTEM_PROMPT
    user_prompt = RUBRIC_final_infer_user_prompt(
        conversation_text, rubric_json, classification_feedback_json,
        corrected_dps_json, coldstart_text
    )

    max_retries = 3
    retry_delay = 10
    progress_placeholder = st.empty()

    for attempt in range(max_retries):
        try:
            thinking_text = ""
            response_text = ""

            if attempt > 0:
                progress_placeholder.info(f"Retry attempt {attempt + 1} of {max_retries}...")

            with client.messages.stream(
                model=MODEL_PRIMARY,
                max_tokens=32000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                thinking={"type": "enabled", "budget_tokens": 10000}
            ) as stream:
                for event in stream:
                    if event.type == "content_block_delta":
                        if hasattr(event.delta, 'thinking'):
                            thinking_text += event.delta.thinking
                        elif hasattr(event.delta, 'text'):
                            response_text += event.delta.text

            progress_placeholder.empty()
            response_text = response_text.strip()

            json_match = re.search(r'\{.*"rubric".*\}', response_text, re.DOTALL)
            if json_match:
                rubric_data = json.loads(json_match.group())
            else:
                rubric_data = json.loads(response_text)

            # Extract explanations before cleaning
            change_explanation = rubric_data.pop("change_explanation", "")
            refinement_summary = rubric_data.get("refinement_summary", "")

            rubric_data["version"] = next_version_number()
            rubric_data["source"] = "inferred_final"

            # Attach explanation so caller can display it
            rubric_data["_change_explanation"] = change_explanation
            rubric_data["_refinement_summary"] = refinement_summary

            # Save to rubric history and activate
            rubric_history = load_rubric_history()
            rubric_history.append(rubric_data)
            save_rubric_history(rubric_history)
            st.session_state.active_rubric_idx = len(rubric_history) - 1
            invalidate_rubric_cache()

            rubric_list = rubric_data.get("rubric", [])
            st.session_state.rubric = rubric_list
            st.session_state.editing_criteria = copy.deepcopy(rubric_list)

            return rubric_data

        except Exception as e:
            error_str = str(e)
            if ('overloaded' in error_str.lower() or 'rate' in error_str.lower()) and attempt < max_retries - 1:
                countdown_placeholder = st.empty()
                for remaining in range(retry_delay, 0, -1):
                    countdown_placeholder.warning(f"Servers busy. Retrying in {remaining}s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(1)
                countdown_placeholder.empty()
                continue
            else:
                st.error(f"Error inferring final rubric: {error_str}")
                return None

    return None


# Page configuration
st.set_page_config(
    page_title="AI Co-Writer",
    page_icon="✍️",
    layout="wide"
)

# Floating scroll-to-top / scroll-to-bottom buttons — injected into parent document
import streamlit.components.v1 as _scroll_components
_scroll_components.html("""
<script>
(function() {
    var doc = window.parent.document;
    // Avoid duplicate injection on reruns
    if (doc.getElementById('st-scroll-buttons')) return;

    var css = doc.createElement('style');
    css.textContent = `
        #st-scroll-buttons {
            position: fixed;
            bottom: 24px;
            right: 24px;
            z-index: 9999;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        #st-scroll-buttons button {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            border: 1px solid rgba(150,150,150,0.3);
            background: rgba(255,255,255,0.92);
            color: #444;
            font-size: 16px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
            transition: background 0.2s;
        }
        #st-scroll-buttons button:hover { background: rgba(230,230,230,1); }
    `;
    doc.head.appendChild(css);

    var wrap = doc.createElement('div');
    wrap.id = 'st-scroll-buttons';

    var btnUp = doc.createElement('button');
    btnUp.innerHTML = '&#9650;';
    btnUp.title = 'Scroll to top';

    var btnDown = doc.createElement('button');
    btnDown.innerHTML = '&#9660;';
    btnDown.title = 'Scroll to bottom';

    function findScroller() {
        var selectors = [
            'section.main',
            '[data-testid="stMain"]',
            '[data-testid="stAppViewContainer"]',
        ];
        for (var i = 0; i < selectors.length; i++) {
            var el = doc.querySelector(selectors[i]);
            if (el && el.scrollHeight > el.clientHeight) return el;
        }
        return doc.documentElement;
    }

    btnUp.addEventListener('click', function() {
        findScroller().scrollTo({top: 0, behavior: 'smooth'});
    });
    btnDown.addEventListener('click', function() {
        var s = findScroller();
        s.scrollTo({top: s.scrollHeight, behavior: 'smooth'});
    });

    wrap.appendChild(btnUp);
    wrap.appendChild(btnDown);
    doc.body.appendChild(wrap);
})();
</script>
""", height=0)

# ========================
# Authentication Setup (Supabase)
# ========================
init_auth_state()

# Get Supabase client
supabase = get_supabase_client()

if supabase is None:
    st.error("Supabase not configured. Please set SUPABASE_URL and SUPABASE_KEY in your environment or Streamlit secrets.")
    st.markdown("### Setup Instructions")
    st.markdown("""
    1. Create a free account at [supabase.com](https://supabase.com)
    2. Create a new project
    3. Go to **Settings > API** to get your URL and anon key
    4. Add to your `.streamlit/secrets.toml`:
    ```toml
    SUPABASE_URL = "your-project-url"
    SUPABASE_KEY = "your-anon-key"
    ```
    5. Run the database schema (click below to copy):
    """)
    with st.expander("Database Schema SQL"):
        st.code(get_schema_sql(), language="sql")
    st.stop()

# Store supabase client in session state for use throughout the app
st.session_state.supabase = supabase

# Check if user is authenticated
if not is_authenticated():
    st.title("AI-Rubric Writer")
    st.markdown("Please login or register to continue.")

    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login", use_container_width=True)

            if submit:
                if email and password:
                    success, message = login_user(supabase, email, password)
                    if success:
                        st.success(message)
                        # Refresh the supabase client with the new session
                        st.session_state.supabase = get_supabase_client()
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("Please enter both email and password.")

    with tab_register:
        with st.form("register_form"):
            reg_name = st.text_input("Full Name")
            reg_email = st.text_input("Email")
            reg_password = st.text_input("Password", type="password")
            reg_password_confirm = st.text_input("Confirm Password", type="password")
            reg_submit = st.form_submit_button("Register", use_container_width=True)

            if reg_submit:
                if not reg_name or not reg_email or not reg_password:
                    st.warning("Please fill in all fields.")
                elif reg_password != reg_password_confirm:
                    st.error("Passwords do not match.")
                elif len(reg_password) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    success, message = register_user(supabase, reg_email, reg_password, reg_name)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)

    st.stop()

# User is authenticated - get user info
current_user = get_current_user()
st.session_state.auth_username = current_user["id"]
st.session_state.auth_name = current_user["name"]
st.session_state.auth_email = current_user["email"]

# Ensure supabase client has the current session
st.session_state.supabase = get_supabase_client()

# Show logout button in sidebar
with st.sidebar:
    st.write(f'Welcome, **{current_user["name"]}**')
    if st.button("Logout", use_container_width=True):
        logout_user(supabase)
        st.rerun()
    st.markdown("---")

# Inject CSS styles for diff highlighting
st.markdown("""
<style>
.diff-container {
    font-family: system-ui, -apple-system, sans-serif;
    line-height: 1.6;
}
.diff-add {
    background-color: #d4edda;
    color: #155724;
    padding: 2px 4px;
    border-radius: 3px;
    font-weight: 500;
}
.diff-del {
    background-color: #f8d7da;
    color: #721c24;
    text-decoration: line-through;
    padding: 2px 4px;
    border-radius: 3px;
}
.diff-container p {
    margin: 0.5em 0;
}

/* Keep chat input at bottom */
div[data-testid="chatInputContainer"] {
    position: fixed !important;
    bottom: 0;
    left: 0;
    right: 0;
    z-index: 999;
    background: white;
    padding: 1rem;
    box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
}

/* Ensure chat input appears above all other content */
div[data-testid="chatInputContainer"] {
    z-index: 999 !important;
}

/* Add padding to main content area to prevent overlap */
main[data-testid="stMain"] {
    padding-bottom: 150px !important;
}

/* Also add padding to main block content */
.block-container {
    padding-bottom: 150px !important;
}

/* Ensure chat messages don't overlap the fixed input */
.stChatMessage {
    z-index: 1 !important;
}

/* Style for new criteria containers */
.criterion-container-new {
    border: 2px solid #FF9800 !important;
    border-radius: 6px !important;
    padding: 4px !important;
    margin-bottom: 8px !important;
    background-color: rgba(255, 152, 0, 0.1) !important;
}

.criterion-container-existing {
    border: 1px solid #2196F3 !important;
    border-radius: 6px !important;
    padding: 4px !important;
    margin-bottom: 8px !important;
    background-color: rgba(33, 150, 243, 0.05) !important;
}

/* Styles for rubric comparison diff highlighting */
.diff-wrap {
    font-family: monospace;
    line-height: 1.4;
}
.diff-wrap .add {
    background-color: #d4edda;
    color: #155724;
    padding: 1px 3px;
    border-radius: 2px;
    font-weight: bold;
}
.diff-wrap .del {
    background-color: #f8d7da;
    color: #721c24;
    text-decoration: line-through;
    padding: 1px 3px;
    border-radius: 2px;
}
.diff-wrap p {
    margin: 0.5em 0;
}

/* Make tabs horizontally scrollable on small screens */
div[data-baseweb="tab-list"] {
    overflow-x: auto !important;
    overflow-y: hidden !important;
    flex-wrap: nowrap !important;
    scrollbar-width: thin;
    -webkit-overflow-scrolling: touch;
}

div[data-baseweb="tab-list"]::-webkit-scrollbar {
    height: 6px;
}

div[data-baseweb="tab-list"]::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 3px;
}

div[data-baseweb="tab-list"]::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 3px;
}

div[data-baseweb="tab-list"]::-webkit-scrollbar-thumb:hover {
    background: #a1a1a1;
}

/* Prevent tabs from wrapping */
button[data-baseweb="tab"] {
    white-space: nowrap !important;
    flex-shrink: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'rubric' not in st.session_state:
    st.session_state.rubric = None

if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = ""

if 'current_rubric_assessment' not in st.session_state:
    st.session_state.current_rubric_assessment = None

# Initialize current_project early so other functions can use it
if 'current_project' not in st.session_state:
    st.session_state.current_project = None  # Project name
if 'current_project_id' not in st.session_state:
    st.session_state.current_project_id = None  # Project UUID from Supabase
if 'survey_responses' not in st.session_state:
    st.session_state.survey_responses = {"task_a": {}, "task_b": {}, "final": {}}

if 'selected_conversation' not in st.session_state:
    st.session_state.selected_conversation = None

if 'active_rubric_idx' not in st.session_state:
    hist = load_rubric_history()
    st.session_state.active_rubric_idx = len(hist) - 1 if hist else None

# Comparison mode
if 'comparison_result' not in st.session_state:
    st.session_state.comparison_result = None  # Store regenerated response for comparison
if 'comparison_rubric_version' not in st.session_state:
    st.session_state.comparison_rubric_version = None  # Which rubric version was used for comparison

# Rubric comparison results (for Compare Rubrics tab)
if 'rubric_comparison_results' not in st.session_state:
    st.session_state.rubric_comparison_results = None

# Message deletion mode for removing unwanted messages from conversation
if 'message_delete_mode' not in st.session_state:
    st.session_state.message_delete_mode = False  # Whether delete mode is active
if 'messages_to_delete' not in st.session_state:
    st.session_state.messages_to_delete = set()  # Set of message indices to delete

# A/B draft comparison: every N-th draft, show rubric-guided vs conversation-only side-by-side
if 'ab_draft_count' not in st.session_state:
    st.session_state.ab_draft_count = 0  # Counts assistant messages containing <draft> tags
if 'ab_comparison_pending' not in st.session_state:
    st.session_state.ab_comparison_pending = None  # Dict with both drafts when A/B triggered

# Layer 1: Silent A/B LLM-as-judge aggregation
if 'ab_judge_results' not in st.session_state:
    st.session_state.ab_judge_results = []  # List of {timestamp, user_chosen, rubric_judge, coldstart_judge, generic_judge, ...}

# Layer 2: Ranking checkpoint state
if 'ranking_checkpoint_results' not in st.session_state:
    st.session_state.ranking_checkpoint_results = []  # List of completed checkpoint results
if 'ranking_checkpoint_pending' not in st.session_state:
    st.session_state.ranking_checkpoint_pending = None  # {step: 1|2|3, writing_task, drafts, shuffle_order}
if 'ranking_checkpoint_auto_triggered' not in st.session_state:
    st.session_state.ranking_checkpoint_auto_triggered = False


# Evaluate: Coverage tab state (9-step workflow)
# Evaluate: Infer tab state (11-step workflow)
if 'infer_coldstart_text' not in st.session_state:
    st.session_state.infer_coldstart_text = ""  # Step 1: User's cold-start preference description
if 'infer_coldstart_saved' not in st.session_state:
    st.session_state.infer_coldstart_saved = False  # Whether Step 1 submitted
if 'infer_user_categorizations' not in st.session_state:
    st.session_state.infer_user_categorizations = {}  # Steps 2-3: {"Criterion Name": "stated"|"real"|"hallucinated"}
if 'infer_categorizations_complete' not in st.session_state:
    st.session_state.infer_categorizations_complete = False  # Whether all criteria categorized
if 'infer_behavioral_result' not in st.session_state:
    st.session_state.infer_behavioral_result = None  # Step 5: LLM behavioral evidence (parsed JSON)
if 'infer_dp_conversation' not in st.session_state:
    st.session_state.infer_dp_conversation = None  # Step 4: Selected conversation for decision points
if 'infer_decision_points' not in st.session_state:
    st.session_state.infer_decision_points = None  # Step 4: Extracted decision points
if 'infer_all_conversations' not in st.session_state:
    st.session_state.infer_all_conversations = []  # List of {messages, decision_points, timestamp, rubric_version}
if 'infer_expanded_dp' not in st.session_state:
    st.session_state.infer_expanded_dp = None  # Step 5: Currently expanded decision point ID
if 'infer_dp_dimension_confirmed' not in st.session_state:
    st.session_state.infer_dp_dimension_confirmed = False  # Step 5: Whether user confirmed dimension mappings
if 'infer_dp_user_mappings' not in st.session_state:
    st.session_state.infer_dp_user_mappings = {}  # Step 5: User-confirmed dimension mappings {dp_id: {"criterion": name, "not_in_rubric_reason": str|None}}
if 'infer_step6_generated_task' not in st.session_state:
    st.session_state.infer_step6_generated_task = None
if 'infer_step6_writing_task' not in st.session_state:
    st.session_state.infer_step6_writing_task = ""
if 'infer_step6_auto_gen_done' not in st.session_state:
    st.session_state.infer_step6_auto_gen_done = False
if 'infer_step6_custom_task_key_version' not in st.session_state:
    st.session_state.infer_step6_custom_task_key_version = 0
if 'infer_step6_drafts' not in st.session_state:
    st.session_state.infer_step6_drafts = None  # {"r_star": str, "r1": str|None, "r0": str, "coldstart": str, "generic": str}
if 'infer_step6_draft_labels' not in st.session_state:
    st.session_state.infer_step6_draft_labels = None  # ordered list of source keys matching A/B/C/D/E shuffle
if 'infer_step6_rubric_versions_used' not in st.session_state:
    st.session_state.infer_step6_rubric_versions_used = None  # {"r_star": ver, "r1": ver|None, "r0": ver}
if 'infer_step6_blind_ratings' not in st.session_state:
    st.session_state.infer_step6_blind_ratings = None  # {"A": 1-5, ...}
if 'infer_step6_user_ranking' not in st.session_state:
    st.session_state.infer_step6_user_ranking = None  # ordered list most→least preferred
if 'infer_step6_user_dimension_checks' not in st.session_state:
    st.session_state.infer_step6_user_dimension_checks = None  # Only for R*, coldstart, generic
if 'infer_step6_llm_evaluations' not in st.session_state:
    st.session_state.infer_step6_llm_evaluations = None
if 'infer_step6_survey' not in st.session_state:
    st.session_state.infer_step6_survey = None  # {"accuracy": str, "rounds_needed": str}
if 'infer_step6_claim2_metrics' not in st.session_state:
    st.session_state.infer_step6_claim2_metrics = None
if 'infer_step6_claim3_metrics' not in st.session_state:
    st.session_state.infer_step6_claim3_metrics = None
if 'infer_pending_rubric' not in st.session_state:
    st.session_state.infer_pending_rubric = None  # Infer tab: Pending inferred rubric awaiting user review

# Chat tab: Criteria classification (Steps 1+2 integrated from Infer tab)
if 'chat_criteria_llm_classification' not in st.session_state:
    st.session_state.chat_criteria_llm_classification = None  # LLM comparison result dict
if 'chat_criteria_user_classifications' not in st.session_state:
    st.session_state.chat_criteria_user_classifications = {}  # {criterion_name: "stated"|"real"|"hallucinated"}
if 'chat_criteria_review_active' not in st.session_state:
    st.session_state.chat_criteria_review_active = False  # True while review UI is showing
if 'chat_criteria_review_confirmed' not in st.session_state:
    st.session_state.chat_criteria_review_confirmed = False  # True after user confirms
if 'chat_classification_feedback' not in st.session_state:
    st.session_state.chat_classification_feedback = {}  # Stored after classification confirm
if 'chat_criteria_hallucination_reasons' not in st.session_state:
    st.session_state.chat_criteria_hallucination_reasons = {}

# Evaluate: Build tab state (5-step workflow)
if 'build_rubric_a_idx' not in st.session_state:
    st.session_state.build_rubric_a_idx = None  # Step 1: Index of Rubric A in history
if 'build_rubric_b_idx' not in st.session_state:
    st.session_state.build_rubric_b_idx = None  # Step 1: Index of Rubric B in history
if 'build_edit_classification' not in st.session_state:
    st.session_state.build_edit_classification = None  # Step 2: Structured diff result
if 'build_writing_task' not in st.session_state:
    st.session_state.build_writing_task = ""  # Step 3: User's writing task description
if 'build_draft_a' not in st.session_state:
    st.session_state.build_draft_a = None  # Step 3: Draft from rubric A
if 'build_draft_b' not in st.session_state:
    st.session_state.build_draft_b = None  # Step 3: Draft from rubric B
if 'build_draft_a_thinking' not in st.session_state:
    st.session_state.build_draft_a_thinking = ""  # Step 3: Thinking from draft A
if 'build_draft_b_thinking' not in st.session_state:
    st.session_state.build_draft_b_thinking = ""  # Step 3: Thinking from draft B
if 'build_blind_labels' not in st.session_state:
    st.session_state.build_blind_labels = None  # Step 3: {"Draft X": "a", "Draft Y": "b"}
if 'build_user_preference' not in st.session_state:
    st.session_state.build_user_preference = None  # Step 3: User's blind preference + per-dimension ratings
if 'build_llm_judge_result' not in st.session_state:
    st.session_state.build_llm_judge_result = None  # Step 4: LLM judge per-dimension scores
if 'build_llm_judge_thinking' not in st.session_state:
    st.session_state.build_llm_judge_thinking = ""  # Step 4: LLM judge thinking
if 'build_self_report' not in st.session_state:
    st.session_state.build_self_report = {}  # Step 5: User's self-report responses
if 'build_self_report_saved' not in st.session_state:
    st.session_state.build_self_report_saved = False  # Step 5: Whether self-report submitted

# Evaluate: Grade tab state (5-step workflow)
if 'grade_writing_task' not in st.session_state:
    st.session_state.grade_writing_task = ""  # Step 1: User's writing task description
if 'grade_violated_dims' not in st.session_state:
    st.session_state.grade_violated_dims = None  # Step 1: List of dimension names selected for violation
if 'grade_draft_good' not in st.session_state:
    st.session_state.grade_draft_good = None  # Step 1: Draft following full rubric
if 'grade_draft_degraded' not in st.session_state:
    st.session_state.grade_draft_degraded = None  # Step 1: Draft with violated dimensions
if 'grade_draft_good_thinking' not in st.session_state:
    st.session_state.grade_draft_good_thinking = ""  # Step 1: Thinking from good draft
if 'grade_draft_degraded_thinking' not in st.session_state:
    st.session_state.grade_draft_degraded_thinking = ""  # Step 1: Thinking from degraded draft
if 'grade_blind_labels' not in st.session_state:
    st.session_state.grade_blind_labels = None  # Step 1: {"Draft X": "good"|"degraded", ...}
if 'grade_user_overall_pref' not in st.session_state:
    st.session_state.grade_user_overall_pref = None  # Step 2: User's overall preference
if 'grade_user_dim_ratings' not in st.session_state:
    st.session_state.grade_user_dim_ratings = {}  # Step 2: Per-dimension ratings for both drafts
if 'grade_rubric_judge_result' not in st.session_state:
    st.session_state.grade_rubric_judge_result = None  # Step 3: Rubric-grounded judge result
if 'grade_rubric_judge_thinking' not in st.session_state:
    st.session_state.grade_rubric_judge_thinking = ""  # Step 3: Rubric-grounded thinking
if 'grade_generic_judge_result' not in st.session_state:
    st.session_state.grade_generic_judge_result = None  # Step 3: Generic judge result
if 'grade_generic_judge_thinking' not in st.session_state:
    st.session_state.grade_generic_judge_thinking = ""  # Step 3: Generic thinking
if 'grade_agreement_results' not in st.session_state:
    st.session_state.grade_agreement_results = None  # Step 4: Computed correlations
if 'grade_saved' not in st.session_state:
    st.session_state.grade_saved = False  # Step 5: Whether results saved

# Alignment tab state
if 'alignment_selected_conversation' not in st.session_state:
    st.session_state.alignment_selected_conversation = None  # Selected conversation file
if 'alignment_selected_draft_idx' not in st.session_state:
    st.session_state.alignment_selected_draft_idx = None  # Index of selected draft in conversation
if 'alignment_draft_content' not in st.session_state:
    st.session_state.alignment_draft_content = None  # The actual draft text
if 'alignment_user_scores' not in st.session_state:
    st.session_state.alignment_user_scores = {}  # User's scores: {criterion_idx: score}
if 'alignment_llm_scores' not in st.session_state:
    st.session_state.alignment_llm_scores = None  # LLM's scores for comparison
if 'alignment_results' not in st.session_state:
    st.session_state.alignment_results = None  # Computed alignment metrics
if 'alignment_evidence_highlights' not in st.session_state:
    st.session_state.alignment_evidence_highlights = []  # Evidence highlights from LLM scoring

# Initialize rubric from active version if not set
if st.session_state.rubric is None and st.session_state.active_rubric_idx is not None:
    active_rubric_dict, _, _ = get_active_rubric()
    if active_rubric_dict:
        rubric_list = active_rubric_dict.get("rubric", [])
        st.session_state.rubric = rubric_list

def simple_markdown_to_html(text):
    """Convert simple markdown formatting to HTML"""
    import re
    # Bold: **text** or __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)
    # Italic: *text* or _text_
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    text = re.sub(r'_(.+?)_', r'<em>\1</em>', text)
    # Line breaks
    text = text.replace('\n', '<br>')
    return text

def classify_rubric_edits(old_rubric_list, new_rubric_list):
    """
    Diff two rubric criteria lists and classify every change by type.
    Returns a structured dict of edits.
    """
    old_map = {}
    for c in (old_rubric_list or []):
        key = c.get('name', '').lower().strip()
        old_map[key] = c

    new_map = {}
    for c in (new_rubric_list or []):
        key = c.get('name', '').lower().strip()
        new_map[key] = c

    edits = {
        "added": [],
        "removed": [],
        "reweighted": [],
        "reworded": [],
        "dimensions_changed": []
    }

    # Find added and modified criteria
    for key, new_c in new_map.items():
        if key not in old_map:
            edits["added"].append({
                "name": new_c.get("name", ""),
                "description": new_c.get("description", ""),
                "weight": new_c.get("weight", new_c.get("priority", 0))
            })
        else:
            old_c = old_map[key]
            # Check weight/priority change
            old_w = old_c.get('weight', old_c.get('priority', 0))
            new_w = new_c.get('weight', new_c.get('priority', 0))
            if old_w != new_w:
                edits["reweighted"].append({
                    "name": new_c.get("name", ""),
                    "old_weight": old_w,
                    "new_weight": new_w
                })
            # Check description change
            old_desc = old_c.get('description', '')
            new_desc = new_c.get('description', '')
            if old_desc != new_desc:
                edits["reworded"].append({
                    "name": new_c.get("name", ""),
                    "field": "description",
                    "old": old_desc,
                    "new": new_desc
                })
            # Check dimension changes
            old_dims = set(d.get('label', '').strip() for d in old_c.get('dimensions', []) if d.get('label', '').strip())
            new_dims = set(d.get('label', '').strip() for d in new_c.get('dimensions', []) if d.get('label', '').strip())
            if old_dims != new_dims:
                edits["dimensions_changed"].append({
                    "name": new_c.get("name", ""),
                    "added_dims": list(new_dims - old_dims),
                    "removed_dims": list(old_dims - new_dims)
                })

    # Find removed criteria
    for key, old_c in old_map.items():
        if key not in new_map:
            edits["removed"].append({
                "name": old_c.get("name", ""),
                "description": old_c.get("description", "")
            })

    return edits

def format_edit_log_message(edit_classification, old_version, new_version, source):
    """
    Format a rubric edit classification into a conversation log message.
    Contains both human-readable text and a machine-parseable HTML comment.
    """
    lines = [f"📋 **Rubric updated: v{old_version} → v{new_version}** (source: {source})"]

    edits = edit_classification
    if edits["added"]:
        for a in edits["added"]:
            lines.append(f"- **Added:** \"{a['name']}\"")
    if edits["removed"]:
        for r in edits["removed"]:
            lines.append(f"- **Removed:** \"{r['name']}\"")
    if edits["reweighted"]:
        for rw in edits["reweighted"]:
            lines.append(f"- **Reweighted:** \"{rw['name']}\" ({rw['old_weight']} → {rw['new_weight']})")
    if edits["reworded"]:
        for rw in edits["reworded"]:
            lines.append(f"- **Reworded:** \"{rw['name']}\" {rw['field']} changed")
    if edits["dimensions_changed"]:
        for dc in edits["dimensions_changed"]:
            parts = []
            if dc["added_dims"]:
                parts.append(f"+{', '.join(dc['added_dims'])}")
            if dc["removed_dims"]:
                parts.append(f"-{', '.join(dc['removed_dims'])}")
            lines.append(f"- **Dimensions:** \"{dc['name']}\" ({'; '.join(parts)})")

    if not any(edits[k] for k in edits):
        lines.append("- No substantive changes detected")

    # Machine-readable log embedded as HTML comment
    log_data = json.dumps({
        "version_from": old_version,
        "version_to": new_version,
        "edits": edit_classification,
        "source": source,
        "reverted": False
    })
    lines.append(f"<!--RUBRIC_EDIT_LOG:{log_data}-->")

    return "\n".join(lines)

def get_effective_edits_from_conversation(messages):
    """
    Walk conversation messages and extract rubric edit logs,
    excluding any that were subsequently reverted.
    Returns a list of edit log dicts (non-reverted only).
    """
    edit_logs = []  # list of (index, parsed_log_dict)
    reverted_versions = set()

    # First pass: collect all revert events
    for msg in messages:
        content = msg.get('content', '')
        if not isinstance(content, str):
            continue
        revert_match = re.search(r'<!--RUBRIC_REVERT_LOG:(.*?)-->', content)
        if revert_match:
            try:
                revert_data = json.loads(revert_match.group(1))
                reverted_to = revert_data.get("reverted_to_version")
                if reverted_to is not None:
                    reverted_versions.add(reverted_to)
            except json.JSONDecodeError:
                pass

    # Second pass: collect edit logs, mark reverted ones
    for msg in messages:
        content = msg.get('content', '')
        if not isinstance(content, str):
            continue
        edit_match = re.search(r'<!--RUBRIC_EDIT_LOG:(.*?)-->', content)
        if edit_match:
            try:
                log_data = json.loads(edit_match.group(1))
                # An edit is "reverted" if a later revert went back to the version before this edit
                version_from = log_data.get("version_from")
                if version_from in reverted_versions:
                    log_data["reverted"] = True
                edit_logs.append(log_data)
            except json.JSONDecodeError:
                pass

    # Return only non-reverted edits
    return [log for log in edit_logs if not log.get("reverted", False)]

def load_conversations(force_reload=False):
    """Load all conversations from Supabase database with caching"""
    project_id = st.session_state.get('current_project_id')
    if not project_id:
        return []

    supabase = st.session_state.get('supabase')
    if not supabase:
        return []

    cache_key = f"conversations_{project_id}"
    if not force_reload and cache_key in st.session_state:
        return st.session_state[cache_key]

    conversations = db_load_conversations(supabase, project_id)
    # Add filename field for compatibility with existing code
    for conv in conversations:
        conv["filename"] = conv["id"]
    st.session_state[cache_key] = conversations
    return conversations

def load_conversation_data(conversation_id):
    """Load conversation data from Supabase database"""
    supabase = st.session_state.get('supabase')
    if not supabase:
        return None

    return load_conversation_by_id(supabase, conversation_id)

def get_available_projects():
    """Get list of available projects for the current user from Supabase"""
    user_id = st.session_state.get('auth_username')
    if not user_id:
        return []

    supabase = st.session_state.get('supabase')
    if not supabase:
        return []

    projects = get_user_projects(supabase, user_id)
    return projects  # Returns list of dicts with 'id', 'name', etc.

def create_new_project(project_name):
    """Create a new project in Supabase database"""
    import string
    valid_chars = string.ascii_letters + string.digits + '-_ '
    if not all(c in valid_chars for c in project_name):
        return False, "Project name can only contain letters, numbers, hyphens, underscores, and spaces"

    user_id = st.session_state.get('auth_username')
    if not user_id:
        return False, "Not authenticated"

    supabase = st.session_state.get('supabase')
    if not supabase:
        return False, "Database connection not available"

    success, message, project_id = db_create_project(supabase, user_id, project_name)
    return success, message, project_id

# Title at the top
st.title("✍️ AI-Rubric Writer")
st.markdown("Collaborate with AI to improve your writing!")

# Create tabs (Evaluate: Build and Evaluate: Grade hidden)
SHOW_BUILD_GRADE_TABS = False
if SHOW_BUILD_GRADE_TABS:
    tab1, tab_survey, tab_infer, tab_grading, tab7, tab8, tab3, tab4 = st.tabs(["💬 Chat", "📋 Evaluate: Survey", "🔎 Evaluate: Infer", "📊 Evaluate: Grading", "🔨 Evaluate: Build", "📝 Evaluate: Grade", "📁 View Rubric", "🔍 Compare Rubrics"])
else:
    tab1, tab_survey, tab_infer, tab_grading, tab3, tab4 = st.tabs(["💬 Chat", "📋 Evaluate: Survey", "🔎 Evaluate: Infer", "📊 Evaluate: Grading", "📁 View Rubric", "🔍 Compare Rubrics"])
    tab7 = tab8 = None

with tab1:

    conversations = load_conversations()

    # Create selection with conversation details
    conversation_options = [("New Conversation", None)]
    if conversations:
        for conv in conversations:
            # Format timestamp for display
            try:
                dt = datetime.fromisoformat(conv["timestamp"])
                formatted_time = dt.strftime("%m/%d %H:%M")
                display = f"{formatted_time} ({conv['messages_count']} msgs)"
            except:
                display = f"{conv['timestamp']} ({conv['messages_count']} msgs)"
            conversation_options.append((display, conv["filename"]))

    # Create selectbox with current selection
    current_selection = st.session_state.selected_conversation or None
    options = [opt[1] for opt in conversation_options]

    # Find index of current selection
    current_index = 0
    if current_selection:
        try:
            current_index = options.index(current_selection)
        except ValueError:
            current_index = 0

    selected_file = st.selectbox(
        "💬 Select conversation:",
        options=options,
        format_func=lambda x: next(opt[0] for opt in conversation_options if opt[1] == x) if x is not None else "New Conversation",
        index=current_index,
        key="conversation_selector"
    )
    
    # Clear the just_saved flag if we're not transitioning conversations
    if selected_file == st.session_state.get("selected_conversation") and st.session_state.get("just_saved"):
        st.session_state.just_saved = None
    
    # Handle "New Conversation" selection
    if selected_file is None and st.session_state.selected_conversation is not None:
        # Explicitly switching FROM a loaded conversation TO "New Conversation"
        st.session_state.messages = []
        st.session_state.rubric = None
        st.session_state.current_analysis = ""
        st.session_state.selected_conversation = None
        st.session_state.comparison_result = None
        st.session_state.comparison_rubric_version = None
        st.session_state.ab_draft_count = 0
        st.session_state.ab_comparison_pending = None
        st.session_state.ranking_checkpoint_pending = None
        st.session_state.ranking_checkpoint_auto_triggered = False
        # Clear DP / inference state
        st.session_state.infer_decision_points = None
        st.session_state.infer_dp_dimension_confirmed = False
        st.session_state.infer_dp_user_mappings = {}
        st.session_state.dp_refinement_result = None
        st.session_state.chat_criteria_llm_classification = None
        st.session_state.chat_criteria_user_classifications = {}
        st.session_state.chat_criteria_review_active = False
        st.session_state.chat_criteria_review_confirmed = False
        st.session_state.chat_classification_feedback = {}
        st.session_state.chat_criteria_hallucination_reasons = {}
        st.rerun()
    elif selected_file and selected_file != st.session_state.selected_conversation:
        # Check if we just saved this conversation - if so, don't reload it
        just_saved = st.session_state.get("just_saved")
        if just_saved and selected_file == just_saved:
            # We just saved this conversation, so keep the current messages
            st.session_state.selected_conversation = selected_file
            st.session_state.just_saved = None  # Clear the flag
        else:
            # Load the selected conversation
            conv_data = load_conversation_data(selected_file)
            if conv_data:
                st.session_state.messages = conv_data.get("messages", [])
                st.session_state.rubric = conv_data.get("rubric", None)
                st.session_state.current_analysis = conv_data.get("analysis", "")  # Load saved analysis
                st.session_state.selected_conversation = selected_file
                # Clear DP / inference state
                st.session_state.infer_decision_points = None
                st.session_state.infer_dp_dimension_confirmed = False
                st.session_state.infer_dp_user_mappings = {}
                st.session_state.dp_refinement_result = None
                st.session_state.chat_criteria_llm_classification = None
                st.session_state.chat_criteria_user_classifications = {}
                st.session_state.chat_criteria_review_active = False
                st.session_state.chat_criteria_review_confirmed = False
                st.session_state.chat_classification_feedback = {}
                st.session_state.chat_criteria_hallucination_reasons = {}
                st.rerun()

    st.divider()

    # --- DP inline rendering setup ---
    import html as _dp_html_lib
    _dp_result = st.session_state.get("infer_decision_points")
    _dp_has_review = any(m.get('is_dp_review') for m in st.session_state.messages)
    _dp_confirmed = st.session_state.get("infer_dp_dimension_confirmed", False)
    # Gate DP display on criteria classification being confirmed (or not applicable)
    _cc_pending = st.session_state.chat_criteria_review_active and not st.session_state.chat_criteria_review_confirmed
    _dp_active = _dp_has_review and not _cc_pending
    _dp_list_all = []
    _dp_by_user_msg = {}  # {user_message_num: [dp, ...]}
    _dp_by_asst_msg = {}  # {assistant_message_num: [dp, ...]}
    _dp_crit_names = []

    if _dp_active and _dp_result:
        _dp_parsed = _dp_result.get("parsed_data", {})
        _dp_list_all = _dp_parsed.get("decision_points", [])
        # Build index: which DPs attach to which message number
        for _dp in _dp_list_all:
            _u_num = _dp.get("user_message_num")
            if _u_num:
                _dp_by_user_msg.setdefault(_u_num, []).append(_dp)
            _a_num = _dp.get("assistant_message_num")
            if _a_num:
                _dp_by_asst_msg.setdefault(_a_num, []).append(_dp)
        # Build criteria list from active rubric
        _dp_rb, _, _ = get_active_rubric()
        if _dp_rb:
            _dp_crit_names = [c.get("name", "") for c in _dp_rb.get("rubric", []) if c.get("name")]

    def _highlight_dp_quotes(text, dp_list, quote_key, color):
        """Highlight DP quotes in message text. Returns modified text with HTML highlights."""
        for dp in dp_list:
            quote = (dp.get(quote_key, '') or '').strip()
            if not quote or len(quote) < 5:
                continue
            dp_id = dp.get('id', 0)
            # Try exact match first, then case-insensitive
            if quote in text:
                highlight = f'<mark style="background:{color};padding:1px 3px;border-radius:3px;" title="DP#{dp_id}">{_dp_html_lib.escape(quote)}</mark>'
                text = text.replace(quote, highlight, 1)
            elif quote.lower() in text.lower():
                # Case-insensitive: find the position and replace preserving original case
                idx = text.lower().find(quote.lower())
                original = text[idx:idx+len(quote)]
                highlight = f'<mark style="background:{color};padding:1px 3px;border-radius:3px;" title="DP#{dp_id}">{_dp_html_lib.escape(original)}</mark>'
                text = text[:idx] + highlight + text[idx+len(quote):]
        return text

    def _chat_auto_match(dp):
        suggested = dp.get("suggested_criterion_name") or dp.get("related_rubric_criterion") or ""
        dim = dp.get("dimension", "")
        if suggested:
            for name in _dp_crit_names:
                if name.lower() == suggested.lower():
                    return name
            for name in _dp_crit_names:
                if name.lower() in suggested.lower() or suggested.lower() in name.lower():
                    return name
        for name in _dp_crit_names:
            if name.lower() in dim.lower() or dim.lower() in name.lower():
                return name
        return suggested or dim or None

    def _render_dp_card(dp):
        """Render a single DP card inline after the message it references."""
        dp_id = dp.get('id', 0)
        dp_id_str = str(dp_id)

        if _dp_confirmed:
            # Confirmed: compact view
            crit = dp.get("confirmed_criterion") or ""
            user_action = dp.get("user_action", "correct")
            if user_action == "correct":
                badge = f'<span style="background:#C8E6C9;padding:2px 8px;border-radius:10px;font-size:0.85em;">Confirmed: {_dp_html_lib.escape(crit)}</span>'
                border_color = "#4CAF50"
            elif user_action == "incorrect":
                orig = dp.get("original_suggestion") or dp.get("dimension", "")
                badge = f'<span style="background:#FFF9C4;padding:2px 8px;border-radius:10px;font-size:0.85em;">Remapped: {_dp_html_lib.escape(orig)} &rarr; {_dp_html_lib.escape(crit)}</span>'
                border_color = "#FF9800"
            elif user_action == "not_in_rubric":
                badge = '<span style="background:#FFCDD2;padding:2px 8px;border-radius:10px;font-size:0.85em;">Not in rubric</span>'
                border_color = "#F44336"
            else:
                badge = _dp_html_lib.escape(crit)
                border_color = "#1976D2"
            st.markdown(
                f'<div id="dp-card-{dp_id}" style="background:linear-gradient(135deg, #E3F2FD 0%, #F3E5F5 100%);'
                f'border-left:5px solid {border_color};padding:10px 14px;margin:8px 0;border-radius:6px;'
                f'font-size:0.9em;box-shadow:0 1px 3px rgba(0,0,0,0.12);scroll-margin-top:80px;">'
                f'<span style="background:#1976D2;color:white;padding:1px 8px;border-radius:10px;font-size:0.8em;font-weight:bold;margin-right:6px;">DP#{dp_id}</span> '
                f'{_dp_html_lib.escape(dp.get("dimension", ""))} &mdash; {badge}'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            # Unconfirmed: editable card with prominent styling
            auto_matched = _chat_auto_match(dp)
            existing_mapping = st.session_state.infer_dp_user_mappings.get(dp_id_str)
            if not existing_mapping and auto_matched:
                st.session_state.infer_dp_user_mappings[dp_id_str] = {"criterion": auto_matched, "not_in_rubric": False}
                existing_mapping = st.session_state.infer_dp_user_mappings[dp_id_str]

            title_crit = auto_matched or dp.get("suggested_criterion_name") or "Unmatched"

            # Render a colored banner above the expander
            st.markdown(
                f'<div id="dp-card-{dp_id}" style="background:linear-gradient(135deg, #E3F2FD 0%, #F3E5F5 100%);'
                f'border-left:5px solid #FF9800;padding:6px 14px;margin:8px 0 0 0;border-radius:6px 6px 0 0;'
                f'font-size:0.85em;box-shadow:0 1px 3px rgba(0,0,0,0.12);scroll-margin-top:80px;">'
                f'<span style="background:#FF9800;color:white;padding:1px 8px;border-radius:10px;font-size:0.8em;font-weight:bold;margin-right:6px;">DP#{dp_id}</span> '
                f'<b>{_dp_html_lib.escape(dp.get("dimension", "Unknown"))}</b> &rarr; {_dp_html_lib.escape(title_crit)} '
                f'<span style="color:#666;font-size:0.9em;">(click to review)</span>'
                f'</div>',
                unsafe_allow_html=True
            )
            with st.expander(f"Review DP#{dp_id}: {title_crit}", expanded=False):
                st.markdown(f"**{dp.get('summary', 'N/A')}**")

                # Determine default action
                if existing_mapping:
                    if existing_mapping.get("not_in_rubric"):
                        default_action_idx = 2
                    elif existing_mapping.get("criterion") != auto_matched and existing_mapping.get("criterion"):
                        default_action_idx = 1
                    else:
                        default_action_idx = 0
                else:
                    default_action_idx = 0

                action = st.radio(
                    f"DP#{dp_id} mapping:",
                    options=["Correct", "Incorrect", "Not in rubric"],
                    index=default_action_idx,
                    key=f"chat_dp_action_{dp_id}",
                    horizontal=True,
                    label_visibility="collapsed"
                )

                if action == "Correct":
                    if auto_matched:
                        st.session_state.infer_dp_user_mappings[dp_id_str] = {"criterion": auto_matched, "not_in_rubric": False}
                    else:
                        st.caption("No auto-match found. Select 'Incorrect' to choose a criterion.")
                elif action == "Incorrect":
                    default_idx = 0
                    if existing_mapping and existing_mapping.get("criterion") and existing_mapping["criterion"] in _dp_crit_names:
                        default_idx = _dp_crit_names.index(existing_mapping["criterion"])
                    elif auto_matched and auto_matched in _dp_crit_names:
                        default_idx = _dp_crit_names.index(auto_matched)

                    if _dp_crit_names:
                        corrected = st.selectbox(
                            f"Select correct criterion for DP#{dp_id}:",
                            options=_dp_crit_names,
                            index=default_idx,
                            key=f"chat_dp_correct_{dp_id_str}",
                        )
                    else:
                        corrected = st.text_input(
                            f"Enter criterion name for DP#{dp_id}:",
                            value=auto_matched or "",
                            key=f"chat_dp_correct_{dp_id_str}",
                        )
                    incorrect_reason = st.text_input(
                        f"Why is this a better match?",
                        value=existing_mapping.get("incorrect_reason", "") if existing_mapping else "",
                        key=f"chat_dp_incorrect_reason_{dp_id_str}",
                    )
                    st.session_state.infer_dp_user_mappings[dp_id_str] = {"criterion": corrected, "not_in_rubric": False, "incorrect_reason": incorrect_reason}
                elif action == "Not in rubric":
                    reason = st.text_input(
                        f"What preference does DP#{dp_id} reflect?",
                        value=existing_mapping.get("not_in_rubric_reason", "") if existing_mapping else "",
                        key=f"chat_dp_notinrubric_{dp_id_str}",
                    )
                    st.session_state.infer_dp_user_mappings[dp_id_str] = {"criterion": None, "not_in_rubric": True, "not_in_rubric_reason": reason}

    # Display chat messages
    _chat_msg_num = 0  # Track message number (1-based, counting user+assistant only)
    for idx, message in enumerate(st.session_state.messages):
        # Skip assessment messages (CHAT_ASSESS_DRAFT_PROMPT and evaluation response) from display
        # They're in conversation history for context but shown only as cards
        if message.get('is_assessment_message'):
            continue

        # Skip the DP review marker message (DPs are now rendered inline on each message)
        if message.get('is_dp_review'):
            continue

        # Skip rubric change log messages (they are in history for LLM context only)
        if message.get('is_rubric_change_log'):
            continue

        if message['role'] == 'system':
            # Only show system messages that belong to the currently selected conversation
            msg_conv_id = message.get("conversation_id")
            if msg_conv_id is not None and msg_conv_id != st.session_state.selected_conversation:
                continue
            # Strip machine-readable HTML comments before display
            display_content = re.sub(r'<!--.*?-->', '', message['content']).strip()
            if display_content:
                st.info(display_content)
        else:
            # In delete mode, show checkbox alongside message
            if st.session_state.message_delete_mode:
                col_check, col_msg = st.columns([0.05, 0.95])
                with col_check:
                    is_selected = idx in st.session_state.messages_to_delete
                    if st.checkbox("Select message", value=is_selected, key=f"delete_msg_{idx}", label_visibility="collapsed"):
                        st.session_state.messages_to_delete.add(idx)
                    else:
                        st.session_state.messages_to_delete.discard(idx)
                with col_msg:
                    with st.chat_message(message['role']):
                        message_id = message.get('message_id', f"{message['role']}_{idx}")
                        _dp_cur_msg_num = _chat_msg_num + 1
                        if message['role'] == 'user':
                            content_to_display = message['content']
                        else:
                            content_to_display = message.get('display_content', message['content'])
                        # Highlight DP quotes in message text
                        _dp_highlighted = False
                        if _dp_active:
                            if message['role'] == 'user' and _dp_cur_msg_num in _dp_by_user_msg:
                                content_to_display = _highlight_dp_quotes(content_to_display, _dp_by_user_msg[_dp_cur_msg_num], 'after_quote', '#A5D6A7')
                                _dp_highlighted = True
                            elif message['role'] == 'assistant' and _dp_cur_msg_num in _dp_by_asst_msg:
                                content_to_display = _highlight_dp_quotes(content_to_display, _dp_by_asst_msg[_dp_cur_msg_num], 'before_quote', '#FFF59D')
                                _dp_highlighted = True
                        if message['role'] == 'assistant' and message.get('thinking'):
                            with st.expander("🧠 Thinking", expanded=False):
                                st.markdown(message['thinking'])
                        # Draft-updated-by-rubric: show What changed, annotated draft with markers, then editable draft
                        if message['role'] == 'assistant' and message.get('rubric_revision'):
                            rr = message['rubric_revision']
                            ann = rr.get('annotated_changes', [])
                            if rr.get('change_summary'):
                                st.markdown("**What changed**")
                                st.markdown(rr['change_summary'])
                            annotated = rr.get('revised_draft_annotated') or rr.get('revised_draft', '')
                            if annotated:
                                st.markdown("**Revised draft** (click a marker to jump to the edit):")
                                st.markdown(_safe_annotated_draft_html(annotated, ann, message_id), unsafe_allow_html=True)
                            if ann:
                                st.markdown("**Edits (by rubric change):**")
                                import html as _html_mod
                                safe_msg_id = re.sub(r'[^a-zA-Z0-9_-]', '_', str(message_id))
                                for i, ac in enumerate(ann, 1):
                                    reason = (ac.get('reason', '') or '').strip()
                                    orig = (ac.get('original_text', '') or '').strip()
                                    new = (ac.get('new_text', '') or '').strip()
                                    reason_esc = _html_mod.escape(reason)
                                    orig_esc = _html_mod.escape(orig[:100] + ('…' if len(orig) > 100 else ''))
                                    new_esc = _html_mod.escape(new[:100] + ('…' if len(new) > 100 else ''))
                                    aid = f"rubric-edit-{safe_msg_id}-{i}"
                                    div_style = "margin:4px 0; scroll-margin-top: 20vh;"
                                    bold_reason = f'<strong>{reason_esc}</strong>'
                                    if not new and orig:
                                        line = f'<div id="{aid}" style="{div_style}">• <strong>[{i}]</strong> {bold_reason} — removed: "{orig_esc}"</div>'
                                    elif not orig and new:
                                        line = f'<div id="{aid}" style="{div_style}">• <strong>[{i}]</strong> {bold_reason} — added: "{new_esc}"</div>'
                                    else:
                                        line = f'<div id="{aid}" style="{div_style}">• <strong>[{i}]</strong> {bold_reason} — "{orig_esc}" → "{new_esc}"</div>'
                                    st.markdown(line, unsafe_allow_html=True)
                                    st.text_input(
                                        "Your feedback (optional)",
                                        value=ac.get('user_feedback', ''),
                                        key=f"rubric_edit_fb_{safe_msg_id}_{i}",
                                        placeholder="Do you not agree? Why? What would you have done differently?",
                                        label_visibility="collapsed"
                                    )
                                _has_feedback = any(
                                    (st.session_state.get(f"rubric_edit_fb_{safe_msg_id}_{j}", "") or "").strip()
                                    for j in range(1, len(ann) + 1)
                                )
                                if st.button("💡Suggest how to change the rubric", key=f"rubric_suggest_btn_{safe_msg_id}", disabled=not _has_feedback):
                                    edits_with_feedback = [
                                        {**ac, "user_feedback": st.session_state.get(f"rubric_edit_fb_{safe_msg_id}_{j}", "") or ""}
                                        for j, ac in enumerate(ann, 1)
                                    ]
                                    # Persist user feedback onto the message's annotated_changes
                                    for j, ac in enumerate(ann):
                                        ac["user_feedback"] = edits_with_feedback[j].get("user_feedback", "")
                                    active_rubric_dict, _, _ = get_active_rubric()
                                    active_rubric_list = (active_rubric_dict.get("rubric", []) or []) if active_rubric_dict else []
                                    edited_rubric_list = list(st.session_state.editing_criteria or [])
                                    active_clean = _rubric_list_for_json(active_rubric_list)
                                    edited_clean = _rubric_list_for_json(edited_rubric_list)
                                    active_json = json.dumps(active_clean, indent=2)
                                    edited_json = json.dumps(edited_clean, indent=2)
                                    try:
                                        with st.spinner("Getting suggestion..."):
                                            prompt1 = RUBRIC_suggest_changes_from_feedback_prompt(active_json, edited_json, edits_with_feedback)
                                            client = anthropic.Anthropic()
                                            r1 = _api_call_with_retry(
                                                model=MODEL_LIGHT,
                                                max_tokens=4096,
                                                messages=[{"role": "user", "content": prompt1}]
                                            )
                                            suggestion_text = ""
                                            for b in r1.content:
                                                if getattr(b, "text", None):
                                                    suggestion_text += b.text
                                            prompt2 = RUBRIC_apply_suggestion_prompt(active_json, edited_json, suggestion_text)
                                            r2 = _api_call_with_retry(
                                                model=MODEL_LIGHT,
                                                max_tokens=8192,
                                                messages=[{"role": "user", "content": prompt2}]
                                            )
                                            raw = ""
                                            for b in r2.content:
                                                if getattr(b, "text", None):
                                                    raw += b.text
                                            json_match = re.search(r'\[[\s\S]*\]', raw)
                                            modified_rubric = json.loads(json_match.group()) if json_match else []
                                            message["rubric_suggestion"] = {
                                                "suggestion_text": suggestion_text,
                                                "modified_rubric": modified_rubric,
                                                "edited_rubric": edited_rubric_list,
                                            }
                                            st.rerun()
                                    except Exception as e:
                                        st.error(f"Suggestion failed: {e}")
                                suggestion_data = message.get("rubric_suggestion")
                                if suggestion_data:
                                    _sg_applied = suggestion_data.get("applied", False)
                                    _sg_label = f"Rubric changes (applied as v{suggestion_data['applied_version']})" if _sg_applied else "How to change the rubric"
                                    with st.expander(_sg_label, expanded=not _sg_applied):
                                        st.markdown(suggestion_data.get("suggestion_text", ""))
                                        if _sg_applied:
                                            st.success(f"Applied as rubric v{suggestion_data['applied_version']}")
                                        else:
                                            st.caption("Apply individual criteria below, or apply all at once at the bottom. Changes appear in **Rubric Configuration** in the sidebar.")
                                            display_rubric_comparison(
                                                suggestion_data.get("edited_rubric", []),
                                                suggestion_data.get("modified_rubric", []),
                                                apply_context={"safe_msg_id": safe_msg_id, "message": message, "message_id": message_id},
                                            )
                        if message.get('is_ranking_checkpoint_result') or message.get('is_dp_confirmation_log') or message.get('is_criteria_classification_log'):
                            st.success(content_to_display)
                        elif message['role'] == 'assistant':
                            if _dp_highlighted:
                                st.markdown(content_to_display, unsafe_allow_html=True)
                            else:
                                has_draft = render_message_with_draft(content_to_display, message_id, wrap_draft_in_expander=bool(message.get('rubric_revision')))
                                if not has_draft:
                                    st.markdown(content_to_display)
                        else:
                            st.markdown(content_to_display, unsafe_allow_html=_dp_highlighted)
                        # Show non-preferred A/B draft inline (blind labels)
                        if message['role'] == 'assistant' and message.get('ab_comparison'):
                            _abc = message['ab_comparison']
                            _abc_chosen = _abc.get('chosen', '')
                            _abc_left_is_rubric = _abc.get('left_is_rubric', True)
                            # Assign blind labels based on original presentation order
                            # Option 1 (left) = Draft A, Option 2 (right) = Draft B
                            if _abc_chosen == 'rubric':
                                _abc_chosen_blind = 'Draft A' if _abc_left_is_rubric else 'Draft B'
                                _abc_other_blind = 'Draft B' if _abc_left_is_rubric else 'Draft A'
                                _abc_other = _abc.get('draft_conversation_only', '')
                            else:
                                _abc_chosen_blind = 'Draft B' if _abc_left_is_rubric else 'Draft A'
                                _abc_other_blind = 'Draft A' if _abc_left_is_rubric else 'Draft B'
                                _abc_other = _abc.get('draft_rubric', '')
                            if _abc_other:
                                st.info(f"You chose **{_abc_chosen_blind}**. {_abc_other_blind} is below.")
                                with st.expander(f"Show {_abc_other_blind}", expanded=False):
                                    st.markdown(strip_draft_tags_for_streaming(_abc_other))
                        if message['role'] == 'assistant' and message.get('rubric_assessment'):
                            assessment = message['rubric_assessment']
                            draft_text = assessment.get('draft_text')
                            display_rubric_assessment(assessment, message_id, draft_text)
                # Track message number and render DPs (outside chat bubble)
                if message['role'] in ('user', 'assistant'):
                    _chat_msg_num += 1
                    if _dp_active and _chat_msg_num in _dp_by_user_msg:
                        for _dp_item in _dp_by_user_msg[_chat_msg_num]:
                            _render_dp_card(_dp_item)
            else:
                with st.chat_message(message['role']):
                    message_id = message.get('message_id', f"{message['role']}_{idx}")
                    _dp_cur_msg_num = _chat_msg_num + 1
                    if message['role'] == 'user':
                        content_to_display = message['content']
                    else:
                        content_to_display = message.get('display_content', message['content'])
                    # Highlight DP quotes in message text
                    _dp_highlighted = False
                    if _dp_active:
                        if message['role'] == 'user' and _dp_cur_msg_num in _dp_by_user_msg:
                            content_to_display = _highlight_dp_quotes(content_to_display, _dp_by_user_msg[_dp_cur_msg_num], 'after_quote', '#A5D6A7')
                            _dp_highlighted = True
                        elif message['role'] == 'assistant' and _dp_cur_msg_num in _dp_by_asst_msg:
                            content_to_display = _highlight_dp_quotes(content_to_display, _dp_by_asst_msg[_dp_cur_msg_num], 'before_quote', '#FFF59D')
                            _dp_highlighted = True
                    if message['role'] == 'assistant' and message.get('thinking'):
                        with st.expander("🧠 Thinking", expanded=False):
                            st.markdown(message['thinking'])
                    if message['role'] == 'assistant' and message.get('rubric_revision'):
                        rr = message['rubric_revision']
                        ann = rr.get('annotated_changes', [])
                        if rr.get('change_summary'):
                            st.markdown("**What changed**")
                            st.markdown(rr['change_summary'])
                        annotated = rr.get('revised_draft_annotated') or rr.get('revised_draft', '')
                        if annotated:
                            st.markdown("**Revised draft** (click a marker to jump to the edit):")
                            st.markdown(_safe_annotated_draft_html(annotated, ann, message_id), unsafe_allow_html=True)
                        if ann:
                            st.markdown("**Edits (by rubric change):**")
                            import html as _html_mod
                            safe_msg_id = re.sub(r'[^a-zA-Z0-9_-]', '_', str(message_id))
                            for i, ac in enumerate(ann, 1):
                                reason = (ac.get('reason', '') or '').strip()
                                orig = (ac.get('original_text', '') or '').strip()
                                new = (ac.get('new_text', '') or '').strip()
                                reason_esc = _html_mod.escape(reason)
                                orig_esc = _html_mod.escape(orig[:100] + ('…' if len(orig) > 100 else ''))
                                new_esc = _html_mod.escape(new[:100] + ('…' if len(new) > 100 else ''))
                                aid = f"rubric-edit-{safe_msg_id}-{i}"
                                div_style = "margin:4px 0; scroll-margin-top: 20vh;"
                                bold_reason = f'<strong>{reason_esc}</strong>'
                                if not new and orig:
                                    line = f'<div id="{aid}" style="{div_style}">• <strong>[{i}]</strong> {bold_reason} — removed: "{orig_esc}"</div>'
                                elif not orig and new:
                                    line = f'<div id="{aid}" style="{div_style}">• <strong>[{i}]</strong> {bold_reason} — added: "{new_esc}"</div>'
                                else:
                                    line = f'<div id="{aid}" style="{div_style}">• <strong>[{i}]</strong> {bold_reason} — "{orig_esc}" → "{new_esc}"</div>'
                                st.markdown(line, unsafe_allow_html=True)
                                st.text_input(
                                    "Your feedback (optional)",
                                    value=ac.get('user_feedback', ''),
                                    key=f"rubric_edit_fb_{safe_msg_id}_{i}",
                                    placeholder="Do you not agree? Why? What would you have done differently?",
                                    label_visibility="collapsed"
                                )
                            _has_feedback = any(
                                (st.session_state.get(f"rubric_edit_fb_{safe_msg_id}_{j}", "") or "").strip()
                                for j in range(1, len(ann) + 1)
                            )
                            if st.button("Suggest how to change the rubric", key=f"rubric_suggest_btn_else_{safe_msg_id}", disabled=not _has_feedback):
                                edits_with_feedback = [
                                    {**ac, "user_feedback": st.session_state.get(f"rubric_edit_fb_{safe_msg_id}_{j}", "") or ""}
                                    for j, ac in enumerate(ann, 1)
                                ]
                                # Persist user feedback onto the message's annotated_changes
                                for j, ac in enumerate(ann):
                                    ac["user_feedback"] = edits_with_feedback[j].get("user_feedback", "")
                                active_rubric_dict, _, _ = get_active_rubric()
                                active_rubric_list = (active_rubric_dict.get("rubric", []) or []) if active_rubric_dict else []
                                edited_rubric_list = list(st.session_state.editing_criteria or [])
                                active_clean = _rubric_list_for_json(active_rubric_list)
                                edited_clean = _rubric_list_for_json(edited_rubric_list)
                                active_json = json.dumps(active_clean, indent=2)
                                edited_json = json.dumps(edited_clean, indent=2)
                                try:
                                    with st.spinner("Getting suggestion..."):
                                        prompt1 = RUBRIC_suggest_changes_from_feedback_prompt(active_json, edited_json, edits_with_feedback)
                                        client = anthropic.Anthropic()
                                        r1 = _api_call_with_retry(
                                            model=MODEL_LIGHT,
                                            max_tokens=4096,
                                            messages=[{"role": "user", "content": prompt1}]
                                        )
                                        suggestion_text = ""
                                        for b in r1.content:
                                            if getattr(b, "text", None):
                                                suggestion_text += b.text
                                        prompt2 = RUBRIC_apply_suggestion_prompt(active_json, edited_json, suggestion_text)
                                        r2 = _api_call_with_retry(
                                            model=MODEL_LIGHT,
                                            max_tokens=8192,
                                            messages=[{"role": "user", "content": prompt2}]
                                        )
                                        raw = ""
                                        for b in r2.content:
                                            if getattr(b, "text", None):
                                                raw += b.text
                                        json_match = re.search(r'\[[\s\S]*\]', raw)
                                        modified_rubric = json.loads(json_match.group()) if json_match else []
                                        message["rubric_suggestion"] = {
                                            "suggestion_text": suggestion_text,
                                            "modified_rubric": modified_rubric,
                                            "edited_rubric": edited_rubric_list,
                                        }
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"Suggestion failed: {e}")
                            suggestion_data = message.get("rubric_suggestion")
                            if suggestion_data:
                                _sg_applied = suggestion_data.get("applied", False)
                                _sg_label = f"Rubric changes (applied as v{suggestion_data['applied_version']})" if _sg_applied else "How to change the rubric"
                                with st.expander(_sg_label, expanded=not _sg_applied):
                                    st.markdown(suggestion_data.get("suggestion_text", ""))
                                    if _sg_applied:
                                        st.success(f"Applied as rubric v{suggestion_data['applied_version']}")
                                    else:
                                        st.caption("Apply individual criteria below, or apply all at once at the bottom. Changes appear in **Rubric Configuration** in the sidebar.")
                                        display_rubric_comparison(
                                            suggestion_data.get("edited_rubric", []),
                                            suggestion_data.get("modified_rubric", []),
                                            apply_context={"safe_msg_id": safe_msg_id, "message": message, "message_id": message_id},
                                        )
                    if message.get('is_ranking_checkpoint_result') or message.get('is_dp_confirmation_log') or message.get('is_criteria_classification_log'):
                        st.success(content_to_display)
                    elif message['role'] == 'assistant':
                        if _dp_highlighted:
                            st.markdown(content_to_display, unsafe_allow_html=True)
                        else:
                            has_draft = render_message_with_draft(content_to_display, message_id, wrap_draft_in_expander=bool(message.get('rubric_revision')))
                            if not has_draft:
                                st.markdown(content_to_display)
                    else:
                        st.markdown(content_to_display, unsafe_allow_html=_dp_highlighted)
                    # Show non-preferred A/B draft inline (blind labels)
                    if message['role'] == 'assistant' and message.get('ab_comparison'):
                        _abc = message['ab_comparison']
                        _abc_chosen = _abc.get('chosen', '')
                        _abc_left_is_rubric = _abc.get('left_is_rubric', True)
                        if _abc_chosen == 'rubric':
                            _abc_chosen_blind = 'Draft A' if _abc_left_is_rubric else 'Draft B'
                            _abc_other_blind = 'Draft B' if _abc_left_is_rubric else 'Draft A'
                            _abc_other = _abc.get('draft_conversation_only', '')
                        else:
                            _abc_chosen_blind = 'Draft B' if _abc_left_is_rubric else 'Draft A'
                            _abc_other_blind = 'Draft A' if _abc_left_is_rubric else 'Draft B'
                            _abc_other = _abc.get('draft_rubric', '')
                        if _abc_other:
                            st.caption(f"You chose {_abc_chosen_blind}. {_abc_other_blind} is below.")
                            with st.expander(f"Show {_abc_other_blind}", expanded=False):
                                st.markdown(strip_draft_tags_for_streaming(_abc_other))
                    if message['role'] == 'assistant' and message.get('rubric_assessment'):
                        assessment = message['rubric_assessment']
                        draft_text = assessment.get('draft_text')
                        display_rubric_assessment(assessment, message_id, draft_text)
                # Track message number and render DPs (outside chat bubble)
                if message['role'] in ('user', 'assistant'):
                    _chat_msg_num += 1
                    if _dp_active and _chat_msg_num in _dp_by_user_msg:
                        for _dp_item in _dp_by_user_msg[_chat_msg_num]:
                            _render_dp_card(_dp_item)

    # --- Rubric Criteria Classification (shown before DP review) ---
    if _dp_has_review and _cc_pending:
        _cc_llm = st.session_state.chat_criteria_llm_classification
        _cc_user = st.session_state.chat_criteria_user_classifications
        if _cc_llm and _cc_user:
            st.divider()

            # Show user's cold-start preferences for reference
            _cs_ref_text = st.session_state.get("infer_coldstart_text", "").strip()
            if _cs_ref_text:
                with st.expander("Your writing preferences (for reference)", expanded=False):
                    st.markdown(f"*You wrote this before the conversation:*\n\n{_cs_ref_text}")

            _cc_stated = [n for n, s in _cc_user.items() if s == "stated"]
            _cc_unstated = [n for n, s in _cc_user.items() if s != "stated"]

            st.markdown("**Rubric Criteria Classification**")
            st.markdown(
                "We compared each rubric criterion against the writing preferences you described earlier. "
                "**Stated** criteria matched something you wrote. **Unstated** criteria were inferred from "
                "your editing behavior but weren't in your original description — please review these below."
            )

            # Build lookup for LLM reasoning per criterion
            _cc_reasoning = {}
            for _cc_comp in _cc_llm.get("criteria_comparison", []):
                _cc_reasoning[_cc_comp.get("criterion_name", "")] = _cc_comp.get("match_reasoning", "")

            # Initialize hallucination reasons in session state if needed
            if 'chat_criteria_hallucination_reasons' not in st.session_state:
                st.session_state.chat_criteria_hallucination_reasons = {}

            if _cc_stated:
                st.markdown(f"**Stated** ({len(_cc_stated)})",
                            help="These criteria matched something in your writing preferences.")
                for _cs_name in _cc_stated:
                    _cs_choice = st.selectbox(
                        _cs_name,
                        ["Stated", "Real", "Hallucinated"],
                        index=0,
                        key=f"cc_chip_{_cs_name}",
                        help="Stated = you mentioned this | Real = you care but didn't mention | Hallucinated = doesn't reflect your preferences"
                    )
                    _cc_user[_cs_name] = _cs_choice.lower()
                    if _cs_choice == "Hallucinated":
                        _cs_reason_text = _cc_reasoning.get(_cs_name, "")
                        if _cs_reason_text:
                            st.caption(f"*Why we inferred this:* {_cs_reason_text}")
                        _cs_existing_reason = st.session_state.get("chat_criteria_hallucination_reasons", {}).get(_cs_name, "")
                        _cs_halluc_reason = st.text_input(
                            f"Why doesn't \"{_cs_name}\" reflect your preferences?",
                            value=_cs_existing_reason,
                            key=f"cc_halluc_reason_{_cs_name}",
                            placeholder="e.g., I never cared about this, the model assumed it from context"
                        )
                        st.session_state.chat_criteria_hallucination_reasons[_cs_name] = _cs_halluc_reason

            if _cc_unstated:
                st.markdown(
                    f"**Unstated** ({len(_cc_unstated)})",
                    help="These criteria were NOT in your original writing preferences but were inferred from the conversation. "
                         "For each one, decide: **Real** = you do care about this even though you didn't mention it. "
                         "**Hallucinated** = this doesn't reflect your actual preferences, the model made it up. "
                         "**Stated** = actually, you did mention this (misclassified)."
                )
                for _cu_name in _cc_unstated:
                    _cu_current = _cc_user.get(_cu_name, "unstated")
                    _cu_default_idx = {"stated": 0, "real": 1, "hallucinated": 2}.get(_cu_current, 1)
                    _cu_choice = st.selectbox(
                        _cu_name,
                        ["Stated", "Real", "Hallucinated"],
                        index=_cu_default_idx,
                        key=f"cc_chip_{_cu_name}",
                        help="Real = you care about this | Hallucinated = model made it up | Stated = you did mention this"
                    )
                    _cc_user[_cu_name] = _cu_choice.lower()
                    if _cu_choice == "Hallucinated":
                        _cu_reason_text = _cc_reasoning.get(_cu_name, "")
                        if _cu_reason_text:
                            st.caption(f"*Why we inferred this:* {_cu_reason_text}")
                        _cu_existing_reason = st.session_state.get("chat_criteria_hallucination_reasons", {}).get(_cu_name, "")
                        _cu_halluc_reason = st.text_input(
                            f"Why doesn't \"{_cu_name}\" reflect your preferences?",
                            value=_cu_existing_reason,
                            key=f"cc_halluc_reason_{_cu_name}",
                            placeholder="e.g., I never cared about this, the model assumed it from context"
                        )
                        st.session_state.chat_criteria_hallucination_reasons[_cu_name] = _cu_halluc_reason

            st.session_state.chat_criteria_user_classifications = _cc_user

            # Confirm button
            if st.button("Confirm & Extract Decision Points", type="primary", use_container_width=True, key="chat_confirm_criteria_pre"):
                _cc_final = st.session_state.chat_criteria_user_classifications
                _cc_llm_data = st.session_state.chat_criteria_llm_classification

                # Calculate agreement rate
                _cc_agreements = 0
                _cc_total = 0
                for _cc_comp in _cc_llm_data.get("criteria_comparison", []):
                    _cc_cname = _cc_comp.get("criterion_name", "")
                    _cc_llm_status = _cc_comp.get("status", "unstated")
                    _cc_user_status = _cc_final.get(_cc_cname, "unstated")
                    _cc_total += 1
                    if (_cc_llm_status == "stated") == (_cc_user_status == "stated"):
                        _cc_agreements += 1
                _cc_agreement_rate = _cc_agreements / _cc_total if _cc_total > 0 else 0

                _cc_stated_count = sum(1 for v in _cc_final.values() if v == "stated")
                _cc_real_count = sum(1 for v in _cc_final.values() if v in ("real", "latent_real", "elicited"))
                _cc_hallucinated_count = sum(1 for v in _cc_final.values() if v == "hallucinated")
                _cc_precision = (_cc_total - _cc_hallucinated_count) / _cc_total if _cc_total > 0 else 1.0

                # Update state
                st.session_state.chat_criteria_review_confirmed = True
                st.session_state.chat_criteria_review_active = False

                # Sync to Infer tab state
                st.session_state.infer_user_categorizations = copy.deepcopy(_cc_final)
                st.session_state.infer_categorizations_complete = True

                # Build log message
                _cc_log_lines = [
                    f"**Criteria classifications confirmed**: {_cc_stated_count} stated, {_cc_real_count} real, {_cc_hallucinated_count} hallucinated.",
                    f"Rubric precision: {_cc_precision:.0%}",
                    ""
                ]
                _cc_halluc_reasons = st.session_state.get("chat_criteria_hallucination_reasons", {})
                for _cc_cname, _cc_cat in _cc_final.items():
                    _cc_icon = {"stated": "✓", "real": "◉", "hallucinated": "✗"}.get(_cc_cat, "?")
                    _cc_label = {"stated": "Stated", "real": "Real", "hallucinated": "Hallucinated"}.get(_cc_cat, _cc_cat)
                    _cc_line = f"- {_cc_icon} **{_cc_cname}**: {_cc_label}"
                    if _cc_cat == "hallucinated" and _cc_halluc_reasons.get(_cc_cname):
                        _cc_line += f" — *{_cc_halluc_reasons[_cc_cname]}*"
                    _cc_log_lines.append(_cc_line)
                _cc_log_content = "\n".join(_cc_log_lines)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": _cc_log_content,
                    "display_content": _cc_log_content,
                    "is_system_generated": True,
                    "is_criteria_classification_log": True,
                    "classification_data": {
                        "classifications": copy.deepcopy(_cc_final),
                        "hallucination_reasons": copy.deepcopy(_cc_halluc_reasons),
                        "llm_classification": copy.deepcopy(_cc_llm_data),
                        "llm_user_agreement": _cc_agreement_rate,
                        "rubric_version": None,
                        "stated_count": _cc_stated_count,
                        "real_count": _cc_real_count,
                        "hallucinated_count": _cc_hallucinated_count,
                        "precision": _cc_precision,
                        "timestamp": datetime.now().isoformat(),
                    },
                    "message_id": f"criteria_class_{int(time.time() * 1000000)}"
                })

                # Save to DB
                _cc_save_sb = st.session_state.get('supabase')
                _cc_save_pid = st.session_state.get('current_project_id')
                if _cc_save_sb and _cc_save_pid:
                    try:
                        _cc_rb_dict, _, _ = get_active_rubric()
                        _cc_rb_ver = _cc_rb_dict.get("version", "?") if _cc_rb_dict else "?"
                        _cc_feedback_record = {
                            "timestamp": datetime.now().isoformat(),
                            "rubric_version": _cc_rb_ver,
                            "classifications": copy.deepcopy(_cc_final),
                            "hallucination_reasons": copy.deepcopy(_cc_halluc_reasons),
                            "llm_classification_summary": _cc_llm_data.get("summary", {}),
                            "llm_user_agreement": _cc_agreement_rate,
                            "stated_count": _cc_stated_count,
                            "real_count": _cc_real_count,
                            "hallucinated_count": _cc_hallucinated_count,
                            "precision": _cc_precision,
                        }
                        save_project_data(_cc_save_sb, _cc_save_pid, "criteria_classification_feedback", _cc_feedback_record)
                    except Exception:
                        pass

                # Store classification feedback (including hallucination reasons) for DP extraction and final rubric
                _cc_halluc_reasons = st.session_state.get("chat_criteria_hallucination_reasons", {})
                _cc_feedback_for_dps = {
                    "classifications": copy.deepcopy(_cc_final),
                    "stated_count": _cc_stated_count,
                    "real_count": _cc_real_count,
                    "hallucinated_count": _cc_hallucinated_count,
                    "hallucination_reasons": {
                        name: reason for name, reason in _cc_halluc_reasons.items()
                        if _cc_final.get(name) == "hallucinated" and reason
                    },
                }
                st.session_state.chat_classification_feedback = _cc_feedback_for_dps

                # No rubric cleaning here — hallucinated criteria are kept in the rubric
                # until the final rubric inference after DP confirmation (step 5)

                # Step 3: Extract DPs with classification feedback
                _cc_conv_msgs = st.session_state.get("infer_dp_messages", [])
                if _cc_conv_msgs:
                    _dp_rb_dict, _, _ = get_active_rubric()
                    _dp_rubric_json = json.dumps(_dp_rb_dict.get("rubric", []), ensure_ascii=False, indent=2) if _dp_rb_dict else "[]"

                    with st.spinner("Extracting decision points with classification context..."):
                        dp_result = extract_decision_points(
                            _cc_conv_msgs,
                            _dp_rubric_json,
                            _cc_feedback_for_dps
                        )
                        if dp_result:
                            # Normalize field names
                            for dp in dp_result.get("parsed_data", {}).get("decision_points", []):
                                if "related_rubric_criterion" in dp and "suggested_criterion_name" not in dp:
                                    dp["suggested_criterion_name"] = dp["related_rubric_criterion"]

                            _dp_result_store = {
                                "thinking": "",
                                "raw_response": "",
                                "parsed_data": dp_result.get("parsed_data", dp_result),
                                "conversation_file": "__from_infer_rubric__"
                            }
                            st.session_state.infer_decision_points = _dp_result_store
                            st.session_state.infer_dp_dimension_confirmed = False
                            st.session_state.infer_dp_user_mappings = {}

                            _dp_list_new = dp_result.get("parsed_data", {}).get("decision_points", [])
                            if _dp_list_new:
                                st.session_state.infer_expanded_dp = _dp_list_new[0].get("id")

                            # Update the DP review message with actual DP data
                            for msg in st.session_state.messages:
                                if msg.get("is_dp_review"):
                                    msg["dp_data"]["decision_points"] = copy.deepcopy(_dp_list_new)
                                    _rb_v = msg["dp_data"].get("rubric_version", "?")
                                    msg["content"] = f"**Rubric v{_rb_v} inferred** — {len(_dp_list_new)} decision points extracted. Review each DP below."
                                    msg["display_content"] = msg["content"]
                                    if _dp_rb_dict:
                                        msg["dp_data"]["rubric"] = copy.deepcopy(_dp_rb_dict.get("rubric", []))
                                    break

                st.rerun()

    # --- DP Confirm button (after all messages) ---
    if _dp_active and _dp_list_all and not _dp_confirmed:
        st.divider()

        # DP jump navigation buttons (color-coded HTML + JS scroll)
        import streamlit.components.v1 as _dp_components
        st.markdown("**Jump to Decision Point:**")
        _dp_btn_html_parts = []
        for _dp_item in _dp_list_all:
            _dp_jump_id = _dp_item.get('id', 0)
            _dp_jump_id_str = str(_dp_jump_id)
            _dp_jump_dim = _dp_item.get('dimension', '')[:30]
            _dp_jump_mapping = st.session_state.infer_dp_user_mappings.get(_dp_jump_id_str, {})
            _dp_jump_auto = _chat_auto_match(_dp_item)
            if _dp_jump_mapping.get("not_in_rubric", False):
                _bg = "#FFCDD2"; _bd = "#F44336"; _tx = "#B71C1C"; _lbl = "Not in rubric"
            elif _dp_jump_mapping.get("criterion") and _dp_jump_mapping["criterion"] != _dp_jump_auto:
                _bg = "#FFF9C4"; _bd = "#FF9800"; _tx = "#E65100"; _lbl = "Remapped"
            elif _dp_jump_mapping.get("criterion"):
                _bg = "#C8E6C9"; _bd = "#4CAF50"; _tx = "#1B5E20"; _lbl = "Correct"
            else:
                _bg = "#E3F2FD"; _bd = "#1976D2"; _tx = "#0D47A1"; _lbl = "Unreviewed"
            _dp_btn_html_parts.append(
                f'<button onclick="jumpToDP(\'DP#{_dp_jump_id}\')" '
                f'title="{_dp_html_lib.escape(_dp_jump_dim)} — {_lbl}" '
                f'style="background:{_bg};border:2px solid {_bd};color:{_tx};border-radius:8px;'
                f'padding:6px 14px;cursor:pointer;font-size:0.85em;font-weight:bold;margin:3px;">'
                f'DP#{_dp_jump_id}</button>'
            )
        _dp_btns_joined = "\n".join(_dp_btn_html_parts)
        _dp_components.html(
            f"""
            <div style="display:flex;flex-wrap:wrap;gap:4px;padding:4px 0;">
                {_dp_btns_joined}
            </div>
            <script>
            function jumpToDP(label) {{
                var doc = window.parent.document;
                var spans = doc.querySelectorAll('span');
                for (var i = 0; i < spans.length; i++) {{
                    if (spans[i].textContent.trim() === label) {{
                        var card = spans[i].closest('div[style*="linear-gradient"]') || spans[i].parentElement;
                        if (card) {{
                            card.scrollIntoView({{behavior: 'smooth', block: 'center'}});
                            var orig = card.style.outline;
                            card.style.outline = '3px solid #FF9800';
                            setTimeout(function() {{ card.style.outline = orig; }}, 2000);
                            break;
                        }}
                    }}
                }}
            }}
            </script>
            """,
            height=50 + (len(_dp_list_all) // 7) * 40,
        )
        # st.divider()

        all_mapped = len(st.session_state.infer_dp_user_mappings) >= len(_dp_list_all)
        confirm_ok = all(
            st.session_state.infer_dp_user_mappings.get(str(dp.get('id', 0)), {}).get("criterion") is not None
            or st.session_state.infer_dp_user_mappings.get(str(dp.get('id', 0)), {}).get("not_in_rubric", False)
            for dp in _dp_list_all
        )

        # Check if any DPs have corrections (incorrect or not_in_rubric)
        _has_corrections = False
        for dp in _dp_list_all:
            dp_id_str = str(dp.get('id', 0))
            mapping = st.session_state.infer_dp_user_mappings.get(dp_id_str, {})
            auto_matched = _chat_auto_match(dp)
            if mapping.get("not_in_rubric", False):
                _has_corrections = True
                break
            if mapping.get("criterion") and mapping["criterion"] != auto_matched:
                _has_corrections = True
                break

        _had_hallucinated = any(
            v == "hallucinated"
            for v in st.session_state.get("chat_classification_feedback", {}).get("classifications", {}).values()
        )
        _needs_final_rubric = _has_corrections or _had_hallucinated
        if _needs_final_rubric:
            _confirm_label = "Confirm DPs & Infer Final Rubric"
        else:
            _confirm_label = "Confirm DPs"
        if st.button(_confirm_label, type="primary", use_container_width=True, key="chat_confirm_dp_and_infer"):
                # Capture source rubric version BEFORE any refinement
                _source_rb_dict, _, _ = get_active_rubric()
                _source_rubric_version = _source_rb_dict.get("version", "?") if _source_rb_dict else "?"

                # Confirm DPs
                _dp_parsed = _dp_result.get("parsed_data", {})
                decision_points = _dp_parsed.get("decision_points", [])
                for dp in decision_points:
                    dp_id_str = str(dp.get('id', 0))
                    mapping = st.session_state.infer_dp_user_mappings.get(dp_id_str, {})
                    crit = mapping.get("criterion")
                    original_suggestion = _chat_auto_match(dp)
                    dp["original_suggestion"] = original_suggestion
                    if crit:
                        dp["confirmed_criterion"] = crit
                        dp["is_not_in_rubric"] = False
                        if crit == original_suggestion:
                            dp["user_action"] = "correct"
                        else:
                            dp["user_action"] = "incorrect"
                            dp["incorrect_reason"] = mapping.get("incorrect_reason", "")
                    else:
                        dp["confirmed_criterion"] = None
                        dp["is_not_in_rubric"] = True
                        dp["not_in_rubric_reason"] = mapping.get("not_in_rubric_reason", "")
                        dp["user_action"] = "not_in_rubric"
                _dp_parsed["decision_points"] = decision_points
                st.session_state.infer_decision_points["parsed_data"] = _dp_parsed
                st.session_state.infer_dp_dimension_confirmed = True

                # Step 5: Final rubric inference with ALL context (if needed)
                if _needs_final_rubric:
                    # Build corrected DPs summary
                    _active_rb_dict, _, _ = get_active_rubric()
                    _active_criteria = _active_rb_dict.get("rubric", []) if _active_rb_dict else []
                    corrected_dps = []
                    for dp in decision_points:
                        dp_summary = {
                            "id": dp.get("id"),
                            "dimension": dp.get("dimension"),
                            "summary": dp.get("summary"),
                            "user_action": dp.get("user_action"),
                            "confirmed_criterion": dp.get("confirmed_criterion"),
                        }
                        if dp.get("user_action") == "incorrect":
                            dp_summary["original_suggestion"] = dp.get("original_suggestion")
                            dp_summary["incorrect_reason"] = dp.get("incorrect_reason", "")
                        elif dp.get("user_action") == "not_in_rubric":
                            dp_summary["not_in_rubric_reason"] = dp.get("not_in_rubric_reason", "")
                        corrected_dps.append(dp_summary)

                    corrected_json = json.dumps(corrected_dps, indent=2)
                    current_rubric_json = json.dumps(_active_criteria, ensure_ascii=False, indent=2)

                    # Gather ALL context for final rubric
                    _final_conv_msgs = st.session_state.get("infer_dp_messages", [])
                    _final_classification = st.session_state.get("chat_classification_feedback", {})
                    _final_classification_json = json.dumps(_final_classification, ensure_ascii=False, indent=2)
                    _final_coldstart = st.session_state.get("infer_coldstart_text", "").strip()

                    with st.spinner("Inferring final rubric with all context..."):
                        refined_rubric_data = infer_final_rubric(
                            _final_conv_msgs,
                            current_rubric_json,
                            _final_classification_json,
                            corrected_json,
                            _final_coldstart
                        )
                        if refined_rubric_data:
                            st.session_state.dp_refinement_result = {
                                "change_explanation": refined_rubric_data.get("_change_explanation", ""),
                                "refinement_summary": refined_rubric_data.get("_refinement_summary", ""),
                                "old_rubric": _active_criteria,
                                "old_version": _active_rb_dict.get("version", "?"),
                                "new_rubric": refined_rubric_data.get("rubric", []),
                                "new_version": refined_rubric_data.get("version", "?"),
                            }
                        else:
                            st.warning("Final rubric inference failed — rubric unchanged.")

                _cur_rb, _, _ = get_active_rubric()
                _result_rubric_version = _cur_rb.get("version", "?") if _cur_rb else "?"
                _infer_entry = {
                    "messages": copy.deepcopy(st.session_state.get("infer_dp_messages", [])),
                    "decision_points": st.session_state.get("infer_decision_points"),
                    "timestamp": datetime.now().isoformat(),
                    "source_rubric_version": _source_rubric_version,
                    "result_rubric_version": _result_rubric_version,
                    "had_corrections": _has_corrections,
                    "had_hallucinated": _had_hallucinated,
                    "num_messages": len(st.session_state.get("infer_dp_messages", [])),
                    "conversation_id": st.session_state.get("selected_conversation", ""),
                    "classification_feedback": copy.deepcopy(st.session_state.get("chat_classification_feedback", {})),
                    "user_categorizations": copy.deepcopy(st.session_state.get("infer_user_categorizations", {})),
                }
                if 'infer_all_conversations' not in st.session_state:
                    st.session_state.infer_all_conversations = []
                st.session_state.infer_all_conversations.append(_infer_entry)

                # Build explicit decision point feedback record
                _dp_feedback_list = []
                for dp in decision_points:
                    _dp_entry = {
                        "id": dp.get("id"),
                        "title": dp.get("title", ""),
                        "dimension": dp.get("dimension", ""),
                        "summary": dp.get("summary", ""),
                        "assistant_message_num": dp.get("assistant_message_num"),
                        "user_message_num": dp.get("user_message_num"),
                        "before_quote": dp.get("before_quote", ""),
                        "after_quote": dp.get("after_quote", ""),
                        "suggested_criterion_name": dp.get("suggested_criterion_name") or dp.get("related_rubric_criterion", ""),
                        "user_action": dp.get("user_action", "unreviewed"),
                        "confirmed_criterion": dp.get("confirmed_criterion"),
                        "original_suggestion": dp.get("original_suggestion", ""),
                    }
                    if dp.get("user_action") == "incorrect":
                        _dp_entry["incorrect_reason"] = dp.get("incorrect_reason", "")
                    elif dp.get("user_action") == "not_in_rubric":
                        _dp_entry["not_in_rubric_reason"] = dp.get("not_in_rubric_reason", "")
                    _dp_feedback_list.append(_dp_entry)

                _dp_feedback_record = {
                    "timestamp": datetime.now().isoformat(),
                    "source_rubric_version": _source_rubric_version,
                    "result_rubric_version": _result_rubric_version,
                    "had_corrections": _has_corrections,
                    "num_decision_points": len(decision_points),
                    "decision_points": _dp_feedback_list,
                }

                _save_sb = st.session_state.get('supabase')
                _save_pid = st.session_state.get('current_project_id')
                if _save_sb and _save_pid:
                    try:
                        # Save infer conversation (replace)
                        _save_sb.table("project_data").delete().eq("project_id", _save_pid).eq("data_type", "infer_conversation").execute()
                        _save_sb.table("project_data").insert({
                            "project_id": _save_pid,
                            "data_type": "infer_conversation",
                            "data": json.dumps(st.session_state.infer_all_conversations),
                            "created_at": datetime.now().isoformat()
                        }).execute()
                        # Save decision point feedback (append)
                        save_project_data(_save_sb, _save_pid, "decision_point_feedback", _dp_feedback_record)
                    except Exception:
                        pass

                # Log the DP confirmation + rubric inference to conversation history
                _cur_rb_log, _, _ = get_active_rubric()
                _dp_log_lines = []
                # Show version transition: DPs inferred from source, rubric refined to result
                if _needs_final_rubric and _source_rubric_version != _result_rubric_version:
                    _dp_log_lines.append(f"**DPs inferred from v{_source_rubric_version}**, final rubric **v{_result_rubric_version}** inferred with all context ({len(decision_points)} decision points).\n")
                else:
                    _dp_log_lines.append(f"**DPs inferred from v{_source_rubric_version}** — {len(decision_points)} decision points confirmed (no rubric changes).\n")
                # Summarize each DP
                for _dp_log in decision_points:
                    _dp_action = _dp_log.get("user_action", "unreviewed")
                    _dp_dim = _dp_log.get("dimension", "")
                    _dp_crit = _dp_log.get("confirmed_criterion") or _dp_log.get("original_suggestion", "—")
                    if _dp_action == "correct":
                        _dp_log_lines.append(f"- **DP#{_dp_log.get('id')}** {_dp_dim} → ✓ {_dp_crit}")
                    elif _dp_action == "incorrect":
                        _dp_orig = _dp_log.get("original_suggestion", "?")
                        _dp_log_lines.append(f"- **DP#{_dp_log.get('id')}** {_dp_dim} → remapped from *{_dp_orig}* to **{_dp_crit}**")
                    elif _dp_action == "not_in_rubric":
                        _dp_log_lines.append(f"- **DP#{_dp_log.get('id')}** {_dp_dim} → not in rubric")
                # Summarize rubric criteria
                if _cur_rb_log:
                    _rb_criteria = _cur_rb_log.get("rubric", [])
                    _dp_log_lines.append(f"\n**Rubric criteria** ({len(_rb_criteria)}):")
                    for _rc_log in _rb_criteria:
                        _dp_log_lines.append(f"- {_rc_log.get('name', 'Unnamed')}: {_rc_log.get('description', '')}")
                _dp_log_content = "\n".join(_dp_log_lines)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": _dp_log_content,
                    "display_content": _dp_log_content,
                    "is_system_generated": True,
                    "is_dp_confirmation_log": True,
                    "dp_data": {
                        "decision_points": copy.deepcopy(decision_points),
                        "source_rubric_version": _source_rubric_version,
                        "result_rubric_version": _result_rubric_version,
                        "rubric": copy.deepcopy(_cur_rb_log.get("rubric", [])) if _cur_rb_log else [],
                        "refinement": copy.deepcopy(st.session_state.get("dp_refinement_result")) if st.session_state.get("dp_refinement_result") else None,
                    },
                    "message_id": f"dp_confirm_{int(time.time() * 1000000)}"
                })

                # Auto-trigger ranking checkpoint after DP confirmation
                if not st.session_state.get("ranking_checkpoint_auto_triggered", False):
                    st.session_state.ranking_checkpoint_auto_triggered = True
                    st.session_state.ranking_checkpoint_pending = {"step": 1}

                st.rerun()

    elif _dp_active and _dp_confirmed:
        pass  # DP confirmation and classification messages are now shown as conversation messages

    # ---- Ranking Checkpoint UI ----
    _rcp = st.session_state.get("ranking_checkpoint_pending")
    if _rcp is not None:
        st.divider()
        st.markdown("### Quick Alignment Check")
        st.markdown("Let's verify your rubric works well on a **new writing task**. We'll generate a few drafts using different approaches — rank them by how well they match what you'd actually want, and we'll measure how well each approach aligns with your preferences.")
        _rcp_step = _rcp.get("step", 1)

        # Show which rubric version is being used
        _rcp_rb_dict, _, _ = get_active_rubric()
        _rcp_rb_ver = _rcp_rb_dict.get("version", "?") if _rcp_rb_dict else "?"
        st.caption(f"Using rubric v{_rcp_rb_ver} for generation and grading")

        if _rcp_step == 1:
            # Step 1: Generate writing task
            st.info("Generating a new writing task based on your conversation...")
            _rcp_conv_text = "\n".join(
                f"{m['role']}: {m.get('content', '')[:500]}"
                for m in st.session_state.messages
                if m.get("role") in ("user", "assistant") and not m.get("is_system_generated")
            )[:5000]
            with st.spinner("Creating writing task..."):
                try:
                    _rcp_task_prompt = GRADING_generate_writing_task_prompt(_rcp_conv_text)
                    _rcp_task_resp = _api_call_with_retry(
                        model=MODEL_LIGHT, max_tokens=500,
                        messages=[{"role": "user", "content": _rcp_task_prompt}]
                    )
                    _rcp_task_text = "".join(b.text for b in _rcp_task_resp.content if b.type == "text").strip()
                    st.session_state.ranking_checkpoint_pending = {"step": 2, "writing_task": _rcp_task_text, "rubric_version": _rcp_rb_ver}
                    st.rerun()
                except Exception as _rcp_e:
                    st.error(f"Failed to generate writing task: {_rcp_e}")
                    if st.button("Cancel checkpoint", key="rcp_cancel_s1"):
                        st.session_state.ranking_checkpoint_pending = None
                        st.rerun()

        elif _rcp_step == 2:
            # Step 2: Generate 3 blind drafts
            st.markdown(f"**Writing task:** {_rcp.get('writing_task', '')}")
            st.info("Generating 3 blind drafts...")
            _rcp_rubric_dict, _, _ = get_active_rubric()
            _rcp_rubric_json = json.dumps(
                _rubric_to_json_serializable(_rcp_rubric_dict), indent=2
            ) if _rcp_rubric_dict else ""
            _rcp_coldstart = st.session_state.get("infer_coldstart_text", "").strip()
            _rcp_task = _rcp["writing_task"]
            _rcp_drafts = {}
            _rcp_ok = True

            with st.spinner("Generating draft 1/3 (rubric-guided)..."):
                try:
                    _rcp_pr = GRADING_generate_draft_from_rubric_prompt(_rcp_task, _rcp_rubric_json)
                    _rcp_resp_r = _api_call_with_retry(
                        model=MODEL_LIGHT, max_tokens=1000,
                        messages=[{"role": "user", "content": _rcp_pr}]
                    )
                    _rcp_drafts["rubric"] = "".join(b.text for b in _rcp_resp_r.content if b.type == "text").strip()
                except Exception as _rcp_e2:
                    st.error(f"Failed to generate rubric draft: {_rcp_e2}")
                    _rcp_ok = False

            if _rcp_ok and _rcp_coldstart:
                with st.spinner("Generating draft 2/3 (cold-start)..."):
                    try:
                        _rcp_pc = GRADING_generate_draft_from_coldstart_prompt(_rcp_task, _rcp_coldstart)
                        _rcp_resp_c = _api_call_with_retry(
                            model=MODEL_LIGHT, max_tokens=1000,
                            messages=[{"role": "user", "content": _rcp_pc}]
                        )
                        _rcp_drafts["coldstart"] = "".join(b.text for b in _rcp_resp_c.content if b.type == "text").strip()
                    except Exception as _rcp_e3:
                        st.error(f"Failed to generate cold-start draft: {_rcp_e3}")
                        _rcp_ok = False
            elif _rcp_ok:
                _rcp_drafts["coldstart"] = ""  # No cold-start text available

            if _rcp_ok:
                with st.spinner("Generating draft 3/3 (generic)..."):
                    try:
                        _rcp_pg = GRADING_generate_draft_generic_prompt(_rcp_task)
                        _rcp_resp_g = _api_call_with_retry(
                            model=MODEL_LIGHT, max_tokens=1000,
                            messages=[{"role": "user", "content": _rcp_pg}]
                        )
                        _rcp_drafts["generic"] = "".join(b.text for b in _rcp_resp_g.content if b.type == "text").strip()
                    except Exception as _rcp_e4:
                        st.error(f"Failed to generate generic draft: {_rcp_e4}")
                        _rcp_ok = False

            if _rcp_ok:
                # Remove empty drafts (e.g. if coldstart had no text)
                _rcp_drafts_filtered = {k: v for k, v in _rcp_drafts.items() if v}
                if len(_rcp_drafts_filtered) < 2:
                    st.error("Need at least 2 drafts to rank. Checkpoint cancelled.")
                    st.session_state.ranking_checkpoint_pending = None
                    st.rerun()
                else:
                    # Shuffle and assign blind labels
                    _rcp_sources = list(_rcp_drafts_filtered.keys())
                    import random as _rcp_rand
                    _rcp_rand.shuffle(_rcp_sources)
                    _rcp_labels = ["A", "B", "C"][:len(_rcp_sources)]
                    _rcp_shuffle_order = list(zip(_rcp_labels, _rcp_sources))
                    st.session_state.ranking_checkpoint_pending = {
                        "step": 3,
                        "writing_task": _rcp_task,
                        "drafts": _rcp_drafts_filtered,
                        "shuffle_order": _rcp_shuffle_order,
                        "rubric_version": _rcp.get("rubric_version", _rcp_rb_ver),
                    }
                    st.rerun()
            else:
                if st.button("Cancel checkpoint", key="rcp_cancel_s2"):
                    st.session_state.ranking_checkpoint_pending = None
                    st.rerun()

        elif _rcp_step == 3:
            # Step 3: Show drafts, user ranks
            st.markdown(f"**Writing task:** {_rcp.get('writing_task', '')}")
            _rcp_shuffle = _rcp.get("shuffle_order", [])
            _rcp_drafts_3 = _rcp.get("drafts", {})
            _rcp_num = len(_rcp_shuffle)

            st.markdown("Read the drafts below and rank them from **best (1)** to **worst** based on how well each matches what you'd actually want.")

            # Display drafts in columns
            _rcp_cols = st.columns(_rcp_num)
            for _rcp_i, (_rcp_label, _rcp_src) in enumerate(_rcp_shuffle):
                with _rcp_cols[_rcp_i]:
                    st.markdown(f"**Draft {_rcp_label}**")
                    with st.container(height=300):
                        st.markdown(_rcp_drafts_3.get(_rcp_src, ""))

            # Ranking inputs
            st.markdown("**Your ranking:**")
            _rcp_rank_cols = st.columns(_rcp_num)
            _rcp_user_ranks = {}
            _rcp_rank_options = list(range(1, _rcp_num + 1))
            for _rcp_i, (_rcp_label, _rcp_src) in enumerate(_rcp_shuffle):
                with _rcp_rank_cols[_rcp_i]:
                    _rcp_user_ranks[_rcp_label] = st.selectbox(
                        f"Draft {_rcp_label}",
                        _rcp_rank_options,
                        index=_rcp_i,
                        key=f"rcp_rank_{_rcp_label}",
                    )

            # Validate no duplicate ranks
            _rcp_rank_vals = list(_rcp_user_ranks.values())
            _rcp_ranks_valid = len(set(_rcp_rank_vals)) == _rcp_num

            _rcp_submit_cols = st.columns([1, 1])
            with _rcp_submit_cols[0]:
                if st.button("Submit ranking", key="rcp_submit", type="primary", disabled=not _rcp_ranks_valid):
                    # Convert user ranks to ordered list of source keys (best first)
                    _rcp_label_to_src = {lab: src for lab, src in _rcp_shuffle}
                    _rcp_ranked_pairs = sorted(_rcp_user_ranks.items(), key=lambda x: x[1])
                    _rcp_user_ranking_ordered = [_rcp_label_to_src[lab] for lab, _ in _rcp_ranked_pairs]

                    _rcp_result = None
                    try:
                        with st.spinner("Evaluating drafts with LLM judges..."):
                            _rcp_result = _process_ranking_checkpoint(_rcp, _rcp_user_ranking_ordered)
                    except Exception as _rcp_err:
                        st.error(f"Error during evaluation: {_rcp_err}")

                    # Build a summary message and save to conversation history
                    _rcp_src_labels = {"rubric": "Rubric-guided", "coldstart": "Cold-start", "generic": "Generic"}
                    _reveal_parts = [f"Draft {lab} = **{_rcp_src_labels.get(src, src)}**" for lab, src in _rcp_shuffle]
                    _user_rank_display = " > ".join(_rcp_src_labels.get(s, s) for s in _rcp_user_ranking_ordered)
                    _rcp_msg_content = (
                        f"**Alignment Check Complete**\n\n"
                        f"{' | '.join(_reveal_parts)}\n\n"
                        f"Your ranking (best → worst): {_user_rank_display}\n\n"
                        f"*Full judge results available in the Evaluate: Grading tab.*"
                    )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": _rcp_msg_content,
                        "display_content": _rcp_msg_content,
                        "is_system_generated": True,
                        "is_ranking_checkpoint_result": True,
                        "ranking_checkpoint_data": _rcp_result,
                        "message_id": f"rcp_result_{int(time.time() * 1000000)}"
                    })
                    # Clear pending state in case _process_ranking_checkpoint didn't complete
                    st.session_state.ranking_checkpoint_pending = None
                    st.rerun()
            with _rcp_submit_cols[1]:
                if st.button("Cancel", key="rcp_cancel_s3"):
                    st.session_state.ranking_checkpoint_pending = None
                    st.rerun()

            if not _rcp_ranks_valid:
                st.warning("Each draft must have a unique rank.")

    # Delete mode confirmation bar (shown at the bottom when in delete mode)
    if st.session_state.message_delete_mode and st.session_state.messages_to_delete:
        st.warning(f"🗑️ **{len(st.session_state.messages_to_delete)} message(s) selected for deletion**")
        del_col1, del_col2 = st.columns(2)
        with del_col1:
            if st.button("✅ Confirm Delete", use_container_width=True, type="primary"):
                # Delete selected messages (in reverse order to preserve indices)
                for idx in sorted(st.session_state.messages_to_delete, reverse=True):
                    if idx < len(st.session_state.messages):
                        del st.session_state.messages[idx]
                st.session_state.messages_to_delete = set()
                st.session_state.message_delete_mode = False
                st.success("Messages deleted successfully!")
                st.rerun()
        with del_col2:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.messages_to_delete = set()
                st.session_state.message_delete_mode = False
                st.rerun()

    # Display rubric update analysis result if pending
    display_rubric_update_result()

    # Display comparison result if it exists
    if st.session_state.comparison_result:
        st.divider()
        col_title, col_close = st.columns([4, 1])
        
        with col_title:
            st.markdown("### ⚖️ Comparison Response")
        
        with col_close:
            if st.button("✖️ Close", key="close_comparison", use_container_width=True):
                st.session_state.comparison_result = None
                st.session_state.comparison_rubric_version = None
                st.rerun()
        
        comp_result = st.session_state.comparison_result
        clean_text = comp_result['clean_text']
        comparison_assessment = comp_result.get('rubric_assessment')

        # Show which rubric version was used
        comp_rubric_version = st.session_state.comparison_rubric_version
        st.caption(f"Response regenerated with Rubric v{comp_rubric_version}")

        # Display the comparison
        with st.chat_message("assistant"):
            # Check for diff markers
            has_diff_markers = re.search(r'\+[^+]+\+', clean_text) or re.search(r'~[^~]+~', clean_text)

            if has_diff_markers:
                # Show diff highlighting
                diff_html = _md_diff_to_html(clean_text)
                st.markdown(diff_html, unsafe_allow_html=True)
            else:
                # Just show the clean text
                st.markdown(clean_text)

            # Display rubric assessment if available
            if comparison_assessment:
                draft_text = comparison_assessment.get('draft_text')
                display_rubric_assessment(comparison_assessment, draft_text=draft_text)

    # --- A/B Draft Comparison UI ---
    if st.session_state.get("ab_comparison_pending"):
        ab = st.session_state.ab_comparison_pending
        st.markdown("### Which draft do you prefer?")
        st.caption("Two versions were generated — pick the one that better matches what you want.")

        left_content = ab["draft_a"] if ab["left_is_rubric"] else ab["draft_b"]
        right_content = ab["draft_b"] if ab["left_is_rubric"] else ab["draft_a"]

        with st.chat_message("assistant"):
            st.markdown("**Option 1**")
            with st.container(height=400):
                st.markdown(strip_draft_tags_for_streaming(left_content))
            if st.button("Prefer this draft", key="ab_prefer_left", type="primary"):
                chosen_is_rubric = ab["left_is_rubric"]
                chosen_content = left_content
                chosen_thinking = ab["thinking_a"] if chosen_is_rubric else ab["thinking_b"]
                _ab_commit_choice(ab, chosen_content, chosen_thinking, "rubric" if chosen_is_rubric else "conversation_only")
                st.rerun()

        with st.chat_message("assistant"):
            st.markdown("**Option 2**")
            with st.container(height=400):
                st.markdown(strip_draft_tags_for_streaming(right_content))
            if st.button("Prefer this draft", key="ab_prefer_right", type="primary"):
                chosen_is_rubric = not ab["left_is_rubric"]
                chosen_content = right_content
                chosen_thinking = ab["thinking_a"] if chosen_is_rubric else ab["thinking_b"]
                _ab_commit_choice(ab, chosen_content, chosen_thinking, "rubric" if chosen_is_rubric else "conversation_only")
                st.rerun()

    # --- Preference prompt: require writing preferences before first message if no rubric exists ---
    _pref_has_project = bool(st.session_state.get("current_project_id"))
    _pref_rubric_hist = load_rubric_history() if _pref_has_project else []
    _pref_has_rubric = len(_pref_rubric_hist) > 0
    _pref_coldstart = st.session_state.get("infer_coldstart_text", "").strip()
    _pref_has_messages = len(st.session_state.messages) > 0
    _pref_blocked = _pref_has_project and not _pref_has_rubric and not _pref_coldstart and not _pref_has_messages

    if _pref_blocked:
        st.markdown("### Before you start writing")
        st.markdown("Describe your writing preferences for this task. What do you care about? Think about tone, structure, formality, length, style, audience — anything that matters to you.")
        _pref_input = st.text_area(
            "Your writing preferences",
            placeholder="e.g., I prefer a warm but professional tone. I like short paragraphs and clear structure with headers. Avoid jargon. Get to the point quickly — no long introductions. Use active voice.",
            height=150,
            key="chat_pref_input",
            label_visibility="collapsed"
        )
        if st.button("Start Writing", type="primary", use_container_width=True, key="chat_pref_submit"):
            if _pref_input.strip():
                st.session_state.infer_coldstart_text = _pref_input.strip()
                # Save cold-start preference text to database
                _cs_sb = st.session_state.get('supabase')
                _cs_pid = st.session_state.get('current_project_id')
                if _cs_sb and _cs_pid:
                    try:
                        _cs_sb.table("project_data").delete().eq("project_id", _cs_pid).eq("data_type", "coldstart_preferences").execute()
                        _cs_sb.table("project_data").insert({
                            "project_id": _cs_pid,
                            "data_type": "coldstart_preferences",
                            "data": json.dumps({"text": _pref_input.strip(), "timestamp": datetime.now().isoformat(), "source": "chat_coldstart"}),
                            "created_at": datetime.now().isoformat()
                        }).execute()
                    except Exception:
                        pass
                st.rerun()
            else:
                st.warning("Please describe your writing preferences before starting.")

    # Create a container for streaming responses BEFORE chat_input
    # This ensures streaming content appears above the input, not below
    streaming_container = st.container()

    # User input (chat input and buttons)
    if prompt := st.chat_input("Type your message here...", disabled=_pref_blocked):
        # Clear comparison when starting a new message
        st.session_state.comparison_result = None
        st.session_state.comparison_rubric_version = None

        # Check if there's feedback to incorporate
        feedback_context = format_feedback_for_context()

        # Prepare user message data (but DON'T add to session state yet - wait for successful API response)
        if feedback_context:
            full_message = feedback_context + prompt
            user_message_data = {
                "role": "user",
                "content": full_message,  # Full message with feedback for API
                "display_content": prompt  # Just the user's prompt for display
            }
        else:
            full_message = prompt
            user_message_data = {"role": "user", "content": full_message}

        # Create placeholder for user message (so we can clear it on error)
        with streaming_container:
            user_message_placeholder = st.empty()

        # Display user message temporarily while waiting for response
        with user_message_placeholder.container():
            with st.chat_message("user"):
                if feedback_context:
                    st.markdown(full_message)
                else:
                    st.markdown(prompt)

        # Prepare message history for API
        api_messages = []

        # Build rubric version lookup for changelog injection
        _rubric_hist = load_rubric_history()
        _rubric_by_version = {r.get("version"): r.get("rubric", []) for r in _rubric_hist}
        _prev_rubric_version = None

        # Include main conversation messages (skip system messages)
        for msg in st.session_state.messages:
            if msg['role'] in ('user', 'assistant'):
                # Inject rubric version changelog on version transitions
                if msg['role'] == 'assistant' and msg.get('rubric_version'):
                    _cur_v = msg['rubric_version']
                    if _prev_rubric_version is not None and _cur_v != _prev_rubric_version:
                        _old_list = _rubric_by_version.get(_prev_rubric_version, [])
                        _new_list = _rubric_by_version.get(_cur_v, [])
                        _changelog = _build_rubric_version_changelog(_old_list, _new_list, _prev_rubric_version, _cur_v)
                        if _changelog:
                            api_messages.append({"role": "assistant", "content": _changelog})
                    _prev_rubric_version = _cur_v

                content_to_send = msg.get('content', msg.get('display_content', ''))

                # Enrich rubric revision messages with metadata context for Draft A
                if msg.get('rubric_revision') and msg['role'] == 'assistant':
                    rr = msg['rubric_revision']
                    extra_parts = []
                    if rr.get('change_summary'):
                        extra_parts.append(f"**What changed:** {rr['change_summary']}")
                    ann = rr.get('annotated_changes', [])
                    if ann:
                        edit_lines = []
                        for _i, ac in enumerate(ann, 1):
                            reason = (ac.get('reason', '') or '').strip()
                            orig = (ac.get('original_text', '') or '').strip()
                            new = (ac.get('new_text', '') or '').strip()
                            fb = (ac.get('user_feedback', '') or '').strip()
                            edit_desc = f"[{_i}] {reason}"
                            if orig and new:
                                edit_desc += f': "{orig[:100]}" → "{new[:100]}"'
                            elif new:
                                edit_desc += f': added "{new[:100]}"'
                            elif orig:
                                edit_desc += f': removed "{orig[:100]}"'
                            if fb:
                                edit_desc += f' | User feedback: "{fb}"'
                            edit_lines.append(edit_desc)
                        extra_parts.append("**Edits by rubric change:**\n" + "\n".join(edit_lines))
                    # Include rubric suggestion if present
                    sg = msg.get('rubric_suggestion')
                    if sg:
                        sg_text = sg.get('suggestion_text', '')
                        if sg_text:
                            applied_note = f" (Applied as rubric v{sg['applied_version']})" if sg.get('applied') else ""
                            extra_parts.append(f"**Rubric suggestion{applied_note}:**\n{sg_text}")
                    if extra_parts:
                        content_to_send = content_to_send + "\n\n" + "\n\n".join(extra_parts)

                api_messages.append({
                    "role": msg['role'],
                    "content": content_to_send
                })

        # Add the new user message to API messages (for the API call)
        api_messages.append({"role": "user", "content": full_message})

        # Show assistant response with streaming (in the streaming container)
        with streaming_container:
            with st.chat_message("assistant"):
                response_placeholder = st.empty()

                # Generate unique message ID for this response (use timestamp to ensure uniqueness)
                import time
                message_id = f"assistant_{int(time.time() * 1000000)}"

                # Get rubric to use
                active_rubric_dict, _, _ = get_active_rubric()
                if active_rubric_dict and isinstance(active_rubric_dict, dict):
                    active_rubric_list = active_rubric_dict.get("rubric", [])
                else:
                    active_rubric_dict = None
                    active_rubric_list = []

                # Build system instruction (pass full dict so it can check source)
                system_instruction = CHAT_build_system_prompt(active_rubric_dict if active_rubric_dict else [])

                # Show which rubric is being used
                if active_rubric_list:
                    source = active_rubric_dict.get("source", "inferred") if active_rubric_dict else "inferred"
                    template_note = f" (template)" if source == "template" else ""
                    st.caption(f"🔍 Using rubric with {len(active_rubric_list)} criteria{template_note}")
                else:
                    st.caption("🔍 No rubric active - system instruction will not include rubric")

                # Create placeholder for thinking display
                thinking_placeholder = st.empty()

                # Status placeholder for retry messages
                status_placeholder = st.empty()

                # Stream the response with retry logic for overloaded errors
                max_retries = 3
                retry_delay = 5  # seconds

                for attempt in range(max_retries):
                    try:
                        with client.messages.stream(
                            max_tokens=32000,
                            system=system_instruction,
                            messages=api_messages,
                            model=MODEL_PRIMARY,
                            thinking={
                                "type": "enabled",
                                "budget_tokens": 10000
                            }
                        ) as stream:
                            # Clear any retry status message
                            status_placeholder.empty()

                            # Stream and filter out analysis tags in real-time
                            # Returns: (main_content without analysis, analysis_content, None for rubric_assessment, thinking_content)
                            main_content, analysis_content, _, thinking_content = stream_without_analysis(stream, response_placeholder, message_id, thinking_placeholder)

                        # Clear the thinking placeholder and show final thinking in expander
                        thinking_placeholder.empty()

                        # Get the currently active rubric version to store with the message
                        active_rubric_dict, active_idx, _ = get_active_rubric()
                        rubric_version = active_rubric_dict.get('version', 1) if active_rubric_dict else None

                        # --- A/B Draft Comparison: check if this response has a draft ---
                        has_draft_tag = bool(re.search(r'<draft>.*?</draft>', main_content, re.DOTALL))
                        trigger_ab = False
                        print(f"[A/B DEBUG] has_draft_tag={has_draft_tag}, ab_draft_count={st.session_state.get('ab_draft_count', 0)}, active_rubric_list={bool(active_rubric_list)}, AB_DRAFT_INTERVAL={AB_DRAFT_INTERVAL}")
                        if has_draft_tag:
                            st.session_state.ab_draft_count = st.session_state.get('ab_draft_count', 0) + 1
                            # Trigger A/B comparison every AB_DRAFT_INTERVAL drafts
                            if st.session_state.ab_draft_count >= AB_DRAFT_INTERVAL and active_rubric_list:
                                trigger_ab = True
                                st.session_state.ab_draft_count = 0
                        print(f"[A/B DEBUG] trigger_ab={trigger_ab}")

                        if trigger_ab:
                            # Generate Draft B (conversation-only, no rubric in system prompt)
                            # Build filtered messages: exclude system-generated messages
                            # (DP reviews, rubric revisions, assessment messages)
                            conv_only_messages = []
                            for msg in st.session_state.messages:
                                if msg['role'] not in ('user', 'assistant'):
                                    continue
                                if msg.get('is_system_generated'):
                                    continue
                                content_to_send = msg.get('content', msg.get('display_content', ''))
                                conv_only_messages.append({"role": msg['role'], "content": content_to_send})
                            # Add the current user message
                            conv_only_messages.append({"role": "user", "content": full_message})

                            # DEBUG: log both message lists
                            print("\n===== A/B DRAFT DEBUG =====")
                            print(f"Draft A (rubric-guided): {len(api_messages)} messages, system prompt length: {len(system_instruction)}")
                            for i, m in enumerate(api_messages):
                                print(f"  [{i}] {m['role']}: {m['content']}...")
                            print(f"\nDraft B (conv-only): {len(conv_only_messages)} messages")
                            for i, m in enumerate(conv_only_messages):
                                print(f"  [{i}] {m['role']}: {m['content']}...")
                            print("===========================\n")

                            response_placeholder.markdown("*Generating alternative draft for comparison...*")
                            conv_only_system = CHAT_build_system_prompt([])
                            try:
                                conv_only_content = ""
                                conv_only_thinking = ""
                                with client.messages.stream(
                                    max_tokens=32000,
                                    system=conv_only_system,
                                    messages=conv_only_messages,
                                    model=MODEL_PRIMARY,
                                    thinking={
                                        "type": "enabled",
                                        "budget_tokens": 10000
                                    }
                                ) as conv_only_stream:
                                    for event in conv_only_stream:
                                        if event.type == "content_block_delta":
                                            if hasattr(event.delta, 'thinking'):
                                                conv_only_thinking += event.delta.thinking
                                            elif hasattr(event.delta, 'text'):
                                                conv_only_content += event.delta.text
                                # Strip analysis tags from Draft B
                                _, conv_only_content = parse_analysis_and_content(conv_only_content)
                                if '<rubric_assessment>' in conv_only_content:
                                    conv_only_content = conv_only_content.split('<rubric_assessment>')[0].strip()
                            except Exception as ab_err:
                                # If Draft B fails, fall through to normal flow
                                print(f"[A/B DEBUG] Draft B generation FAILED: {ab_err}")
                                conv_only_content = None

                            if conv_only_content:
                                print(f"[A/B DEBUG] Draft B generated successfully, length={len(conv_only_content)}")
                                # Randomly assign left/right positions
                                left_is_rubric = random.choice([True, False])

                                # Store A/B comparison state — DON'T add messages yet
                                st.session_state.ab_comparison_pending = {
                                    "draft_a": main_content,              # rubric-guided
                                    "draft_b": conv_only_content,         # conversation-only
                                    "left_is_rubric": left_is_rubric,
                                    "user_message_data": user_message_data,
                                    "message_id": message_id,
                                    "rubric_version": rubric_version,
                                    "thinking_a": thinking_content,
                                    "thinking_b": conv_only_thinking,
                                    "analysis_content": analysis_content,
                                    "feedback_context": bool(feedback_context),
                                }

                                st.rerun()
                                break  # Success, exit retry loop

                        # Normal flow: store messages and rerun
                        message_data = {
                            "role": "assistant",
                            "content": main_content,
                            "display_content": main_content,
                            "message_id": message_id,
                            "rubric_version": rubric_version,
                            "rubric_assessment": None,
                            "thinking": thinking_content
                        }

                        # SUCCESS - Now add both user and assistant messages to session state
                        st.session_state.messages.append(user_message_data)
                        st.session_state.messages.append(message_data)

                        # Clear the feedback after successful response
                        if feedback_context:
                            st.session_state.assessment_feedback = {}

                        # Update analysis in session state and rerun to show in sidebar
                        st.session_state.current_analysis = analysis_content

                        st.rerun()
                        break  # Success, exit retry loop

                    except Exception as e:
                        error_str = str(e)

                        # Check if it's an overloaded error
                        if 'overloaded' in error_str.lower() or 'Overloaded' in error_str:
                            if attempt < max_retries - 1:
                                # Show retry message with countdown
                                for seconds_left in range(retry_delay, 0, -1):
                                    status_placeholder.warning(
                                        f"⏳ Claude's servers are experiencing high demand. "
                                        f"Retrying in {seconds_left} seconds... (Attempt {attempt + 2}/{max_retries})"
                                    )
                                    time.sleep(1)
                                status_placeholder.info("🔄 Retrying now...")
                            else:
                                # Final attempt failed - clear the user message and show error
                                user_message_placeholder.empty()
                                status_placeholder.empty()
                                st.error(
                                    "Claude's servers are currently overloaded. "
                                    "Please wait a moment and try again."
                                )
                        else:
                            # Non-overload error, don't retry - clear user message
                            user_message_placeholder.empty()
                            st.error(f"Error occurred: {error_str}")
                            break
    
    # Buttons below chat input
    if st.session_state.messages:
        btn_col1, btn_col2, btn_col3, btn_col4, btn_col5 = st.columns(5)

        with btn_col1:
            save_button = st.button("💾 Save Conversation", use_container_width=True)
            if save_button:
                try:
                    # Wait for background A/B judge to finish (if running) so results embed in messages
                    _ab_evt = st.session_state.get("_ab_judge_done_event")
                    if _ab_evt is not None and not _ab_evt.is_set():
                        with st.spinner("Waiting for judge evaluation to finish..."):
                            _ab_evt.wait(timeout=15)  # wait up to 15s
                    # Persist any rubric edit feedback from widget state onto messages before saving
                    for _sv_msg in st.session_state.messages:
                        if _sv_msg.get('rubric_revision') and _sv_msg.get('role') == 'assistant':
                            _sv_ann = _sv_msg['rubric_revision'].get('annotated_changes', [])
                            _sv_mid = _sv_msg.get('message_id', '')
                            _sv_safe_id = re.sub(r'[^a-zA-Z0-9_-]', '_', str(_sv_mid)) if _sv_mid else ''
                            if _sv_safe_id and _sv_ann:
                                for _sv_j, _sv_ac in enumerate(_sv_ann, 1):
                                    _sv_fb = (st.session_state.get(f"rubric_edit_fb_{_sv_safe_id}_{_sv_j}", "") or "").strip()
                                    if _sv_fb:
                                        _sv_ac["user_feedback"] = _sv_fb
                    # Save the conversation to Supabase
                    conv_id = save_message_log(st.session_state.messages, st.session_state.rubric, st.session_state.current_analysis)

                    if conv_id:
                        # Update selected conversation — the selectbox will pick this
                        # up on the next rerun via the just_saved / selected_conversation sync
                        st.session_state.just_saved = conv_id
                        st.session_state.selected_conversation = conv_id
                        st.toast("Conversation saved!")
                        st.rerun()
                    else:
                        st.error("Failed to save conversation")
                except Exception as e:
                    st.error(f"Error saving log: {str(e)}")
        
        with btn_col2:
            infer_button = st.button("🔍 Infer Rubric", use_container_width=True)
            if infer_button:
                if not st.session_state.messages:
                    st.error("No conversation to infer rubric from!")
                else:
                    # Single call: infer rubric + extract DPs together
                    # Filter out non-conversation messages so numbering matches chat display
                    _infer_filtered = [
                        m for m in copy.deepcopy(st.session_state.messages)
                        if m.get('role') in ('user', 'assistant')
                        and not m.get('is_assessment_message')
                        and not m.get('is_dp_review')
                    ]
                    # Inject rubric version changelog messages between version transitions
                    _infer_hist = load_rubric_history()
                    _infer_by_ver = {r.get("version"): r.get("rubric", []) for r in _infer_hist}
                    _infer_prev_v = None
                    conversation_for_infer = []
                    for _inf_msg in _infer_filtered:
                        if _inf_msg['role'] == 'assistant' and _inf_msg.get('rubric_version'):
                            _inf_cur_v = _inf_msg['rubric_version']
                            if _infer_prev_v is not None and _inf_cur_v != _infer_prev_v:
                                _inf_cl = _build_rubric_version_changelog(
                                    _infer_by_ver.get(_infer_prev_v, []),
                                    _infer_by_ver.get(_inf_cur_v, []),
                                    _infer_prev_v, _inf_cur_v
                                )
                                if _inf_cl:
                                    conversation_for_infer.append({"role": "assistant", "content": _inf_cl, "_synthetic_changelog": True})
                            _infer_prev_v = _inf_cur_v
                        conversation_for_infer.append(_inf_msg)
                    st.session_state.infer_dp_messages = copy.deepcopy(conversation_for_infer)
                    st.session_state.infer_dp_conversation = "__from_infer_rubric__"

                    # Step 1: Infer rubric ONLY (no DPs yet)
                    with st.spinner("Inferring rubric from conversation..."):
                        rubric_data = infer_rubric_only(conversation_for_infer)
                        if rubric_data:
                            _rb_ver = rubric_data.get("version", "?")

                            # Clear stale DP state — DPs will be extracted after classification
                            st.session_state.infer_decision_points = None
                            st.session_state.infer_dp_dimension_confirmed = False
                            st.session_state.infer_dp_user_mappings = {}
                            st.session_state.chat_classification_feedback = {}
                            st.session_state.chat_criteria_hallucination_reasons = {}

                            # Remove existing DP review messages
                            st.session_state.messages = [m for m in st.session_state.messages if not m.get('is_dp_review')]

                            # Append rubric-inferred message (no DPs yet)
                            _rb_criteria_log = rubric_data.get("rubric", [])
                            _rb_review_content = f"**Rubric v{_rb_ver} inferred.** Review criteria classifications below."
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": _rb_review_content,
                                "display_content": _rb_review_content,
                                "is_dp_review": True,
                                "is_system_generated": True,
                                "message_id": f"dp_review_{int(time.time() * 1000000)}",
                                "dp_data": {
                                    "decision_points": [],  # Empty — DPs extracted after classification
                                    "rubric_version": _rb_ver,
                                    "rubric": copy.deepcopy(_rb_criteria_log),
                                },
                            })

                            # Step 1b: Run LLM criteria classification with conversation context
                            _cs_text_infer = st.session_state.get("infer_coldstart_text", "").strip()
                            if _cs_text_infer and rubric_data.get("rubric"):
                                _class_conv_text = _build_conversation_text(conversation_for_infer)
                                with st.spinner("Classifying criteria against your writing preferences..."):
                                    try:
                                        _class_rubric_json = json.dumps(rubric_data.get("rubric", []), ensure_ascii=False, indent=2)
                                        _class_prompt = RUBRIC_compare_to_coldstart_prompt(_class_rubric_json, _cs_text_infer, _class_conv_text)
                                        _class_response = _api_call_with_retry(
                                            model=MODEL_PRIMARY,
                                            max_tokens=16000,
                                            messages=[{"role": "user", "content": _class_prompt}],
                                            thinking={"type": "enabled", "budget_tokens": 8000}
                                        )
                                        _class_text = ""
                                        for block in _class_response.content:
                                            if block.type == "text":
                                                _class_text += block.text
                                        _class_json_match = re.search(r'\{[\s\S]*\}', _class_text)
                                        if _class_json_match:
                                            _class_parsed = json.loads(_class_json_match.group())
                                            st.session_state.chat_criteria_llm_classification = _class_parsed
                                            _class_user = {}
                                            for _cc in _class_parsed.get("criteria_comparison", []):
                                                _cc_name = _cc.get("criterion_name", "")
                                                _cc_status = _cc.get("status", "unstated")
                                                _class_user[_cc_name] = _cc_status
                                            st.session_state.chat_criteria_user_classifications = _class_user
                                            st.session_state.chat_criteria_review_active = True
                                            st.session_state.chat_criteria_review_confirmed = False
                                    except Exception:
                                        pass

                            st.success(f"Rubric v{_rb_ver} inferred. Review criteria classifications in the chat.")
                            st.rerun()
                        else:
                            st.error("Failed to infer rubric from conversation.")
        
        with btn_col3:
            # Only show assess button if there's an assistant message and active rubric
            active_rubric_dict, _, _ = get_active_rubric()

            has_assistant_message = any(msg['role'] == 'assistant' for msg in st.session_state.messages)

            if has_assistant_message and active_rubric_dict:
                assess_button = st.button("📊 Assess Draft", use_container_width=True)
                if assess_button:
                    # Get the last assistant message with a draft
                    last_assistant_msg = None
                    last_assistant_idx = None
                    draft_content = None

                    # Pattern to extract draft content
                    draft_pattern = r'<draft>(.*?)</draft>'

                    # Look in main messages (from most recent to oldest)
                    for i in range(len(st.session_state.messages) - 1, -1, -1):
                        if st.session_state.messages[i]['role'] == 'assistant':
                            msg_content = st.session_state.messages[i].get('display_content', st.session_state.messages[i].get('content', ''))
                            match = re.search(draft_pattern, msg_content, re.DOTALL)
                            if match:
                                last_assistant_msg = st.session_state.messages[i]
                                last_assistant_idx = i
                                draft_content = match.group(1).strip()
                                break

                    # Alert user if no draft found
                    if draft_content is None:
                        st.warning("⚠️ No draft found in recent messages. The assistant message must contain text wrapped in `<draft></draft>` tags to be assessed.")
                    elif last_assistant_msg:
                        with st.spinner("Evaluating draft against rubric..."):
                            try:
                                from prompts import CHAT_ASSESS_DRAFT_PROMPT

                                # Build the assessment prompt with the draft content
                                assessment_prompt = f"""{CHAT_ASSESS_DRAFT_PROMPT}
                                    ## Draft to Assess

                                    <draft>
                                    {draft_content}
                                    </draft>
                                    """

                                # Build the conversation history with assessment prompt as the last message
                                assessment_messages = []

                                # Include all user/assistant messages (skip system messages)
                                for msg in st.session_state.messages:
                                    if msg['role'] in ('user', 'assistant'):
                                        assessment_messages.append({
                                            "role": msg['role'],
                                            "content": msg['content']
                                        })

                                # Add the assessment prompt as the last user message
                                assessment_messages.append({
                                    "role": "user",
                                    "content": assessment_prompt
                                })

                                # Get rubric for system instruction
                                rubric_for_system = active_rubric_dict if active_rubric_dict else []

                                system_instruction = CHAT_build_system_prompt(rubric_for_system)

                                # Make API call with full conversation context
                                assessment_response = _api_call_with_retry(
                                    max_tokens=16000,
                                    system=system_instruction,
                                    messages=assessment_messages,
                                    model=MODEL_PRIMARY,
                                    thinking={
                                        "type": "enabled",
                                        "budget_tokens": 8000
                                    }
                                )

                                # Extract thinking and text from response
                                thinking_text = ""
                                assessment_text = ""
                                for block in assessment_response.content:
                                    if block.type == "thinking":
                                        thinking_text = block.thinking
                                    elif block.type == "text":
                                        assessment_text = block.text

                                rubric_assessment = parse_rubric_assessment(assessment_text)
                                # Store thinking and draft text in the assessment
                                if rubric_assessment:
                                    rubric_assessment['thinking'] = thinking_text
                                    rubric_assessment['draft_text'] = draft_content  # Store for evidence highlighting

                                # Update the last assistant message with the assessment (for UI display)
                                st.session_state.messages[last_assistant_idx]['rubric_assessment'] = rubric_assessment

                                # Add the assessment to conversation history
                                # First add the assessment prompt as a user message (hidden from display)
                                st.session_state.messages.append({
                                    "role": "user",
                                    "content": assessment_prompt,
                                    "is_assessment_message": True,
                                    "is_system_generated": True,
                                })

                                # Then add the full assessment text as an assistant message (hidden from display)
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": assessment_text,
                                    "display_content": assessment_text,
                                    "message_id": str(uuid.uuid4()),
                                    "rubric_version": active_rubric_dict.get("version") if active_rubric_dict else None,
                                    "rubric_assessment": rubric_assessment,
                                    "is_assessment_message": True,
                                    "is_system_generated": True,
                                })

                                st.success("✓ Draft assessed successfully!")
                                st.rerun()

                            except Exception as e:
                                st.error(f"Error during assessment: {str(e)}")

        with btn_col4:
            # Toggle delete mode button
            if st.session_state.message_delete_mode:
                if st.button("✖️ Cancel Delete", use_container_width=True, type="secondary"):
                    st.session_state.message_delete_mode = False
                    st.session_state.messages_to_delete = set()
                    st.rerun()
            else:
                if st.button("🗑️ Delete Messages", use_container_width=True):
                    st.session_state.message_delete_mode = True
                    st.session_state.messages_to_delete = set()
                    st.rerun()

        with btn_col5:
            clear_button = st.button("🗑️ Clear All", use_container_width=True)
            if clear_button:
                st.session_state.messages = []
                st.session_state.current_analysis = ""
                st.session_state.selected_conversation = None
                st.session_state.comparison_result = None
                st.session_state.comparison_rubric_version = None
                st.session_state.message_delete_mode = False
                st.session_state.messages_to_delete = set()
                st.session_state.ab_draft_count = 0
                st.session_state.ab_comparison_pending = None
                st.rerun()

    # Comparison mode UI (only show if there are messages and an assistant response)
# Sidebar for rubric input
with st.sidebar:

    # Project Selector at the top
    st.header("📁 Project")

    # Get available projects from Supabase
    available_projects = get_available_projects()

    # Project selector
    if available_projects:
        # Create mapping of names to IDs
        project_names = [p['name'] for p in available_projects]
        project_id_map = {p['name']: p['id'] for p in available_projects}

        # Find current index
        current_idx = 0
        if st.session_state.current_project in project_names:
            current_idx = project_names.index(st.session_state.current_project)

        selected_project_name = st.selectbox(
            "Select Project:",
            options=project_names,
            index=current_idx,
            key="project_selector"
        )

        # Update current_project if selection changed
        if selected_project_name != st.session_state.current_project:
            st.session_state.current_project = selected_project_name
            st.session_state.current_project_id = project_id_map[selected_project_name]
            # Reset session state
            st.session_state.messages = []
            st.session_state.selected_conversation = None
            st.session_state.ab_draft_count = 0
            st.session_state.ab_comparison_pending = None
            if 'active_rubric_idx' in st.session_state:
                del st.session_state.active_rubric_idx

            # Load the new project's rubric
            active_rubric_dict, active_idx, _ = get_active_rubric()
            if active_rubric_dict and active_rubric_dict.get("rubric"):
                rubric_list = active_rubric_dict.get("rubric", [])
                st.session_state.rubric = rubric_list
                st.session_state.editing_criteria = copy.deepcopy(rubric_list)
            else:
                st.session_state.rubric = []
                st.session_state.editing_criteria = []

            # Load survey responses from database
            _supabase = st.session_state.get('supabase')
            _new_pid = project_id_map[selected_project_name]
            if _supabase and _new_pid:
                _loaded_survey = load_project_data(_supabase, _new_pid, "survey_responses")
                if _loaded_survey:
                    # save_project_data appends to a list, so get the latest entry
                    if isinstance(_loaded_survey, list) and len(_loaded_survey) > 0:
                        _latest = _loaded_survey[-1]
                        if isinstance(_latest, dict) and "task_a" in _latest:
                            st.session_state.survey_responses = _latest
                        else:
                            st.session_state.survey_responses = {"task_a": {}, "task_b": {}, "final": {}}
                    elif isinstance(_loaded_survey, dict) and "task_a" in _loaded_survey:
                        st.session_state.survey_responses = _loaded_survey
                    else:
                        st.session_state.survey_responses = {"task_a": {}, "task_b": {}, "final": {}}
                else:
                    st.session_state.survey_responses = {"task_a": {}, "task_b": {}, "final": {}}
            else:
                st.session_state.survey_responses = {"task_a": {}, "task_b": {}, "final": {}}
            # Load cold-start preferences from database
            st.session_state.infer_coldstart_text = ""
            st.session_state.infer_coldstart_saved = False
            if _supabase and _new_pid:
                try:
                    _cs_raw = _supabase.table("project_data").select("data").eq("project_id", _new_pid).eq("data_type", "coldstart_preferences").execute()
                    if _cs_raw.data and _cs_raw.data[0].get("data"):
                        _cs_data = _cs_raw.data[0]["data"]
                        if isinstance(_cs_data, str):
                            _cs_data = json.loads(_cs_data)
                        if isinstance(_cs_data, dict) and _cs_data.get("text"):
                            st.session_state.infer_coldstart_text = _cs_data["text"]
                            st.session_state.infer_coldstart_saved = True
                except Exception:
                    pass

            # Load infer conversations from database
            if _supabase and _new_pid:
                try:
                    _infer_conv_raw = _supabase.table("project_data").select("data").eq("project_id", _new_pid).eq("data_type", "infer_conversation").execute()
                    if _infer_conv_raw.data and _infer_conv_raw.data[0].get("data"):
                        _raw_ic = _infer_conv_raw.data[0]["data"]
                        _infer_conv_loaded = json.loads(_raw_ic) if isinstance(_raw_ic, str) else _raw_ic
                        if isinstance(_infer_conv_loaded, str):
                            _infer_conv_loaded = json.loads(_infer_conv_loaded)
                        if isinstance(_infer_conv_loaded, list):
                            # New format: list of infer conversations
                            st.session_state.infer_all_conversations = _infer_conv_loaded
                            # Default to the latest one
                            if _infer_conv_loaded:
                                _latest_conv = _infer_conv_loaded[-1]
                                st.session_state.infer_dp_messages = _latest_conv.get("messages", [])
                                st.session_state.infer_dp_conversation = "__from_infer_rubric__"
                                if _latest_conv.get("decision_points"):
                                    st.session_state.infer_decision_points = _latest_conv["decision_points"]
                        elif isinstance(_infer_conv_loaded, dict) and "messages" in _infer_conv_loaded:
                            # Legacy format: single conversation dict
                            st.session_state.infer_all_conversations = [_infer_conv_loaded]
                            st.session_state.infer_dp_messages = _infer_conv_loaded["messages"]
                            st.session_state.infer_dp_conversation = "__from_infer_rubric__"
                            if _infer_conv_loaded.get("decision_points"):
                                st.session_state.infer_decision_points = _infer_conv_loaded["decision_points"]
                        else:
                            st.session_state.infer_all_conversations = []
                            st.session_state.infer_dp_messages = []
                    else:
                        st.session_state.infer_all_conversations = []
                        st.session_state.infer_dp_messages = []
                except Exception:
                    st.session_state.infer_all_conversations = []
                    st.session_state.infer_dp_messages = []

            # Load A/B judge results and ranking checkpoint results from database
            st.session_state.ab_judge_results = []
            st.session_state.ranking_checkpoint_results = []
            st.session_state.ranking_checkpoint_pending = None
            st.session_state.ranking_checkpoint_auto_triggered = False
            if _supabase and _new_pid:
                try:
                    _ab_loaded = load_project_data(_supabase, _new_pid, "ab_judge_results")
                    if isinstance(_ab_loaded, list):
                        st.session_state.ab_judge_results = _ab_loaded
                except Exception:
                    pass
                try:
                    _rk_loaded = load_project_data(_supabase, _new_pid, "ranking_checkpoint")
                    if isinstance(_rk_loaded, list):
                        st.session_state.ranking_checkpoint_results = _rk_loaded
                except Exception:
                    pass

            st.rerun()

        # Ensure current_project_id is set if we have a current project
        if st.session_state.current_project and not st.session_state.current_project_id:
            st.session_state.current_project_id = project_id_map.get(st.session_state.current_project)

        # Load infer conversations from DB on startup if not already loaded
        # (handles the case where user logs back in and project is already selected)
        _startup_pid = st.session_state.get('current_project_id')
        _startup_sb = st.session_state.get('supabase')
        if _startup_pid and _startup_sb and not st.session_state.get('infer_all_conversations'):
            try:
                _startup_raw = _startup_sb.table("project_data").select("data").eq("project_id", _startup_pid).eq("data_type", "infer_conversation").execute()
                if _startup_raw.data and _startup_raw.data[0].get("data"):
                    _raw_data = _startup_raw.data[0]["data"]
                    # Handle both string (text column) and already-parsed (jsonb column)
                    if isinstance(_raw_data, str):
                        _startup_loaded = json.loads(_raw_data)
                    else:
                        _startup_loaded = _raw_data
                    # May be double-encoded: a string inside jsonb
                    if isinstance(_startup_loaded, str):
                        _startup_loaded = json.loads(_startup_loaded)
                    if isinstance(_startup_loaded, list) and _startup_loaded:
                        st.session_state.infer_all_conversations = _startup_loaded
                        _latest = _startup_loaded[-1]
                        st.session_state.infer_dp_messages = _latest.get("messages", [])
                        st.session_state.infer_dp_conversation = "__from_infer_rubric__"
                        if _latest.get("decision_points"):
                            st.session_state.infer_decision_points = _latest["decision_points"]
                    elif isinstance(_startup_loaded, dict) and "messages" in _startup_loaded:
                        st.session_state.infer_all_conversations = [_startup_loaded]
                        st.session_state.infer_dp_messages = _startup_loaded["messages"]
                        st.session_state.infer_dp_conversation = "__from_infer_rubric__"
                        if _startup_loaded.get("decision_points"):
                            st.session_state.infer_decision_points = _startup_loaded["decision_points"]
            except Exception as _e:
                st.warning(f"Could not load infer conversations: {_e}")

        # Also load survey responses on startup if not already loaded
        if _startup_pid and _startup_sb and not st.session_state.get('survey_responses', {}).get('task_a', {}).get('completed'):
            try:
                _startup_survey = load_project_data(_startup_sb, _startup_pid, "survey_responses")
                if _startup_survey:
                    if isinstance(_startup_survey, list):
                        _startup_survey = _startup_survey[-1]
                    if isinstance(_startup_survey, dict):
                        st.session_state.survey_responses = _startup_survey
            except Exception:
                pass

        # Load A/B judge results and ranking checkpoint results on startup if not already loaded
        if _startup_pid and _startup_sb and not st.session_state.get('ab_judge_results'):
            try:
                _ab_startup = load_project_data(_startup_sb, _startup_pid, "ab_judge_results")
                if isinstance(_ab_startup, list):
                    st.session_state.ab_judge_results = _ab_startup
            except Exception:
                pass
        if _startup_pid and _startup_sb and not st.session_state.get('ranking_checkpoint_results'):
            try:
                _rk_startup = load_project_data(_startup_sb, _startup_pid, "ranking_checkpoint")
                if isinstance(_rk_startup, list):
                    st.session_state.ranking_checkpoint_results = _rk_startup
            except Exception:
                pass
    else:
        st.info("No projects found. Create one below!")

    # Delete project section
    if available_projects and st.session_state.current_project:
        with st.expander("🗑️ Delete Project"):
            st.warning(f"This will permanently delete **{st.session_state.current_project}** and all its data (conversations, rubrics, evaluations).")
            confirm_name = st.text_input(
                "Type the project name to confirm:",
                placeholder=st.session_state.current_project,
                key="delete_project_confirm"
            )
            if st.button("Delete Project", type="primary", use_container_width=True, key="delete_project_btn"):
                if confirm_name.strip() == st.session_state.current_project:
                    supabase = st.session_state.get('supabase')
                    user_id = st.session_state.get('auth_username')
                    project_id = st.session_state.current_project_id
                    # Fallback: resolve project_id from map if not set
                    if not project_id and st.session_state.current_project in project_id_map:
                        project_id = project_id_map[st.session_state.current_project]
                        st.session_state.current_project_id = project_id
                    if supabase and user_id and project_id:
                        success, message = delete_project(supabase, user_id, project_id)
                        if success:
                            st.session_state.current_project = None
                            st.session_state.current_project_id = None
                            st.session_state.rubric = []
                            st.session_state.editing_criteria = []
                            st.session_state.messages = []
                            st.session_state.selected_conversation = None
                            st.session_state.survey_responses = {"task_a": {}, "task_b": {}, "final": {}}
                            st.session_state.ab_draft_count = 0
                            st.session_state.ab_comparison_pending = None
                            if 'active_rubric_idx' in st.session_state:
                                del st.session_state.active_rubric_idx
                            if 'delete_project_confirm' in st.session_state:
                                del st.session_state.delete_project_confirm
                            st.rerun()
                        else:
                            st.error(message)
                    else:
                        _missing = []
                        if not supabase: _missing.append("database connection")
                        if not user_id: _missing.append("user ID")
                        if not project_id: _missing.append("project ID")
                        st.error(f"Could not delete project. Missing: {', '.join(_missing)}. Try refreshing the page.")
                else:
                    st.error("Project name does not match. Please type the exact project name to confirm deletion.")

    # Create new project section
    # Initialize expander state if not exists
    if 'create_project_expanded' not in st.session_state:
        st.session_state.create_project_expanded = False

    with st.expander("➕ Create New Project", expanded=st.session_state.create_project_expanded):
        new_project_name = st.text_input(
            "Project Name:",
            placeholder="e.g., my-essay-project",
            key="new_project_name"
        )

        if st.button("Create Project", use_container_width=True):
            if new_project_name.strip():
                success, message, project_id = create_new_project(new_project_name.strip())
                if success:
                    st.success(message)
                    # Update current project
                    st.session_state.current_project = new_project_name.strip()
                    st.session_state.current_project_id = project_id
                    # Reset session state
                    st.session_state.rubric = []
                    st.session_state.editing_criteria = []
                    st.session_state.messages = []
                    st.session_state.selected_conversation = None
                    st.session_state.infer_coldstart_text = ""
                    st.session_state.ab_draft_count = 0
                    st.session_state.ab_comparison_pending = None
                    st.session_state.infer_coldstart_saved = False
                    st.session_state.chat_criteria_llm_classification = None
                    st.session_state.chat_criteria_user_classifications = {}
                    st.session_state.chat_criteria_review_active = False
                    st.session_state.chat_criteria_review_confirmed = False
                    st.session_state.chat_classification_feedback = {}
                    st.session_state.chat_criteria_hallucination_reasons = {}
                    # Clear the text input and collapse expander
                    if 'new_project_name' in st.session_state:
                        del st.session_state.new_project_name
                    st.session_state.create_project_expanded = False
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.error("Please enter a project name")

    # Export/Import Project section - Note: Data is stored in cloud database
    # with st.expander("📦 Export / Import Project"):
    #     st.info("Your data is stored securely in the cloud and syncs automatically across devices.")
    #     st.markdown("**Export** and **Import** features coming soon for cloud storage.")

    st.divider()

    st.header("📋 Rubric Configuration")

    # Get active rubric
    active_rubric_dict, active_idx, rubric_history = get_active_rubric()

    # Show general rubric selector when no rubric history exists
    if not rubric_history:
        general_rubrics = load_general_rubrics()
        if general_rubrics:
            st.markdown("### 📚 Start with a Template Rubric")
            st.caption("Select a pre-built rubric to get started, or have a conversation and use 'Infer Rubric' to create a custom one.")

            rubric_options = ["-- Select a template --"] + list(general_rubrics.keys())
            selected_template = st.selectbox(
                "Choose a rubric template:",
                options=rubric_options,
                key="general_rubric_selector"
            )

            if selected_template and selected_template != "-- Select a template --":
                template_data = general_rubrics[selected_template]

                # Show preview of the selected rubric
                with st.expander("📋 Preview rubric", expanded=False):
                    writing_type = template_data.get("writing_type", "Not specified")
                    st.markdown(f"**Writing Type:** {writing_type[:200]}..." if len(writing_type) > 200 else f"**Writing Type:** {writing_type}")
                    rubric_list = template_data.get("rubric", [])
                    st.markdown(f"**Criteria:** {len(rubric_list)} items")
                    for criterion in rubric_list[:5]:  # Show first 5
                        st.markdown(f"- {criterion.get('name', 'Unnamed')}")
                    if len(rubric_list) > 5:
                        st.markdown(f"*... and {len(rubric_list) - 5} more*")

                if st.button("✅ Use this rubric", key="use_general_rubric", type="primary"):
                    # Add the selected rubric to history with version 1
                    new_rubric = template_data.copy()
                    new_rubric["version"] = 1
                    hist = [new_rubric]
                    save_rubric_history(hist)
                    st.session_state.active_rubric_idx = 0
                    st.session_state.rubric = new_rubric.get("rubric", [])
                    st.session_state.editing_criteria = copy.deepcopy(new_rubric.get("rubric", []))
                    st.success(f"✓ '{selected_template}' rubric loaded!")
                    st.rerun()

            st.divider()

    # Initialize editing criteria if needed
    if "editing_criteria" not in st.session_state:
        if active_rubric_dict:
            rubric_list = active_rubric_dict.get("rubric", [])
            st.session_state.editing_criteria = copy.deepcopy(rubric_list) if rubric_list else []
        else:
            st.session_state.editing_criteria = []
    # Bump this when editing_criteria is updated from outside the sidebar (e.g. Apply suggestion) so widget keys change and inputs show new values
    if "editing_criteria_ui_version" not in st.session_state:
        st.session_state.editing_criteria_ui_version = 0

    # Version selector
    if rubric_history:
        version_options = [f"v{r.get('version', 1)}" for r in rubric_history]
        selected_version = st.selectbox(
            "Active Rubric Version:",
            options=version_options,
            index=active_idx if active_idx is not None else 0,
            key="rubric_version_selector"
        )
        if selected_version:
            new_idx = version_options.index(selected_version)
            if new_idx != active_idx:
                st.session_state.active_rubric_idx = new_idx
                # Update session state rubric
                active_rubric_dict, _, _ = get_active_rubric()
                rubric_list = active_rubric_dict.get("rubric", []) if active_rubric_dict else []
                st.session_state.rubric = rubric_list
                # Reset editing criteria with deep copy to avoid modifying original
                if active_rubric_dict:
                    st.session_state.editing_criteria = copy.deepcopy(rubric_list)
                st.rerun()

    # Display current rubric
    if st.session_state.editing_criteria:
        st.markdown("### Current Criteria")

        version_key = st.session_state.active_rubric_idx if st.session_state.active_rubric_idx is not None else 0
        ui_ver = st.session_state.get("editing_criteria_ui_version", 0)

        # Sort criteria by priority for display
        sorted_criteria = sorted(
            enumerate(st.session_state.editing_criteria),
            key=lambda x: int(x[1].get("priority", x[1].get("weight", x[0] + 1)))
        )

        num_criteria = len(sorted_criteria)

        for rank, (original_idx, criterion) in enumerate(sorted_criteria, start=1):
            criterion_name = criterion.get('name', 'Unnamed Criterion')

            # Row with up arrow, down arrow, and expander
            up_col, down_col, expander_col = st.columns([0.04, 0.04, 0.92])

            with up_col:
                up_key = f"move_up_{rank}_{version_key}"
                up_disabled = (rank == 1)
                if st.button("▲", key=up_key, disabled=up_disabled, help="Move up"):
                    if not up_disabled:
                        # Swap priorities with criterion above
                        for r, (idx, c) in enumerate(sorted_criteria, start=1):
                            if r == rank:
                                c["priority"] = rank - 1
                            elif r == rank - 1:
                                c["priority"] = rank
                        st.rerun()

            with down_col:
                down_key = f"move_down_{rank}_{version_key}"
                down_disabled = (rank == num_criteria)
                if st.button("▼", key=down_key, disabled=down_disabled, help="Move down"):
                    if not down_disabled:
                        # Swap priorities with criterion below
                        for r, (idx, c) in enumerate(sorted_criteria, start=1):
                            if r == rank:
                                c["priority"] = rank + 1
                            elif r == rank + 1:
                                c["priority"] = rank
                        st.rerun()

            with expander_col:
                expander_label = f"#{rank} - {criterion_name}"
                with st.expander(expander_label, expanded=False):
                    # Description (editable); ui_ver in key so widgets refresh when criteria updated from Apply suggestion
                    desc_key = f"criterion_desc_{original_idx}_{version_key}_{ui_ver}"
                    description = st.text_area(
                        "Description",
                        value=criterion.get("description", ""),
                        key=desc_key,
                        placeholder="Description of this criterion...",
                        height=100
                    )

                    # Dimensions (editable)
                    st.markdown("**Dimensions**")

                    # Always read dimensions from session state (not from criterion snapshot)
                    dimensions = st.session_state.editing_criteria[original_idx].get("dimensions", [])
                    num_dims = len(dimensions)

                    # Display dimensions with text inputs and delete buttons
                    dim_labels_updated = {}
                    dim_to_delete = None

                    for dim_idx in range(num_dims):
                        dim = st.session_state.editing_criteria[original_idx]["dimensions"][dim_idx]
                        # Use dimension's unique ID in key, not index
                        dim_id = dim.get("id", f"generated_{dim_idx}")

                        col1, col2 = st.columns([0.88, 0.12])
                        with col1:
                            dim_key = f"criterion_dim_{original_idx}_{dim_id}_{version_key}_{ui_ver}"
                            dim_label = st.text_input(
                                f"Dimension {dim_idx + 1}",
                                value=dim.get("label", ""),
                                key=dim_key,
                                label_visibility="collapsed",
                                placeholder=f"Dimension {dim_idx + 1} label..."
                            )
                            dim_labels_updated[dim_idx] = dim_label
                        with col2:
                            # Store the dim_id (not index) in the button key
                            remove_key = f"remove_dim_{original_idx}_{dim_id}_{version_key}_{ui_ver}"
                            if st.button("➖", key=remove_key, help="Remove dimension"):
                                # Mark this dimension ID for deletion
                                dim_to_delete = dim_id

                    # Update all labels first
                    for dim_idx, label in dim_labels_updated.items():
                        if dim_idx < len(st.session_state.editing_criteria[original_idx]["dimensions"]):
                            st.session_state.editing_criteria[original_idx]["dimensions"][dim_idx]["label"] = label

                    # Now handle deletion by ID (not by index)
                    if dim_to_delete is not None:
                        dims = st.session_state.editing_criteria[original_idx]["dimensions"]
                        # Find the dimension with matching ID and remove it
                        for i, d in enumerate(dims):
                            if d.get("id", f"generated_{i}") == dim_to_delete:
                                del st.session_state.editing_criteria[original_idx]["dimensions"][i]
                                break
                        st.rerun()

                    # Add dimension button
                    add_dim_key = f"add_dim_{original_idx}_{version_key}_{ui_ver}"
                    if st.button("➕ Add Dimension", key=add_dim_key):
                        # Generate a unique ID using timestamp
                        new_dim_id = f"dim_{int(time.time() * 1000)}"
                        st.session_state.editing_criteria[original_idx]["dimensions"].append({"id": new_dim_id, "label": ""})
                        st.rerun()

                    # Update the criterion description in the session state
                    st.session_state.editing_criteria[original_idx]["description"] = description

                    # Remove criterion button
                    if st.button("🗑️ Remove Criterion", key=f"remove_{original_idx}"):
                        st.session_state.editing_criteria.pop(original_idx)
                        st.rerun()

        # Check if there are unsaved changes
        def has_rubric_changes():
            """Compare editing_criteria with the original saved rubric to detect changes."""
            if not rubric_history or active_idx is None:
                return False
            original_rubric = rubric_history[active_idx].get("rubric", [])
            editing = st.session_state.editing_criteria

            # Different number of criteria
            if len(original_rubric) != len(editing):
                return True

            # Compare each criterion
            for orig, edit in zip(original_rubric, editing):
                # Check name
                if orig.get("name", "") != edit.get("name", ""):
                    return True
                # Check description
                if orig.get("description", "") != edit.get("description", ""):
                    return True
                # Check priority
                if orig.get("priority", 0) != edit.get("priority", 0):
                    return True
                # Check dimensions
                orig_dims = orig.get("dimensions", [])
                edit_dims = edit.get("dimensions", [])
                if len(orig_dims) != len(edit_dims):
                    return True
                for od, ed in zip(orig_dims, edit_dims):
                    if od.get("label", "") != ed.get("label", ""):
                        return True
            return False

        has_changes = has_rubric_changes()

        # Show any draft regeneration error from a previous Log Changes attempt
        if st.session_state.get("draft_regeneration_error"):
            st.error("Draft regeneration failed: " + st.session_state.draft_regeneration_error)
            del st.session_state.draft_regeneration_error

        # Log Changes, Save Version, and Reset buttons
        log_col, save_col, reset_col = st.columns(3)
        with log_col:
            if st.button("📝 Log Changes", use_container_width=True, disabled=not has_changes):
                # Log edits to conversation WITHOUT saving a new version
                if rubric_history and active_idx is not None:
                    current_version_num = rubric_history[active_idx].get("version", active_idx + 1)
                    old_rubric_for_diff = rubric_history[active_idx].get("rubric", [])
                    edit_class = classify_rubric_edits(old_rubric_for_diff, st.session_state.editing_criteria)
                    log_msg = format_edit_log_message(edit_class, current_version_num, f"{current_version_num}*", "editing")
                    if st.session_state.selected_conversation is not None:
                        st.session_state.messages.append({
                            "role": "system",
                            "content": log_msg,
                            "conversation_id": st.session_state.selected_conversation,
                        })
                    st.session_state.rubric = copy.deepcopy(st.session_state.editing_criteria)

                    draft_appended = False
                    last_draft, _ = get_last_draft_from_messages()
                    if last_draft:
                        regenerate_result = regenerate_draft_from_rubric_changes(
                            old_rubric_for_diff,
                            st.session_state.editing_criteria,
                            last_draft
                        )
                        if regenerate_result and regenerate_result.get("revised_draft") and not regenerate_result.get("error"):
                            revised_draft = regenerate_result["revised_draft"]
                            rubric_revision = {
                                "change_summary": regenerate_result.get("change_summary", ""),
                                "annotated_changes": regenerate_result.get("annotated_changes", []),
                                "revised_draft": revised_draft,
                                "revised_draft_annotated": regenerate_result.get("revised_draft_annotated") or regenerate_result.get("revised_draft_with_markers") or revised_draft,
                            }
                            draft_msg = {
                                "role": "assistant",
                                "content": f"<draft>{revised_draft}</draft>\n\n*Draft updated based on rubric changes.*",
                                "display_content": f"<draft>{revised_draft}</draft>\n\n*Draft updated based on rubric changes.*",
                                "thinking": regenerate_result.get("thinking", ""),
                                "rubric_revision": rubric_revision,
                                "rubric_version": current_version_num,
                                "is_system_generated": True,
                            }
                            st.session_state.messages.append(draft_msg)
                            draft_appended = True
                        else:
                            err = regenerate_result.get("error", "Unknown error") if isinstance(regenerate_result, dict) else "Regeneration failed"
                            st.session_state.draft_regeneration_error = err
                            st.toast("Changes logged. Draft regeneration failed — see error above.")
                            st.rerun()
                    else:
                        # No prior draft: generate a new one from the last user message and updated rubric
                        last_user_content = None
                        for idx in range(len(st.session_state.messages) - 1, -1, -1):
                            m = st.session_state.messages[idx]
                            if m.get("role") == "user":
                                last_user_content = m.get("content") or m.get("display_content", "")
                                break
                        if last_user_content:
                            try:
                                system = CHAT_build_system_prompt(st.session_state.editing_criteria) + "\n\nWhen the user asks for a draft, output ONLY the draft text wrapped in <draft></draft>. No preamble or follow-up."
                                req = "Write a draft for the following request. Output ONLY the draft text inside <draft></draft> tags.\n\n" + (last_user_content[:8000] or "Write a short passage.")
                                resp = _api_call_with_retry(
                                    model=MODEL_PRIMARY,
                                    max_tokens=4096,
                                    system=system,
                                    messages=[{"role": "user", "content": req}]
                                )
                                raw = "".join(b.text for b in resp.content if b.type == "text")
                                m = re.search(r"<draft>(.*?)</draft>", raw, re.DOTALL)
                                if m:
                                    new_draft = m.group(1).strip()
                                    draft_msg = {
                                        "role": "assistant",
                                        "content": f"<draft>{new_draft}</draft>\n\n*Draft generated with updated rubric (no prior draft to revise).*",
                                        "display_content": f"<draft>{new_draft}</draft>\n\n*Draft generated with updated rubric (no prior draft to revise).*",
                                        "rubric_revision": {"change_summary": "Draft generated using the updated rubric.", "annotated_changes": [], "revised_draft": new_draft, "revised_draft_annotated": new_draft},
                                        "rubric_version": current_version_num,
                                        "is_system_generated": True,
                                    }
                                    st.session_state.messages.append(draft_msg)
                                    draft_appended = True
                            except Exception as e:
                                st.toast(f"Changes logged. Could not generate draft: {str(e)}")
                                st.rerun()

                    st.toast("Changes logged & draft updated!" if draft_appended else "Changes logged. No draft in conversation to update — send a message in Chat and get a draft first.")
                    st.rerun()
        with save_col:
            if st.button("💾 Save Version", use_container_width=True, type="primary", disabled=not has_changes):
                # Save as a NEW version in rubric history
                if rubric_history is not None:
                    saved_criteria = copy.deepcopy(st.session_state.editing_criteria)

                    new_version = next_version_number()
                    new_rubric_entry = {
                        "version": new_version,
                        "rubric": saved_criteria,
                        "source": "edited"
                    }

                    hist = load_rubric_history()
                    hist.append(new_rubric_entry)
                    save_rubric_history(hist)

                    # Log edit to conversation
                    old_version_num = rubric_history[active_idx].get("version", active_idx + 1) if rubric_history and active_idx is not None else 0
                    old_rubric_for_diff = rubric_history[active_idx].get("rubric", []) if rubric_history and active_idx is not None else []
                    edit_class = classify_rubric_edits(old_rubric_for_diff, saved_criteria)
                    log_msg = format_edit_log_message(edit_class, old_version_num, new_version, "edited")
                    if st.session_state.selected_conversation is not None:
                        st.session_state.messages.append({
                            "role": "system",
                            "content": log_msg,
                            "conversation_id": st.session_state.selected_conversation,
                        })

                    # Update session state to point to new version
                    st.session_state.active_rubric_idx = len(hist) - 1
                    st.session_state.rubric = saved_criteria
                    st.session_state.editing_criteria = copy.deepcopy(saved_criteria)

                    st.toast(f"Saved as v{new_version}!")
                    st.rerun()
        with reset_col:
            if st.button("↩️ Revert", use_container_width=True, type="secondary", disabled=not has_changes):
                # Reset editing criteria to the original saved version
                if rubric_history and active_idx is not None:
                    original_rubric = rubric_history[active_idx].get("rubric", [])
                    st.session_state.editing_criteria = copy.deepcopy(original_rubric)
                    st.session_state.rubric = copy.deepcopy(original_rubric)
                    # Log revert to conversation (tagged so it only shows when this conversation is selected)
                    revert_version = rubric_history[active_idx].get("version", active_idx + 1)
                    revert_log = json.dumps({"reverted_to_version": revert_version})
                    st.session_state.messages.append({
                        "role": "system",
                        "content": f"↩️ **Rubric edits reverted** (back to v{revert_version})\n<!--RUBRIC_REVERT_LOG:{revert_log}-->",
                        "conversation_id": st.session_state.selected_conversation,
                    })
                    st.toast("Rubric reset to saved version!")
                    st.rerun()

        # Delete Version button
        if st.button("🗑️ Delete Version", use_container_width=True, type="secondary"):
            # Get the rubric to delete
            rubric_to_delete = rubric_history[active_idx]
            deleted_version = rubric_to_delete.get("version", "?")
            rubric_id = rubric_to_delete.get("id")

            # Delete from database
            supabase = st.session_state.get('supabase')
            if supabase and rubric_id:
                if delete_rubric_version(supabase, rubric_id):
                    # Invalidate cache and reload
                    invalidate_rubric_cache()
                    new_history = load_rubric_history(force_reload=True)

                    if new_history:
                        # Still have versions left — select the previous one
                        new_idx = active_idx - 1 if active_idx > 0 else 0
                        st.session_state.active_rubric_idx = new_idx
                        rubric_list = new_history[new_idx].get("rubric", [])
                        st.session_state.rubric = rubric_list
                        st.session_state.editing_criteria = copy.deepcopy(rubric_list)
                    else:
                        # Deleted the last version — clear rubric state
                        st.session_state.active_rubric_idx = 0
                        st.session_state.rubric = []
                        st.session_state.editing_criteria = []

                    st.toast(f"Version {deleted_version} deleted!")
                    st.rerun()
                else:
                    st.error("Failed to delete rubric version")
            elif not supabase:
                st.error("Cannot delete: not connected to database")
            elif not rubric_id:
                st.error("Cannot delete: this version was not saved to database yet")

        # Fallback for when Update button condition is false
        if active_rubric_dict and active_idx is None:
            rubric_list = active_rubric_dict.get("rubric", [])
            st.session_state.rubric = rubric_list
    else:
        st.info("No rubric loaded. Use 'Infer Rubric' to create one from your conversation.")


# View Rubric Tab

# View Rubric Tab
with tab3:
    st.subheader("📁 View Rubric")
    st.markdown("View the active rubric version with all criteria and checkable dimensions.")

    # Get active rubric
    active_rubric_dict, active_idx, rubric_history = get_active_rubric()

    if not active_rubric_dict or not active_rubric_dict.get("rubric"):
        st.warning("No active rubric available. Please create or select a rubric first.")
    else:
        # Display version, writing type, and user goals at the top
        col_meta1, col_meta2, col_meta3 = st.columns([1, 1, 2])

        with col_meta1:
            version = active_rubric_dict.get("version", 1)
            st.metric("Version", f"v{version}")

        with col_meta2:
            _source_raw = active_rubric_dict.get("source", "unknown")
            _source_labels = {
                "inferred": ("Inferred", "#E3F2FD", "#1565C0"),
                "inferred_from_evaluation": ("Inferred", "#E3F2FD", "#1565C0"),
                "edited": ("User-edited", "#E8F5E9", "#2E7D32"),
                "edit_feedback": ("User-edited", "#E8F5E9", "#2E7D32"),
                "chat_edit": ("User-edited", "#E8F5E9", "#2E7D32"),
            }
            _src_label, _src_bg, _src_fg = _source_labels.get(_source_raw, ("Unknown", "#f5f5f5", "#666"))
            st.markdown(f'<span style="display:inline-block;padding:4px 12px;background:{_src_bg};color:{_src_fg};border-radius:12px;font-size:0.85em;font-weight:600;">{_src_label}</span>', unsafe_allow_html=True)
            st.caption(f"Source: {_source_raw}")

        with col_meta3:
            writing_type = active_rubric_dict.get("writing_type", "Not specified")
            st.markdown(f"**Writing Type:** {writing_type}")

        user_goals = active_rubric_dict.get("user_goals_summary", "")
        if user_goals:
            st.markdown("**User Goals:**")
            st.info(user_goals)

        # Achievement level explanation
        st.markdown("**Achievement Levels** *(based on dimensions met)*")
        st.markdown("⭐⭐⭐ **Excellent**: 90%+ | ⭐⭐ **Good**: 75-89% | ⭐ **Fair**: 50-74% | ◇ **Needs Work**: 25-49% | ☆ **Weak**: <25%")

        st.markdown("---")

        # Prepare rubric list
        rubric_list = active_rubric_dict.get("rubric", [])

        # Display priority rankings as a simple numbered list
        if rubric_list:
            # Sort by priority (1 = most important = shown first)
            sorted_criteria = sorted(rubric_list, key=lambda c: c.get('priority', c.get('weight', 99)))

            st.markdown("### Priority Rankings")
            for criterion in sorted_criteria:
                priority = criterion.get('priority', criterion.get('weight', 0))
                name = criterion.get('name', 'Unnamed')
                st.markdown(f"**{priority}.** {name}")

            st.markdown("---")

        # Group criteria by category
        from collections import defaultdict
        categories = defaultdict(list)

        for criterion in rubric_list:
            category = criterion.get('category', 'Uncategorized')
            categories[category].append(criterion)

        # Display each category group
        for category_name, criteria in categories.items():
            # Category header box
            st.markdown(f"""
                <div style="border: 1px solid #2196F3; border-radius: 6px; padding: 8px; margin-bottom: 16px; background-color: rgba(33, 150, 243, 0.1);">
                    <div style="font-size: 18px; font-weight: 600; margin-bottom: 8px; color: #2196F3; text-transform: capitalize; text-align: center;">{category_name}</div>
            """, unsafe_allow_html=True)

            # Display criteria within this category
            for criterion in criteria:
                criterion_name = criterion.get('name', 'Unnamed Criterion')
                priority = criterion.get('priority', criterion.get('weight', 0))
                description = criterion.get('description', 'No description provided')

                # Display criterion name, priority, and description
                st.markdown(f"**{criterion_name}**")
                st.markdown(f"*Priority: #{priority}*")
                st.markdown(f"{description}")

                # Expander for dimensions (checkable items)
                dimensions = criterion.get('dimensions', [])
                if dimensions:
                    with st.expander(f"Dimensions", expanded=False):
                        for dim in dimensions:
                            dim_label = dim.get('label', 'Unnamed dimension')
                            st.markdown(f"• {dim_label}")
                else:
                    # Fallback for old rubrics with achievement levels
                    exemplary = criterion.get('exemplary', '')
                    if exemplary:
                        with st.expander("📊 Achievement Levels", expanded=False):
                            proficient = criterion.get('proficient', 'Not specified')
                            developing = criterion.get('developing', 'Not specified')
                            beginning = criterion.get('beginning', 'Not specified')

                            st.markdown("**🌟 Exemplary**")
                            st.markdown(exemplary)
                            st.markdown("")

                            st.markdown("**✅ Proficient**")
                            st.markdown(proficient)
                            st.markdown("")

                            st.markdown("**📈 Developing**")
                            st.markdown(developing)
                            st.markdown("")

                            st.markdown("**🔰 Beginning**")
                            st.markdown(beginning)

                st.markdown("<br>", unsafe_allow_html=True)

            # Close the category box
            st.markdown("</div>", unsafe_allow_html=True)

# Compare Rubrics Tab
with tab4:
    st.subheader("🔍 Compare Rubrics")
    st.markdown("Select two different rubric versions to see how they would affect the same writing task.")
    
    # Get rubric history
    _, _, rubric_history = get_active_rubric()
    
    if len(rubric_history) < 2:
        st.warning("You need at least 2 rubrics to compare. Create more rubric versions first.")
    else:
        _cmp_source_labels = {"inferred": "inferred", "inferred_from_evaluation": "inferred", "edited": "edited", "edit_feedback": "edited", "chat_edit": "edited"}
        rubric_options = [f"v{r.get('version', 1)} ({_cmp_source_labels.get(r.get('source', ''), r.get('source', 'unknown'))})" for r in rubric_history]
        col1, col2 = st.columns(2)

        with col1:
            rubric_a_idx = st.selectbox("Rubric A:", options=list(range(len(rubric_history))),
                                        format_func=lambda x: rubric_options[x], key="tab2_rubric_a_select")
        with col2:
            rubric_b_idx = st.selectbox("Rubric B:", options=list(range(len(rubric_history))),
                                        format_func=lambda x: rubric_options[x], key="tab2_rubric_b_select")

        # Display selected rubrics
        st.markdown("---")
        col_rubric_a, col_rubric_b = st.columns(2)

        with col_rubric_a:
            rubric_a_version = rubric_history[rubric_a_idx].get('version', 1)
            _src_a = _cmp_source_labels.get(rubric_history[rubric_a_idx].get("source", ""), "unknown")
            st.markdown(f"### 📋 Rubric v{rubric_a_version} ({_src_a})")
            display_rubric_criteria(rubric_history[rubric_a_idx], st, comparison_rubric_data=rubric_history[rubric_b_idx])

        with col_rubric_b:
            rubric_b_version = rubric_history[rubric_b_idx].get('version', 1)
            _src_b = _cmp_source_labels.get(rubric_history[rubric_b_idx].get("source", ""), "unknown")
            st.markdown(f"### 📋 Rubric v{rubric_b_version} ({_src_b})")
            display_rubric_criteria(rubric_history[rubric_b_idx], st, comparison_rubric_data=rubric_history[rubric_a_idx])
        
        # Comparison task input
        with st.form("compare_form", clear_on_submit=False):
            compare_input = st.text_area("Writing task:", height=100,
                                        placeholder="Enter the writing task you want to compare with different rubrics...")
            compare_submit = st.form_submit_button("Generate Comparison")
        
        if compare_submit and compare_input.strip():
            if rubric_a_idx == rubric_b_idx:
                st.error("Please select different rubrics to compare.")
            else:
                # Generate comparison
                with st.spinner("Generating comparison..."):
                    # Create clean copies without _diff metadata (which contains non-serializable sets)
                    def clean_rubric_for_api(rubric_data):
                        cleaned = copy.deepcopy(rubric_data)
                        if 'rubric' in cleaned:
                            for criterion in cleaned['rubric']:
                                if '_diff' in criterion:
                                    del criterion['_diff']
                        return cleaned

                    clean_rubric_a = clean_rubric_for_api(rubric_history[rubric_a_idx])
                    clean_rubric_b = clean_rubric_for_api(rubric_history[rubric_b_idx])
                    result = compare_rubrics(compare_input, clean_rubric_a, clean_rubric_b)

                    # Store results in session state
                    st.session_state.rubric_comparison_results = {
                        "base_txt": result.get("base_txt", ""),
                        "a_txt": result.get("a_txt", ""),
                        "b_txt": result.get("b_txt", ""),
                        "key_diffs": result.get("key_diffs", ""),
                        "summary": result.get("summary", ""),
                        "thinking": result.get("thinking", ""),
                        "rubric_a_idx": rubric_a_idx,
                        "rubric_b_idx": rubric_b_idx
                    }
                    st.rerun()

    # Display comparison results
    if st.session_state.rubric_comparison_results:
        results = st.session_state.rubric_comparison_results

        st.subheader("Comparison Results")

        # Show thinking if available
        if results.get("thinking"):
            with st.expander("🧠 Thinking", expanded=False):
                st.markdown(results["thinking"])

        # Helper to escape HTML and format paragraphs
        def escape_and_format(text):
            if not text:
                return ""
            escaped = (text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))
            # Convert markdown bullet points to HTML
            lines = escaped.split("\n")
            formatted_lines = []
            for line in lines:
                if line.strip().startswith("- "):
                    formatted_lines.append(f"<li>{line.strip()[2:]}</li>")
                elif line.strip():
                    formatted_lines.append(f"<p>{line}</p>")
            return "".join(formatted_lines)

        # Key differences and summary at the top
        col_diff, col_summary = st.columns(2)

        with col_diff:
            st.markdown("### Key Differences")
            key_diffs_html = escape_and_format(results["key_diffs"])
            st.markdown(f"""
            <div style="
                height: 200px;
                overflow-y: auto;
                border: 1px solid #ccc;
                padding: 10px;
                background-color: #f9f9f9;
                border-radius: 5px;
            ">
            {key_diffs_html}
            </div>
            """, unsafe_allow_html=True)

        with col_summary:
            st.markdown("### Summary")
            summary_html = escape_and_format(results["summary"])
            st.markdown(f"""
            <div style="
                height: 200px;
                overflow-y: auto;
                border: 1px solid #ccc;
                padding: 10px;
                background-color: #f9f9f9;
                border-radius: 5px;
            ">
            {summary_html}
            </div>
            """, unsafe_allow_html=True)

        # Separator
        st.markdown("---")

        # Create three columns for side-by-side comparison
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Base Draft")
            st.markdown("---")
            # Scrollable container for base draft
            # Escape HTML to prevent rendering issues
            base_txt_escaped = (results["base_txt"]
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))
            # Convert paragraphs
            base_txt_html = "".join(f"<p>{p}</p>" for p in base_txt_escaped.split("\n\n") if p.strip())
            st.markdown(f"""
            <div style="
                height: 400px;
                overflow-y: auto;
                border: 1px solid #ccc;
                padding: 10px;
                background-color: #f9f9f9;
                border-radius: 5px;
            ">
            {base_txt_html}
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("### Rubric A Revision")
            st.markdown("---")
            # Scrollable container for rubric A with diff highlighting
            diff_a = _md_diff_to_html_compare(results["a_txt"])
            st.markdown(f"""
            <div style="
                height: 400px;
                overflow-y: auto;
                border: 1px solid #ccc;
                padding: 10px;
                background-color: #f9f9f9;
                border-radius: 5px;
            ">
            {diff_a}
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("### Rubric B Revision")
            st.markdown("---")
            # Scrollable container for rubric B with diff highlighting
            diff_b = _md_diff_to_html_compare(results["b_txt"])
            st.markdown(f"""
            <div style="
                height: 400px;
                overflow-y: auto;
                border: 1px solid #ccc;
                padding: 10px;
                background-color: #f9f9f9;
                border-radius: 5px;
            ">
            {diff_b}
            </div>
            """, unsafe_allow_html=True)

# ============ EVALUATE: INFER TAB ============
with tab_infer:
    st.header("🔎 Evaluate: Infer")
    st.markdown("Measure what the rubric surfaces beyond what you can state upfront, and test whether it predicts your preferences.")

    # Helper function for rank correlation
    def compute_rank_correlation(user_rankings, predicted_rankings):
        """Compute Kendall's tau correlation between user and predicted rankings."""
        if not user_rankings or not predicted_rankings:
            return None, None

        all_user_ranks = []
        all_predicted_ranks = []
        all_alts = ["alt_1", "alt_2", "alt_3"]

        for dp_id in user_rankings:
            if dp_id not in predicted_rankings:
                continue
            user_rank = user_rankings[dp_id].get("ranking", [])
            pred_rank = predicted_rankings[dp_id].get("ranking", [])

            for alt_id in all_alts:
                user_pos = user_rank.index(alt_id) + 1 if alt_id in user_rank else 4
                pred_pos = pred_rank.index(alt_id) + 1 if alt_id in pred_rank else 4
                all_user_ranks.append(user_pos)
                all_predicted_ranks.append(pred_pos)

        if len(all_user_ranks) < 3:
            return None, None

        tau, p_value = kendalltau(all_user_ranks, all_predicted_ranks)
        return tau, p_value

    # Check for active rubric
    infer_rubric_dict, infer_rubric_idx, _ = get_active_rubric()

    if not infer_rubric_dict:
        st.warning("No active rubric found. Please create a rubric first.")
    else:
        infer_rubric_list = infer_rubric_dict.get("rubric", [])
        infer_rubric_version = infer_rubric_dict.get("version", infer_rubric_idx + 1)

        # Header with rubric info and reset
        col_info, col_reset = st.columns([3, 1])
        with col_info:
            st.success(f"Using rubric: **Version {infer_rubric_version}** ({len(infer_rubric_list)} criteria)")
        with col_reset:
            if st.button("🔄 Reset All", use_container_width=True, key="infer_reset"):
                st.session_state.infer_coldstart_text = ""
                st.session_state.infer_coldstart_saved = False
                st.session_state.infer_user_categorizations = {}
                st.session_state.infer_categorizations_complete = False
                st.session_state.infer_dp_conversation = None
                st.session_state.infer_decision_points = None
                st.session_state.infer_expanded_dp = None
                st.session_state.infer_dp_dimension_confirmed = False
                st.session_state.infer_dp_user_mappings = {}
                st.session_state.infer_step6_generated_task = None
                st.session_state.infer_step6_writing_task = ""
                st.session_state.infer_step6_auto_gen_done = False
                st.session_state.infer_step6_custom_task_key_version = 0
                st.session_state.infer_step6_drafts = None
                st.session_state.infer_step6_draft_labels = None
                st.session_state.infer_step6_rubric_versions_used = None
                st.session_state.infer_step6_blind_ratings = None
                st.session_state.infer_step6_user_ranking = None
                st.session_state.infer_step6_user_dimension_checks = None
                st.session_state.infer_step6_llm_evaluations = None
                st.session_state.infer_step6_survey = None
                st.session_state.infer_step6_claim2_metrics = None
                st.session_state.infer_step6_claim3_metrics = None
                st.session_state.chat_criteria_llm_classification = None
                st.session_state.chat_criteria_user_classifications = {}
                st.session_state.chat_criteria_review_active = False
                st.session_state.chat_criteria_review_confirmed = False
                st.session_state.chat_classification_feedback = {}
                st.session_state.chat_criteria_hallucination_reasons = {}
                st.rerun()

        # ==================== INFERENCE SESSION SELECTOR ====================
        _all_infer_sessions = st.session_state.get("infer_all_conversations", [])

        # Also check if there's a current (uncommitted) session with data
        _current_cats = st.session_state.get("infer_user_categorizations", {})
        _current_has_class = bool(_current_cats) and st.session_state.get("infer_categorizations_complete", False)
        _current_has_dps = bool(st.session_state.get("infer_decision_points"))
        _has_current_session = _current_has_class or _current_has_dps

        # Build list of sessions for selector
        _infer_session_options = []
        for _si, _sess in enumerate(_all_infer_sessions):
            _sess_ts = _sess.get("timestamp", "")
            _sess_conv = _sess.get("conversation_id", "")
            _sess_src_v = _sess.get("source_rubric_version", "?")
            _sess_res_v = _sess.get("result_rubric_version", "?")
            _sess_n_msgs = _sess.get("num_messages", 0)
            try:
                _sess_dt = datetime.fromisoformat(_sess_ts)
                _sess_time_str = _sess_dt.strftime("%m/%d %H:%M")
            except Exception:
                _sess_time_str = _sess_ts[:16] if _sess_ts else "Unknown"
            _sess_conv_label = _sess_conv if _sess_conv else "Unknown conversation"
            _sess_label = f"{_sess_time_str}  |  v{_sess_src_v} → v{_sess_res_v}  |  {_sess_n_msgs} msgs"
            _infer_session_options.append((_sess_label, _si))

        # Add current in-progress session if it has data and isn't already saved
        if _has_current_session:
            _infer_session_options.append(("Current session (in progress)", -1))

        if not _infer_session_options:
            st.info("No inference sessions yet. Complete the rubric inference flow in the **Chat** tab to populate this dashboard.")
        else:
            # Default to most recent: current session if exists, otherwise last saved
            _infer_default_idx = len(_infer_session_options) - 1

            if len(_infer_session_options) > 1:
                _selected_session_key = st.selectbox(
                    "Select inference session:",
                    options=range(len(_infer_session_options)),
                    format_func=lambda i: _infer_session_options[i][0],
                    index=_infer_default_idx,
                    key="infer_session_selector"
                )
            else:
                _selected_session_key = 0

            _selected_session_value = _infer_session_options[_selected_session_key][1]

            # Load data from selected session
            if _selected_session_value == -1:
                # Current in-progress session — read from session state
                _dash_cats = st.session_state.get("infer_user_categorizations", {})
                _dash_has_classifications = _current_has_class
                _dash_has_dps = _current_has_dps
                _dash_dp_data = st.session_state.get("infer_decision_points")
                _dash_classification_feedback = st.session_state.get("chat_classification_feedback", {})
            else:
                # Saved session — read from infer_all_conversations entry
                _selected_sess = _all_infer_sessions[_selected_session_value]
                _dash_cats = _selected_sess.get("user_categorizations", {})
                _dash_has_classifications = bool(_dash_cats)
                _dash_dp_data = _selected_sess.get("decision_points")
                _dash_has_dps = bool(_dash_dp_data)
                _dash_classification_feedback = _selected_sess.get("classification_feedback", {})

            # ==================== CLASSIFICATION SUMMARY ====================
            if _dash_has_classifications:
                st.divider()
                st.subheader("Classification Summary")
                st.markdown("How rubric criteria break down by source category.")

                _dash_total = len(_dash_cats)
                _dash_stated = sum(1 for v in _dash_cats.values() if v == "stated")
                _dash_real = sum(1 for v in _dash_cats.values() if v in ("real", "latent_real", "elicited"))
                _dash_halluc = sum(1 for v in _dash_cats.values() if v == "hallucinated")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Stated", f"{_dash_stated}/{_dash_total}",
                              delta=f"{_dash_stated/_dash_total*100:.0f}%" if _dash_total > 0 else "0%")
                with col2:
                    st.metric("Real", f"{_dash_real}/{_dash_total}",
                              delta=f"{_dash_real/_dash_total*100:.0f}%" if _dash_total > 0 else "0%")
                with col3:
                    st.metric("Hallucinated", f"{_dash_halluc}/{_dash_total}",
                              delta=f"{_dash_halluc/_dash_total*100:.0f}%" if _dash_total > 0 else "0%",
                              delta_color="inverse")

                # Breakdown table
                _dash_display_cat = {"stated": "Stated", "real": "Real", "latent_real": "Real", "elicited": "Real", "hallucinated": "Hallucinated"}
                _dash_ratio_data = []
                for crit_name, category in _dash_cats.items():
                    _dash_ratio_data.append({"Criterion": crit_name, "Category": _dash_display_cat.get(category, category)})
                if _dash_ratio_data:
                    st.table(_dash_ratio_data)

                # Hallucination reasons (from classification feedback)
                _dash_halluc_reasons = _dash_classification_feedback.get("hallucination_reasons", {})
                if _dash_halluc_reasons:
                    st.markdown("**Hallucination feedback:**")
                    for _hr_name, _hr_reason in _dash_halluc_reasons.items():
                        if _hr_reason:
                            st.markdown(f"- **{_hr_name}**: {_hr_reason}")

                st.markdown(
                    f"**Key Insight**: Of {_dash_total} rubric criteria, **{_dash_stated}** were stated upfront, "
                    f"**{_dash_real}** were real (user-endorsed but absent from your preference description), and **{_dash_halluc}** were hallucinated."
                )
                if _dash_total > 0 and _dash_halluc < _dash_total:
                    _dash_precision = (_dash_total - _dash_halluc) / _dash_total
                    st.markdown(f"**Rubric precision** (non-hallucinated / total): **{_dash_precision:.0%}**")

            # ==================== DECISION POINTS SUMMARY ====================
            if _dash_has_dps:
                st.divider()
                st.subheader("Decision Points Summary")
                _dash_dp_parsed = _dash_dp_data.get("parsed_data", {}) if isinstance(_dash_dp_data, dict) else {}
                _dash_dps = _dash_dp_parsed.get("decision_points", [])
                if _dash_dps:
                    _dash_active_dps = sum(1 for dp in _dash_dps if not dp.get("is_not_in_rubric"))
                    _dash_nir_dps = sum(1 for dp in _dash_dps if dp.get("is_not_in_rubric"))
                    _dash_dp_summary = f"**{_dash_active_dps}** decision points confirmed"
                    if _dash_nir_dps > 0:
                        _dash_dp_summary += f" ({_dash_nir_dps} not in rubric)"
                    st.success(_dash_dp_summary)

                    for dp in _dash_dps:
                        dp_id = dp.get('id', 0)
                        crit = dp.get("confirmed_criterion") or ""
                        user_action = dp.get("user_action", "correct")
                        original = dp.get("original_suggestion") or dp.get("dimension", "")
                        if user_action == "correct":
                            action_note = f"✅ {crit}"
                        elif user_action == "incorrect":
                            action_note = f"✏️ {original} → {crit}"
                        elif user_action == "not_in_rubric":
                            action_note = f"❌ Not in rubric"
                        else:
                            action_note = crit
                        with st.expander(f"DP#{dp_id}: {dp.get('dimension', 'Unknown')} — {action_note}", expanded=False):
                            col_b, col_a = st.columns(2)
                            with col_b:
                                st.caption("Original:")
                                st.info(dp.get('before_quote', 'N/A')[:200])
                            with col_a:
                                st.caption("Your Action:")
                                st.success(dp.get('after_quote', 'N/A')[:200])
                            if user_action == "incorrect" and dp.get("incorrect_reason"):
                                st.caption(f"Reason: {dp['incorrect_reason']}")
                            elif user_action == "not_in_rubric" and dp.get("not_in_rubric_reason"):
                                st.caption(f"Reason: {dp['not_in_rubric_reason']}")
                else:
                    st.info("Decision points were recorded but no parsed data is available.")


# ============ EVALUATE: BUILD TAB (hidden when SHOW_BUILD_GRADE_TABS is False) ============
if SHOW_BUILD_GRADE_TABS and tab7 is not None:
    with tab7:
        st.header("🔨 Evaluate: Build")
        st.markdown("Measure whether user rubric editing improves outcomes compared to inference-only rubrics.")
    
        # Load rubric history
        build_rubric_history = load_rubric_history()
    
        if not build_rubric_history or len(build_rubric_history) < 2:
            st.warning("You need at least 2 rubric versions to use this tab. Edit your rubric to create additional versions.")
        else:
            # Reset button
            col_info, col_reset = st.columns([3, 1])
            with col_info:
                st.success(f"**{len(build_rubric_history)}** rubric versions available for comparison")
            with col_reset:
                if st.button("🔄 Reset All", use_container_width=True, key="build_reset"):
                    st.session_state.build_rubric_a_idx = None
                    st.session_state.build_rubric_b_idx = None
                    st.session_state.build_edit_classification = None
                    st.session_state.build_writing_task = ""
                    st.session_state.build_draft_a = None
                    st.session_state.build_draft_b = None
                    st.session_state.build_draft_a_thinking = ""
                    st.session_state.build_draft_b_thinking = ""
                    st.session_state.build_blind_labels = None
                    st.session_state.build_user_preference = None
                    st.session_state.build_llm_judge_result = None
                    st.session_state.build_llm_judge_thinking = ""
                    st.session_state.build_self_report = {}
                    st.session_state.build_self_report_saved = False
                    st.rerun()
    
            # Progress indicator
            def get_build_step():
                if st.session_state.build_rubric_a_idx is None or st.session_state.build_rubric_b_idx is None:
                    return 1
                if not st.session_state.build_edit_classification:
                    return 2
                if not st.session_state.build_draft_a or not st.session_state.build_draft_b:
                    return 3
                if not st.session_state.build_user_preference:
                    return 3
                if not st.session_state.build_llm_judge_result:
                    return 4
                if not st.session_state.build_self_report_saved:
                    return 5
                return 5
    
            build_current_step = get_build_step()
            build_steps = [
                "1. Choose Two Rubrics",
                "2. Classify Edits",
                "3. Generate & Evaluate Drafts",
                "4. LLM Judge Scoring",
                "5. Self-Report"
            ]
            st.progress(build_current_step / 5)
            st.markdown(f"**Step {build_current_step}/5: {build_steps[build_current_step - 1]}**")
            st.divider()
    
            # ==================== STEP 1: CHOOSE TWO RUBRICS ====================
            st.subheader("Step 1: Choose Two Rubrics")
            st.markdown("Select two rubric versions to compare. Typically: one edited by you (R_edited) and one inferred from conversation only (R_inferred).")
    
            build_rubric_options = [f"v{r.get('version', i+1)} ({r.get('source', 'unknown')})" for i, r in enumerate(build_rubric_history)]
    
            col_a, col_b = st.columns(2)
            with col_a:
                build_a_idx = st.selectbox(
                    "Rubric A:",
                    options=list(range(len(build_rubric_history))),
                    format_func=lambda x: build_rubric_options[x],
                    index=st.session_state.build_rubric_a_idx if st.session_state.build_rubric_a_idx is not None else 0,
                    key="build_rubric_a_select"
                )
            with col_b:
                build_b_idx = st.selectbox(
                    "Rubric B:",
                    options=list(range(len(build_rubric_history))),
                    format_func=lambda x: build_rubric_options[x],
                    index=st.session_state.build_rubric_b_idx if st.session_state.build_rubric_b_idx is not None else min(1, len(build_rubric_history) - 1),
                    key="build_rubric_b_select"
                )
    
            if build_a_idx == build_b_idx:
                st.warning("Please select two different rubric versions.")
            else:
                # Update state if changed
                if build_a_idx != st.session_state.build_rubric_a_idx or build_b_idx != st.session_state.build_rubric_b_idx:
                    st.session_state.build_rubric_a_idx = build_a_idx
                    st.session_state.build_rubric_b_idx = build_b_idx
                    # Reset downstream state
                    st.session_state.build_edit_classification = None
                    st.session_state.build_draft_a = None
                    st.session_state.build_draft_b = None
                    st.session_state.build_blind_labels = None
                    st.session_state.build_user_preference = None
                    st.session_state.build_llm_judge_result = None
    
                build_rubric_a = build_rubric_history[build_a_idx]
                build_rubric_b = build_rubric_history[build_b_idx]
                build_rubric_a_list = build_rubric_a.get("rubric", [])
                build_rubric_b_list = build_rubric_b.get("rubric", [])
    
                # Show rubrics side by side
                with st.expander("📋 View Selected Rubrics", expanded=False):
                    col_ra, col_rb = st.columns(2)
                    with col_ra:
                        st.markdown(f"**Rubric A: v{build_rubric_a.get('version', '?')}** ({build_rubric_a.get('source', 'unknown')})")
                        for c in build_rubric_a_list:
                            st.markdown(f"- **{c.get('name', '?')}** (wt: {c.get('weight', c.get('priority', '?'))}): {c.get('description', '')[:100]}...")
                    with col_rb:
                        st.markdown(f"**Rubric B: v{build_rubric_b.get('version', '?')}** ({build_rubric_b.get('source', 'unknown')})")
                        for c in build_rubric_b_list:
                            st.markdown(f"- **{c.get('name', '?')}** (wt: {c.get('weight', c.get('priority', '?'))}): {c.get('description', '')[:100]}...")
    
                # ==================== STEP 2: CLASSIFY EDITS ====================
                st.divider()
                st.subheader("Step 2: Classify Edits Between Rubrics")
                st.markdown(f"Structural diff: Rubric B (v{build_rubric_b.get('version', '?')}) → Rubric A (v{build_rubric_a.get('version', '?')})")
    
                if not st.session_state.build_edit_classification:
                    if st.button("🔍 Classify Edits", type="primary", use_container_width=True, key="build_classify_btn"):
                        edits = classify_rubric_edits(build_rubric_b_list, build_rubric_a_list)
                        st.session_state.build_edit_classification = edits
                        st.rerun()
    
                if st.session_state.build_edit_classification:
                    edits = st.session_state.build_edit_classification
                    total_edits = sum(len(edits[k]) for k in edits)
    
                    # Summary metrics
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Added", len(edits["added"]))
                    with col2:
                        st.metric("Removed", len(edits["removed"]))
                    with col3:
                        st.metric("Reweighted", len(edits["reweighted"]))
                    with col4:
                        st.metric("Reworded", len(edits["reworded"]))
                    with col5:
                        st.metric("Dims Changed", len(edits["dimensions_changed"]))
    
                    st.markdown(f"**Total substantive edits: {total_edits}**")
    
                    # Detailed breakdown
                    if total_edits > 0:
                        edit_table = []
                        for a in edits["added"]:
                            edit_table.append({"Type": "Added", "Criterion": a["name"], "Detail": a.get("description", "")[:80]})
                        for r in edits["removed"]:
                            edit_table.append({"Type": "Removed", "Criterion": r["name"], "Detail": r.get("description", "")[:80]})
                        for rw in edits["reweighted"]:
                            edit_table.append({"Type": "Reweighted", "Criterion": rw["name"], "Detail": f"{rw['old_weight']} → {rw['new_weight']}"})
                        for rw in edits["reworded"]:
                            edit_table.append({"Type": "Reworded", "Criterion": rw["name"], "Detail": f"{rw['field']} changed"})
                        for dc in edits["dimensions_changed"]:
                            detail_parts = []
                            if dc["added_dims"]:
                                detail_parts.append(f"+{', '.join(dc['added_dims'][:3])}")
                            if dc["removed_dims"]:
                                detail_parts.append(f"-{', '.join(dc['removed_dims'][:3])}")
                            edit_table.append({"Type": "Dims Changed", "Criterion": dc["name"], "Detail": "; ".join(detail_parts)})
                        st.table(edit_table)
    
                    # Also show conversation-logged edits if available
                    conv_edits = get_effective_edits_from_conversation(st.session_state.messages)
                    if conv_edits:
                        with st.expander(f"📝 Conversation Edit Log ({len(conv_edits)} logged edits)", expanded=False):
                            for i, log in enumerate(conv_edits):
                                st.markdown(f"**Edit {i+1}:** v{log.get('version_from', '?')} → v{log.get('version_to', '?')} (source: {log.get('source', '?')})")
                                log_edits = log.get("edits", {})
                                for edit_type, items in log_edits.items():
                                    if items:
                                        st.markdown(f"  - {edit_type}: {len(items)} change(s)")
    
                # ==================== STEP 3: GENERATE DRAFTS & BLIND EVALUATION ====================
                if st.session_state.build_edit_classification:
                    st.divider()
                    st.subheader("Step 3: Generate Drafts & Blind Evaluation")
    
                    # Writing task input
                    st.markdown("Enter a writing task. Both rubrics will generate a draft for the same task.")
                    build_task_input = st.text_area(
                        "Writing task:",
                        value=st.session_state.build_writing_task,
                        placeholder="e.g., Write a professional email declining a meeting invitation while maintaining a positive relationship...",
                        height=120,
                        key="build_task_input"
                    )
    
                    if not st.session_state.build_draft_a or not st.session_state.build_draft_b:
                        if st.button("✍️ Generate Drafts from Both Rubrics", type="primary", use_container_width=True, key="build_generate_btn"):
                            if not build_task_input.strip():
                                st.error("Please enter a writing task first.")
                            else:
                                st.session_state.build_writing_task = build_task_input.strip()
    
                                with st.spinner("Generating Draft A (from Rubric A)..."):
                                    try:
                                        clean_a = copy.deepcopy(build_rubric_a)
                                        if 'rubric' in clean_a:
                                            for c in clean_a['rubric']:
                                                if '_diff' in c:
                                                    del c['_diff']
                                        rubric_a_json = json.dumps(clean_a, indent=2)
    
                                        system_a = CHAT_build_system_prompt(clean_a)
                                        response_a = _api_call_with_retry(
                                            model=MODEL_PRIMARY,
                                            max_tokens=8000,
                                            system=system_a,
                                            messages=[{"role": "user", "content": f"Please write a complete draft for this task. Output ONLY the draft text — no preamble, no follow-up notes, no meta-commentary.\n\n{st.session_state.build_writing_task}"}],
                                            thinking={"type": "enabled", "budget_tokens": 6000}
                                        )
    
                                        thinking_a = ""
                                        draft_a_text = ""
                                        for block in response_a.content:
                                            if block.type == "thinking":
                                                thinking_a = block.thinking
                                            elif block.type == "text":
                                                draft_a_text = block.text
    
                                        # Strip <draft> tags if present
                                        draft_a_clean = re.sub(r'</?draft>', '', draft_a_text).strip()
                                        st.session_state.build_draft_a = draft_a_clean
                                        st.session_state.build_draft_a_thinking = thinking_a
    
                                    except Exception as e:
                                        st.error(f"Error generating Draft A: {str(e)}")
    
                                if st.session_state.build_draft_a:
                                    with st.spinner("Generating Draft B (from Rubric B)..."):
                                        try:
                                            clean_b = copy.deepcopy(build_rubric_b)
                                            if 'rubric' in clean_b:
                                                for c in clean_b['rubric']:
                                                    if '_diff' in c:
                                                        del c['_diff']
    
                                            system_b = CHAT_build_system_prompt(clean_b)
                                            response_b = _api_call_with_retry(
                                                model=MODEL_PRIMARY,
                                                max_tokens=8000,
                                                system=system_b,
                                                messages=[{"role": "user", "content": f"Please write a complete draft for this task. Output ONLY the draft text — no preamble, no follow-up notes, no meta-commentary.\n\n{st.session_state.build_writing_task}"}],
                                                thinking={"type": "enabled", "budget_tokens": 6000}
                                            )
    
                                            thinking_b = ""
                                            draft_b_text = ""
                                            for block in response_b.content:
                                                if block.type == "thinking":
                                                    thinking_b = block.thinking
                                                elif block.type == "text":
                                                    draft_b_text = block.text
    
                                            draft_b_clean = re.sub(r'</?draft>', '', draft_b_text).strip()
                                            st.session_state.build_draft_b = draft_b_clean
                                            st.session_state.build_draft_b_thinking = thinking_b
    
                                            # Randomize blind labels
                                            if random.random() < 0.5:
                                                st.session_state.build_blind_labels = {"Draft X": "a", "Draft Y": "b"}
                                            else:
                                                st.session_state.build_blind_labels = {"Draft X": "b", "Draft Y": "a"}
    
                                            st.rerun()
    
                                        except Exception as e:
                                            st.error(f"Error generating Draft B: {str(e)}")
    
                    # Display drafts for blind evaluation
                    if st.session_state.build_draft_a and st.session_state.build_draft_b and st.session_state.build_blind_labels:
                        blind = st.session_state.build_blind_labels
                        # Map blind labels to actual drafts
                        draft_x = st.session_state.build_draft_a if blind["Draft X"] == "a" else st.session_state.build_draft_b
                        draft_y = st.session_state.build_draft_a if blind["Draft Y"] == "a" else st.session_state.build_draft_b
    
                        st.markdown("### Blind Evaluation")
                        st.markdown("Read both drafts below. You do NOT know which rubric produced which.")
    
                        col_dx, col_dy = st.columns(2)
                        with col_dx:
                            st.markdown("#### Draft X")
                            with st.container(height=400):
                                st.markdown(draft_x)
                        with col_dy:
                            st.markdown("#### Draft Y")
                            with st.container(height=400):
                                st.markdown(draft_y)
    
                        # Overall preference
                        st.markdown("### Overall Preference")
                        pref_options = ["Prefer Draft X", "Prefer Draft Y", "No preference"]
                        existing_pref = st.session_state.build_user_preference or {}
                        overall_pref = st.radio(
                            "Which draft do you prefer overall?",
                            pref_options,
                            index=pref_options.index(existing_pref.get("overall", "No preference")) if existing_pref.get("overall") in pref_options else 2,
                            key="build_overall_pref",
                            horizontal=True
                        )
    
                        # Per-dimension satisfaction ratings
                        st.markdown("### Per-Dimension Satisfaction")
                        st.markdown("For each criterion, rate how well each draft satisfies it (1 = poorly, 5 = excellently).")
    
                        # Use Rubric A's criteria as the union set for evaluation
                        all_criteria_names = []
                        for c in build_rubric_a_list:
                            if c.get("name") not in all_criteria_names:
                                all_criteria_names.append(c.get("name"))
                        for c in build_rubric_b_list:
                            if c.get("name") not in all_criteria_names:
                                all_criteria_names.append(c.get("name"))
    
                        dim_ratings = existing_pref.get("dimension_ratings", {})
                        rating_options = ["1", "2", "3", "4", "5"]
    
                        for crit_idx, crit_name in enumerate(all_criteria_names):
                            # Find description from either rubric
                            crit_desc = ""
                            for c in build_rubric_a_list + build_rubric_b_list:
                                if c.get("name") == crit_name:
                                    crit_desc = c.get("description", "")
                                    break
    
                            existing_crit = dim_ratings.get(crit_name, {})
    
                            with st.expander(f"**{crit_name}**", expanded=True):
                                if crit_desc:
                                    st.caption(crit_desc[:150])
                                col_rx, col_ry = st.columns(2)
                                with col_rx:
                                    dim_ratings.setdefault(crit_name, {})
                                    dim_ratings[crit_name]["draft_x"] = st.radio(
                                        f"Draft X — {crit_name}",
                                        rating_options,
                                        index=rating_options.index(str(existing_crit.get("draft_x", "3"))) if str(existing_crit.get("draft_x", "3")) in rating_options else 2,
                                        key=f"build_dim_x_{crit_idx}",
                                        horizontal=True,
                                        label_visibility="collapsed"
                                    )
                                with col_ry:
                                    dim_ratings[crit_name]["draft_y"] = st.radio(
                                        f"Draft Y — {crit_name}",
                                        rating_options,
                                        index=rating_options.index(str(existing_crit.get("draft_y", "3"))) if str(existing_crit.get("draft_y", "3")) in rating_options else 2,
                                        key=f"build_dim_y_{crit_idx}",
                                        horizontal=True,
                                        label_visibility="collapsed"
                                    )
                                st.caption("Draft X ↑ · Draft Y ↑ (1=poorly, 5=excellently)")
    
                        # Save preference
                        if st.button("💾 Save Evaluation", type="primary", use_container_width=True, key="build_save_pref"):
                            st.session_state.build_user_preference = {
                                "overall": overall_pref,
                                "dimension_ratings": dim_ratings,
                                "blind_labels": st.session_state.build_blind_labels
                            }
                            st.rerun()
    
                # ==================== STEP 4: LLM JUDGE SCORING ====================
                if st.session_state.build_user_preference:
                    st.divider()
                    st.subheader("Step 4: LLM Judge Scoring")
                    st.markdown("An LLM judge scores both drafts per-dimension, using your satisfaction ratings as the reference standard.")
    
                    # Reveal which draft was which
                    blind = st.session_state.build_blind_labels
                    pref = st.session_state.build_user_preference
                    overall = pref.get("overall", "No preference")
    
                    # Map overall preference to rubric
                    if overall == "Prefer Draft X":
                        preferred_rubric = "A" if blind["Draft X"] == "a" else "B"
                    elif overall == "Prefer Draft Y":
                        preferred_rubric = "A" if blind["Draft Y"] == "a" else "B"
                    else:
                        preferred_rubric = "tie"
    
                    st.info(f"**Reveal:** Draft X = Rubric {'A' if blind['Draft X'] == 'a' else 'B'}, Draft Y = Rubric {'A' if blind['Draft Y'] == 'a' else 'B'}. You preferred: **{overall}** (Rubric {preferred_rubric})")
    
                    if not st.session_state.build_llm_judge_result:
                        if st.button("🔬 Run LLM Judge", type="primary", use_container_width=True, key="build_judge_btn"):
                            with st.spinner("LLM judge is scoring both drafts per-dimension..."):
                                try:
                                    # Prepare rubric criteria JSON (union of both)
                                    all_criteria = []
                                    edits = st.session_state.build_edit_classification or {}
                                    edited_names = set()
                                    for k in ["added", "removed", "reweighted", "reworded", "dimensions_changed"]:
                                        for item in edits.get(k, []):
                                            edited_names.add(item.get("name", ""))
    
                                    for c in build_rubric_a_list:
                                        all_criteria.append({
                                            "name": c.get("name", ""),
                                            "description": c.get("description", ""),
                                            "weight": c.get("weight", c.get("priority", 0)),
                                            "was_edited": c.get("name", "") in edited_names
                                        })
    
                                    criteria_json = json.dumps(all_criteria, indent=2)
                                    ratings_json = json.dumps(pref.get("dimension_ratings", {}), indent=2)
    
                                    prompt = GRADING_judge_per_dimension_prompt(
                                        st.session_state.build_draft_a,
                                        st.session_state.build_draft_b,
                                        criteria_json,
                                        ratings_json
                                    )
    
                                    response = _api_call_with_retry(
                                        model=MODEL_PRIMARY,
                                        max_tokens=16000,
                                        messages=[{"role": "user", "content": prompt}],
                                        thinking={"type": "enabled", "budget_tokens": 8000}
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
                                        parsed = json.loads(json_match.group())
                                        st.session_state.build_llm_judge_result = parsed
                                        st.session_state.build_llm_judge_thinking = thinking_text
                                        st.rerun()
                                    else:
                                        st.error("Could not parse LLM judge response.")
                                        st.text(response_text)
    
                                except Exception as e:
                                    st.error(f"Error during LLM judging: {str(e)}")
    
                    # Display judge results
                    if st.session_state.build_llm_judge_result:
                        judge = st.session_state.build_llm_judge_result
    
                        if st.session_state.build_llm_judge_thinking:
                            with st.expander("🧠 Judge Thinking", expanded=False):
                                st.markdown(st.session_state.build_llm_judge_thinking)
    
                        overall_j = judge.get("overall", {})
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Draft A (Rubric A) Avg", f"{overall_j.get('draft_a_avg', 0):.1f}")
                        with col2:
                            st.metric("Draft B (Rubric B) Avg", f"{overall_j.get('draft_b_avg', 0):.1f}")
                        with col3:
                            st.metric("Winner", f"Rubric {overall_j.get('overall_winner', '?')}")
    
                        if overall_j.get("win_pattern"):
                            st.markdown(f"**Pattern:** {overall_j['win_pattern']}")
    
                        # Per-criterion table
                        st.markdown("### Per-Criterion Scores")
                        edits = st.session_state.build_edit_classification or {}
                        edited_names = set()
                        for k in ["added", "removed", "reweighted", "reworded", "dimensions_changed"]:
                            for item in edits.get(k, []):
                                edited_names.add(item.get("name", ""))
    
                        judge_table = []
                        edited_wins_a = 0
                        edited_wins_b = 0
                        unedited_wins_a = 0
                        unedited_wins_b = 0
    
                        for pc in judge.get("per_criterion", []):
                            was_edited = pc.get("criterion_name", "") in edited_names
                            winner = pc.get("winner", "tie")
                            judge_table.append({
                                "Criterion": pc.get("criterion_name", ""),
                                "Edited?": "Yes" if was_edited else "No",
                                "Draft A": pc.get("draft_a_score", 0),
                                "Draft B": pc.get("draft_b_score", 0),
                                "Winner": winner,
                                "Aligns w/ User": "Yes" if pc.get("aligns_with_user_rating") else "No"
                            })
                            if was_edited:
                                if winner == "A":
                                    edited_wins_a += 1
                                elif winner == "B":
                                    edited_wins_b += 1
                            else:
                                if winner == "A":
                                    unedited_wins_a += 1
                                elif winner == "B":
                                    unedited_wins_b += 1
    
                        st.table(judge_table)
    
                        # Attribution analysis
                        st.markdown("### Edit Attribution Analysis")
                        st.markdown(
                            f"On **edited dimensions**: Rubric A wins {edited_wins_a}, Rubric B wins {edited_wins_b}. "
                            f"On **unedited dimensions**: Rubric A wins {unedited_wins_a}, Rubric B wins {unedited_wins_b}."
                        )
                        if edited_wins_a > edited_wins_b:
                            st.success("Rubric A (edited) shows stronger performance specifically on dimensions that were edited — gains are attributable to user corrections.")
                        elif edited_wins_b > edited_wins_a:
                            st.warning("Rubric B performed better on edited dimensions — editing may not have improved these specific areas.")
                        else:
                            st.info("Tied on edited dimensions — no clear attribution signal.")
    
                # ==================== STEP 5: POST-TASK SELF-REPORT ====================
                if st.session_state.build_llm_judge_result:
                    st.divider()
                    st.subheader("Step 5: Post-Task Self-Report")
                    st.markdown("Reflect on whether seeing and editing the rubric shaped your preferences.")
    
                    report = st.session_state.build_self_report
    
                    st.markdown("**Q1: Did seeing the rubric help you realize preferences you wouldn't have known about?**")
                    q1_options = ["1 - Not at all", "2", "3", "4", "5 - Absolutely"]
                    report["q1"] = st.radio(
                        "Realize preferences",
                        q1_options,
                        index=q1_options.index(report.get("q1", "3")) if report.get("q1") in q1_options else 2,
                        key="build_q1",
                        label_visibility="collapsed",
                        horizontal=True
                    )
    
                    st.markdown("**Q2: Did seeing the rubric help you understand your preferences more clearly?**")
                    q2_options = ["1 - Not at all", "2", "3", "4", "5 - Absolutely"]
                    report["q2"] = st.radio(
                        "Understand preferences",
                        q2_options,
                        index=q2_options.index(report.get("q2", "3")) if report.get("q2") in q2_options else 2,
                        key="build_q2",
                        label_visibility="collapsed",
                        horizontal=True
                    )
    
                    st.markdown("**Q3: Did seeing the rubric help you express preferences you otherwise couldn't articulate?**")
                    q3_options = ["1 - Not at all", "2", "3", "4", "5 - Absolutely"]
                    report["q3"] = st.radio(
                        "Express preferences",
                        q3_options,
                        index=q3_options.index(report.get("q3", "3")) if report.get("q3") in q3_options else 2,
                        key="build_q3",
                        label_visibility="collapsed",
                        horizontal=True
                    )
    
                    st.markdown("**Q4: Describe any specific preferences the rubric helped you discover or clarify.**")
                    report["q4"] = st.text_area(
                        "Specific preferences",
                        value=report.get("q4", ""),
                        placeholder="e.g., I didn't realize I cared so much about paragraph transitions until I saw it in the rubric...",
                        key="build_q4",
                        label_visibility="collapsed",
                        height=150
                    )
    
                    st.session_state.build_self_report = report
    
                    if st.button("💾 Save Build Evaluation", type="primary", use_container_width=True, key="save_build_eval"):
                        st.session_state.build_self_report_saved = True
    
                        # Compile all data for export
                        blind = st.session_state.build_blind_labels
                        pref = st.session_state.build_user_preference
                        export_data = {
                            "timestamp": datetime.now().isoformat(),
                            "rubric_a": {
                                "version": build_rubric_a.get("version"),
                                "source": build_rubric_a.get("source"),
                                "criteria_count": len(build_rubric_a_list)
                            },
                            "rubric_b": {
                                "version": build_rubric_b.get("version"),
                                "source": build_rubric_b.get("source"),
                                "criteria_count": len(build_rubric_b_list)
                            },
                            "edit_classification": st.session_state.build_edit_classification,
                            "writing_task": st.session_state.build_writing_task,
                            "blind_labels": blind,
                            "user_preference": pref,
                            "llm_judge_result": st.session_state.build_llm_judge_result,
                            "self_report": report,
                            "conversation_edits": get_effective_edits_from_conversation(st.session_state.messages)
                        }
    
                        project_id = st.session_state.get('current_project_id')
                        if not project_id:
                            st.error("No project selected. Please select a project first.")
                        else:
                            supabase = st.session_state.get('supabase')
                            if supabase and save_project_data(supabase, project_id, "build_evaluation", export_data):
                                st.success("✅ Build evaluation saved successfully!")
                            else:
                                st.error("Failed to save evaluation.")

# ============ EVALUATE: GRADE TAB (hidden when SHOW_BUILD_GRADE_TABS is False) ============
if SHOW_BUILD_GRADE_TABS and tab8 is not None:
    with tab8:
        st.header("📝 Evaluate: Grade")
        st.markdown("Measure how well an LLM replicates your quality judgments when using your rubric vs. generic criteria.")
    
        # Get active rubric
        grade_rubric_dict, grade_rubric_idx, _ = get_active_rubric()
    
        if not grade_rubric_dict:
            st.warning("No active rubric found. Please create a rubric first.")
        else:
            grade_rubric_list = grade_rubric_dict.get("rubric", [])
            if len(grade_rubric_list) < 2:
                st.warning("Your rubric needs at least 2 criteria for this evaluation.")
            else:
                # Reset button
                col_grade_info, col_grade_reset = st.columns([3, 1])
                with col_grade_info:
                    st.success(f"Active rubric: **v{grade_rubric_dict.get('version', '?')}** with **{len(grade_rubric_list)}** criteria")
                with col_grade_reset:
                    if st.button("🔄 Reset All", use_container_width=True, key="grade_reset"):
                        st.session_state.grade_writing_task = ""
                        st.session_state.grade_violated_dims = None
                        st.session_state.grade_draft_good = None
                        st.session_state.grade_draft_degraded = None
                        st.session_state.grade_draft_good_thinking = ""
                        st.session_state.grade_draft_degraded_thinking = ""
                        st.session_state.grade_blind_labels = None
                        st.session_state.grade_user_overall_pref = None
                        st.session_state.grade_user_dim_ratings = {}
                        st.session_state.grade_rubric_judge_result = None
                        st.session_state.grade_rubric_judge_thinking = ""
                        st.session_state.grade_generic_judge_result = None
                        st.session_state.grade_generic_judge_thinking = ""
                        st.session_state.grade_agreement_results = None
                        st.session_state.grade_saved = False
                        st.rerun()
    
                # Progress indicator
                def get_grade_step():
                    if not st.session_state.grade_draft_good or not st.session_state.grade_draft_degraded:
                        return 1
                    if not st.session_state.grade_user_overall_pref:
                        return 2
                    if not st.session_state.grade_rubric_judge_result or not st.session_state.grade_generic_judge_result:
                        return 3
                    if not st.session_state.grade_agreement_results:
                        return 4
                    return 5
    
                grade_current_step = get_grade_step()
                grade_steps = [
                    "1. Generate Draft Pair",
                    "2. Rate Drafts",
                    "3. LLM Judge Evaluation",
                    "4. Agreement Analysis",
                    "5. Dimension Gap Analysis"
                ]
                st.progress(grade_current_step / 5)
                st.markdown(f"**Step {grade_current_step}/5: {grade_steps[grade_current_step - 1]}**")
                st.divider()
    
                # ==================== STEP 1: GENERATE DRAFT PAIR ====================
                st.subheader("Step 1: Generate Draft Pair")
                st.markdown("Select which rubric dimensions the degraded draft should violate, then enter a writing task.")
    
                # Checkboxes for dimensions to violate
                st.markdown("**Select dimensions to violate** (the degraded draft will deliberately underperform on these):")
                violated_selections = {}
                num_cols = min(3, len(grade_rubric_list))
                checkbox_cols = st.columns(num_cols)
                for ci, criterion in enumerate(grade_rubric_list):
                    crit_name = criterion.get("name", f"Criterion {ci+1}")
                    with checkbox_cols[ci % num_cols]:
                        violated_selections[crit_name] = st.checkbox(
                            crit_name,
                            value=crit_name in (st.session_state.grade_violated_dims or []),
                            key=f"grade_violate_{ci}",
                            help=criterion.get("description", "")[:100]
                        )
    
                selected_violated = [name for name, checked in violated_selections.items() if checked]
    
                if selected_violated:
                    st.caption(f"Selected for violation: {', '.join(selected_violated)}")
                else:
                    st.caption("Select at least 1 dimension (2-3 recommended)")
    
                # Writing task input
                grade_task_input = st.text_area(
                    "Writing task:",
                    value=st.session_state.grade_writing_task,
                    placeholder="e.g., Write a professional email declining a meeting invitation while maintaining a positive relationship...",
                    height=120,
                    key="grade_task_input"
                )
    
                if not st.session_state.grade_draft_good or not st.session_state.grade_draft_degraded:
                    if st.button("✍️ Generate Draft Pair", type="primary", use_container_width=True, key="grade_generate_btn"):
                        if not grade_task_input.strip():
                            st.error("Please enter a writing task first.")
                        elif not selected_violated:
                            st.error("Please select at least 1 dimension to violate.")
                        else:
                            st.session_state.grade_writing_task = grade_task_input.strip()
                            st.session_state.grade_violated_dims = selected_violated
    
                            # Generate good draft
                            with st.spinner("Generating good draft (following full rubric)..."):
                                try:
                                    clean_rubric = copy.deepcopy(grade_rubric_dict)
                                    if 'rubric' in clean_rubric:
                                        for c in clean_rubric['rubric']:
                                            if '_diff' in c:
                                                del c['_diff']
    
                                    system_good = CHAT_build_system_prompt(clean_rubric)
                                    response_good = _api_call_with_retry(
                                        model=MODEL_PRIMARY,
                                        max_tokens=8000,
                                        system=system_good,
                                        messages=[{"role": "user", "content": f"Please write a complete draft for this task. Output ONLY the draft text — no preamble, no follow-up notes, no meta-commentary.\n\n{st.session_state.grade_writing_task}"}],
                                        thinking={"type": "enabled", "budget_tokens": 6000}
                                    )
    
                                    thinking_good = ""
                                    draft_good_text = ""
                                    for block in response_good.content:
                                        if block.type == "thinking":
                                            thinking_good = block.thinking
                                        elif block.type == "text":
                                            draft_good_text = block.text
    
                                    draft_good_clean = re.sub(r'</?draft>', '', draft_good_text).strip()
                                    st.session_state.grade_draft_good = draft_good_clean
                                    st.session_state.grade_draft_good_thinking = thinking_good
    
                                except Exception as e:
                                    st.error(f"Error generating good draft: {str(e)}")
    
                            # Generate degraded draft
                            if st.session_state.grade_draft_good:
                                with st.spinner("Generating degraded draft (violating selected dimensions)..."):
                                    try:
                                        rubric_json = json.dumps(grade_rubric_list, indent=2)
                                        violated_dims_info = []
                                        for crit in grade_rubric_list:
                                            if crit.get("name") in selected_violated:
                                                violated_dims_info.append({
                                                    "name": crit.get("name"),
                                                    "description": crit.get("description", ""),
                                                    "weight": crit.get("weight", 0)
                                                })
                                        violated_json = json.dumps(violated_dims_info, indent=2)
    
                                        degraded_prompt = GRADING_generate_degraded_draft_prompt(
                                            st.session_state.grade_writing_task,
                                            rubric_json,
                                            violated_json
                                        )
    
                                        response_degraded = _api_call_with_retry(
                                            model=MODEL_PRIMARY,
                                            max_tokens=8000,
                                            messages=[{"role": "user", "content": degraded_prompt}],
                                            thinking={"type": "enabled", "budget_tokens": 6000}
                                        )
    
                                        thinking_degraded = ""
                                        draft_degraded_text = ""
                                        for block in response_degraded.content:
                                            if block.type == "thinking":
                                                thinking_degraded = block.thinking
                                            elif block.type == "text":
                                                draft_degraded_text = block.text
    
                                        draft_degraded_clean = re.sub(r'</?draft>', '', draft_degraded_text).strip()
                                        st.session_state.grade_draft_degraded = draft_degraded_clean
                                        st.session_state.grade_draft_degraded_thinking = thinking_degraded
    
                                        # Randomize blind labels
                                        if random.random() < 0.5:
                                            st.session_state.grade_blind_labels = {"Draft X": "good", "Draft Y": "degraded"}
                                        else:
                                            st.session_state.grade_blind_labels = {"Draft X": "degraded", "Draft Y": "good"}
    
                                        st.rerun()
    
                                    except Exception as e:
                                        st.error(f"Error generating degraded draft: {str(e)}")
    
                # Display drafts for blind evaluation
                if st.session_state.grade_draft_good and st.session_state.grade_draft_degraded and st.session_state.grade_blind_labels:
                    blind = st.session_state.grade_blind_labels
                    draft_x = st.session_state.grade_draft_good if blind["Draft X"] == "good" else st.session_state.grade_draft_degraded
                    draft_y = st.session_state.grade_draft_good if blind["Draft Y"] == "good" else st.session_state.grade_draft_degraded
    
                    st.markdown("### Drafts Generated")
                    st.markdown("Read both drafts below. You do NOT know which one follows the full rubric.")
    
                    col_gx, col_gy = st.columns(2)
                    with col_gx:
                        st.markdown("#### Draft X")
                        with st.container(height=400):
                            st.markdown(draft_x)
                    with col_gy:
                        st.markdown("#### Draft Y")
                        with st.container(height=400):
                            st.markdown(draft_y)
    
                    # ==================== STEP 2: USER RATES DRAFTS ====================
                    st.divider()
                    st.subheader("Step 2: Rate Drafts")
    
                    # Overall preference
                    st.markdown("### Overall Preference")
                    grade_pref_options = ["Prefer Draft X", "Prefer Draft Y", "No preference"]
                    existing_overall = st.session_state.grade_user_overall_pref
                    grade_overall_pref = st.radio(
                        "Which draft do you prefer overall?",
                        grade_pref_options,
                        index=grade_pref_options.index(existing_overall) if existing_overall in grade_pref_options else 2,
                        key="grade_overall_pref_radio",
                        horizontal=True
                    )
    
                    # Per-dimension satisfaction ratings
                    st.markdown("### Per-Dimension Satisfaction")
                    st.markdown("For each criterion, rate how well each draft satisfies it (1 = poorly, 5 = excellently).")
    
                    grade_dim_ratings = st.session_state.grade_user_dim_ratings.copy() if st.session_state.grade_user_dim_ratings else {}
                    grade_rating_options = ["1", "2", "3", "4", "5"]
    
                    for crit_idx, criterion in enumerate(grade_rubric_list):
                        crit_name = criterion.get("name", f"Criterion {crit_idx+1}")
                        crit_desc = criterion.get("description", "")
    
                        existing_crit = grade_dim_ratings.get(crit_name, {})
    
                        with st.expander(f"**{crit_name}**", expanded=True):
                            if crit_desc:
                                st.caption(crit_desc[:150])
                            col_grx, col_gry = st.columns(2)
                            with col_grx:
                                grade_dim_ratings.setdefault(crit_name, {})
                                grade_dim_ratings[crit_name]["draft_x"] = st.radio(
                                    f"Draft X — {crit_name}",
                                    grade_rating_options,
                                    index=grade_rating_options.index(str(existing_crit.get("draft_x", "3"))) if str(existing_crit.get("draft_x", "3")) in grade_rating_options else 2,
                                    key=f"grade_dim_x_{crit_idx}",
                                    horizontal=True,
                                    label_visibility="collapsed"
                                )
                            with col_gry:
                                grade_dim_ratings[crit_name]["draft_y"] = st.radio(
                                    f"Draft Y — {crit_name}",
                                    grade_rating_options,
                                    index=grade_rating_options.index(str(existing_crit.get("draft_y", "3"))) if str(existing_crit.get("draft_y", "3")) in grade_rating_options else 2,
                                    key=f"grade_dim_y_{crit_idx}",
                                    horizontal=True,
                                    label_visibility="collapsed"
                                )
                            st.caption("Draft X ↑ · Draft Y ↑ (1=poorly, 5=excellently)")
    
                    # Save ratings
                    if not st.session_state.grade_user_overall_pref:
                        if st.button("💾 Save Ratings", type="primary", use_container_width=True, key="grade_save_ratings"):
                            st.session_state.grade_user_overall_pref = grade_overall_pref
                            st.session_state.grade_user_dim_ratings = grade_dim_ratings
                            st.rerun()
    
                    # ==================== STEP 3: LLM JUDGE EVALUATION ====================
                    if st.session_state.grade_user_overall_pref:
                        st.divider()
                        st.subheader("Step 3: LLM Judge Evaluation")
                        st.markdown("Two LLM judges evaluate the same drafts: one using your rubric + conversation context, one using only generic criteria.")
    
                        if not st.session_state.grade_rubric_judge_result or not st.session_state.grade_generic_judge_result:
                            if st.button("🔬 Run LLM Judges", type="primary", use_container_width=True, key="grade_judge_btn"):
                                # Prepare drafts (use actual A/B, not blind labels)
                                actual_draft_a = st.session_state.grade_draft_good
                                actual_draft_b = st.session_state.grade_draft_degraded
    
                                # Rubric-grounded judge
                                with st.spinner("Running rubric-grounded judge..."):
                                    try:
                                        rubric_criteria_json = json.dumps(grade_rubric_list, indent=2)
    
                                        # Build conversation context (last 10 non-system messages)
                                        conv_messages = []
                                        for msg in st.session_state.messages:
                                            if msg.get("role") != "system" or "<!--" not in msg.get("content", ""):
                                                conv_messages.append(msg)
                                        recent_conv = conv_messages[-10:] if len(conv_messages) > 10 else conv_messages
                                        conv_context = "\n".join([f"[{m.get('role', 'unknown')}]: {m.get('content', '')[:500]}" for m in recent_conv])
    
                                        rubric_judge_prompt = GRADING_rubric_judge_prompt(
                                            actual_draft_a,
                                            actual_draft_b,
                                            rubric_criteria_json,
                                            conv_context
                                        )
    
                                        response_rubric = _api_call_with_retry(
                                            model=MODEL_PRIMARY,
                                            max_tokens=16000,
                                            messages=[{"role": "user", "content": rubric_judge_prompt}],
                                            thinking={"type": "enabled", "budget_tokens": 8000}
                                        )
    
                                        rubric_thinking = ""
                                        rubric_text = ""
                                        for block in response_rubric.content:
                                            if block.type == "thinking":
                                                rubric_thinking = block.thinking
                                            elif block.type == "text":
                                                rubric_text = block.text
    
                                        json_match = re.search(r'\{[\s\S]*\}', rubric_text)
                                        if json_match:
                                            st.session_state.grade_rubric_judge_result = json.loads(json_match.group())
                                            st.session_state.grade_rubric_judge_thinking = rubric_thinking
                                        else:
                                            st.error("Failed to parse rubric judge response.")
    
                                    except Exception as e:
                                        st.error(f"Error running rubric-grounded judge: {str(e)}")
    
                                # Generic judge
                                if st.session_state.grade_rubric_judge_result:
                                    with st.spinner("Running generic judge..."):
                                        try:
                                            generic_prompt = GRADING_generic_judge_prompt(actual_draft_a, actual_draft_b)
    
                                            response_generic = _api_call_with_retry(
                                                model=MODEL_PRIMARY,
                                                max_tokens=16000,
                                                messages=[{"role": "user", "content": generic_prompt}],
                                                thinking={"type": "enabled", "budget_tokens": 8000}
                                            )
    
                                            generic_thinking = ""
                                            generic_text = ""
                                            for block in response_generic.content:
                                                if block.type == "thinking":
                                                    generic_thinking = block.thinking
                                                elif block.type == "text":
                                                    generic_text = block.text
    
                                            json_match = re.search(r'\{[\s\S]*\}', generic_text)
                                            if json_match:
                                                st.session_state.grade_generic_judge_result = json.loads(json_match.group())
                                                st.session_state.grade_generic_judge_thinking = generic_thinking
                                            else:
                                                st.error("Failed to parse generic judge response.")
    
                                            st.rerun()
    
                                        except Exception as e:
                                            st.error(f"Error running generic judge: {str(e)}")
    
                        # Display judge results
                        if st.session_state.grade_rubric_judge_result and st.session_state.grade_generic_judge_result:
                            rubric_result = st.session_state.grade_rubric_judge_result
                            generic_result = st.session_state.grade_generic_judge_result
    
                            col_rj, col_gj = st.columns(2)
                            with col_rj:
                                st.markdown("#### Rubric-Grounded Scores")
                                st.markdown(f"**Overall preference:** Draft {rubric_result.get('overall_preference', '?')}")
                                if rubric_result.get("overall_reasoning"):
                                    st.caption(rubric_result["overall_reasoning"])
                                for pc in rubric_result.get("per_criterion", []):
                                    with st.expander(f"{pc.get('criterion_name', '?')} — A:{pc.get('draft_a_score', '?')} / B:{pc.get('draft_b_score', '?')}"):
                                        st.markdown(pc.get("reasoning", ""))
    
                            with col_gj:
                                st.markdown("#### Generic Scores")
                                st.markdown(f"**Overall preference:** Draft {generic_result.get('overall_preference', '?')}")
                                if generic_result.get("overall_reasoning"):
                                    st.caption(generic_result["overall_reasoning"])
                                for pc in generic_result.get("per_criterion", []):
                                    with st.expander(f"{pc.get('criterion_name', '?')} — A:{pc.get('draft_a_score', '?')} / B:{pc.get('draft_b_score', '?')}"):
                                        st.markdown(pc.get("reasoning", ""))
    
                            # ==================== STEP 4: AGREEMENT ANALYSIS ====================
                            st.divider()
                            st.subheader("Step 4: Agreement Analysis")
    
                            # Compute agreement if not already done
                            if not st.session_state.grade_agreement_results:
                                blind = st.session_state.grade_blind_labels
                                user_pref = st.session_state.grade_user_overall_pref
                                user_dim = st.session_state.grade_user_dim_ratings
    
                                # Map user preference to A/B (good/degraded)
                                if user_pref == "Prefer Draft X":
                                    user_prefers = blind["Draft X"]  # "good" or "degraded"
                                elif user_pref == "Prefer Draft Y":
                                    user_prefers = blind["Draft Y"]
                                else:
                                    user_prefers = "tie"
    
                                # Map to A/B: A=good, B=degraded
                                if user_prefers == "good":
                                    user_overall_ab = "A"
                                elif user_prefers == "degraded":
                                    user_overall_ab = "B"
                                else:
                                    user_overall_ab = "tie"
    
                                rubric_overall = rubric_result.get("overall_preference", "tie")
                                generic_overall = generic_result.get("overall_preference", "tie")
    
                                # Overall agreement
                                rubric_overall_agrees = (user_overall_ab == rubric_overall)
                                generic_overall_agrees = (user_overall_ab == generic_overall)
    
                                # Per-dimension agreement (rubric-grounded)
                                # Map user dim ratings from blind X/Y to actual A(good)/B(degraded)
                                rubric_dim_agreements = {}
                                violated_dims = st.session_state.grade_violated_dims or []
    
                                for crit_name, ratings in user_dim.items():
                                    # User rated draft_x and draft_y; map to good/degraded
                                    user_x_score = int(ratings.get("draft_x", 3))
                                    user_y_score = int(ratings.get("draft_y", 3))
    
                                    # Map X/Y to A(good)/B(degraded) scores
                                    if blind["Draft X"] == "good":
                                        user_a_score = user_x_score
                                        user_b_score = user_y_score
                                    else:
                                        user_a_score = user_y_score
                                        user_b_score = user_x_score
    
                                    # User's preference for this dimension
                                    if user_a_score > user_b_score:
                                        user_dim_pref = "A"
                                    elif user_b_score > user_a_score:
                                        user_dim_pref = "B"
                                    else:
                                        user_dim_pref = "tie"
    
                                    # Find rubric judge's scores for this criterion
                                    judge_dim_pref = "tie"
                                    judge_a_score = None
                                    judge_b_score = None
                                    for pc in rubric_result.get("per_criterion", []):
                                        if pc.get("criterion_name") == crit_name:
                                            judge_a_score = pc.get("draft_a_score", 3)
                                            judge_b_score = pc.get("draft_b_score", 3)
                                            if judge_a_score > judge_b_score:
                                                judge_dim_pref = "A"
                                            elif judge_b_score > judge_a_score:
                                                judge_dim_pref = "B"
                                            break
    
                                    agrees = (user_dim_pref == judge_dim_pref)
                                    rubric_dim_agreements[crit_name] = {
                                        "user_pref": user_dim_pref,
                                        "judge_pref": judge_dim_pref,
                                        "agrees": agrees,
                                        "violated": crit_name in violated_dims,
                                        "user_a_score": user_a_score,
                                        "user_b_score": user_b_score,
                                        "judge_a_score": judge_a_score,
                                        "judge_b_score": judge_b_score,
                                        "user_score_gap": abs(user_a_score - user_b_score)
                                    }
    
                                # Compute overall Kendall's tau using per-dimension scores
                                user_ranks_list = []
                                rubric_ranks_list = []
                                generic_ranks_list = []
    
                                for crit_name, info in rubric_dim_agreements.items():
                                    # User scores (A, B)
                                    user_ranks_list.append(info["user_a_score"])
                                    user_ranks_list.append(info["user_b_score"])
                                    # Rubric judge scores
                                    if info["judge_a_score"] is not None:
                                        rubric_ranks_list.append(info["judge_a_score"])
                                        rubric_ranks_list.append(info["judge_b_score"])
                                    else:
                                        rubric_ranks_list.append(3)
                                        rubric_ranks_list.append(3)
    
                                # For generic: use average generic score for each draft
                                generic_per_crit = generic_result.get("per_criterion", [])
                                generic_a_avg = sum(pc.get("draft_a_score", 3) for pc in generic_per_crit) / max(len(generic_per_crit), 1)
                                generic_b_avg = sum(pc.get("draft_b_score", 3) for pc in generic_per_crit) / max(len(generic_per_crit), 1)
    
                                # Compute Kendall's tau
                                rubric_tau, rubric_p = (None, None)
                                if len(user_ranks_list) >= 4:
                                    rubric_tau, rubric_p = kendalltau(user_ranks_list, rubric_ranks_list)
    
                                # Per-dimension agreement rates
                                total_dims = len(rubric_dim_agreements)
                                rubric_agree_count = sum(1 for v in rubric_dim_agreements.values() if v["agrees"])
                                rubric_agree_rate = rubric_agree_count / total_dims if total_dims > 0 else 0
    
                                st.session_state.grade_agreement_results = {
                                    "rubric_overall_agrees": rubric_overall_agrees,
                                    "generic_overall_agrees": generic_overall_agrees,
                                    "user_overall_ab": user_overall_ab,
                                    "rubric_overall": rubric_overall,
                                    "generic_overall": generic_overall,
                                    "rubric_tau": rubric_tau,
                                    "rubric_p": rubric_p,
                                    "rubric_agree_rate": rubric_agree_rate,
                                    "rubric_agree_count": rubric_agree_count,
                                    "total_dims": total_dims,
                                    "rubric_dim_agreements": rubric_dim_agreements,
                                    "generic_a_avg": generic_a_avg,
                                    "generic_b_avg": generic_b_avg
                                }
    
                            # Display agreement results
                            agreement = st.session_state.grade_agreement_results
    
                            st.markdown("### Overall Agreement")
                            col_m1, col_m2, col_m3 = st.columns(3)
                            with col_m1:
                                st.metric("User Prefers", f"Draft {agreement['user_overall_ab']}")
                            with col_m2:
                                rubric_match = "✅ Agrees" if agreement["rubric_overall_agrees"] else "❌ Disagrees"
                                st.metric("Rubric Judge", f"Draft {agreement['rubric_overall']}", delta=rubric_match, delta_color="off")
                            with col_m3:
                                generic_match = "✅ Agrees" if agreement["generic_overall_agrees"] else "❌ Disagrees"
                                st.metric("Generic Judge", f"Draft {agreement['generic_overall']}", delta=generic_match, delta_color="off")
    
                            st.markdown("### Per-Dimension Agreement (Rubric-Grounded)")
                            col_tau, col_rate = st.columns(2)
                            with col_tau:
                                tau_val = agreement.get("rubric_tau")
                                if tau_val is not None:
                                    interpretation = "Strong" if abs(tau_val) > 0.6 else ("Moderate" if abs(tau_val) > 0.3 else "Weak")
                                    st.metric("Kendall's τ", f"{tau_val:.3f}", delta=interpretation, delta_color="off")
                                else:
                                    st.metric("Kendall's τ", "N/A")
                            with col_rate:
                                rate = agreement.get("rubric_agree_rate", 0)
                                count = agreement.get("rubric_agree_count", 0)
                                total = agreement.get("total_dims", 0)
                                st.metric("Dimension Agreement", f"{count}/{total} ({rate:.0%})")
    
                            # ==================== STEP 5: DIMENSION GAP ANALYSIS + SAVE ====================
                            st.divider()
                            st.subheader("Step 5: Dimension Gap Analysis")
                            st.markdown("Which dimensions show the largest gap between rubric-grounded and generic evaluation? These are where personalization matters most.")
    
                            dim_agreements = agreement.get("rubric_dim_agreements", {})
    
                            # Build gap table
                            gap_rows = []
                            for crit_name, info in dim_agreements.items():
                                gap_rows.append({
                                    "Dimension": crit_name,
                                    "Violated?": "Yes" if info["violated"] else "No",
                                    "User Prefers": f"Draft {info['user_pref']}",
                                    "Rubric Judge": f"Draft {info['judge_pref']}",
                                    "Agree?": "✅" if info["agrees"] else "❌",
                                    "User Score Gap": info["user_score_gap"],
                                    "User A": info["user_a_score"],
                                    "User B": info["user_b_score"],
                                    "Judge A": info.get("judge_a_score", "?"),
                                    "Judge B": info.get("judge_b_score", "?")
                                })
    
                            # Sort: disagreements first, then by user score gap descending
                            gap_rows.sort(key=lambda r: (r["Agree?"] == "✅", -r["User Score Gap"]))
    
                            if gap_rows:
                                st.dataframe(
                                    gap_rows,
                                    column_config={
                                        "Dimension": st.column_config.TextColumn("Dimension", width="medium"),
                                        "Violated?": st.column_config.TextColumn("Violated?", width="small"),
                                        "User Prefers": st.column_config.TextColumn("User", width="small"),
                                        "Rubric Judge": st.column_config.TextColumn("Rubric Judge", width="small"),
                                        "Agree?": st.column_config.TextColumn("Agree?", width="small"),
                                        "User Score Gap": st.column_config.NumberColumn("Score Gap", width="small"),
                                        "User A": st.column_config.NumberColumn("User A", width="small"),
                                        "User B": st.column_config.NumberColumn("User B", width="small"),
                                        "Judge A": st.column_config.NumberColumn("Judge A", width="small"),
                                        "Judge B": st.column_config.NumberColumn("Judge B", width="small")
                                    },
                                    use_container_width=True,
                                    hide_index=True
                                )
    
                            # Summary statistics
                            violated_agree = sum(1 for v in dim_agreements.values() if v["violated"] and v["agrees"])
                            violated_total = sum(1 for v in dim_agreements.values() if v["violated"])
                            non_violated_agree = sum(1 for v in dim_agreements.values() if not v["violated"] and v["agrees"])
                            non_violated_total = sum(1 for v in dim_agreements.values() if not v["violated"])
    
                            st.markdown("### Summary")
                            col_s1, col_s2, col_s3 = st.columns(3)
                            with col_s1:
                                st.metric(
                                    "Rubric Judge Agreement",
                                    f"{agreement.get('rubric_agree_count', 0)}/{agreement.get('total_dims', 0)}"
                                )
                            with col_s2:
                                if violated_total > 0:
                                    st.metric("On Violated Dims", f"{violated_agree}/{violated_total}")
                                else:
                                    st.metric("On Violated Dims", "N/A")
                            with col_s3:
                                if non_violated_total > 0:
                                    st.metric("On Non-Violated Dims", f"{non_violated_agree}/{non_violated_total}")
                                else:
                                    st.metric("On Non-Violated Dims", "N/A")
    
                            # Generic vs rubric comparison
                            st.markdown("### Rubric-Grounded vs. Generic")
                            st.markdown(f"- **Rubric-grounded judge** agreed with your overall preference: {'✅ Yes' if agreement['rubric_overall_agrees'] else '❌ No'}")
                            st.markdown(f"- **Generic judge** agreed with your overall preference: {'✅ Yes' if agreement['generic_overall_agrees'] else '❌ No'}")
                            st.markdown(f"- **Rubric-grounded per-dimension agreement:** {agreement.get('rubric_agree_rate', 0):.0%}")
    
                            # Identify dimensions where personalization matters most
                            disagreement_dims = [name for name, info in dim_agreements.items() if not info["agrees"]]
                            if disagreement_dims and not agreement["generic_overall_agrees"]:
                                st.info(f"**Dimensions where personalization matters most:** {', '.join(disagreement_dims)} — the rubric-grounded judge disagreed with you here, and the generic judge also failed to match your overall preference.")
                            elif disagreement_dims:
                                st.info(f"**Dimensions with rubric-judge disagreement:** {', '.join(disagreement_dims)}")
                            elif not agreement["generic_overall_agrees"]:
                                st.success("The rubric-grounded judge agreed with you on all dimensions, while the generic judge disagreed on overall preference — the rubric captured your preferences well.")
                            else:
                                st.success("Both judges agreed with your preferences across the board.")
    
                            # Save button
                            st.divider()
                            if not st.session_state.grade_saved:
                                if st.button("💾 Save Grade Evaluation", type="primary", use_container_width=True, key="grade_save_btn"):
                                    export_data = {
                                        "timestamp": datetime.now().isoformat(),
                                        "rubric_version": grade_rubric_dict.get("version", "?"),
                                        "writing_task": st.session_state.grade_writing_task,
                                        "violated_dims": st.session_state.grade_violated_dims,
                                        "blind_labels": st.session_state.grade_blind_labels,
                                        "user_overall_preference": st.session_state.grade_user_overall_pref,
                                        "user_dim_ratings": st.session_state.grade_user_dim_ratings,
                                        "rubric_judge_result": st.session_state.grade_rubric_judge_result,
                                        "generic_judge_result": st.session_state.grade_generic_judge_result,
                                        "agreement_results": {
                                            k: v for k, v in agreement.items()
                                            if k != "rubric_dim_agreements"
                                        },
                                        "dimension_agreements": agreement.get("rubric_dim_agreements", {})
                                    }
    
                                    project_id = st.session_state.get('current_project_id')
                                    if not project_id:
                                        st.error("No project selected. Please select a project first.")
                                    else:
                                        supabase = st.session_state.get('supabase')
                                        if supabase and save_project_data(supabase, project_id, "grade_evaluation", export_data):
                                            st.session_state.grade_saved = True
                                            st.success("✅ Grade evaluation saved successfully!")
                                        else:
                                            st.error("Failed to save evaluation.")
                            else:
                                st.success("✅ Grade evaluation has been saved.")
    
# ============ EVALUATE: GRADING TAB ============
with tab_grading:
    st.header("📊 Evaluate: Grading")

    # --- Auto-pull preference description from Task A Q4 ---
    _gr_coldstart = st.session_state.get("infer_coldstart_text", "").strip()
    _gr_tab_blocked = False
    if not _gr_coldstart:
        _gr_task_a = st.session_state.get("survey_responses", {}).get("task_a", {})
        _gr_q4 = _gr_task_a.get("q4", "").strip()
        _gr_a_done = _gr_task_a.get("completed", False)
        if not _gr_a_done or not _gr_q4:
            st.warning("You haven't completed the **Task A survey** yet (or Q4 is empty). Please go to the **Evaluate: Survey** tab, select **Task A (without rubric)**, and fill in Q4 with your writing preferences before continuing.")
            _gr_tab_blocked = True
        else:
            st.session_state.infer_coldstart_text = _gr_q4
            _gr_coldstart = _gr_q4

    # Get active rubric and full history
    _gr_rubric_dict, _gr_rubric_idx, _ = get_active_rubric()
    _gr_hist = load_rubric_history()

    if _gr_tab_blocked:
        pass  # Warning already shown above
    elif not _gr_rubric_dict:
        st.warning("No active rubric found. Please create a rubric first in the **Evaluate: Infer** tab.")
    else:
        # ============ SECTION 1: A/B Agreement Summary ============
        st.subheader("Section 1: A/B Judge Agreement")
        _ab_results = st.session_state.get("ab_judge_results", [])
        if not _ab_results:
            st.info("No A/B judge data yet. Make A/B draft choices in the Chat tab to collect agreement data.")
        else:
            _ab_total = len(_ab_results)
            _ab_conditions = ["rubric", "coldstart", "generic"]
            _ab_agree_counts = {}
            _ab_valid_counts = {}
            for _cond in _ab_conditions:
                _agree = 0
                _valid = 0
                for _abr in _ab_results:
                    _judge = _abr.get(f"{_cond}_judge", "skipped")
                    if _judge == "skipped":
                        continue
                    _valid += 1
                    if _judge == _abr.get("user_chosen", ""):
                        _agree += 1
                _ab_agree_counts[_cond] = _agree
                _ab_valid_counts[_cond] = _valid

            _ab_metric_cols = st.columns(3)
            _ab_cond_labels = {"rubric": "Rubric-grounded", "coldstart": "Cold-start", "generic": "Generic"}
            for _ci, _cond in enumerate(_ab_conditions):
                with _ab_metric_cols[_ci]:
                    _v = _ab_valid_counts[_cond]
                    _a = _ab_agree_counts[_cond]
                    if _v > 0:
                        _pct = _a / _v * 100
                        st.metric(_ab_cond_labels[_cond], f"{_pct:.0f}%", f"{_a}/{_v} agree")
                    else:
                        st.metric(_ab_cond_labels[_cond], "—", "no data")

            with st.expander("Detailed A/B judge log", expanded=False):
                for _abi, _abr in enumerate(reversed(_ab_results)):
                    _ts_raw = _abr.get("timestamp", "")
                    _ts = datetime.fromtimestamp(_ts_raw).strftime("%m/%d %H:%M") if isinstance(_ts_raw, (int, float)) else str(_ts_raw)[:16].replace("T", " ")
                    _user = _abr.get("user_chosen", "?")
                    _details = []
                    for _cond in _ab_conditions:
                        _j = _abr.get(f"{_cond}_judge", "skipped")
                        _match = "agree" if _j == _user else ("skipped" if _j == "skipped" else "disagree")
                        _details.append(f"{_ab_cond_labels[_cond]}: {_j} ({_match})")
                    st.markdown(f"**#{_ab_total - _abi}** ({_ts}) — User chose **{_user}** | {' | '.join(_details)}")

        st.divider()

        # ============ SECTION 2: Ranking Checkpoint Results ============
        st.subheader("Section 2: Ranking Checkpoint Results")
        _rk_results = st.session_state.get("ranking_checkpoint_results", [])
        if not _rk_results:
            st.info("No ranking checkpoint data yet. Complete a ranking checkpoint in the Chat tab to see results here.")
        else:
            _rk_src_labels = {"rubric": "Rubric-guided", "coldstart": "Cold-start", "generic": "Generic"}
            _rk_cond_labels = {"rubric": "Rubric-grounded judge", "coldstart": "Cold-start judge", "generic": "Generic judge"}

            # Per-checkpoint display
            for _rki, _rkr in enumerate(reversed(_rk_results)):
                _rk_ts = _rkr.get("timestamp", "")[:16].replace("T", " ")
                _rk_ver = _rkr.get("rubric_version", "?")
                _rk_expanded = _rki == 0  # Expand the most recent
                with st.expander(f"Checkpoint #{len(_rk_results) - _rki} ({_rk_ts}, rubric v{_rk_ver})", expanded=_rk_expanded):
                    st.markdown(f"**Writing task:** {_rkr.get('writing_task', '')}")

                    _rk_user_rank = _rkr.get("user_ranking", [])
                    _rk_llm_rankings = _rkr.get("llm_rankings") or {}
                    _rk_kendall = _rkr.get("kendall_tau") or {}
                    _rk_borda = _rkr.get("borda_distance") or {}
                    _rk_cond_order = [c for c in ["rubric", "coldstart", "generic"] if c in _rk_llm_rankings]

                    # Get all draft sources that appear in rankings
                    _rk_all_sources = list(dict.fromkeys(_rk_user_rank))  # preserve order from user ranking

                    # Build ranking table: rows = judges, columns = draft sources, cells = rank position
                    _rk_src_headers = "".join(
                        f'<th style="text-align:center;padding:8px 14px;font-weight:600;">{_rk_src_labels.get(s, s)}</th>'
                        for s in _rk_all_sources
                    )
                    _rk_table = f'''<table style="width:100%;border-collapse:separate;border-spacing:0 4px;margin:12px 0;">
<tr style="background:#f8f9fa;">
  <th style="text-align:left;padding:8px 14px;"></th>
  {_rk_src_headers}
</tr>'''

                    # User row
                    _rk_table += '<tr style="background:white;"><td style="padding:8px 14px;font-weight:600;">You</td>'
                    for _src in _rk_all_sources:
                        _pos = _rk_user_rank.index(_src) + 1 if _src in _rk_user_rank else "—"
                        _medal = {1: "1st", 2: "2nd", 3: "3rd"}.get(_pos, f"{_pos}th") if isinstance(_pos, int) else "—"
                        _rk_table += f'<td style="text-align:center;padding:8px 14px;background:#f5f5f5;border-radius:4px;font-weight:600;">{_medal}</td>'
                    _rk_table += '</tr>'

                    # Judge rows
                    for _cond in _rk_cond_order:
                        _llm_rank = _rk_llm_rankings.get(_cond, [])
                        _rk_table += f'<tr style="background:white;"><td style="padding:8px 14px;font-weight:600;">{_rk_cond_labels.get(_cond, _cond)}</td>'
                        for _src in _rk_all_sources:
                            _pos = _llm_rank.index(_src) + 1 if _src in _llm_rank else "—"
                            _medal = {1: "1st", 2: "2nd", 3: "3rd"}.get(_pos, f"{_pos}th") if isinstance(_pos, int) else "—"
                            # Highlight if matches user's rank for this source
                            _user_pos = _rk_user_rank.index(_src) + 1 if _src in _rk_user_rank else None
                            _bg = "#E8F5E9" if _pos == _user_pos else "#f5f5f5"
                            _rk_table += f'<td style="text-align:center;padding:8px 14px;background:{_bg};border-radius:4px;">{_medal}</td>'
                        _rk_table += '</tr>'

                    _rk_table += '</table>'
                    st.markdown(_rk_table, unsafe_allow_html=True)
                    st.caption("Green cells = judge agrees with your rank for that draft.")

                    # Tau and Borda distance per condition
                    st.markdown("**Alignment metrics:**")
                    _rk_metric_cols = st.columns(len(_rk_cond_order))
                    for _ci, _cond in enumerate(_rk_cond_order):
                        with _rk_metric_cols[_ci]:
                            _tau = _rk_kendall.get(_cond)
                            _bd = _rk_borda.get(_cond)
                            _tau_str = f"{_tau:.2f}" if _tau is not None else "—"
                            _bd_str = f"{_bd:.1f}" if _bd is not None else "—"
                            st.metric(_rk_cond_labels.get(_cond, _cond), f"tau = {_tau_str}", f"Borda dist = {_bd_str}")

            # Explanation of metrics
            st.divider()
            st.markdown("**Understanding the metrics:**")
            st.markdown(
                "- **Kendall's tau** measures rank correlation between your ranking and the judge's ranking. "
                "Ranges from -1 (completely opposite) to +1 (perfect agreement). A tau of 0 means no correlation.\n"
                "- **Borda distance** measures how far apart the rankings are in total points. "
                "Lower is better — 0 means the rankings are identical. "
                "Each position earns Borda points (1st=2, 2nd=1, 3rd=0 for 3 drafts), and the distance sums the absolute differences per draft."
            )

        st.divider()

        # ============ SECTION 3: Full Evaluation (Advanced) ============
        _gr_show_full = st.checkbox("Show Full Evaluation (Advanced)", value=False, key="grading_show_full_eval")

        if _gr_show_full:
            _gr_rubric_list = _gr_rubric_dict.get("rubric", [])

            # --- Determine rubric versions for draft generation ---
            _gr_r_star = _gr_hist[-1] if _gr_hist else _gr_rubric_dict  # latest
            _gr_r0 = _gr_hist[0] if _gr_hist else _gr_rubric_dict  # first version
            _gr_r1 = None
            _gr_r1_idx = None
            if len(_gr_hist) >= 3:
                # Prefer an edited (user-saved) version for R₁; fall back to midpoint
                _edited_sources = {"edited", "edit_feedback", "chat_edit"}
                _edited_versions = [(i, h) for i, h in enumerate(_gr_hist) if h.get("source") in _edited_sources and 0 < i < len(_gr_hist) - 1]
                if _edited_versions:
                    # Pick the edited version closest to the midpoint
                    mid = len(_gr_hist) // 2
                    _gr_r1_idx = min(_edited_versions, key=lambda x: abs(x[0] - mid))[0]
                else:
                    _gr_r1_idx = len(_gr_hist) // 2
                _gr_r1 = _gr_hist[_gr_r1_idx]
            _gr_num_drafts = 5 if _gr_r1 else (4 if len(_gr_hist) >= 2 else 3)

            st.subheader("Step 1: New Writing Task")

            # Build conversation text from dp_messages for task generation
            _gr_dp_messages = st.session_state.get("infer_dp_messages", [])
            _gr_step5_text = ""
            _gr_msg_num = 1
            for _msg in _gr_dp_messages:
                _role = _msg.get('role', 'unknown')
                _content = _msg.get('content', '')
                if _role == 'user':
                    _gr_step5_text += f"\n\n[Message #{_gr_msg_num}] USER:\n{_content}"
                    _gr_msg_num += 1
                elif _role == 'assistant':
                    _gr_step5_text += f"\n\n[Message #{_gr_msg_num}] ASSISTANT:\n{_content}"
                    _gr_msg_num += 1

            # Build project task examples
            _gr_project_task_examples = ""
            _gr_all_convs = load_conversations()
            if _gr_all_convs:
                _task_snippets = []
                for _conv in _gr_all_convs[:10]:
                    _conv_data = load_conversation_data(_conv.get("filename") or _conv.get("id"))
                    if _conv_data:
                        _msgs = _conv_data.get("messages", [])
                        for _m in _msgs:
                            if _m.get("role") == "user" and _m.get("content", "").strip():
                                _snippet = _m["content"].strip()[:300]
                                if len(_m["content"].strip()) > 300:
                                    _snippet += "..."
                                _task_snippets.append(_snippet)
                                break
                if _task_snippets:
                    _gr_project_task_examples = "\n\n---\n\n".join(
                        f"**Task {i+1}:**\n{s}" for i, s in enumerate(_task_snippets)
                    )

            # --- Step 1: Generate writing task automatically ---
            if not st.session_state.infer_step6_writing_task:
                if not st.session_state.infer_step6_auto_gen_done:
                    with st.spinner("Generating writing task..."):
                        try:
                            prompt = GRADING_generate_writing_task_prompt(_gr_step5_text, _gr_project_task_examples)
                            resp = _api_call_with_retry(model=MODEL_PRIMARY, max_tokens=500, messages=[{"role": "user", "content": prompt}])
                            text = "".join(b.text for b in resp.content if b.type == "text")
                            task_text = text.strip()
                            if task_text:
                                st.session_state.infer_step6_generated_task = task_text
                                st.session_state.infer_step6_writing_task = task_text
                                st.session_state.infer_step6_auto_gen_done = True
                                st.rerun()
                            else:
                                st.session_state.infer_step6_auto_gen_done = True
                                st.error("No task text returned.")
                        except Exception as e:
                            st.session_state.infer_step6_auto_gen_done = True
                            st.error(str(e))
                else:
                    st.markdown("Generate a **new** writing task that matches the type of writing in your project conversations, but a different specific scenario.")
                    if st.button("📝 Generate Writing Task", use_container_width=True, type="primary", key="grading_gen_task_btn"):
                        with st.spinner("Generating writing task..."):
                            try:
                                prompt = GRADING_generate_writing_task_prompt(_gr_step5_text, _gr_project_task_examples)
                                resp = _api_call_with_retry(model=MODEL_PRIMARY, max_tokens=500, messages=[{"role": "user", "content": prompt}])
                                text = "".join(b.text for b in resp.content if b.type == "text")
                                task_text = text.strip()
                                if task_text:
                                    st.session_state.infer_step6_generated_task = task_text
                                    st.session_state.infer_step6_writing_task = task_text
                                    st.rerun()
                                else:
                                    st.error("No task text returned.")
                            except Exception as e:
                                st.error(str(e))

            # --- Step 2: Show task + edit/regenerate ---
            if st.session_state.infer_step6_writing_task:
                st.markdown("**Writing task:**")
                st.info(st.session_state.infer_step6_writing_task)
                with st.expander("✏️ Use a different task (type your own or edit)", expanded=False):
                    custom_task = st.text_area(
                        "Your task text",
                        value=st.session_state.infer_step6_writing_task,
                        key=f"grading_custom_task_{st.session_state.infer_step6_custom_task_key_version}",
                        placeholder="e.g. Write a brief professional email declining a meeting and proposing an alternative."
                    )
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("Use my task", key="grading_use_custom"):
                            if custom_task.strip():
                                st.session_state.infer_step6_writing_task = custom_task.strip()
                                st.session_state.infer_step6_drafts = None
                                st.session_state.infer_step6_draft_labels = None
                                st.rerun()
                    with c2:
                        if st.button("🔄 Regenerate task", key="grading_regen_task"):
                            with st.spinner("Regenerating..."):
                                try:
                                    prompt = GRADING_generate_writing_task_prompt(_gr_step5_text, _gr_project_task_examples)
                                    resp = _api_call_with_retry(model=MODEL_PRIMARY, max_tokens=500, messages=[{"role": "user", "content": prompt}])
                                    text = "".join(b.text for b in resp.content if b.type == "text")
                                    if text.strip():
                                        st.session_state.infer_step6_generated_task = text.strip()
                                        st.session_state.infer_step6_writing_task = text.strip()
                                        st.session_state.infer_step6_custom_task_key_version += 1
                                        st.session_state.infer_step6_drafts = None
                                        st.session_state.infer_step6_draft_labels = None
                                        st.rerun()
                                except Exception as e:
                                    st.error(str(e))

                # --- Step 3: Generate 4-5 drafts ---
                if not st.session_state.infer_step6_drafts:
                    _draft_desc_parts = ["R* (final rubric v{})".format(_gr_r_star.get("version", "?"))]
                    if _gr_r1:
                        _draft_desc_parts.append("R₁ (midpoint rubric v{})".format(_gr_r1.get("version", "?")))
                    if len(_gr_hist) >= 2:
                        _draft_desc_parts.append("R₀ (initial rubric v{})".format(_gr_r0.get("version", "?")))
                    _draft_desc_parts += ["Cold-start", "Generic"]
                    st.markdown(f"Generate **{_gr_num_drafts} drafts** for this task: {', '.join(_draft_desc_parts)}.")
                    if st.button(f"📄 Generate {_gr_num_drafts} Drafts", use_container_width=True, type="primary", key="grading_gen_drafts_btn"):
                        task = st.session_state.infer_step6_writing_task
                        with st.spinner(f"Generating {_gr_num_drafts} drafts..."):
                            try:
                                generated = {}
                                versions_used = {}
                                # R* draft
                                r_star_json = json.dumps(_rubric_to_json_serializable(_gr_r_star), indent=2)
                                pr = GRADING_generate_draft_from_rubric_prompt(task, r_star_json)
                                r = _api_call_with_retry(model=MODEL_PRIMARY, max_tokens=4000, messages=[{"role": "user", "content": pr}])
                                generated["r_star"] = "".join(b.text for b in r.content if b.type == "text").strip()
                                versions_used["r_star"] = _gr_r_star.get("version", 1)

                                # R₁ draft (if exists)
                                if _gr_r1:
                                    r1_json = json.dumps(_rubric_to_json_serializable(_gr_r1), indent=2)
                                    pr1 = GRADING_generate_draft_from_rubric_prompt(task, r1_json)
                                    r1_resp = _api_call_with_retry(model=MODEL_PRIMARY, max_tokens=4000, messages=[{"role": "user", "content": pr1}])
                                    generated["r1"] = "".join(b.text for b in r1_resp.content if b.type == "text").strip()
                                    versions_used["r1"] = _gr_r1.get("version", 1)

                                # R₀ draft (only if different from R*)
                                if len(_gr_hist) >= 2:
                                    r0_json = json.dumps(_rubric_to_json_serializable(_gr_r0), indent=2)
                                    pr0 = GRADING_generate_draft_from_rubric_prompt(task, r0_json)
                                    r0_resp = _api_call_with_retry(model=MODEL_PRIMARY, max_tokens=4000, messages=[{"role": "user", "content": pr0}])
                                    generated["r0"] = "".join(b.text for b in r0_resp.content if b.type == "text").strip()
                                    versions_used["r0"] = _gr_r0.get("version", 1)

                                # Cold-start draft
                                pc = GRADING_generate_draft_from_coldstart_prompt(task, _gr_coldstart)
                                c_resp = _api_call_with_retry(model=MODEL_PRIMARY, max_tokens=4000, messages=[{"role": "user", "content": pc}])
                                generated["coldstart"] = "".join(b.text for b in c_resp.content if b.type == "text").strip()

                                # Generic draft
                                pg = GRADING_generate_draft_generic_prompt(task)
                                g_resp = _api_call_with_retry(model=MODEL_PRIMARY, max_tokens=4000, messages=[{"role": "user", "content": pg}])
                                generated["generic"] = "".join(b.text for b in g_resp.content if b.type == "text").strip()

                                # Shuffle for blind grading
                                source_keys = list(generated.keys())
                                random.shuffle(source_keys)
                                st.session_state.infer_step6_drafts = generated
                                st.session_state.infer_step6_draft_labels = source_keys
                                st.session_state.infer_step6_rubric_versions_used = versions_used
                                st.session_state.infer_step6_blind_ratings = None
                                st.session_state.infer_step6_user_ranking = None
                                st.session_state.infer_step6_user_dimension_checks = None
                                st.session_state.infer_step6_llm_evaluations = None
                                st.session_state.infer_step6_survey = None
                                st.session_state.infer_step6_claim2_metrics = None
                                st.session_state.infer_step6_claim3_metrics = None
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))

                # --- Step 2: Show drafts + blind grading ---
                if st.session_state.infer_step6_drafts:
                    drafts = st.session_state.infer_step6_drafts
                    labels = st.session_state.infer_step6_draft_labels
                    num_d = len(labels)
                    letters = [chr(65 + i) for i in range(num_d)]  # A, B, C, D, E
                    _gr_letter_to_src = {letters[i]: labels[i] for i in range(num_d)}

                    st.subheader("Step 2: Blind Grading")

                    # Show drafts
                    st.markdown("""
                    <style>
                    textarea[disabled] {
                        color: #1e1e1e !important;
                        -webkit-text-fill-color: #1e1e1e !important;
                        background-color: #f5f5f5 !important;
                        font-size: 15px !important;
                        line-height: 1.5 !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)

                    # Show drafts in rows of 3
                    draft_height = 300
                    for row_start in range(0, num_d, 3):
                        row_letters = letters[row_start:row_start + 3]
                        cols = st.columns(len(row_letters))
                        for ci, letter in enumerate(row_letters):
                            with cols[ci]:
                                st.markdown(f"**Version {letter}**")
                                src_key = _gr_letter_to_src[letter]
                                st.text_area("", value=drafts[src_key], height=draft_height, key=f"grading_draft_{letter}", disabled=True, label_visibility="collapsed")

                    st.divider()
                    st.markdown("**Blind overall satisfaction** — Rate each draft without looking at the rubric. Giving multiple drafts the same score is perfectly fine.")
                    st.caption("1 = Does not reflect what I want at all | 2 = Misses most of what I care about | 3 = Gets some things right but misses important aspects | 4 = Mostly reflects what I want with minor issues | 5 = Fully reflects what I want")

                    if st.session_state.infer_step6_blind_ratings is None:
                        st.session_state.infer_step6_blind_ratings = {l: 3 for l in letters}
                    _blind = st.session_state.infer_step6_blind_ratings

                    # Rating sliders in rows of 3
                    _blind_vals = {}
                    for row_start in range(0, num_d, 3):
                        row_letters = letters[row_start:row_start + 3]
                        cols = st.columns(len(row_letters))
                        for ci, letter in enumerate(row_letters):
                            with cols[ci]:
                                _blind_vals[letter] = st.slider(f"Draft {letter}", 1, 5, _blind.get(letter, 3), key=f"grading_blind_{letter}")

                    # Ranking
                    st.markdown("**Rank all drafts** from most preferred (1) to least preferred ({})".format(num_d))
                    if st.session_state.infer_step6_user_ranking is None:
                        st.session_state.infer_step6_user_ranking = []
                    _rank_vals = {}
                    rank_cols = st.columns(num_d)
                    for ci, letter in enumerate(letters):
                        with rank_cols[ci]:
                            _rank_vals[letter] = st.selectbox(
                                f"Draft {letter} rank",
                                options=list(range(1, num_d + 1)),
                                key=f"grading_rank_{letter}"
                            )

                    # --- Step 3: Rubric criterion checks (3 drafts only: R*, coldstart, generic) ---
                    st.divider()
                    st.subheader("Step 3: Rubric Criterion Checks")
                    st.markdown("Using the **same drafts from the Blind Grading step above**, mark each criterion as Met or Not met for **three of them** only. The remaining drafts are excluded from this step to avoid circularity.")

                    # Find which letters map to the 3 evaluation drafts
                    _eval_srcs = ["r_star", "coldstart", "generic"]
                    _eval_letters = [l for l in letters if _gr_letter_to_src[l] in _eval_srcs]
                    _eval_src_labels = {"r_star": "R*", "coldstart": "Cold-start", "generic": "Generic"}

                    # Show the 3 evaluated drafts for reference
                    _eval_cols = st.columns(len(_eval_letters))
                    for ci, letter in enumerate(_eval_letters):
                        with _eval_cols[ci]:
                            src_key = _gr_letter_to_src[letter]
                            st.markdown(f"**Draft {letter}**")
                            st.text_area("", value=drafts[src_key], height=250, key=f"grading_draft_1e_{letter}", disabled=True, label_visibility="collapsed")

                    if st.session_state.infer_step6_user_dimension_checks is None:
                        st.session_state.infer_step6_user_dimension_checks = {l: {} for l in _eval_letters}
                    dim_checks = st.session_state.infer_step6_user_dimension_checks
                    for l in _eval_letters:
                        if l not in dim_checks:
                            dim_checks[l] = {}

                    if _gr_rubric_list:
                        opts = ["✓ Met", "✗ Not met", "—"]
                        for crit_idx, criterion in enumerate(_gr_rubric_list):
                            cname = criterion.get("name", "")
                            if not cname:
                                continue
                            vals = [dim_checks.get(l, {}).get(cname, None) for l in _eval_letters]
                            criterion_done = all(v is not None for v in vals)
                            expander_label = f"{'✅' if criterion_done else '⬜'} {cname}"
                            with st.expander(expander_label, expanded=not criterion_done):
                                if criterion.get("description"):
                                    st.caption(criterion.get("description"))
                                ucols = st.columns(len(_eval_letters))
                                for ci, letter in enumerate(_eval_letters):
                                    with ucols[ci]:
                                        val = dim_checks.get(letter, {}).get(cname, None)
                                        idx = 0 if val is True else (1 if val is False else 2)
                                        opt = st.radio(f"Draft {letter}", opts, index=idx, key=f"grading_crit_{letter}_{crit_idx}", horizontal=True)
                                        dim_checks.setdefault(letter, {})[cname] = None if opt == "—" else (opt == "✓ Met")

                    # --- Run LLM evaluations (3 drafts only) ---
                    if st.session_state.infer_step6_llm_evaluations is None:
                        # st.divider()
                        # st.subheader("Step 6: LLM Evaluations")
                        if st.button("👾 Run LLM evaluations", type="primary", key="grading_run_llm_evals"):
                            # Save current blind ratings
                            st.session_state.infer_step6_blind_ratings = _blind_vals
                            # Save ranking
                            ranked = sorted(_rank_vals.items(), key=lambda x: x[1])
                            st.session_state.infer_step6_user_ranking = [r[0] for r in ranked]
                            st.session_state.infer_step6_user_dimension_checks = {k: dict(v) for k, v in dim_checks.items()}

                            task_desc = st.session_state.infer_step6_writing_task
                            rubric_json = json.dumps(_rubric_to_json_serializable(_gr_r_star), indent=2)
                            # Map eval letters to actual draft texts
                            _eval_draft_texts = []
                            for l in _eval_letters:
                                src = _gr_letter_to_src[l]
                                _eval_draft_texts.append(drafts[src])

                            with st.spinner("Evaluating 3 drafts under rubric, cold-start, and generic conditions..."):
                                try:
                                    prompt = GRADING_unified_eval_prompt(task_desc, rubric_json, _gr_coldstart, *_eval_draft_texts[:3])
                                    r = _api_call_with_retry(model=MODEL_PRIMARY, max_tokens=16000, messages=[{"role": "user", "content": prompt}])
                                    txt = "".join(b.text for b in r.content if b.type == "text")
                                    js = re.search(r'\{[\s\S]*\}', txt)
                                    if js:
                                        parsed = json.loads(js.group())
                                        evals = {}
                                        # Map parsed A/B/C back to our eval letters
                                        for i, letter in enumerate(_eval_letters[:3]):
                                            parsed_key = chr(65 + i)  # A, B, C from the prompt
                                            block = parsed.get(parsed_key, {})
                                            evals[letter] = {
                                                "rubric": block.get("rubric", {}),
                                                "coldstart": block.get("coldstart", {}),
                                                "generic": block.get("generic", {}),
                                            }
                                        st.session_state.infer_step6_llm_evaluations = evals
                                        st.rerun()
                                    else:
                                        st.error("Could not parse evaluation JSON from response.")
                                except json.JSONDecodeError as e:
                                    st.error(f"Invalid JSON in evaluation response: {e}")
                                except Exception as e:
                                    st.error(str(e))

                    # --- Step 7+: Computation, survey, display, save (after LLM evals) ---
                    if st.session_state.infer_step6_llm_evaluations is not None:
                        evals = st.session_state.infer_step6_llm_evaluations
                        blind_ratings = st.session_state.infer_step6_blind_ratings or {}
                        user_ranking = st.session_state.infer_step6_user_ranking or []

                        # Helper functions (used in both computation and display)
                        def _gr_pct_to_15(pct):
                            if pct >= 0.90: return 5
                            if pct >= 0.75: return 4
                            if pct >= 0.50: return 3
                            if pct >= 0.25: return 2
                            return 1

                        def _gr_score_from_llm_eval(eval_data):
                            if not eval_data or "error" in eval_data:
                                return None
                            cs = eval_data.get("criteria_scores", [])
                            if not cs:
                                return None
                            total_met = sum(c.get("dimensions_met", 0) for c in cs)
                            total_dim = sum(c.get("dimensions_total", 0) for c in cs)
                            if total_dim == 0:
                                return None
                            return _gr_pct_to_15(total_met / total_dim)

                        def _cohens_kappa(user_checks, llm_checks):
                            n = len(user_checks)
                            if n == 0:
                                return None
                            agree = sum(1 for u, l in zip(user_checks, llm_checks) if u == l)
                            p_o = agree / n
                            p_u1 = sum(user_checks) / n
                            p_l1 = sum(llm_checks) / n
                            p_e = p_u1 * p_l1 + (1 - p_u1) * (1 - p_l1)
                            if p_e == 1:
                                return 1.0
                            return (p_o - p_e) / (1 - p_e)

                        # --- Step 7: Automatic computation ---
                        if st.session_state.infer_step6_claim2_metrics is None or st.session_state.infer_step6_claim3_metrics is None:
                            # ---- CLAIM 2: Convergence trajectory ----
                            # Map source labels to trajectory order
                            trajectory_order = ["coldstart"]
                            if "r0" in drafts:
                                trajectory_order.append("r0")
                            if "r1" in drafts:
                                trajectory_order.append("r1")
                            trajectory_order.append("r_star")

                            trajectory_scores = {}
                            for src in trajectory_order:
                                # Find which letter this source maps to
                                letter = next((l for l, s in _gr_letter_to_src.items() if s == src), None)
                                if letter and letter in blind_ratings:
                                    trajectory_scores[src] = blind_ratings[letter]

                            # Improvement per step
                            improvements = []
                            traj_keys = list(trajectory_scores.keys())
                            for i in range(1, len(traj_keys)):
                                prev = trajectory_scores[traj_keys[i - 1]]
                                curr = trajectory_scores[traj_keys[i]]
                                improvements.append({
                                    "from": traj_keys[i - 1],
                                    "to": traj_keys[i],
                                    "delta": curr - prev
                                })

                            # Diminishing returns check
                            diminishing = False
                            if len(improvements) >= 2:
                                deltas = [imp["delta"] for imp in improvements]
                                diminishing = all(deltas[i] <= deltas[i - 1] for i in range(1, len(deltas)))

                            # Rubric diffs between versions
                            rubric_diffs = []
                            if len(_gr_hist) >= 2:
                                rubric_diffs.append({
                                    "from_version": _gr_r0.get("version", 1),
                                    "to_version": (_gr_r1 or _gr_r_star).get("version", "?"),
                                    "label": "R₀ → {}".format("R₁" if _gr_r1 else "R*"),
                                    "edits": classify_rubric_edits(_gr_r0.get("rubric", []), (_gr_r1 or _gr_r_star).get("rubric", []))
                                })
                            if _gr_r1 and len(_gr_hist) >= 3:
                                rubric_diffs.append({
                                    "from_version": _gr_r1.get("version", "?"),
                                    "to_version": _gr_r_star.get("version", "?"),
                                    "label": "R₁ → R*",
                                    "edits": classify_rubric_edits(_gr_r1.get("rubric", []), _gr_r_star.get("rubric", []))
                                })

                            claim2 = {
                                "trajectory_scores": trajectory_scores,
                                "improvements": improvements,
                                "diminishing": diminishing,
                                "rubric_diffs": rubric_diffs,
                            }
                            st.session_state.infer_step6_claim2_metrics = claim2

                            # ---- CLAIM 3: Alignment metrics ----
                            # Kendall tau: user blind scores vs LLM scores per condition (3 eval drafts)
                            corr = {}
                            for cond in ["rubric", "coldstart", "generic"]:
                                pairs = []
                                for letter in _eval_letters:
                                    user_s = blind_ratings.get(letter)
                                    llm_s = _gr_score_from_llm_eval(evals.get(letter, {}).get(cond))
                                    if user_s is not None and llm_s is not None:
                                        pairs.append((user_s, llm_s))
                                tau = None
                                if len(pairs) >= 2:
                                    try:
                                        tau, _ = kendalltau([p[0] for p in pairs], [p[1] for p in pairs])
                                    except Exception:
                                        pass
                                corr[cond] = {"tau": tau, "pairs": pairs}

                            # Dimension agreement rates per condition
                            dim_agree = {}
                            _dc = st.session_state.infer_step6_user_dimension_checks or {}
                            user_binary_all = []
                            llm_binary_all = []
                            for cond in ["rubric", "coldstart", "generic"]:
                                total_agree = total_count = 0
                                for letter in _eval_letters:
                                    uc = _dc.get(letter, {})
                                    eval_data = evals.get(letter, {}).get(cond, {})
                                    for c in (eval_data.get("criteria_scores") or []):
                                        cname = c.get("name", "")
                                        user_met = uc.get(cname)
                                        if user_met is None:
                                            continue
                                        dims_detail = c.get("dimensions_detail", [])
                                        llm_met = all(d.get("met", False) for d in dims_detail) if dims_detail else None
                                        if llm_met is None:
                                            continue
                                        total_count += 1
                                        if user_met == llm_met:
                                            total_agree += 1
                                        if cond == "rubric":
                                            user_binary_all.append(int(user_met))
                                            llm_binary_all.append(int(llm_met))
                                dim_agree[cond] = {"rate": (total_agree / total_count) if total_count else None, "agree": total_agree, "count": total_count}

                            # Cohen's kappa (rubric condition only)
                            kappa = _cohens_kappa(user_binary_all, llm_binary_all) if user_binary_all else None

                            # Top-draft match per condition
                            user_top = user_ranking[0] if user_ranking else None
                            top_match = {}
                            for cond in ["rubric", "coldstart", "generic"]:
                                cond_scores = []
                                for letter in _eval_letters:
                                    s = _gr_score_from_llm_eval(evals.get(letter, {}).get(cond))
                                    if s is not None:
                                        cond_scores.append((letter, s))
                                if cond_scores:
                                    cond_scores.sort(key=lambda x: -x[1])
                                    top_match[cond] = (cond_scores[0][0] == user_top) if user_top else None
                                else:
                                    top_match[cond] = None

                            # Failure modes: where generic disagrees with user but rubric agrees
                            failure_modes = []
                            for letter in _eval_letters:
                                uc = _dc.get(letter, {})
                                rub_eval = evals.get(letter, {}).get("rubric", {})
                                gen_eval = evals.get(letter, {}).get("generic", {})
                                for c in (gen_eval.get("criteria_scores") or []):
                                    cname = c.get("name", "")
                                    user_met = uc.get(cname)
                                    if user_met is None:
                                        continue
                                    gen_dims = c.get("dimensions_detail", [])
                                    gen_met = all(d.get("met", False) for d in gen_dims) if gen_dims else None
                                    if gen_met is None or gen_met == user_met:
                                        continue
                                    # Generic disagrees with user — check if rubric agrees
                                    rub_crit = next((rc for rc in (rub_eval.get("criteria_scores") or []) if rc.get("name") == cname), None)
                                    rub_met = None
                                    if rub_crit:
                                        rub_dims = rub_crit.get("dimensions_detail", [])
                                        rub_met = all(d.get("met", False) for d in rub_dims) if rub_dims else None
                                    corrected = rub_met == user_met if rub_met is not None else None
                                    failure_modes.append({
                                        "letter": letter,
                                        "criterion": cname,
                                        "user": user_met,
                                        "generic": gen_met,
                                        "rubric_corrected": corrected
                                    })

                            claim3 = {
                                "correlations": corr,
                                "dimension_agreement": dim_agree,
                                "cohens_kappa": kappa,
                                "top_match": top_match,
                                "failure_modes": failure_modes,
                            }
                            st.session_state.infer_step6_claim3_metrics = claim3

                        # --- Step 8: Post-task survey ---
                        st.divider()
                        st.subheader("Step 4: Post-Task Survey")

                        if st.session_state.infer_step6_survey is None:
                            st.session_state.infer_step6_survey = {}
                        _survey = st.session_state.infer_step6_survey

                        st.markdown("**Does this rubric accurately represent what you care about in your writing?**")
                        accuracy_opts = [
                            "Yes, very accurately",
                            "Mostly, with minor gaps",
                            "Somewhat, it captures the basics but misses nuance",
                            "No, it does not represent my preferences well"
                        ]
                        _survey["accuracy"] = st.radio(
                            "Rubric accuracy",
                            accuracy_opts,
                            index=accuracy_opts.index(_survey["accuracy"]) if _survey.get("accuracy") in accuracy_opts else 0,
                            key="grading_survey_accuracy",
                            label_visibility="collapsed"
                        )

                        st.markdown("**How many more rounds of refinement do you think you would need before the rubric fully captures your preferences?**")
                        rounds_opts = [
                            "It already does",
                            "One more round",
                            "Two to three more rounds",
                            "Many more rounds"
                        ]
                        _survey["rounds_needed"] = st.radio(
                            "Rounds needed",
                            rounds_opts,
                            index=rounds_opts.index(_survey["rounds_needed"]) if _survey.get("rounds_needed") in rounds_opts else 0,
                            key="grading_survey_rounds",
                            label_visibility="collapsed"
                        )

                        # --- Step 9: Results Display ---
                        st.divider()
                        st.subheader("Step 5: Results")

                        claim2 = st.session_state.infer_step6_claim2_metrics or {}
                        claim3 = st.session_state.infer_step6_claim3_metrics or {}

                        # ---- CLAIM 2 DISPLAY ----
                        st.markdown("### Convergence Trajectory")

                        traj = claim2.get("trajectory_scores", {})
                        if len(traj) >= 2:
                            import pandas as pd
                            import altair as alt
                            traj_display = {"coldstart": "Cold-start", "r0": "R₀", "r1": "R₁", "r_star": "R*"}
                            ordered_keys = [k for k in ["coldstart", "r0", "r1", "r_star"] if k in traj]
                            stage_labels = [traj_display.get(k, k) for k in ordered_keys]
                            chart_data = pd.DataFrame({
                                "Stage": stage_labels,
                                "Satisfaction": [traj[k] for k in ordered_keys]
                            })
                            chart = alt.Chart(chart_data).mark_line(point=True).encode(
                                x=alt.X("Stage", sort=stage_labels, title="Stage"),
                                y=alt.Y("Satisfaction", scale=alt.Scale(domain=[1, 5]), title="Satisfaction (1-5)")
                            ).properties(height=300)
                            st.altair_chart(chart, use_container_width=True)
                        else:
                            st.info("Only one rubric version available — convergence trajectory requires at least two versions.")

                        # Rubric diffs
                        diffs = claim2.get("rubric_diffs", [])
                        if diffs:
                            st.markdown("**Rubric changes at each transition:**")
                            for diff_info in diffs:
                                with st.expander(f"{diff_info['label']} (v{diff_info['from_version']} → v{diff_info['to_version']})"):
                                    edits = diff_info["edits"]
                                    if edits.get("added"):
                                        for a in edits["added"]:
                                            st.markdown(f"➕ **Added:** {a['name']}")
                                    if edits.get("removed"):
                                        for r in edits["removed"]:
                                            st.markdown(f"➖ **Removed:** {r['name']}")
                                    if edits.get("reweighted"):
                                        for w in edits["reweighted"]:
                                            st.markdown(f"🔃 **Reweighted:** {w['name']} ({w['old_weight']} → {w['new_weight']})")
                                    if edits.get("reworded"):
                                        for rw in edits["reworded"]:
                                            st.markdown(f"✏️ **Reworded:** {rw['name']}")
                                    if edits.get("dimensions_changed"):
                                        for dc in edits["dimensions_changed"]:
                                            parts = []
                                            if dc.get("added_dims"):
                                                parts.append(f"+{len(dc['added_dims'])} dims")
                                            if dc.get("removed_dims"):
                                                parts.append(f"-{len(dc['removed_dims'])} dims")
                                            st.markdown(f"📐 **Dimensions changed:** {dc['name']} ({', '.join(parts)})")
                                    if not any(edits.get(k) for k in ["added", "removed", "reweighted", "reworded", "dimensions_changed"]):
                                        st.caption("No changes detected.")

                        # ---- CLAIM 3 DISPLAY ----
                        st.divider()
                        st.markdown("### Rubric–User Alignment")

                        # Order columns: R* (final) first, then cold-start, then generic
                        _display_order = ["r_star", "coldstart", "generic"]
                        _display_letters = []
                        for src_key in _display_order:
                            for l in _eval_letters:
                                if _gr_letter_to_src[l] == src_key:
                                    _display_letters.append(l)
                                    break

                        # Draft reveal pills
                        src_label_map = {"r_star": "R* (final rubric)", "r1": "R₁ (midpoint)", "r0": "R₀ (initial)", "coldstart": "cold-start", "generic": "generic"}
                        pill_colors = {"r_star": ("#E3F2FD", "#1565C0"), "r1": ("#E8EAF6", "#283593"), "r0": ("#F3E5F5", "#6A1B9A"), "coldstart": ("#FFF3E0", "#E65100"), "generic": ("#F3E5F5", "#7B1FA2")}
                        pills_html = '<div style="display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap;">'
                        for letter in _display_letters:
                            src = _gr_letter_to_src[letter]
                            label = src_label_map.get(src, src)
                            bg, border = pill_colors.get(src, ("#f5f5f5", "#999"))
                            pills_html += f'<span style="display:inline-flex;align-items:center;gap:6px;padding:6px 14px;background:{bg};border:1.5px solid {border};border-radius:20px;font-size:0.9em;"><strong>Draft {letter}</strong> <span style="color:#666;">→ {label}</span></span>'
                        pills_html += '</div>'
                        st.markdown(pills_html, unsafe_allow_html=True)

                        # Score matrix (4 rows × 3 evaluated draft columns)
                        def _gr_score_cell(s):
                            if s is None:
                                return '<td style="text-align:center;padding:12px 18px;color:#999;font-size:1.1em;">—</td>'
                            colors = {5: ("#E8F5E9", "#2E7D32", "⭐⭐⭐"), 4: ("#E3F2FD", "#1565C0", "⭐⭐"), 3: ("#FFF3E0", "#E65100", "⭐"), 2: ("#FBE9E7", "#BF360C", "◇"), 1: ("#FFEBEE", "#C62828", "☆")}
                            bg, fg, icon = colors.get(s, ("#f5f5f5", "#333", ""))
                            return f'<td style="text-align:center;padding:12px 18px;background:{bg};color:{fg};font-weight:700;font-size:1.2em;border-radius:6px;">{s} {icon}</td>'

                        def _gr_user_score(letter):
                            d = (st.session_state.infer_step6_user_dimension_checks or {}).get(letter, {})
                            vals = [v for v in d.values() if v is not None]
                            if not vals:
                                return None
                            return _gr_pct_to_15(sum(1 for v in vals if v is True) / len(vals))

                        # Build score rows in fixed order: R* → cold-start → generic
                        user_scores = [_gr_user_score(l) for l in _display_letters]
                        rub_scores = [_gr_score_from_llm_eval(evals.get(l, {}).get("rubric")) for l in _display_letters]
                        cs_scores = [_gr_score_from_llm_eval(evals.get(l, {}).get("coldstart")) for l in _display_letters]
                        gen_scores = [_gr_score_from_llm_eval(evals.get(l, {}).get("generic")) for l in _display_letters]

                        rows = [
                            ("👤 You", "rubric criterion checks", user_scores),
                            ("👾 LLM — rubric", "R* rubric-grounded", rub_scores),
                            ("👾 LLM — cold-start", "cold-start-grounded", cs_scores),
                            ("👾 LLM — generic", "standard quality", gen_scores),
                        ]
                        # Build header
                        header_cells = "".join(
                            f'<th style="text-align:center;padding:10px 14px;font-size:0.95em;font-weight:600;">Draft {l} ({_eval_src_labels.get(_gr_letter_to_src[l], "")})</th>'
                            for l in _display_letters
                        )
                        table_html = f'''<table style="width:100%;border-collapse:separate;border-spacing:0 6px;margin:12px 0;">
    <tr style="background:#f8f9fa;">
      <th style="text-align:left;padding:10px 14px;font-size:0.85em;color:#666;"></th>
      {header_cells}
    </tr>'''
                        for label, desc, scores in rows:
                            table_html += f'<tr style="background:white;"><td style="padding:10px 14px;"><strong>{label}</strong><br><span style="font-size:0.8em;color:#888;">{desc}</span></td>'
                            for s in scores:
                                table_html += _gr_score_cell(s)
                            table_html += '</tr>'
                        table_html += '</table>'
                        st.markdown(table_html, unsafe_allow_html=True)

                        # Score legend
                        legend_html = '<div style="display:flex;gap:6px;flex-wrap:wrap;margin:8px 0 4px 0;">'
                        for val, label, bg, fg in [(5, "Excellent", "#E8F5E9", "#2E7D32"), (4, "Good", "#E3F2FD", "#1565C0"), (3, "Fair", "#FFF3E0", "#E65100"), (2, "Needs Work", "#FBE9E7", "#BF360C"), (1, "Weak", "#FFEBEE", "#C62828")]:
                            legend_html += f'<span style="padding:3px 10px;background:{bg};color:{fg};border-radius:12px;font-size:0.8em;font-weight:600;">{val} = {label}</span>'
                        legend_html += '</div>'
                        st.markdown(legend_html, unsafe_allow_html=True)

    #                     # --- Agreement summary: how similar is each LLM condition to your evaluation ---
                        # st.markdown("**How closely does each LLM grading approach match yours?**")
                        # st.caption("Match % = percentage of drafts where the LLM's overall score (derived from criterion checks) matches your score exactly.")
    #
                        # _match_data = []
                        # for cond_key, cond_label in [("rubric", "Rubric-grounded"), ("coldstart", "Cold-start-grounded"), ("generic", "Generic")]:
                            # matches = 0
                            # total = 0
                            # for l in _display_letters:
                                # u = _gr_user_score(l)
                                # llm_s = _gr_score_from_llm_eval(evals.get(l, {}).get(cond_key))
                                # if u is not None and llm_s is not None:
                                    # total += 1
                                    # if u == llm_s:
                                        # matches += 1
                            # pct = (matches / total * 100) if total > 0 else 0
                            # _match_data.append((cond_label, matches, total, pct))
    #
                        # match_cols = st.columns(3)
                        # for ci, (cond_label, matches, total, pct) in enumerate(_match_data):
                            # with match_cols[ci]:
                                # if pct >= 67:
                                    # color = "#2E7D32"
                                # elif pct >= 34:
                                    # color = "#E65100"
                                # else:
                                    # color = "#C62828"
                                # st.markdown(
                                    # f'<div style="text-align:center;padding:12px;background:#fafafa;border-radius:10px;border:1px solid #e0e0e0;">'
                                    # f'<div style="font-size:0.85em;color:#666;">{cond_label}</div>'
                                    # f'<div style="font-size:1.8em;font-weight:700;color:{color};">{pct:.0f}%</div>'
                                    # f'<div style="font-size:0.8em;color:#999;">{matches}/{total} drafts match</div>'
                                    # f'</div>',
                                    # unsafe_allow_html=True
                                # )
    #
    #                     # --- LLM scoring breakdowns (3 collapsed expanders) ---
                        # st.markdown("**LLM Scoring Breakdowns**")
                        # st.caption("Expand to see how each LLM condition evaluated the drafts.")
    #
                        # _cond_info = [
                            # ("rubric", "Rubric-grounded", "Evaluated against your final rubric criteria."),
                            # ("coldstart", "Cold-start-grounded", "Evaluated against dimensions extracted from your stated preferences."),
                            # ("generic", "Generic", "Evaluated against standard writing quality criteria."),
                        # ]
                        # import html as _html_mod_bd
                        # for _cond_key, _cond_label, _cond_desc in _cond_info:
                            # with st.expander(f"👾 {_cond_label}", expanded=False):
                                # st.caption(_cond_desc)
    #                             # Build one table per draft
                                # for _dl in _display_letters:
                                    # _dl_src = _gr_letter_to_src[_dl]
                                    # _dl_src_label = src_label_map.get(_dl_src, _dl_src)
                                    # _eval_block = evals.get(_dl, {}).get(_cond_key, {})
                                    # if not _eval_block or "error" in _eval_block:
                                        # st.info(f"Draft {_dl} ({_dl_src_label}): No evaluation data")
                                        # continue
                                    # _cs_list = _eval_block.get("criteria_scores", [])
                                    # _overall = _eval_block.get("overall_assessment", "")
                                    # _total_met = sum(c.get("dimensions_met", 0) for c in _cs_list)
                                    # _total_dim = sum(c.get("dimensions_total", 0) for c in _cs_list)
                                    # _score = _gr_score_from_llm_eval(_eval_block)
                                    # _score_colors = {5: "#2E7D32", 4: "#1565C0", 3: "#E65100", 2: "#BF360C", 1: "#C62828"}
                                    # _sc = _score_colors.get(_score, "#666")
                                    # _score_html = f'<span style="color:{_sc};font-weight:700;">{_score}/5</span>' if _score else "—"
    #                                 # Draft header
                                    # _tbl = f'<div style="margin-bottom:16px;">'
                                    # _tbl += f'<div style="display:flex;align-items:baseline;gap:8px;margin-bottom:6px;">'
                                    # _tbl += f'<strong>Draft {_dl}</strong> <span style="color:#666;font-size:0.85em;">({_dl_src_label})</span>'
                                    # _tbl += f'<span style="margin-left:auto;font-size:0.9em;">{_total_met}/{_total_dim} dimensions met → {_score_html}</span>'
                                    # _tbl += f'</div>'
    #                                 # Criteria table
                                    # _tbl += '<table style="width:100%;border-collapse:collapse;font-size:0.85em;">'
                                    # _tbl += '<tr style="background:#f8f9fa;"><th style="text-align:left;padding:6px 10px;border-bottom:1px solid #e0e0e0;">Criterion</th><th style="text-align:center;padding:6px 10px;border-bottom:1px solid #e0e0e0;width:80px;">Score</th><th style="text-align:left;padding:6px 10px;border-bottom:1px solid #e0e0e0;">Dimensions</th></tr>'
                                    # for _cr in _cs_list:
                                        # _cr_name = _html_mod_bd.escape(_cr.get("name", ""))
                                        # _cr_met = _cr.get("dimensions_met", 0)
                                        # _cr_total = _cr.get("dimensions_total", 0)
                                        # _cr_icon = "✅" if _cr_met == _cr_total and _cr_total > 0 else ("⚠️" if _cr_met > 0 else "❌")
    #                                     # Build compact dimension chips
                                        # _dim_chips = ""
                                        # for _dim in _cr.get("dimensions_detail", []):
                                            # _d_label = _html_mod_bd.escape(_dim.get("label", ""))
                                            # _d_met = _dim.get("met", False)
                                            # if _d_met:
                                                # _dim_chips += f'<span style="display:inline-block;padding:2px 8px;margin:2px;background:#E8F5E9;color:#2E7D32;border-radius:10px;font-size:0.9em;">✓ {_d_label}</span>'
                                            # else:
                                                # _dim_chips += f'<span style="display:inline-block;padding:2px 8px;margin:2px;background:#FFEBEE;color:#C62828;border-radius:10px;font-size:0.9em;">✗ {_d_label}</span>'
                                        # _tbl += f'<tr><td style="padding:6px 10px;border-bottom:1px solid #f0f0f0;vertical-align:top;"><strong>{_cr_name}</strong></td>'
                                        # _tbl += f'<td style="text-align:center;padding:6px 10px;border-bottom:1px solid #f0f0f0;vertical-align:top;">{_cr_icon} {_cr_met}/{_cr_total}</td>'
                                        # _tbl += f'<td style="padding:6px 10px;border-bottom:1px solid #f0f0f0;">{_dim_chips}</td></tr>'
                                    # _tbl += '</table>'
                                    # if _overall:
                                        # _overall_esc = _html_mod_bd.escape(_overall)
                                        # _tbl += f'<div style="margin-top:6px;padding:6px 10px;background:#f8f9fa;border-radius:6px;font-size:0.85em;color:#555;font-style:italic;">{_overall_esc}</div>'
                                    # _tbl += '</div>'
                                    # st.markdown(_tbl, unsafe_allow_html=True)
    #
                        # --- Step 10: Save to database ---
                        st.divider()
                        st.markdown("**Save all evaluation data** to the database.")
                        if st.button("💾 Save evaluation to database", type="primary", key="grading_tab_save_eval"):
                            project_id = st.session_state.get("current_project_id")
                            if not project_id:
                                st.error("No project selected.")
                            else:
                                supabase = st.session_state.get("supabase")
                                if not supabase:
                                    st.error("Not connected to database.")
                                else:
                                    dim_checks_serializable = None
                                    if st.session_state.get("infer_step6_user_dimension_checks"):
                                        dim_checks_serializable = {}
                                        for letter, d in st.session_state.infer_step6_user_dimension_checks.items():
                                            dim_checks_serializable[letter] = {k if isinstance(k, str) else f"{k[0]}|{k[1]}": v for k, v in d.items()}
                                    export_data = {
                                        "timestamp": datetime.now().isoformat(),
                                        "rubric_versions_used": st.session_state.get("infer_step6_rubric_versions_used"),
                                        "coldstart_text": _gr_coldstart,
                                        "writing_task": st.session_state.get("infer_step6_writing_task"),
                                        "drafts": {k: v[:500] + "..." if len(v) > 500 else v for k, v in (st.session_state.get("infer_step6_drafts") or {}).items()},
                                        "draft_labels": st.session_state.get("infer_step6_draft_labels"),
                                        "blind_ratings": st.session_state.get("infer_step6_blind_ratings"),
                                        "user_ranking": st.session_state.get("infer_step6_user_ranking"),
                                        "user_dimension_checks": dim_checks_serializable,
                                        "llm_evaluations": st.session_state.get("infer_step6_llm_evaluations"),
                                        "survey": st.session_state.get("infer_step6_survey"),
                                        "claim2_metrics": st.session_state.get("infer_step6_claim2_metrics"),
                                        "claim3_metrics": st.session_state.get("infer_step6_claim3_metrics"),
                                    }
                                    if save_project_data(supabase, project_id, "grading_evaluation", export_data):
                                        st.success("Evaluation saved to database.")
                                    else:
                                        st.error("Failed to save.")








# ============ EVALUATE: SURVEY TAB ============
with tab_survey:
    st.header("📋 Evaluate: Survey")
    st.caption("Complete surveys after each task to track your experience")

    # Initialize survey session state
    if "survey_responses" not in st.session_state:
        st.session_state.survey_responses = {
            "task_a": {},
            "task_b": {},
            "final": {}
        }

    # Get active rubric for final survey
    survey_rubric_dict, _, _ = get_active_rubric()

    # Task selection
    survey_task = st.radio(
        "Select which survey to complete:",
        ["Task A (without rubric)", "Task B (with rubric, visible)", "Final Review"],
        horizontal=True,
        key="survey_task_select"
    )

    st.markdown("---")

    # ============ TASK A SURVEY ============
    if survey_task == "Task A (without rubric)":
        st.subheader("Task A: Without Rubric")
        st.markdown("*Complete this after working on a task without any rubric assistance.*")

        task_a = st.session_state.survey_responses["task_a"]

        # Q1
        st.markdown("**Q1: How well did the model understand what you wanted from the start?**")
        q1_options = ["1 - Not at all", "2", "3", "4", "5 - Perfectly"]
        task_a["q1"] = st.radio(
            "Understanding rating",
            q1_options,
            index=q1_options.index(task_a.get("q1", "3")) if task_a.get("q1") in q1_options else 2,
            key="task_a_q1",
            label_visibility="collapsed",
            horizontal=True
        )

        # Q2
        st.markdown("**Q2: How much effort did you spend getting the model to match your style?**")
        q2_options = ["1 - None", "2", "3", "4", "5 - A lot"]
        task_a["q2"] = st.radio(
            "Effort rating",
            q2_options,
            index=q2_options.index(task_a.get("q2", "3")) if task_a.get("q2") in q2_options else 2,
            key="task_a_q2",
            label_visibility="collapsed",
            horizontal=True
        )

        # Q3
        st.markdown("**Q3: Is there anything the model kept getting wrong?**")
        task_a["q3"] = st.text_area(
            "What went wrong",
            value=task_a.get("q3", ""),
            placeholder="Describe any recurring issues or misunderstandings...",
            key="task_a_q3",
            label_visibility="collapsed",
            height=100
        )

        # Q4
        st.markdown("**Q4: Write down all the writing preferences you'd want an AI assistant to know about if you were setting it up from scratch.**")
        st.caption("Tone, structure, style, formality, length — anything you care about when writing in your selected topic. Do NOT look at any rubric or open the sidebar.")
        task_a["q4"] = st.text_area(
            "Writing preferences",
            value=task_a.get("q4", ""),
            placeholder="List your writing preferences here...",
            key="task_a_q4",
            label_visibility="collapsed",
            height=150
        )

        if st.button("Save Task A Survey", type="primary", key="save_task_a"):
            task_a["completed"] = True
            task_a["timestamp"] = datetime.now().isoformat()
            # Save to database
            project_id = st.session_state.get('current_project_id')
            if project_id:
                supabase = st.session_state.get('supabase')
                if supabase:
                    try:
                        save_project_data(supabase, project_id, "survey_responses", st.session_state.survey_responses)
                        st.toast("Task A survey saved!")
                    except Exception as e:
                        st.error(f"Failed to save: {e}")
                else:
                    st.toast("Task A survey saved locally!")
            else:
                st.toast("Task A survey saved locally!")
            st.rerun()

    # ============ TASK B SURVEY ============
    elif survey_task == "Task B (with rubric, visible)":
        st.subheader("Task B: With Rubric (Visible)")
        st.markdown("*Complete this after working on a task where you could see and edit the rubric.*")

        task_b = st.session_state.survey_responses["task_b"]

        # Q1
        st.markdown("**Q1: How well did the model understand what you wanted from the start?**")
        q1_options = ["1 - Not at all", "2", "3", "4", "5 - Perfectly"]
        task_b["q1"] = st.radio(
            "Understanding rating",
            q1_options,
            index=q1_options.index(task_b.get("q1", "3")) if task_b.get("q1") in q1_options else 2,
            key="task_b_q1",
            label_visibility="collapsed",
            horizontal=True
        )

        # Q2
        st.markdown("**Q2: How much effort did you spend getting the model to match your style?**")
        q2_options = ["1 - None", "2", "3", "4", "5 - A lot"]
        task_b["q2"] = st.radio(
            "Effort rating",
            q2_options,
            index=q2_options.index(task_b.get("q2", "3")) if task_b.get("q2") in q2_options else 2,
            key="task_b_q2",
            label_visibility="collapsed",
            horizontal=True
        )

        # Q3
        st.markdown("**Q3: Is there anything the model kept getting wrong?**")
        task_b["q3"] = st.text_area(
            "What went wrong",
            value=task_b.get("q3", ""),
            placeholder="Describe any recurring issues or misunderstandings...",
            key="task_b_q3",
            label_visibility="collapsed",
            height=100
        )

        # Q4
        st.markdown("**Q4: Compared to the previous task, how did this one feel?**")
        q4_options = ["Much better", "Somewhat better", "About the same", "Somewhat worse", "Much worse"]
        task_b["q4"] = st.radio(
            "Comparison",
            q4_options,
            index=q4_options.index(task_b.get("q4", "About the same")) if task_b.get("q4") in q4_options else 2,
            key="task_b_q4",
            label_visibility="collapsed"
        )

        # Q5
        st.markdown("**Q5: Did you look at the rubric? Was it useful?**")
        task_b["q5"] = st.text_area(
            "Rubric usefulness",
            value=task_b.get("q5", ""),
            placeholder="Describe whether and how you used the rubric...",
            key="task_b_q5",
            label_visibility="collapsed",
            height=100
        )

        # Q6
        st.markdown("**Q6: Did the rubric show you anything about the model's behavior you wouldn't have noticed otherwise?**")
        task_b["q6"] = st.text_area(
            "Rubric insights",
            value=task_b.get("q6", ""),
            placeholder="Any insights or surprises from seeing the rubric...",
            key="task_b_q6",
            label_visibility="collapsed",
            height=100
        )

        if st.button("Save Task B Survey", type="primary", key="save_task_b"):
            task_b["completed"] = True
            task_b["timestamp"] = datetime.now().isoformat()
            # Save to database
            project_id = st.session_state.get('current_project_id')
            if project_id:
                supabase = st.session_state.get('supabase')
                if supabase:
                    try:
                        save_project_data(supabase, project_id, "survey_responses", st.session_state.survey_responses)
                        st.toast("Task B survey saved!")
                    except Exception as e:
                        st.error(f"Failed to save: {e}")
                else:
                    st.toast("Task B survey saved locally!")
            else:
                st.toast("Task B survey saved locally!")
            st.rerun()

    # ============ FINAL REVIEW ============
    elif survey_task == "Final Review":
        st.subheader("Final Review: Rubric Accuracy")
        st.markdown("*Review what the system learned about your preferences.*")

        final = st.session_state.survey_responses["final"]

        if not survey_rubric_dict or not survey_rubric_dict.get("rubric"):
            st.warning("No active rubric found. Please create or select a rubric first.")
        else:
            rubric_list = survey_rubric_dict.get("rubric", [])

            st.markdown("**Q1: This is what the system learned about your preferences. Walk me through it — what's right, what's wrong, what surprises you?**")
            st.markdown("*For each criterion, rate its accuracy and provide a brief explanation.*")

            st.markdown("---")

            # Initialize criteria responses if needed
            if "criteria_ratings" not in final:
                final["criteria_ratings"] = {}

            for idx, criterion in enumerate(rubric_list):
                crit_name = criterion.get("name", f"Criterion {idx + 1}")
                crit_desc = criterion.get("description", "")
                crit_dims = criterion.get("dimensions", [])

                with st.expander(f"**{crit_name}**", expanded=True):
                    st.markdown(f"*{crit_desc}*")

                    if crit_dims:
                        st.markdown("**Dimensions:**")
                        for dim in crit_dims:
                            st.markdown(f"- {dim.get('label', '')}")

                    st.markdown("---")

                    # Initialize this criterion's response
                    if crit_name not in final["criteria_ratings"]:
                        final["criteria_ratings"][crit_name] = {"accuracy": "Partially right", "explanation": ""}

                    # Accuracy rating
                    accuracy_options = ["Accurate", "Partially right", "Inaccurate"]
                    final["criteria_ratings"][crit_name]["accuracy"] = st.radio(
                        "How accurate is this criterion?",
                        accuracy_options,
                        index=accuracy_options.index(final["criteria_ratings"][crit_name].get("accuracy", "Partially right")),
                        key=f"final_accuracy_{idx}",
                        horizontal=True
                    )

                    # Brief explanation
                    final["criteria_ratings"][crit_name]["explanation"] = st.text_input(
                        "Brief explanation (what's right/wrong):",
                        value=final["criteria_ratings"][crit_name].get("explanation", ""),
                        key=f"final_explanation_{idx}",
                        placeholder="e.g., 'This is spot on' or 'I actually prefer the opposite'"
                    )

            st.markdown("---")

            # Q2
            st.markdown("**Q2: Is there anything here you wouldn't have thought to mention yourself?**")
            final["q2"] = st.text_area(
                "Unexpected insights",
                value=final.get("q2", ""),
                placeholder="Any preferences the system captured that you didn't realize you had, or wouldn't have articulated...",
                key="final_q2",
                label_visibility="collapsed",
                height=100
            )

            if st.button("Save Final Review", type="primary", key="save_final"):
                final["completed"] = True
                final["timestamp"] = datetime.now().isoformat()
                # Save to database
                project_id = st.session_state.get('current_project_id')
                if project_id:
                    supabase = st.session_state.get('supabase')
                    if supabase:
                        try:
                            save_project_data(supabase, project_id, "survey_responses", st.session_state.survey_responses)
                            st.toast("Final review saved!")
                        except Exception as e:
                            st.error(f"Failed to save: {e}")
                    else:
                        st.toast("Final review saved locally!")
                else:
                    st.toast("Final review saved locally!")
                st.rerun()

    # ============ SURVEY SUMMARY & EXPORT ============
    st.markdown("---")
    st.subheader("Survey Progress")

    col1, col2, col3 = st.columns(3)
    with col1:
        a_done = st.session_state.survey_responses["task_a"].get("completed", False)
        st.markdown(f"**Task A:** {'✅ Complete' if a_done else '⬜ Incomplete'}")
    with col2:
        b_done = st.session_state.survey_responses["task_b"].get("completed", False)
        st.markdown(f"**Task B:** {'✅ Complete' if b_done else '⬜ Incomplete'}")
    with col3:
        f_done = st.session_state.survey_responses["final"].get("completed", False)
        st.markdown(f"**Final:** {'✅ Complete' if f_done else '⬜ Incomplete'}")

    # Save to database option
    if any([a_done, b_done, f_done]):
        project_id = st.session_state.get('current_project_id')
        if project_id:
            supabase = st.session_state.get('supabase')
            if supabase:
                if st.button("Save All Surveys to Database", key="save_all_surveys"):
                    try:
                        save_project_data(supabase, project_id, "survey_responses", st.session_state.survey_responses)
                        st.success("All survey responses saved to database!")
                    except Exception as e:
                        st.error(f"Failed to save: {e}")
