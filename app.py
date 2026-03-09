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
    send_otp, verify_otp,
    logout_user, get_current_user, is_authenticated,
    get_user_projects, create_project as db_create_project, delete_project,
    save_conversation, load_conversations as db_load_conversations, delete_conversation,
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
    GRADING_rubric_judge_3draft_prompt,
    GRADING_generic_judge_prompt,
    GRADING_generate_draft_from_coldstart_prompt,
    RUBRIC_suggest_changes_from_feedback_prompt,
    RUBRIC_apply_suggestion_prompt,
    GRADING_generate_writing_task_prompt,
    GRADING_generate_draft_generic_prompt,
    GRADING_unified_eval_prompt,
    RUBRIC_INFER_ONLY_SYSTEM_PROMPT,
    RUBRIC_infer_only_user_prompt,
    RUBRIC_EXTRACT_DPS_SYSTEM_PROMPT,
    RUBRIC_extract_dps_user_prompt,
    RUBRIC_FINAL_INFER_SYSTEM_PROMPT,
    RUBRIC_final_infer_user_prompt,
    ALIGNMENT_diagnostic_suggest_and_apply_prompt,
    ALIGNMENT_generate_annotated_draft_prompt,
    ALIGNMENT_verify_suggested_rubric_prompt,
    PROBE_identify_uncertainty_prompt,
    PROBE_generate_variant_draft_prompt,
    PROBE_refine_criterion_prompt,
)

from scipy.stats import kendalltau
import random
import warnings
import logging

# Suppress Streamlit deprecation warnings from terminal
warnings.filterwarnings("ignore", message=".*use_container_width.*")
warnings.filterwarnings("ignore", message=".*Please replace.*")
logging.getLogger("streamlit.runtime.caching").setLevel(logging.ERROR)
st_logger = logging.getLogger("streamlit")
st_logger.setLevel(logging.ERROR)

# ── Model configuration ──────────────────────────────────────────────────────
# Change these to switch all API calls at once.
MODEL_PRIMARY = "claude-opus-4-6"    # Main model for chat, rubric inference, grading, etc.
MODEL_LIGHT   = "claude-sonnet-4-6"      # Lighter model for suggestions, small helper calls
PROBE_FALLBACK_INTERVAL = 1          # After N drafts with no probe, nudge harder

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


def split_draft_into_sentences(draft_text: str) -> list:
    """Split draft text into sentences, preserving paragraph structure."""
    paragraphs = draft_text.split('\n\n')
    sentences = []
    for para in paragraphs:
        if not para.strip():
            continue
        # Split on sentence-ending punctuation followed by space + uppercase letter
        para_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', para.strip())
        sentences.extend([s.strip() for s in para_sentences if s.strip()])
    return sentences


def replace_sentences_in_draft(draft: str, replacements: list) -> str:
    """Replace each sentence in draft with its corresponding replacement in-place.

    replacements: list of {"original": str, "replacement": str} dicts.
    Each original sentence is found in the draft and swapped for its replacement,
    preserving all surrounding text.
    """
    result = draft
    for item in replacements:
        original = item.get("original", "")
        replacement = item.get("replacement", "")
        if original and original in result:
            result = result.replace(original, replacement, 1)
    return result


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

def _safe_annotated_draft_html(annotated_text: str, annotated_changes: list, message_id: str = "", clickable: bool = True) -> str:
    """
    Turn revised_draft_annotated into safe HTML: allow <ins>/<del>, replace [1],[2] with
    markers. annotated_changes is used for [N] tooltips and anchor ids.
    If a [N] marker has no nearby ins/del, attempt to inject them from annotated_changes data.

    Args:
        clickable: If True, markers are clickable links that jump to edit details.
                   If False, markers are styled spans with hover tooltips only.
    """
    import html as html_lib
    if not annotated_text:
        return ""
    # --- Fallback injection: fix markers that lack ins/del tags ---
    _patched = annotated_text
    for _fi in range(1, len(annotated_changes) + 1):
        _f_marker = f"[{_fi}]"
        _f_pos = _patched.find(_f_marker)
        if _f_pos == -1:
            continue
        # Check if <ins> or <del> exists within 500 chars before the marker
        _f_window = _patched[max(0, _f_pos - 500):_f_pos]
        if '<ins>' in _f_window or '<del>' in _f_window or '</ins>' in _f_window or '</del>' in _f_window:
            continue  # Already has proper tags
        _f_ac = annotated_changes[_fi - 1]
        _f_orig = (_f_ac.get('original_text', '') or '').strip()
        _f_new = (_f_ac.get('new_text', '') or '').strip()
        if _f_orig and _f_new:
            # Replacement: find new_text near the marker and wrap it
            # Search in a larger window to handle long replacements
            _f_search_start = max(0, _f_pos - len(_f_new) - 100)
            _f_search = _patched[_f_search_start:_f_pos]
            _f_idx = _f_search.find(_f_new)
            if _f_idx != -1:
                _f_abs = _f_search_start + _f_idx
                _patched = _patched[:_f_abs] + f'<del>{_f_orig}</del><ins>{_f_new}</ins>' + _patched[_f_abs + len(_f_new):]
            else:
                # Fallback: just inject del+ins before the marker
                _f_pos2 = _patched.find(_f_marker)
                if _f_pos2 != -1:
                    _patched = _patched[:_f_pos2] + f'<del>{_f_orig}</del><ins>{_f_new}</ins>' + _patched[_f_pos2:]
        elif _f_new and not _f_orig:
            _f_search_start = max(0, _f_pos - len(_f_new) - 100)
            _f_search = _patched[_f_search_start:_f_pos]
            _f_idx = _f_search.find(_f_new)
            if _f_idx != -1:
                _f_abs = _f_search_start + _f_idx
                _patched = _patched[:_f_abs] + f'<ins>{_f_new}</ins>' + _patched[_f_abs + len(_f_new):]
            else:
                _f_pos2 = _patched.find(_f_marker)
                if _f_pos2 != -1:
                    _patched = _patched[:_f_pos2] + f'<ins>{_f_new}</ins>' + _patched[_f_pos2:]
        elif _f_orig and not _f_new:
            # Deletion: inject del tag before the marker
            _f_pos2 = _patched.find(_f_marker)  # Re-find after potential earlier patches
            if _f_pos2 != -1:
                _patched = _patched[:_f_pos2] + f'<del>{_f_orig}</del>' + _patched[_f_pos2:]
    escaped = html_lib.escape(_patched)
    escaped = escaped.replace("&lt;ins&gt;", "<ins>").replace("&lt;/ins&gt;", "</ins>")
    escaped = escaped.replace("&lt;del&gt;", "<del>").replace("&lt;/del&gt;", "</del>")
    safe_msg_id = re.sub(r'[^a-zA-Z0-9_-]', '_', str(message_id))
    for i in range(1, len(annotated_changes) + 1):
        reason = annotated_changes[i - 1].get("reason", "") if i <= len(annotated_changes) else ""
        reason_esc = html_lib.escape(reason)
        marker = f"[{i}]"
        if clickable:
            edit_anchor_id = f"rubric-edit-{safe_msg_id}-{i}"
            replacement = f'<a href="#{edit_anchor_id}" class="rubric-marker" title="{reason_esc}" style="background:#e3f2fd;color:#1565c0;padding:0 4px;border-radius:3px;font-size:0.85em;text-decoration:none;">[{i}]</a>'
        else:
            replacement = f'<span class="rubric-marker" title="{reason_esc}" style="background:#e3f2fd;color:#1565c0;padding:0 4px;border-radius:3px;font-size:0.85em;cursor:default;">[{i}]</span>'
        escaped = escaped.replace(marker, replacement)
    return (
        '<div class="annotated-draft" style="background:#fafafa;border:1px solid #e0e0e0;border-radius:6px;padding:12px;margin:8px 0;">'
        '<style>.annotated-draft ins { background:#c8e6c9; font-weight:bold; text-decoration:none; padding:0 2px; } '
        '.annotated-draft del { background:#ffcdd2; text-decoration:line-through; padding:0 2px; }</style>'
        f'{escaped}</div>'
    )

def render_message_with_draft(content: str, message_id: str, wrap_draft_in_expander: bool = False, editable: bool = True):
    """
    Render a message that may contain <draft> tags.
    Draft sections are rendered as editable text areas (when editable=True) or read-only text (when editable=False).
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
            if editable:
                _draft_label = "📝 **Your Draft** — edit directly, or select sentences below to rephrase with AI"
            else:
                _draft_label = "📝 **Draft Preview**"
            draft_container = st.expander(_draft_label, expanded=not wrap_draft_in_expander) if wrap_draft_in_expander else st.container()
            with draft_container:
                if not wrap_draft_in_expander:
                    st.markdown(_draft_label)

                # Get current value to display
                current_value = st.session_state[draft_key][edit_key]

                if not editable:
                    # --- Read-only mode: show draft as plain text ---
                    st.text_area(
                        label="Draft (read-only)",
                        value=current_value,
                        height=300,
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"readonly_textarea_{edit_key}",
                    )
                    draft_idx += 1
                    continue

                # --- Rephrase selection state ---
                rephrase_selected_key = f"rephrase_selected_{message_id}_{draft_idx}"
                if rephrase_selected_key not in st.session_state:
                    st.session_state[rephrase_selected_key] = set()

                # --- Editable text area ---
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

                # --- Rephrase sentence selector (always visible) ---
                _reph_sentences = split_draft_into_sentences(edited_draft)
                _reph_selected = set()
                _reph_instr = ""
                if _reph_sentences:
                    _sentence_options = [f"[{i+1}] {s}" for i, s in enumerate(_reph_sentences)]
                    _prev_sel = st.session_state[rephrase_selected_key]
                    _default = [_sentence_options[i] for i in _prev_sel if i < len(_sentence_options)]
                    _chosen = st.multiselect(
                        "Select sentences to rephrase",
                        options=_sentence_options,
                        default=_default,
                        key=f"reph_multi_{edit_key}_v{st.session_state[reset_counter_key]}",
                    )
                    _reph_selected = {_sentence_options.index(c) for c in _chosen}
                    st.session_state[rephrase_selected_key] = _reph_selected

                    # Show instruction + regenerate when sentences are selected
                    if _reph_selected:
                        _sel_text = " ".join(_reph_sentences[i] for i in sorted(_reph_selected))
                        st.markdown(f"**Selected ({len(_reph_selected)} sentence{'s' if len(_reph_selected) > 1 else ''}):**")
                        st.info(_sel_text[:500] + ("..." if len(_sel_text) > 500 else ""))

                        _reph_instr = st.text_input(
                            "How should this be rephrased?",
                            placeholder="e.g., make more formal, shorten, add detail, rephrase...",
                            key=f"rephrase_instr_{edit_key}_v{st.session_state[reset_counter_key]}"
                        )

                # Check if draft has been modified from original
                original_draft = st.session_state[original_key].get(orig_key, draft_content)
                has_changes = edited_draft != original_draft

                # Buttons row — Rephrase only enabled when sentences are selected + instruction provided
                _can_rephrase = bool(_reph_selected) and bool(_reph_instr)
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("🔄 Rephrase", key=f"regen_btn_{edit_key}", disabled=not _can_rephrase, type="primary"):
                        _sel_sentence_list = [_reph_sentences[i] for i in sorted(_reph_selected)]
                        _sel_text = " ".join(_sel_sentence_list)
                        active_rubric_dict, _, _ = get_active_rubric()
                        _rub_list = active_rubric_dict.get("rubric", []) if active_rubric_dict else []
                        with st.spinner("Rephrasing selected text..."):
                            _regen_result = regenerate_selected_text(
                                full_draft=edited_draft,
                                selected_sentences=_sel_sentence_list,
                                instruction=_reph_instr,
                                rubric_list=_rub_list
                            )
                        if _regen_result and _regen_result.get("replacements") and not _regen_result.get("error"):
                            _new_draft = replace_sentences_in_draft(
                                edited_draft, _regen_result["replacements"]
                            )
                            # Reset original message's text area back to its original content
                            st.session_state[draft_key][edit_key] = original_draft
                            st.session_state[reset_counter_key] += 1
                            st.session_state[rephrase_selected_key] = set()
                            # Log inline rephrase to conversation
                            import uuid as _uuid_mod
                            _reph_log_id_user = f"inline_rephrase_req_{_uuid_mod.uuid4().hex[:8]}"
                            _reph_log_id = f"inline_rephrase_{_uuid_mod.uuid4().hex[:8]}"
                            st.session_state.messages.append({
                                "role": "user",
                                "content": f"**Rephrase request:** {_reph_instr}\n\n**Selected text:** {_sel_text}",
                                "message_id": _reph_log_id_user,
                                "is_inline_rephrase_request": True,
                            })
                            _reph_explanation = _regen_result.get("explanation", "").strip()
                            _reph_content = ""
                            if _reph_explanation:
                                _reph_content += f"{_reph_explanation}\n\n"
                            _reph_content += f"<draft>\n{_new_draft}\n</draft>"
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": _reph_content,
                                "message_id": _reph_log_id,
                                "is_inline_rephrase": True,
                                "inline_rephrase_data": {
                                    "original_text": _sel_text,
                                    "replacements": _regen_result["replacements"],
                                    "instruction": _reph_instr,
                                    "explanation": _regen_result.get("explanation", ""),
                                },
                            })
                            _reph_pid = st.session_state.get("current_project_id")
                            if _reph_pid:
                                save_project_data(supabase, _reph_pid, "inline_rephrase", {
                                    "timestamp": datetime.now().isoformat(),
                                    "conversation_id": st.session_state.get("selected_conversation", ""),
                                    "message_id": _reph_log_id,
                                    "original_draft": edited_draft,
                                    "selected_text": _sel_text,
                                    "instruction": _reph_instr,
                                    "replacements": _regen_result["replacements"],
                                    "new_draft": _new_draft,
                                    "explanation": _regen_result.get("explanation", ""),
                                })
                            _auto_save_conversation()
                            if _regen_result.get("explanation"):
                                st.toast(_regen_result["explanation"])
                            st.rerun()
                        else:
                            st.error(f"Regeneration failed: {_regen_result.get('error', 'Unknown error')}")
                with col2:
                    if st.button("💾 Save", key=f"save_{edit_key}", disabled=not has_changes):
                        if has_changes:
                            import uuid
                            new_message_id = f"assistant_{uuid.uuid4().hex[:8]}"
                            new_message_content = f"The user has made edits to the draft. Here's the edited draft:\n\n<draft>{edited_draft}</draft>"
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": new_message_content,
                                "display_content": new_message_content,
                                "id": new_message_id
                            })
                            st.session_state[draft_key][edit_key] = original_draft
                            st.session_state[reset_counter_key] += 1
                            _auto_save_conversation()
                            st.success("Draft saved as new message!")
                            st.rerun()
                with col3:
                    if st.button("↩️ Reset", key=f"reset_{edit_key}", disabled=not has_changes):
                        st.session_state[draft_key][edit_key] = original_draft
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
            thinking={"type": "enabled", "budget_tokens": 4000}
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


def _probe_commit_choice(probe_state, chosen_label, user_reason=""):
    """Commit the user's uncertainty probe choice and launch background criterion refinement.

    Args:
        probe_state: dict from st.session_state.probe_pending
        chosen_label: "a", "b", or "skip"
        user_reason: optional free-text reason from user
    """
    import threading

    # Find the assistant message that the probe is attached to
    target_msg_id = probe_state.get("message_id")
    target_message = None
    for msg in reversed(st.session_state.messages):
        if msg.get("message_id") == target_msg_id:
            target_message = msg
            break

    # --- Build probe log message (rendered like DP confirmation) ---
    _probe_crit_name = probe_state.get("criterion_name", "")
    _probe_reason = probe_state.get("uncertainty_reason", "")
    _probe_interp_a = probe_state.get("interpretation_a", "")
    _probe_interp_b = probe_state.get("interpretation_b", "")
    _probe_variant_a = probe_state.get("variant_a", "")
    _probe_variant_b = probe_state.get("variant_b", "")
    _probe_dim_varied = probe_state.get("dimension_varied", "")

    def _build_probe_log(choice_label, choice_reason=""):
        """Build probe log content (summary only, drafts stored separately).

        Returns a dict with structured parts so the display layer can render
        them with better formatting (headers, expanders, etc.).
        The 'summary' key is a short one-liner for st.info().
        The 'full_markdown' key is the full text for backward compat / saving.
        """
        # --- short summary line for the header ---
        if choice_label == "skip":
            _choice_str = "Skipped"
        else:
            _choice_str = "Version A" if choice_label == "a" else "Version B"

        summary = f"Rubric Clarity Check — **\"{_probe_crit_name}\"** → You chose **{_choice_str}**"

        # --- full markdown (stored in message content for DB persistence) ---
        lines = [f"**Rubric Clarity Check: \"{_probe_crit_name}\"**\n"]

        # Split reason into bullet points by sentence boundaries
        if _probe_reason:
            lines.append("**Why probed:**")
            import re as _re_probe
            _reason_sentences = [s.strip() for s in _re_probe.split(r'(?<=[.!?])\s+', _probe_reason) if s.strip()]
            for _rs in _reason_sentences:
                lines.append(f"- {_rs}")
            if _probe_dim_varied:
                lines.append(f"- Dimension varied: {_probe_dim_varied}")
        elif _probe_dim_varied:
            lines.append(f"**Dimension varied:** {_probe_dim_varied}")

        lines.append(f"\n**Interpretation A:** {_probe_interp_a}")
        lines.append(f"\n**Interpretation B:** {_probe_interp_b}")
        lines.append("")
        if choice_label == "skip":
            lines.append("**Your choice:** Skipped")
        else:
            chosen_display = "Version A" if choice_label == "a" else "Version B"
            lines.append(f"**Your choice:** {chosen_display}")
            if choice_reason:
                lines.append(f"**Your reason:** {choice_reason}")
        full_markdown = "\n".join(lines)

        return {"summary": summary, "full_markdown": full_markdown}

    def _make_probe_log_msg(choice_label, choice_reason=""):
        _plm_data = _build_probe_log(choice_label, choice_reason)
        return {
            "role": "assistant",
            "content": _plm_data["full_markdown"],
            "display_content": _plm_data["full_markdown"],
            "probe_log_summary": _plm_data["summary"],
            "is_system_generated": True,
            "is_probe_log": True,
            "probe_log_data": {
                "criterion_name": _probe_crit_name,
                "variant_a": _probe_variant_a,
                "variant_b": _probe_variant_b,
                "user_choice": choice_label,
                "source_message_id": target_msg_id,
                "reason": _probe_reason,
                "dimension_varied": _probe_dim_varied,
                "interpretation_a": _probe_interp_a,
                "interpretation_b": _probe_interp_b,
            },
            "message_id": f"probe_log_{int(time.time() * 1000000)}",
        }

    if chosen_label == "skip":
        # Store minimal probe result on the message
        if target_message:
            target_message["probe_result"] = {
                "criterion_name": _probe_crit_name,
                "user_choice": "skip",
                "user_reason": "",
                "variant_a": _probe_variant_a,
                "variant_b": _probe_variant_b,
                "interpretation_a": _probe_interp_a,
                "interpretation_b": _probe_interp_b,
                "updated_criterion": None,
                "applied": False,
                "applied_version": None,
            }
        # Insert probe log message BEFORE the assistant draft message
        _skip_log_msg = _make_probe_log_msg("skip")
        _skip_target_idx = None
        for _i, _m in enumerate(st.session_state.messages):
            if _m.get("message_id") == target_msg_id:
                _skip_target_idx = _i
                break
        if _skip_target_idx is not None:
            st.session_state.messages.insert(_skip_target_idx, _skip_log_msg)
        else:
            st.session_state.messages.append(_skip_log_msg)
        # Persist skip result
        _save_sb = st.session_state.get("supabase")
        _save_pid = st.session_state.get("current_project_id")
        if _save_sb and _save_pid:
            try:
                save_project_data(_save_sb, _save_pid, "probe_results", {
                    "timestamp": datetime.now().isoformat(),
                    "rubric_version": probe_state.get("rubric_version"),
                    "conversation_id": st.session_state.get("selected_conversation", ""),
                    "criterion_name": _probe_crit_name,
                    "criterion_index": probe_state.get("criterion_index", -1),
                    "interpretation_a": _probe_interp_a,
                    "interpretation_b": _probe_interp_b,
                    "uncertainty_reason": _probe_reason,
                    "user_choice": "skip",
                    "user_reason": "",
                    "updated_criterion": None,
                    "rubric_updated": False,
                })
            except Exception:
                pass
        st.session_state.probe_pending = None
        _auto_save_conversation()
        return

    # User chose a or b
    _orig_slot = probe_state.get("original_slot", "a")
    # Determine which interpretation was chosen based on slot mapping:
    # original draft follows interpretation_a, alternative follows interpretation_b
    if chosen_label == _orig_slot:
        # User chose the original draft → they prefer interpretation_a
        chosen_interpretation = probe_state.get("interpretation_a", "")
    else:
        # User chose the alternative draft → they prefer interpretation_b
        chosen_interpretation = probe_state.get("interpretation_b", "")

    # If user chose the alternative draft, replace the assistant message's draft
    if chosen_label != _orig_slot and target_message:
        _alt_draft = probe_state.get("alternative_draft", "")
        if _alt_draft:
            _orig_content = target_message.get("content", "")
            _orig_display = target_message.get("display_content", _orig_content)
            # Replace the <draft>...</draft> content with the preferred alternative
            _new_content = re.sub(
                r'<draft>.*?</draft>',
                f'<draft>\n{_alt_draft}\n</draft>',
                _orig_content, count=1, flags=re.DOTALL
            )
            _new_display = re.sub(
                r'<draft>.*?</draft>',
                f'<draft>\n{_alt_draft}\n</draft>',
                _orig_display, count=1, flags=re.DOTALL
            )
            target_message["content"] = _new_content
            target_message["display_content"] = _new_display

    # Store probe result on the message (updated_criterion will be filled by background thread)
    probe_result = {
        "criterion_name": _probe_crit_name,
        "user_choice": chosen_label,
        "user_reason": user_reason,
        "variant_a": _probe_variant_a,
        "variant_b": _probe_variant_b,
        "interpretation_a": _probe_interp_a,
        "interpretation_b": _probe_interp_b,
        "original_slot": _orig_slot,
        "chose_original": chosen_label == _orig_slot,
        "updated_criterion": None,  # filled by background thread
        "applied": False,
        "applied_version": None,
    }
    if target_message:
        target_message["probe_result"] = probe_result

    # Insert probe log message BEFORE the assistant draft message (so it appears above)
    _probe_log_msg = _make_probe_log_msg(chosen_label, user_reason)
    _target_idx = None
    for _i, _m in enumerate(st.session_state.messages):
        if _m.get("message_id") == target_msg_id:
            _target_idx = _i
            break
    if _target_idx is not None:
        st.session_state.messages.insert(_target_idx, _probe_log_msg)
    else:
        st.session_state.messages.append(_probe_log_msg)

    # Get criterion JSON for refinement
    rubric_dict, _, _ = get_active_rubric()
    rubric_list = rubric_dict.get("rubric", []) if rubric_dict else []
    criterion_index = probe_state.get("criterion_index", -1)
    criterion_json = ""
    if 0 <= criterion_index < len(rubric_list):
        criterion_json = json.dumps(rubric_list[criterion_index], indent=2)
    else:
        # Fallback: find by name
        crit_name = _probe_crit_name.lower().strip()
        for c in rubric_list:
            if c.get("name", "").lower().strip() == crit_name:
                criterion_json = json.dumps(c, indent=2)
                break

    # Launch background thread for criterion refinement
    _refine_args = {
        "criterion_json": criterion_json,
        "chosen_interpretation": chosen_interpretation,
        "user_reason": user_reason,
        "message_data_ref": target_message,
        "probe_log_ref": _probe_log_msg,
        "results_list_ref": st.session_state.get("probe_results", []),
        "probe_state": probe_state,
        "supabase": st.session_state.get("supabase"),
        "project_id": st.session_state.get("current_project_id"),
        "conversation_id": st.session_state.get("selected_conversation", ""),
        "rubric_version": rubric_dict.get("version") if rubric_dict else None,
    }

    _refine_event = threading.Event()

    def _refine_background(args, done_event):
        try:
            _run_probe_refine_bg(args)
        except Exception as e:
            # print(f"[PROBE] Refine background failed: {e}")
            pass
        finally:
            done_event.set()

    threading.Thread(target=_refine_background, args=(_refine_args, _refine_event), daemon=True).start()
    st.session_state._probe_refine_done_event = _refine_event

    # Clear pending state
    st.session_state.probe_pending = None
    _auto_save_conversation()


def _run_probe_refine_bg(args):
    """Background-thread-safe: refine a single rubric criterion based on user's probe choice.
    Does NOT access st.session_state. All data passed via args dict."""
    criterion_json = args.get("criterion_json", "")
    chosen_interpretation = args.get("chosen_interpretation", "")
    user_reason = args.get("user_reason", "")
    probe_state = args.get("probe_state", {})

    if not criterion_json or not chosen_interpretation:
        return

    updated_criterion = None
    try:
        prompt = PROBE_refine_criterion_prompt(criterion_json, chosen_interpretation, user_reason)
        resp = _api_call_with_retry(
            model=MODEL_PRIMARY, max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        js_match = re.search(r'\{[\s\S]*\}', text)
        if js_match:
            parsed = json.loads(js_match.group())
            updated_criterion = parsed.get("updated_criterion")
    except Exception as e:
        # print(f"[PROBE] Criterion refinement failed: {e}")

        pass
    # Write updated criterion into the message's probe_result (thread-safe dict update)
    message_data_ref = args.get("message_data_ref")
    if message_data_ref and "probe_result" in message_data_ref:
        message_data_ref["probe_result"]["updated_criterion"] = updated_criterion

    # Also write suggestion into the probe log message for conversation persistence
    probe_log_ref = args.get("probe_log_ref")
    if probe_log_ref and "probe_log_data" in probe_log_ref:
        probe_log_ref["probe_log_data"]["suggested_update"] = updated_criterion
        if updated_criterion:
            probe_log_ref["probe_log_data"]["suggestion_summary"] = updated_criterion.get("description", "")

    # Build and persist result
    result = {
        "timestamp": datetime.now().isoformat(),
        "rubric_version": args.get("rubric_version"),
        "conversation_id": args.get("conversation_id", ""),
        "criterion_name": probe_state.get("criterion_name", ""),
        "criterion_index": probe_state.get("criterion_index", -1),
        "interpretation_a": probe_state.get("interpretation_a", ""),
        "interpretation_b": probe_state.get("interpretation_b", ""),
        "uncertainty_reason": probe_state.get("uncertainty_reason", ""),
        "user_choice": probe_state.get("user_choice", ""),
        "user_reason": user_reason,
        "updated_criterion": updated_criterion,
        "rubric_updated": False,  # will be set to True when user clicks Apply
    }

    results_list = args.get("results_list_ref")
    if results_list is not None:
        results_list.append(result)

    # Persist to database
    _save_sb = args.get("supabase")
    _save_pid = args.get("project_id")
    if _save_sb and _save_pid:
        try:
            save_project_data(_save_sb, _save_pid, "probe_results", result)
        except Exception:
            pass


def _run_grade_retest_bg(args):
    """Background-thread-safe: re-run rubric and generic judges for test-retest reliability.
    Does NOT access st.session_state. All data passed via args dict."""
    from scipy.stats import kendalltau as _kt

    draft_a = args.get("draft_good", "")
    draft_b = args.get("draft_degraded", "")
    rubric_json = args.get("rubric_criteria_json", "")
    conv_context = args.get("conv_context", "")
    original_rubric = args.get("original_rubric_result")
    original_generic = args.get("original_generic_result")

    if not draft_a or not draft_b:
        return

    retest_rubric = None
    retest_generic = None

    # Re-run rubric judge
    try:
        prompt = GRADING_rubric_judge_prompt(draft_a, draft_b, rubric_json, conv_context)
        resp = _api_call_with_retry(
            model=MODEL_PRIMARY, max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        js = re.search(r'\{[\s\S]*\}', text)
        if js:
            retest_rubric = json.loads(js.group())
    except Exception as e:
        # print(f"[RETEST] Rubric judge retest failed: {e}")

        pass
    # Re-run generic judge
    try:
        prompt = GRADING_generic_judge_prompt(draft_a, draft_b)
        resp = _api_call_with_retry(
            model=MODEL_PRIMARY, max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        js = re.search(r'\{[\s\S]*\}', text)
        if js:
            retest_generic = json.loads(js.group())
    except Exception as e:
        # print(f"[RETEST] Generic judge retest failed: {e}")

        pass
    # Compute test-retest metrics
    retest_data = {
        "timestamp": args.get("grade_eval_timestamp", ""),
        "retest_rubric_result": retest_rubric,
        "retest_generic_result": retest_generic,
        "original_rubric_result": original_rubric,
        "original_generic_result": original_generic,
        "metrics": {}
    }

    # --- Rubric judge test-retest ---
    if retest_rubric and original_rubric:
        orig_scores = {}
        for c in original_rubric.get("per_criterion", []):
            orig_scores[c.get("criterion_name", "")] = (
                c.get("draft_a_score", 3), c.get("draft_b_score", 3)
            )
        retest_scores = {}
        for c in retest_rubric.get("per_criterion", []):
            retest_scores[c.get("criterion_name", "")] = (
                c.get("draft_a_score", 3), c.get("draft_b_score", 3)
            )
        rubric_run1, rubric_run2 = [], []
        for crit in orig_scores:
            if crit in retest_scores:
                rubric_run1.extend(orig_scores[crit])
                rubric_run2.extend(retest_scores[crit])
        retest_data["metrics"]["rubric_run1_scores"] = rubric_run1
        retest_data["metrics"]["rubric_run2_scores"] = rubric_run2
        if len(rubric_run1) >= 4:
            tau, p = _kt(rubric_run1, rubric_run2)
            retest_data["metrics"]["rubric_tau"] = tau
            retest_data["metrics"]["rubric_tau_p"] = p
        # Keep legacy key for backward compat with Panel 4
        retest_data["metrics"]["retest_tau"] = retest_data["metrics"].get("rubric_tau")

    # --- Generic judge test-retest ---
    if retest_generic and original_generic:
        orig_g_scores = {}
        for c in original_generic.get("per_criterion", []):
            orig_g_scores[c.get("criterion_name", "")] = (
                c.get("draft_a_score", 3), c.get("draft_b_score", 3)
            )
        retest_g_scores = {}
        for c in retest_generic.get("per_criterion", []):
            retest_g_scores[c.get("criterion_name", "")] = (
                c.get("draft_a_score", 3), c.get("draft_b_score", 3)
            )
        generic_run1, generic_run2 = [], []
        for crit in orig_g_scores:
            if crit in retest_g_scores:
                generic_run1.extend(orig_g_scores[crit])
                generic_run2.extend(retest_g_scores[crit])
        retest_data["metrics"]["generic_run1_scores"] = generic_run1
        retest_data["metrics"]["generic_run2_scores"] = generic_run2
        if len(generic_run1) >= 4:
            tau, p = _kt(generic_run1, generic_run2)
            retest_data["metrics"]["generic_tau"] = tau
            retest_data["metrics"]["generic_tau_p"] = p

    # Save to database
    sb = args.get("supabase")
    pid = args.get("project_id")
    if sb and pid:
        try:
            _retest_data_type = args.get("data_type", "grade_retest")
            save_project_data(sb, pid, _retest_data_type, retest_data)
            # print(f"[RETEST] Saved {_retest_data_type} data. Tau={retest_data['metrics'].get('retest_tau', 'N/A')}")
        except Exception as e:
            # print(f"[RETEST] Failed to save: {e}")

            pass
    # Also append to session state list ref if provided
    results_list = args.get("results_list_ref")
    if results_list is not None:
        results_list.append(retest_data)


def _process_alignment_diagnostic(rcp, user_ranking, user_reason="", status_callback=None, pipeline_messages=None):
    """Run per-criterion diagnostic comparing drafts (2 or 3 drafts).

    Returns a result dict with per-criterion classifications and suggested rubric changes.

    Args:
        rcp: ranking checkpoint pending state dict with "drafts", "writing_task", etc.
        user_ranking: list of source keys in ranked order, e.g. ["rubric", "preference", "generic"]
                      or ["rubric", "generic"] for 2-draft mode
        user_reason: optional free-text reason from user
        status_callback: optional callable(str) to update UI progress status
        pipeline_messages: optional list of prior conversation messages (from draft generation)
                          to continue the conversation context
    """
    def _update_status(msg):
        if status_callback:
            status_callback(msg)
    drafts = rcp["drafts"]
    rubric_draft = drafts.get("rubric", "")
    generic_draft = drafts.get("generic", "")
    preference_draft = drafts.get("preference", "")
    is_3draft = bool(preference_draft)

    rubric_dict, _, _ = get_active_rubric()
    rubric_json = json.dumps(
        _rubric_to_json_serializable(rubric_dict), indent=2
    ) if rubric_dict else ""
    rubric_list = rubric_dict.get("rubric", []) if rubric_dict else []

    # Build conversation context for the rubric judge
    conv_text = _build_conversation_text(st.session_state.get("messages", []))
    conv_context = conv_text[-3000:] if len(conv_text) > 3000 else conv_text

    # --- Shared conversation context for the diagnostic pipeline ---
    # Continues from draft generation messages if provided, so the model has
    # full context of the writing task, drafts, scoring, and suggestions.
    _pipeline_messages = list(pipeline_messages) if pipeline_messages else []

    # --- Call rubric judge (per-criterion scores) ---
    _update_status("Scoring each draft against your rubric criteria...")
    rubric_judge_result = None
    try:
        if is_3draft:
            prompt_rubric = GRADING_rubric_judge_3draft_prompt(
                rubric_draft, generic_draft, preference_draft, rubric_json, conv_context
            )
        else:
            prompt_rubric = GRADING_rubric_judge_prompt(rubric_draft, generic_draft, rubric_json, conv_context)
        _pipeline_messages.append({"role": "user", "content": prompt_rubric})
        resp = _api_call_with_retry(
            model=MODEL_PRIMARY, max_tokens=2000,
            messages=_pipeline_messages
        )
        _judge_text = "".join(b.text for b in resp.content if b.type == "text")
        _pipeline_messages.append({"role": "assistant", "content": _judge_text})
        js_match = re.search(r'\{[\s\S]*\}', _judge_text)
        if js_match:
            rubric_judge_result = json.loads(js_match.group())
    except Exception as e:
        pass

    # --- Call generic judge (per-dimension scores, for retest reliability) ---
    _update_status("Running general quality comparison...")
    generic_judge_result = None
    try:
        prompt_generic = GRADING_generic_judge_prompt(rubric_draft, generic_draft)
        resp = _api_call_with_retry(
            model=MODEL_PRIMARY, max_tokens=2000,
            messages=[{"role": "user", "content": prompt_generic}]
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        js_match = re.search(r'\{[\s\S]*\}', text)
        if js_match:
            generic_judge_result = json.loads(js_match.group())
    except Exception as e:
        pass

    # --- Classify each rubric criterion ---
    # In judge prompt: Draft A = rubric, Draft B = generic, Draft C = preference (if 3-draft)
    criteria_analysis = []
    if rubric_judge_result and rubric_judge_result.get("per_criterion"):
        for crit in rubric_judge_result["per_criterion"]:
            rubric_score = crit.get("draft_a_score", 3)
            generic_score = crit.get("draft_b_score", 3)
            preference_score = crit.get("draft_c_score", None) if is_3draft else None
            gap = rubric_score - generic_score

            if is_3draft and preference_score is not None:
                # 3-draft classification
                if rubric_score >= generic_score and rubric_score >= preference_score and gap > 0:
                    classification = "DIFFERENTIATING"
                elif preference_score > rubric_score and preference_score >= generic_score:
                    classification = "PREFERENCE_GAP"
                elif generic_score > rubric_score and generic_score >= (preference_score or 0):
                    classification = "UNDERPERFORMING"
                elif rubric_score == generic_score == preference_score:
                    classification = "REDUNDANT"
                else:
                    classification = "REDUNDANT"
            else:
                # 2-draft classification (same as before)
                if gap > 0:
                    classification = "DIFFERENTIATING"
                elif gap < 0:
                    classification = "UNDERPERFORMING"
                else:
                    classification = "REDUNDANT"

            ca_entry = {
                "name": crit.get("criterion_name", "Unknown"),
                "rubric_score": rubric_score,
                "generic_score": generic_score,
                "gap": gap,
                "classification": classification,
                "reasoning": crit.get("reasoning", ""),
            }
            if preference_score is not None:
                ca_entry["preference_score"] = preference_score
            criteria_analysis.append(ca_entry)

    # Find matching priority from rubric_list
    rubric_priority_map = {}
    for c in rubric_list:
        rubric_priority_map[c.get("name", "").lower().strip()] = c.get("priority", 99)
    for ca in criteria_analysis:
        ca["priority"] = rubric_priority_map.get(ca["name"].lower().strip(), 99)

    # Sort by priority
    criteria_analysis.sort(key=lambda x: x["priority"])

    # --- Generate rubric improvements + reasons in a single call ---
    suggestion_text = ""
    suggested_rubric = None
    # Build ranking description
    _source_names = {"rubric": "rubric-guided", "generic": "generic (no rubric)", "preference": "preference-based (from original stated preferences)"}
    ranking_desc = "The user ranked the drafts: " + " > ".join(
        f"{i+1}. {_source_names.get(s, s)}" for i, s in enumerate(user_ranking)
    )
    _update_status("Identifying rubric improvements based on your ranking...")
    suggestion_reasons = {}  # criterion_name -> reason string
    try:
        rubric_judge_json = json.dumps(rubric_judge_result, indent=2) if rubric_judge_result else "{}"

        suggest_apply_prompt = ALIGNMENT_diagnostic_suggest_and_apply_prompt(
            rubric_json, rubric_judge_json, ranking_desc, user_reason
        )
        _pipeline_messages.append({"role": "user", "content": suggest_apply_prompt})
        resp = _api_call_with_retry(
            model=MODEL_LIGHT, max_tokens=4000,
            messages=_pipeline_messages
        )
        _sa_raw = "".join(b.text for b in resp.content if b.type == "text").strip()
        _pipeline_messages.append({"role": "assistant", "content": _sa_raw})

        # Parse the combined JSON response containing both "reasons" and "rubric"
        _sa_json_match = re.search(r'\{[\s\S]*\}', _sa_raw)
        if _sa_json_match:
            _sa_parsed = json.loads(_sa_json_match.group())

            # Extract per-criterion reasons
            _all_reasons = _sa_parsed.get("reasons", {})
            # Fallback: try old "criteria" key
            if not _all_reasons:
                _all_reasons = _sa_parsed.get("criteria", {})
            suggestion_reasons = {k: v for k, v in _all_reasons.items() if v}

            # Extract the modified rubric array
            _rubric_arr = _sa_parsed.get("rubric", [])
            if isinstance(_rubric_arr, list) and len(_rubric_arr) > 0:
                suggested_rubric = _rubric_arr
            else:
                # Fallback: try to find a JSON array in the raw text
                _arr_match = re.search(r'\[[\s\S]*\]', _sa_raw)
                if _arr_match:
                    suggested_rubric = json.loads(_arr_match.group())

            # --- Post-process: fix mismatched reasons ---
            # If the model said "working well / kept as-is" but actually modified the criterion,
            # ask the model to explain why it made the change (it has full pipeline context).
            if suggested_rubric and rubric_list:
                _keep_phrases = ["keep unchanged", "no change", "working well", "kept as-is",
                                 "keep as-is", "no changes needed", "performing well",
                                 "leave as-is", "leave unchanged", "kept unchanged"]
                _orig_map = {c.get("name", "").lower().strip(): c for c in rubric_list}
                _new_map = {c.get("name", "").lower().strip(): c for c in suggested_rubric}
                _mismatched_criteria = []
                for _cr_name, _cr_reason in list(suggestion_reasons.items()):
                    _cr_key = _cr_name.lower().strip()
                    _is_keep = any(p in _cr_reason.lower() for p in _keep_phrases)
                    if _is_keep and _cr_key in _orig_map and _cr_key in _new_map:
                        _orig = _orig_map[_cr_key]
                        _new = _new_map[_cr_key]
                        _actually_changed = (
                            _orig.get("description", "") != _new.get("description", "")
                            or _orig.get("priority") != _new.get("priority")
                            or sorted([d.get("label", "") for d in _orig.get("dimensions", [])]) != sorted([d.get("label", "") for d in _new.get("dimensions", [])])
                        )
                        if _actually_changed:
                            _mismatched_criteria.append(_cr_name)

                # Ask the model for real reasons for mismatched criteria
                if _mismatched_criteria:
                    try:
                        _fix_names = ", ".join(f'"{c}"' for c in _mismatched_criteria)
                        _fix_prompt = (
                            f"You said the following criteria were 'working well' or 'kept as-is', "
                            f"but you actually modified them in the rubric you returned: {_fix_names}.\n\n"
                            f"For each one, explain in 1-2 sentences WHY you changed it. "
                            f"Do not describe what changed — the user can see the diff. Explain the reasoning.\n\n"
                            f"Return ONLY a JSON object mapping criterion name to reason, e.g.:\n"
                            f'{{"Criterion A": "reason", "Criterion B": "reason"}}'
                        )
                        _pipeline_messages.append({"role": "user", "content": _fix_prompt})
                        _fix_resp = _api_call_with_retry(
                            model=MODEL_LIGHT, max_tokens=1000,
                            messages=_pipeline_messages
                        )
                        _fix_raw = "".join(b.text for b in _fix_resp.content if b.type == "text").strip()
                        _pipeline_messages.append({"role": "assistant", "content": _fix_raw})
                        _fix_match = re.search(r'\{[\s\S]*\}', _fix_raw)
                        if _fix_match:
                            _fix_reasons = json.loads(_fix_match.group())
                            for _fc_name, _fc_reason in _fix_reasons.items():
                                if _fc_reason and isinstance(_fc_reason, str):
                                    suggestion_reasons[_fc_name] = _fc_reason
                    except Exception:
                        # If follow-up fails, use a generic fallback
                        for _mc in _mismatched_criteria:
                            suggestion_reasons[_mc] = "Adjusted based on alignment diagnostic scores."

            # --- Check if suggested_rubric actually differs from the original ---
            # The model may return a rubric array with cosmetic rephrasing even when
            # all reasons say "keep unchanged". Detect this and discard the suggestion.
            if suggested_rubric and rubric_list:
                _orig_map_chk = {c.get("name", "").lower().strip(): c for c in rubric_list}
                _new_map_chk = {c.get("name", "").lower().strip(): c for c in suggested_rubric}
                _has_real_change = False
                # Check for added criteria
                for _nk in _new_map_chk:
                    if _nk not in _orig_map_chk:
                        _has_real_change = True
                        break
                # Check for removed criteria
                if not _has_real_change:
                    for _ok in _orig_map_chk:
                        if _ok not in _new_map_chk:
                            _has_real_change = True
                            break
                # Check for modified criteria
                if not _has_real_change:
                    for _ck in _orig_map_chk:
                        if _ck in _new_map_chk and _criterion_changed(_orig_map_chk[_ck], _new_map_chk[_ck]):
                            _has_real_change = True
                            break
                if not _has_real_change:
                    # No real changes — discard the suggested rubric
                    suggested_rubric = None

            # Filter out "keep unchanged" reasons — only show reasons for actual changes
            if suggestion_reasons:
                _keep_filter_phrases = ["keep unchanged", "no change", "working well", "kept as-is",
                                        "keep as-is", "no changes needed", "performing well",
                                        "leave as-is", "leave unchanged", "kept unchanged"]
                suggestion_reasons = {
                    k: v for k, v in suggestion_reasons.items()
                    if not any(p in v.lower() for p in _keep_filter_phrases)
                }

            # Build suggestion_text summary from reasons (for display / verification)
            _text_parts = []
            for _cr_name, _cr_reason in suggestion_reasons.items():
                _text_parts.append(f"- **{_cr_name}**: {_cr_reason}")
            if _text_parts:
                suggestion_text = "\n".join(_text_parts)
        else:
            suggestion_text = _sa_raw  # fallback

    except Exception as e:
        pass

    # --- Generate an annotated preview draft using the suggested rubric ---
    suggested_draft = ""
    suggested_annotated_changes = []
    _update_status("Generating a new draft with the improved rubric...")
    if suggested_rubric and rcp.get("writing_task"):
        try:
            _sd_rubric_json = json.dumps(suggested_rubric, indent=2)
            _sd_prompt = ALIGNMENT_generate_annotated_draft_prompt(
                writing_task=rcp["writing_task"],
                suggested_rubric_json=_sd_rubric_json,
                original_rubric_draft=rubric_draft,
                suggestion_reasons=suggestion_reasons,
            )
            _pipeline_messages.append({"role": "user", "content": _sd_prompt})
            _sd_resp = _api_call_with_retry(
                model=MODEL_LIGHT, max_tokens=2000,
                messages=_pipeline_messages
            )
            _sd_raw = "".join(b.text for b in _sd_resp.content if b.type == "text").strip()
            _pipeline_messages.append({"role": "assistant", "content": _sd_raw})

            # Parse JSON response
            _sd_json_match = re.search(r'\{[\s\S]*\}', _sd_raw)
            if _sd_json_match:
                _sd_parsed = json.loads(_sd_json_match.group())
                suggested_draft = _sd_parsed.get("revised_draft", "")
                suggested_annotated_changes = _sd_parsed.get("annotated_changes", [])
            else:
                # Fallback: treat entire response as plain draft text
                suggested_draft = _sd_raw

            # DEBUG: print both drafts and annotated_changes for verification
            # print("\n" + "="*80)
            # print("[DEBUG] ALIGNMENT DRAFT GENERATION — PRIMARY PATH")
            # print("="*80)
            # print(f"\n--- ORIGINAL RUBRIC DRAFT ---\n{rubric_draft}")
            # print(f"\n--- SUGGESTED DRAFT (revised_draft) ---\n{suggested_draft}")
            # print(f"\n--- ANNOTATED CHANGES ({len(suggested_annotated_changes)} entries) ---")
            for _dbg_i, _dbg_ac in enumerate(suggested_annotated_changes, 1):
                # print(f"  [{_dbg_i}] original_text: {_dbg_ac.get('original_text', '')!r}")
                # print(f"       new_text:      {_dbg_ac.get('new_text', '')!r}")
                pass
                # print(f"       reason:        {_dbg_ac.get('reason', '')}")
            # print("="*80 + "\n")
        except Exception as e:
            pass

    # --- Verify suggested rubric against user preferences ---
    _update_status("Verifying the new rubric against your preferences...")
    verification_result = None
    _coldstart_prefs = st.session_state.get("infer_coldstart_text", "").strip()
    if suggested_rubric and suggested_draft and _coldstart_prefs:
        try:
            _verify_prompt = ALIGNMENT_verify_suggested_rubric_prompt(
                user_preferences=_coldstart_prefs,
                writing_task=rcp.get("writing_task", ""),
                rubric_draft=rubric_draft,
                generic_draft=generic_draft,
                preference_draft=preference_draft,
                user_ranking_description=ranking_desc,
                suggested_rubric_json=json.dumps(suggested_rubric, indent=2),
                suggested_draft=suggested_draft,
                original_suggestion_text=suggestion_text,
            )
            _pipeline_messages.append({"role": "user", "content": _verify_prompt})
            _verify_resp = _api_call_with_retry(
                model=MODEL_LIGHT, max_tokens=2000,
                messages=_pipeline_messages
            )
            _verify_text = "".join(b.text for b in _verify_resp.content if b.type == "text")
            _pipeline_messages.append({"role": "assistant", "content": _verify_text})
            _verify_match = re.search(r'\{[\s\S]*\}', _verify_text)
            if _verify_match:
                verification_result = json.loads(_verify_match.group())

                # If refinements needed, apply them and regenerate draft
                if verification_result.get("verdict") == "needs_refinement" and verification_result.get("refinements"):
                    _update_status("Refining the rubric based on verification...")
                    refinements = verification_result["refinements"]
                    # Convert refinements to bullet-point text for the apply prompt
                    _refine_bullets = []
                    for ref in refinements:
                        action = ref.get("action", "reword")
                        cname = ref.get("criterion_name", "")
                        new_text = ref.get("suggested_text", "")
                        reason = ref.get("reason", "")
                        if action == "add":
                            _refine_bullets.append(f"- Add a new criterion '{cname}': {new_text}. ({reason})")
                        elif action == "remove":
                            _refine_bullets.append(f"- Remove the criterion '{cname}'. ({reason})")
                        elif action == "adjust_weight":
                            _refine_bullets.append(f"- Adjust the priority/weight of '{cname}': {new_text}. ({reason})")
                        else:  # reword
                            _refine_bullets.append(f"- Reword '{cname}' to: {new_text}. ({reason})")
                    _refine_text = "\n".join(_refine_bullets)

                    # Apply refinements to the suggested rubric
                    try:
                        _sr_json = json.dumps(suggested_rubric, indent=2)
                        _apply_refine_prompt = RUBRIC_apply_suggestion_prompt(_sr_json, _sr_json, _refine_text)
                        _pipeline_messages.append({"role": "user", "content": _apply_refine_prompt})
                        _apply_refine_resp = _api_call_with_retry(
                            model=MODEL_LIGHT, max_tokens=4000,
                            messages=_pipeline_messages
                        )
                        _apply_refine_text = "".join(b.text for b in _apply_refine_resp.content if b.type == "text").strip()
                        _pipeline_messages.append({"role": "assistant", "content": _apply_refine_text})
                        _refine_js_match = re.search(r'\[[\s\S]*\]', _apply_refine_text)
                        if _refine_js_match:
                            refined_rubric = json.loads(_refine_js_match.group())
                            suggested_rubric = refined_rubric
                            suggestion_text = suggestion_text + "\n\n**Verification refinements:**\n" + _refine_text

                            # Regenerate annotated draft with refined rubric
                            _update_status("Regenerating draft with the refined rubric...")
                            _rd_rubric_json = json.dumps(refined_rubric, indent=2)
                            _rd_prompt = ALIGNMENT_generate_annotated_draft_prompt(
                                writing_task=rcp["writing_task"],
                                suggested_rubric_json=_rd_rubric_json,
                                original_rubric_draft=rubric_draft,
                                suggestion_reasons=suggestion_reasons,
                            )
                            _pipeline_messages.append({"role": "user", "content": _rd_prompt})
                            _rd_resp = _api_call_with_retry(
                                model=MODEL_LIGHT, max_tokens=2000,
                                messages=_pipeline_messages
                            )
                            _rd_raw = "".join(b.text for b in _rd_resp.content if b.type == "text").strip()
                            _pipeline_messages.append({"role": "assistant", "content": _rd_raw})

                            _rd_json_match = re.search(r'\{[\s\S]*\}', _rd_raw)
                            if _rd_json_match:
                                _rd_parsed = json.loads(_rd_json_match.group())
                                suggested_draft = _rd_parsed.get("revised_draft", "")
                                suggested_annotated_changes = _rd_parsed.get("annotated_changes", [])
                            else:
                                suggested_draft = _rd_raw

                            # DEBUG: print both drafts and annotated_changes for verification
                            # print("\n" + "="*80)
                            # print("[DEBUG] ALIGNMENT DRAFT GENERATION — REFINEMENT PATH")
                            # print("="*80)
                            # print(f"\n--- ORIGINAL RUBRIC DRAFT ---\n{rubric_draft}")
                            # print(f"\n--- SUGGESTED DRAFT (revised_draft) ---\n{suggested_draft}")
                            # print(f"\n--- ANNOTATED CHANGES ({len(suggested_annotated_changes)} entries) ---")
                            for _dbg_i, _dbg_ac in enumerate(suggested_annotated_changes, 1):
                                # print(f"  [{_dbg_i}] original_text: {_dbg_ac.get('original_text', '')!r}")
                                # print(f"       new_text:      {_dbg_ac.get('new_text', '')!r}")
                                # print(f"       reason:        {_dbg_ac.get('reason', '')}")
                                pass
                            # print("="*80 + "\n")
                    except Exception as e:
                        pass
        except Exception as e:
            pass

    # --- Build result ---
    # Convert ranking to legacy user_preference for backward compatibility
    _legacy_pref = user_ranking[0] if user_ranking else "tie"
    _update_status("Finalizing diagnostic results...")
    result = {
        "timestamp": datetime.now().isoformat(),
        "writing_task": rcp.get("writing_task", ""),
        "drafts": drafts,
        "shuffle_order": rcp.get("shuffle_order", []),
        "user_preference": _legacy_pref,
        "user_ranking": user_ranking,
        "user_reason": user_reason,
        "rubric_version": rubric_dict.get("version") if rubric_dict else None,
        "criteria_analysis": criteria_analysis,
        "rubric_judge_result": rubric_judge_result,
        "generic_judge_result": generic_judge_result,
        "suggestion_text": suggestion_text,
        "suggestion_reasons": suggestion_reasons,
        "suggested_rubric": suggested_rubric,
        "suggested_draft": suggested_draft,
        "suggested_annotated_changes": suggested_annotated_changes,
        "is_3draft": is_3draft,
        "verification_result": verification_result,
        "pipeline_messages": _pipeline_messages,
    }

    # Store the pipeline conversation so the main chat can build on it
    st.session_state.alignment_pipeline_messages = _pipeline_messages

    st.session_state.ranking_checkpoint_results.append(result)
    st.session_state.ranking_checkpoint_pending = None

    # Persist to database
    _save_sb = st.session_state.get('supabase')
    _save_pid = st.session_state.get('current_project_id')
    if _save_sb and _save_pid:
        try:
            save_project_data(_save_sb, _save_pid, "alignment_diagnostic", result)
        except Exception:
            pass

    # Launch background retest for reliability measurement
    if rubric_judge_result and generic_judge_result:
        import threading as _diag_rt_threading
        _diag_rt_args = {
            "draft_good": rubric_draft,
            "draft_degraded": generic_draft,
            "rubric_criteria_json": rubric_json,
            "conv_context": conv_context,
            "original_rubric_result": rubric_judge_result,
            "original_generic_result": generic_judge_result,
            "supabase": _save_sb,
            "project_id": _save_pid,
            "grade_eval_timestamp": result["timestamp"],
            "data_type": "diagnostic_retest",
            "results_list_ref": st.session_state.get("diagnostic_retest_history", []),
        }
        _diag_rt_threading.Thread(
            target=_run_grade_retest_bg,
            args=(_diag_rt_args,),
            daemon=True
        ).start()

    return result


def _annotated_diff_html(old_text: str, new_text: str, annotated_changes: list, message_id: str = "") -> str:
    """
    Generate diff HTML between old and new text, with [N] markers
    linked to annotated_changes reasons via hover tooltips.

    Uses a two-pass approach:
    1. Line-level diff to identify changed regions (prevents false "equal"
       matches when similar text appears in different positions).
    2. Word-level diff within each changed line pair for fine-grained display.
    """
    import difflib
    import html as html_lib

    if not old_text and not new_text:
        return '<span style="color:#9e9e9e;font-style:italic;">Not specified</span>'
    if not old_text:
        return f'<ins>{html_lib.escape(new_text)}</ins>'
    if not new_text:
        return f'<del>{html_lib.escape(old_text)}</del>'
    if old_text.strip() == new_text.strip():
        return html_lib.escape(new_text)

    # Split into lines for first pass
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    line_matcher = difflib.SequenceMatcher(None, old_lines, new_lines)

    # Build list of annotated_changes that haven't been matched yet
    _ac_available = list(range(len(annotated_changes)))

    def _match_change(old_fragment: str, new_fragment: str):
        """Try to match a diff region to an annotated_changes entry. Returns index or None."""
        old_frag_lower = old_fragment.lower().strip()
        new_frag_lower = new_fragment.lower().strip()
        best_idx = None
        best_score = 0.0
        for idx in _ac_available:
            ac = annotated_changes[idx]
            ac_orig = (ac.get('original_text', '') or '').strip().lower()
            ac_new = (ac.get('new_text', '') or '').strip().lower()
            score = 0.0
            if ac_orig and old_frag_lower:
                if ac_orig in old_frag_lower or old_frag_lower in ac_orig:
                    overlap = min(len(ac_orig), len(old_frag_lower)) / max(len(ac_orig), len(old_frag_lower), 1)
                    score += 0.5 + 0.5 * overlap
            if ac_new and new_frag_lower:
                if ac_new in new_frag_lower or new_frag_lower in ac_new:
                    overlap = min(len(ac_new), len(new_frag_lower)) / max(len(ac_new), len(new_frag_lower), 1)
                    score += 0.5 + 0.5 * overlap
            if ac_orig and not ac_new and old_frag_lower and not new_frag_lower:
                if ac_orig in old_frag_lower or old_frag_lower in ac_orig:
                    score += 0.5
            if ac_new and not ac_orig and new_frag_lower and not old_frag_lower:
                if ac_new in new_frag_lower or new_frag_lower in ac_new:
                    score += 0.5
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is not None and best_score >= 0.3:
            _ac_available.remove(best_idx)
            return best_idx
        return None

    def _marker_span(match_idx):
        """Build the [N] marker HTML for a matched annotated_change."""
        ac = annotated_changes[match_idx]
        reason = (ac.get('reason', '') or '').strip()
        reason_esc = html_lib.escape(reason)
        n = match_idx + 1
        return (f' <span class="rubric-marker" title="{reason_esc}" '
                f'style="background:#e3f2fd;color:#1565c0;padding:0 4px;'
                f'border-radius:3px;font-size:0.85em;cursor:default;">[{n}]</span>')

    def _word_diff_html(old_block: str, new_block: str, try_match: bool = True):
        """Word-level diff within a changed block. Returns HTML string."""
        old_w = old_block.split()
        new_w = new_block.split()
        w_matcher = difflib.SequenceMatcher(None, old_w, new_w)
        parts = []
        # Try to match the whole block to an annotated_change
        block_match = _match_change(old_block, new_block) if try_match else None
        for tag, i1, i2, j1, j2 in w_matcher.get_opcodes():
            if tag == 'equal':
                parts.append(html_lib.escape(' '.join(old_w[i1:i2])))
            elif tag == 'replace':
                parts.append(f'<del>{html_lib.escape(" ".join(old_w[i1:i2]))}</del>')
                parts.append(f'<ins>{html_lib.escape(" ".join(new_w[j1:j2]))}</ins>')
            elif tag == 'delete':
                parts.append(f'<del>{html_lib.escape(" ".join(old_w[i1:i2]))}</del>')
            elif tag == 'insert':
                parts.append(f'<ins>{html_lib.escape(" ".join(new_w[j1:j2]))}</ins>')
        result = ' '.join(parts)
        if block_match is not None:
            result += _marker_span(block_match)
        return result

    result_parts = []

    for tag, i1, i2, j1, j2 in line_matcher.get_opcodes():
        if tag == 'equal':
            result_parts.append(html_lib.escape(''.join(old_lines[i1:i2])))
        elif tag == 'replace':
            # For replaced line groups, do word-level diff within them
            old_block = ''.join(old_lines[i1:i2])
            new_block = ''.join(new_lines[j1:j2])
            result_parts.append(_word_diff_html(old_block, new_block))
        elif tag == 'delete':
            old_block = ''.join(old_lines[i1:i2])
            match_idx = _match_change(old_block, "")
            marker = _marker_span(match_idx) if match_idx is not None else ""
            result_parts.append(f'<del>{html_lib.escape(old_block)}</del>{marker}')
        elif tag == 'insert':
            new_block = ''.join(new_lines[j1:j2])
            match_idx = _match_change("", new_block)
            marker = _marker_span(match_idx) if match_idx is not None else ""
            result_parts.append(f'<ins>{html_lib.escape(new_block)}</ins>{marker}')

    diff_html = ''.join(result_parts)
    # Preserve line breaks in HTML
    diff_html = diff_html.replace('\n', '<br>')
    return (
        '<div class="annotated-draft" style="background:#fafafa;border:1px solid #e0e0e0;border-radius:6px;padding:12px;margin:8px 0;line-height:1.7;">'
        '<style>.annotated-draft ins { background:#c8e6c9; font-weight:bold; text-decoration:none; padding:0 2px; } '
        '.annotated-draft del { background:#ffcdd2; text-decoration:line-through; padding:0 2px; }</style>'
        f'{diff_html}</div>'
    )


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


def _mark_probe_rubric_updated(criterion_name: str):
    """Mark the matching probe_results entry as rubric_updated=True, update the probe log message, and re-save to DB."""
    _pr_list = st.session_state.get("probe_results", [])
    _crit_lower = (criterion_name or "").lower().strip()
    for _pr in reversed(_pr_list):  # most recent first
        if _pr.get("criterion_name", "").lower().strip() == _crit_lower and not _pr.get("rubric_updated"):
            _pr["rubric_updated"] = True
            break
    # Also update the probe log message in the conversation
    for _msg in reversed(st.session_state.get("messages", [])):
        if _msg.get("is_probe_log"):
            _pld = _msg.get("probe_log_data", {})
            if _pld.get("criterion_name", "").lower().strip() == _crit_lower:
                _pld["rubric_applied"] = True
                break
    _auto_save_conversation()
    # Re-save full list to DB
    _save_sb = st.session_state.get("supabase")
    _save_pid = st.session_state.get("current_project_id")
    if _save_sb and _save_pid:
        try:
            from auth_supabase import save_project_data as _spd
            # Overwrite the full list (save_project_data appends, so we need to use update directly)
            existing = _save_sb.table("project_data").select("id").eq("project_id", _save_pid).eq("data_type", "probe_results").execute()
            if existing.data:
                _save_sb.table("project_data").update({
                    "data": json.dumps(_pr_list),
                    "updated_at": datetime.now().isoformat()
                }).eq("id", existing.data[0]["id"]).execute()
        except Exception as _e:
            # print(f"[PROBE] Failed to persist rubric_updated: {_e}")


            pass
def display_rubric_comparison(current_rubric: list, updated_rubric: list, apply_context: dict = None, criterion_reasons: dict = None):
    """
    Display a comparison of current and updated rubric using collapsible sections.
    Each criterion is shown as an expander with status badge visible when collapsed.
    Word-level diffing highlights specific changes.
    criterion_reasons: optional dict mapping criterion name -> reason string for why the change is suggested.
    If apply_context is provided, adds an "Apply all suggestions" button.
    apply_context = {"safe_msg_id", "message", "message_id"}.
    """
    # Build a map of current criteria by name for comparison
    current_map = {c.get('name', '').lower().strip(): c for c in current_rubric}
    updated_map = {c.get('name', '').lower().strip(): c for c in updated_rubric}
    safe_msg_id = (apply_context or {}).get("safe_msg_id", "")
    message = (apply_context or {}).get("message")
    message_id = (apply_context or {}).get("message_id", "")
    # Build case-insensitive reasons lookup
    _reasons_map = {}
    if criterion_reasons:
        for _rk, _rv in criterion_reasons.items():
            _reasons_map[_rk.lower().strip()] = _rv

    def _find_reason(crit_name_key):
        """Look up reason by exact match, then fuzzy substring match."""
        if crit_name_key in _reasons_map:
            return _reasons_map[crit_name_key]
        # Fuzzy: check if any reason key is contained in the criterion name or vice versa
        for _rk, _rv in _reasons_map.items():
            if _rk in crit_name_key or crit_name_key in _rk:
                return _rv
        return ""


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
            _crit_reason = _find_reason(name_key)
            expander_label = f"➕ {name} (Priority #{priority})"

            with st.expander(expander_label, expanded=False):
                st.markdown(status_html, unsafe_allow_html=True)
                if _crit_reason:
                    st.info(f"💡 {_crit_reason}")
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

        elif _criterion_changed(current_map[name_key], criterion):
            # MODIFIED criterion
            old = current_map[name_key]
            old_priority = old.get('priority', old.get('weight', 0))

            status_html = '<span class="status-badge status-modified">MODIFIED</span>'
            _crit_reason = _find_reason(name_key)
            expander_label = f"🔄 {name} (Priority #{priority})"

            with st.expander(expander_label, expanded=False):
                st.markdown(status_html, unsafe_allow_html=True)
                if _crit_reason:
                    st.info(f"💡 {_crit_reason}")

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

        else:
            # UNCHANGED criterion
            status_html = '<span class="status-badge status-unchanged">UNCHANGED</span>'
            _crit_reason = _find_reason(name_key)
            expander_label = f"⚪ {name} (Priority #{priority})"

            with st.expander(expander_label, expanded=False):
                st.markdown(status_html, unsafe_allow_html=True)
                if _crit_reason:
                    st.success(f"✓ {_crit_reason}")
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
            _crit_reason = _find_reason(name_key)
            expander_label = f"❌ {name} (Priority #{priority}) - REMOVED"

            with st.expander(expander_label, expanded=False):
                st.markdown(status_html, unsafe_allow_html=True)
                if _crit_reason:
                    st.info(f"💡 {_crit_reason}")
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

    # Apply all suggestions (when in apply_context)
    if apply_context and message is not None and updated_rubric:
        # Check if there are actual differences before showing Apply All
        _any_change = False
        _cur_map_check = {c.get('name', '').lower().strip(): c for c in current_rubric}
        _upd_map_check = {c.get('name', '').lower().strip(): c for c in updated_rubric}
        for _uk in _upd_map_check:
            if _uk not in _cur_map_check:
                _any_change = True
                break
            if _criterion_changed(_cur_map_check[_uk], _upd_map_check[_uk]):
                _any_change = True
                break
        if not _any_change:
            for _ck in _cur_map_check:
                if _ck not in _upd_map_check:
                    _any_change = True
                    break
        st.markdown("---")
        _acol, _ = st.columns([0.3, 0.7])
        with _acol:
            if st.button("✅ Apply all", key=f"apply_all_{safe_msg_id}", type="primary", width="stretch", disabled=not _any_change):
                # Merge updated_rubric into the full current rubric (not replace)
                # This handles cases where updated_rubric is a subset (e.g., single criterion from probe)
                full_criteria = list(st.session_state.editing_criteria or [])
                _updated_map = {c.get('name', '').lower().strip(): c for c in updated_rubric}
                _current_names = {c.get('name', '').lower().strip() for c in current_rubric}
                # Update existing criteria that match
                for i, c in enumerate(full_criteria):
                    c_key = (c.get('name') or '').lower().strip()
                    if c_key in _updated_map:
                        full_criteria[i] = copy.deepcopy(_updated_map[c_key])
                # Add new criteria (in updated but not in current)
                for c in updated_rubric:
                    c_key = c.get('name', '').lower().strip()
                    if c_key not in _current_names:
                        full_criteria.append(copy.deepcopy(c))
                # Remove criteria (in current but not in updated — only if current_rubric was the full set)
                if len(current_rubric) == len(full_criteria):
                    _removed_names = _current_names - set(_updated_map.keys())
                    full_criteria = [c for c in full_criteria if (c.get('name') or '').lower().strip() not in _removed_names]
                new_criteria = copy.deepcopy(full_criteria)
                hist = load_rubric_history()
                new_version = next_version_number()
                hist.append({"version": new_version, "rubric": copy.deepcopy(new_criteria), "source": "edit_feedback", "conversation_id": st.session_state.get("selected_conversation")})
                save_rubric_history(hist)
                st.session_state.active_rubric_idx = len(hist) - 1
                st.session_state.rubric = new_criteria
                st.session_state.editing_criteria = new_criteria
                st.session_state.editing_criteria_ui_version = st.session_state.get("editing_criteria_ui_version", 0) + 1
                # Force the sidebar version selector to update to the new version
                st.session_state["rubric_version_selector"] = f"v{new_version}"
                if message and "rubric_suggestion" in message:
                    message["rubric_suggestion"]["applied"] = True
                    message["rubric_suggestion"]["applied_version"] = new_version
                    message["rubric_version"] = new_version
                if message and "probe_result" in message:
                    message["probe_result"]["applied"] = True
                    message["probe_result"]["applied_version"] = new_version
                    _mark_probe_rubric_updated(message["probe_result"].get("criterion_name", ""))
                # If this is a conversation-start alignment check, inject the suggested draft
                if message and message.get("_ac_pending_draft"):
                    _ac_sd = message.get("_ac_suggested_draft", "")
                    if _ac_sd:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Here is your starting draft, generated using the **updated rubric** (with the improvements you just applied):\n\n<draft>\n{_ac_sd}\n</draft>",
                            "message_id": f"ac_draft_{int(time.time() * 1000000)}",
                        })
                    message["_ac_pending_draft"] = False
                # Save rubric application event to database
                _apply_pid = st.session_state.get("current_project_id")
                if _apply_pid:
                    _apply_source = "rubric_suggestion" if (message and "rubric_suggestion" in message) else "probe_result" if (message and "probe_result" in message) else "unknown"
                    save_project_data(supabase, _apply_pid, "rubric_edit_applied", {
                        "timestamp": datetime.now().isoformat(),
                        "source": _apply_source,
                        "new_rubric_version": new_version,
                        "conversation_id": st.session_state.get("selected_conversation", ""),
                        "message_id": message_id,
                        "previous_rubric": _rubric_list_for_json(current_rubric),
                        "applied_rubric": _rubric_list_for_json(new_criteria),
                        "criteria_changes": {
                            "added": [c.get("name", "") for c in updated_rubric if c.get("name", "").lower().strip() not in {x.get("name", "").lower().strip() for x in current_rubric}],
                            "removed": [c.get("name", "") for c in current_rubric if c.get("name", "").lower().strip() not in {x.get("name", "").lower().strip() for x in updated_rubric}],
                            "modified": [c.get("name", "") for c in updated_rubric if c.get("name", "").lower().strip() in current_map and _criterion_changed(current_map[c.get("name", "").lower().strip()], c)],
                        },
                    })
                    # Also persist as log_changes for analysis
                    _apply_edit_class = classify_rubric_edits(current_rubric, updated_rubric)
                    _apply_old_ver = hist[-2].get("version", 0) if len(hist) >= 2 else 0
                    save_project_data(supabase, _apply_pid, "log_changes", {
                        "timestamp": datetime.now().isoformat(),
                        "source": f"apply_{_apply_source}",
                        "edit_classification": _apply_edit_class,
                        "rubric_version": new_version,
                        "old_version": _apply_old_ver,
                        "draft_regenerated": False,
                    })
                # Use the preview draft from suggestion if available, otherwise regenerate
                _apply_last_draft, _ = get_last_draft_from_messages()
                _apply_preview = message.get("rubric_suggestion", {}).get("preview_draft") if message else None
                if _apply_preview and _apply_preview.get("revised_draft"):
                    # Use the already-generated preview draft
                    _apply_new_draft = _apply_preview["revised_draft"]
                    _apply_draft_msg = {
                        "role": "assistant",
                        "content": f"Here's an updated draft based on the rubric changes you just applied (v{new_version}):\n\n<draft>{_apply_new_draft}</draft>",
                        "display_content": f"Here's an updated draft based on the rubric changes you just applied (v{new_version}):\n\n<draft>{_apply_new_draft}</draft>",
                        "rubric_version": new_version,
                        "is_system_generated": True,
                        "is_post_apply_draft": True,
                        "post_apply_data": {
                            "previous_draft": _apply_last_draft or "",
                            "new_draft": _apply_new_draft,
                            "rubric_version": new_version,
                        },
                        "message_id": f"post_apply_{int(time.time() * 1000000)}",
                    }
                    st.session_state.messages.append(_apply_draft_msg)
                elif _apply_last_draft and current_rubric:
                    # Fallback: regenerate draft (for non-suggestion Apply All, e.g. probe results)
                    _conv_parts = []
                    for _cm in st.session_state.messages:
                        _cm_role = _cm.get("role", "")
                        _cm_content = _cm.get("display_content") or _cm.get("content", "")
                        if _cm_role in ("user", "assistant") and _cm_content:
                            _conv_parts.append(f"[{_cm_role.upper()}]: {_cm_content[:2000]}")
                    _conv_history = "\n\n".join(_conv_parts[-20:]) if _conv_parts else None

                    _apply_suggestion_text = None
                    if message:
                        _rs = message.get("rubric_suggestion", {})
                        if _rs:
                            _apply_suggestion_text = _rs.get("suggestion_text", "")
                        _dd = message.get("diagnostic_data", {})
                        if not _apply_suggestion_text and _dd:
                            _apply_suggestion_text = _dd.get("suggestion_text", "")

                    _apply_edit_fb = None
                    if message:
                        _rr = message.get("rubric_revision", {})
                        _ann = _rr.get("annotated_changes", []) if _rr else []
                        _fb_parts = []
                        for _fb_ac in _ann:
                            _fb_text = _fb_ac.get("user_feedback", "")
                            if _fb_text and _fb_text.strip():
                                _fb_parts.append(f"- Edit: \"{_fb_ac.get('original_text', '')}\" → \"{_fb_ac.get('new_text', '')}\"\n  User feedback: {_fb_text}")
                        if _fb_parts:
                            _apply_edit_fb = "\n".join(_fb_parts)

                    _apply_regen = regenerate_draft_from_rubric_changes(
                        current_rubric, new_criteria, _apply_last_draft,
                        conversation_history=_conv_history,
                        rubric_suggestion_text=_apply_suggestion_text,
                        user_edit_feedback=_apply_edit_fb,
                    )
                    if _apply_regen and _apply_regen.get("revised_draft") and not _apply_regen.get("error"):
                        _apply_new_draft = _apply_regen["revised_draft"]
                        _apply_draft_msg = {
                            "role": "assistant",
                            "content": f"Here's an updated draft based on the rubric changes you just applied (v{new_version}):\n\n<draft>{_apply_new_draft}</draft>",
                            "display_content": f"Here's an updated draft based on the rubric changes you just applied (v{new_version}):\n\n<draft>{_apply_new_draft}</draft>",
                            "rubric_version": new_version,
                            "is_system_generated": True,
                            "is_post_apply_draft": True,
                            "post_apply_data": {
                                "previous_draft": _apply_last_draft,
                                "new_draft": _apply_new_draft,
                                "rubric_version": new_version,
                            },
                            "message_id": f"post_apply_{int(time.time() * 1000000)}",
                        }
                        st.session_state.messages.append(_apply_draft_msg)
                st.toast(f"All suggestions applied and saved as v{new_version}.")
                _auto_save_conversation()
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
                            "coaching_notes": rubric_updates.get('rationale', 'Rubric updated based on draft edit analysis'),
                            "conversation_id": st.session_state.get("selected_conversation"),
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

                        # Persist log_changes for analysis
                        _lc_pid2 = st.session_state.get('current_project_id')
                        _lc_sb2 = st.session_state.get('supabase')
                        if _lc_sb2 and _lc_pid2:
                            try:
                                save_project_data(_lc_sb2, _lc_pid2, "log_changes", {
                                    "timestamp": datetime.now().isoformat(),
                                    "source": "draft_edit",
                                    "edit_classification": edit_class,
                                    "rubric_version": new_version,
                                    "old_version": old_ver,
                                    "draft_regenerated": False,
                                })
                            except Exception:
                                pass

                        # Update active rubric
                        st.session_state.active_rubric_idx = len(hist) - 1
                        st.session_state.rubric = modified_rubric
                        # Also update editing_criteria so Rubric Configuration shows the new version
                        st.session_state.editing_criteria = copy.deepcopy(modified_rubric)
                        st.session_state["rubric_version_selector"] = f"v{new_version}"

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

def save_message_log(messages, rubric, analysis=None, conversation_id=None):
    """Save conversation to Supabase database.

    If conversation_id is provided, updates the existing conversation.
    Otherwise, inserts a new one.
    """
    project_id = st.session_state.get('current_project_id')
    if not project_id:
        raise ValueError("No project selected. Please select a project first.")

    supabase = st.session_state.get('supabase')
    if not supabase:
        raise ValueError("Database connection not available.")

    # Save to Supabase (update if existing, insert if new)
    conv_id = save_conversation(supabase, project_id, messages, rubric, analysis or "",
                                conversation_id=conversation_id)
    # Invalidate conversations cache so the list refreshes
    cache_key = f"conversations_{project_id}"
    if cache_key in st.session_state:
        del st.session_state[cache_key]
    return conv_id


def _auto_save_conversation():
    """Silently auto-save the current conversation.

    If selected_conversation exists, updates it. Otherwise inserts a new one
    and sets selected_conversation so future saves update the same row.
    Does nothing if there are no messages or no project selected.
    """
    if not st.session_state.get("messages"):
        # print("[AUTO-SAVE] Skipped: no messages")
        return
    if not st.session_state.get("current_project_id"):
        # print("[AUTO-SAVE] Skipped: no project_id")
        pass
        return
    if not st.session_state.get("supabase"):
        # print("[AUTO-SAVE] Skipped: no supabase")
        pass
        return
    try:
        _existing_id = st.session_state.get("selected_conversation")
        # print(f"[AUTO-SAVE] Saving... existing_id={_existing_id}, msg_count={len(st.session_state.messages)}")
        pass
        conv_id = save_message_log(
            st.session_state.messages,
            st.session_state.get("rubric", []),
            st.session_state.get("current_analysis", ""),
            conversation_id=_existing_id,
        )
        if conv_id and conv_id != _existing_id:
            # New insert (or re-insert after stale ID) — record the ID
            st.session_state.selected_conversation = conv_id
            # print(f"[AUTO-SAVE] New conversation saved: {conv_id}")
        elif conv_id:
            # print(f"[AUTO-SAVE] Updated conversation: {conv_id}")
            pass
        else:
            # print("[AUTO-SAVE] save_message_log returned None")
            pass
    except Exception as e:
        # print(f"[AUTO-SAVE] ERROR: {e}")  # Log but don't interrupt the user


        pass
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
    """Build numbered conversation text from messages for rubric/DP prompts.

    Includes rich structured data from special message types so the model
    sees the full picture: drafts, probes, rubric changes, alignment checks,
    user feedback, accept/revert decisions, etc.
    """
    conversation_text = ""
    msg_num = 1
    for msg in messages:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        # Synthetic changelog messages are context-only — include but don't number
        if msg.get('_synthetic_changelog'):
            conversation_text += f"\n\n[Rubric Version Change]\n{content}"
            continue

        # --- Build enriched content for special message types ---
        enriched = content

        # Probe log: include probe details (why, interpretations, drafts, user choice)
        if msg.get('is_probe_log'):
            pld = msg.get("probe_log_data", {})
            parts = [content]
            if pld.get("variant_a"):
                parts.append(f"\n[Probe Draft — Version A]\n{pld['variant_a']}")
            if pld.get("variant_b"):
                parts.append(f"\n[Probe Draft — Version B]\n{pld['variant_b']}")
            enriched = "\n".join(parts)

        # Rubric revision (Log Changes): include what changed, edit details, user feedback
        elif msg.get('rubric_revision'):
            rr = msg['rubric_revision']
            parts = [content]
            if rr.get('change_summary'):
                parts.append(f"\n[Rubric Change Summary]\n{rr['change_summary']}")
            if rr.get('annotated_changes'):
                parts.append("\n[Edit Details]")
                for _ac_i, _ac in enumerate(rr['annotated_changes'], 1):
                    parts.append(f"  [{_ac_i}] {_ac.get('original_text', '')} → {_ac.get('new_text', '')}  |  Reason: {_ac.get('reason', '')}")
            if rr.get('user_feedback'):
                parts.append("\n[User Feedback on Edits]")
                for _fb_key, _fb_val in rr['user_feedback'].items():
                    if _fb_val:
                        parts.append(f"  Edit {_fb_key}: {_fb_val}")
            if rr.get('_decision'):
                parts.append(f"\n[User Decision] {rr['_decision']}")
            enriched = "\n".join(parts)

        # Alignment diagnostic: include rubric suggestion and suggestion reasons
        elif msg.get('is_alignment_diagnostic'):
            parts = [content]
            rs = msg.get('rubric_suggestion', {})
            if rs:
                if rs.get('suggestion_reasons'):
                    parts.append("\n[Rubric Suggestion Reasons]")
                    _sr_data = rs['suggestion_reasons']
                    if isinstance(_sr_data, dict):
                        for _sr_name, _sr_reason in _sr_data.items():
                            parts.append(f"  - {_sr_name}: {_sr_reason}")
                    elif isinstance(_sr_data, list):
                        for _sr in _sr_data:
                            if isinstance(_sr, dict):
                                parts.append(f"  - {_sr.get('criterion_name', '')}: {_sr.get('reason', '')}")
                            else:
                                parts.append(f"  - {_sr}")
                if rs.get('updated_rubric'):
                    parts.append(f"\n[Suggested Rubric]\n{json.dumps(rs['updated_rubric'], ensure_ascii=False, indent=2)}")
                if rs.get('_user_applied'):
                    parts.append("\n[User Applied Suggestion] yes")
                elif rs.get('_user_dismissed'):
                    parts.append("\n[User Dismissed Suggestion]")
                if rs.get('suggested_draft'):
                    parts.append(f"\n[Suggested Draft from Rubric Suggestion]\n{rs['suggested_draft']}")
                if rs.get('original_rubric_draft'):
                    parts.append(f"\n[Original Draft Before Suggestion]\n{rs['original_rubric_draft']}")
            # Include user feedback on suggested edits if stored on the diagnostic
            diag = msg.get('diagnostic_data', {})
            if diag.get('user_edit_feedback'):
                parts.append("\n[User Feedback on Suggested Edits]")
                for _fk, _fv in diag['user_edit_feedback'].items():
                    if _fv:
                        parts.append(f"  Edit {_fk}: {_fv}")
            enriched = "\n".join(parts)

        # Criteria classification log: include what was stated/real/hallucinated
        elif msg.get('is_criteria_classification_log'):
            cd = msg.get('classification_data', {})
            parts = [content]
            if cd.get('classifications'):
                parts.append("\n[Criteria Classifications]")
                for _cn, _cat in cd['classifications'].items():
                    _line = f"  - {_cn}: {_cat}"
                    if _cat == "hallucinated" and cd.get('hallucination_reasons', {}).get(_cn):
                        _line += f" — reason: {cd['hallucination_reasons'][_cn]}"
                    parts.append(_line)
            enriched = "\n".join(parts)

        # DP confirmation log: include decision points and rubric mappings
        elif msg.get('is_dp_confirmation_log'):
            dpd = msg.get('dp_data', {})
            parts = [content]
            if dpd.get('decision_points'):
                parts.append("\n[Confirmed Decision Points]")
                for _dp in dpd['decision_points']:
                    parts.append(f"  - Message #{_dp.get('message_number', '?')}: {_dp.get('text', '')}")
                    parts.append(f"    Criterion: {_dp.get('criterion_name', 'unmapped')}")
                    if _dp.get('user_override'):
                        parts.append(f"    User override: {_dp['user_override']}")
            enriched = "\n".join(parts)

        # DP review: include inferred rubric criteria
        elif msg.get('is_dp_review'):
            dpd = msg.get('dp_data', {})
            parts = [content]
            if dpd.get('rubric'):
                parts.append(f"\n[Inferred Rubric v{dpd.get('rubric_version', '?')}]")
                for _rc in dpd['rubric']:
                    parts.append(f"  - {_rc.get('name', 'Unnamed')}: {_rc.get('description', '')}")
            enriched = "\n".join(parts)

        # Inline rephrase: include original/replacement text and instruction
        elif msg.get('is_inline_rephrase'):
            ird = msg.get('inline_rephrase_data', {})
            parts = [content]
            if ird:
                parts.append(f"\n[Inline Rephrase]")
                parts.append(f"  Original text: {ird.get('original_text', '')}")
                parts.append(f"  Replacement: {ird.get('replacement_text', '')}")
                parts.append(f"  Instruction: {ird.get('instruction', '')}")
            enriched = "\n".join(parts)

        # Format based on role
        if role == 'user':
            conversation_text += f"\n\n[Message #{msg_num}] USER:\n{enriched}"
            msg_num += 1
        elif role == 'assistant':
            conversation_text += f"\n\n[Message #{msg_num}] ASSISTANT:\n{enriched}"
            msg_num += 1
        elif role == 'system':
            conversation_text += f"\n\n[Message #{msg_num}] SYSTEM:\n{enriched}"
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
            rubric_data["conversation_id"] = st.session_state.get("selected_conversation")

            # Normalize priorities to unique sequential 1..N
            _infer_criteria = rubric_data.get("rubric", [])
            if _infer_criteria:
                _indexed = [(i, c) for i, c in enumerate(_infer_criteria)]
                _indexed.sort(key=lambda x: (x[1].get("priority", 999), x[0]))
                for _rank, (_, _c) in enumerate(_indexed, 1):
                    _c["priority"] = _rank

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
            rubric_data["conversation_id"] = st.session_state.get("selected_conversation")

            # Normalize priorities to unique sequential 1..N
            _final_criteria = rubric_data.get("rubric", [])
            if _final_criteria:
                _final_indexed = [(i, c) for i, c in enumerate(_final_criteria)]
                _final_indexed.sort(key=lambda x: (x[1].get("priority", 999), x[0]))
                for _final_rank, (_, _fc) in enumerate(_final_indexed, 1):
                    _fc["priority"] = _final_rank

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
    import hashlib as _auth_hashlib
    st.title("AI-Rubric Writer")
    st.markdown("Enter your name and email to get started.")

    # Check if we need to handle a legacy user (has old password)
    _legacy_email = st.session_state.get("_auth_legacy_email")

    if _legacy_email:
        # Legacy user migration: ask for old password one time
        st.info(f"Welcome back! We've simplified login — no more passwords. Please enter your existing password one last time for **{_legacy_email}** to migrate your account.")
        with st.form("legacy_form"):
            _legacy_pw = st.text_input("Your existing password", type="password")
            _legacy_submit = st.form_submit_button("Migrate & Continue", width="stretch")

            if _legacy_submit:
                if _legacy_pw:
                    success, message = login_user(supabase, _legacy_email, _legacy_pw)
                    if success:
                        st.session_state.pop("_auth_legacy_email", None)
                        st.session_state.supabase = get_supabase_client()
                        # Update password to auto-generated one for future logins
                        try:
                            _auto_pw = _auth_hashlib.sha256((_legacy_email + "_rubricllm_auto").encode()).hexdigest()[:24]
                            supabase_client = get_supabase_client()
                            supabase_client.auth.update_user({"password": _auto_pw})
                        except Exception:
                            pass  # Non-critical — will just ask for password again next time
                        st.rerun()
                    else:
                        st.error("Incorrect password. Please try again.")
                else:
                    st.warning("Please enter your password.")
        if st.button("Use a different email"):
            st.session_state.pop("_auth_legacy_email", None)
            st.rerun()
    else:
        with st.form("auth_form"):
            _auth_name = st.text_input("Name")
            _auth_email = st.text_input("Email")
            _auth_submit = st.form_submit_button("Continue", width="stretch")

            if _auth_submit:
                if not _auth_name or not _auth_email:
                    st.warning("Please enter both your name and email.")
                elif "@" not in _auth_email:
                    st.warning("Please enter a valid email address.")
                else:
                    _auth_email = _auth_email.strip().lower()
                    _auth_name = _auth_name.strip()
                    # Generate a deterministic password from the email (invisible to the user)
                    _auto_pw = _auth_hashlib.sha256((_auth_email + "_rubricllm_auto").encode()).hexdigest()[:24]
                    # Try to log in with auto password
                    success, message = login_user(supabase, _auth_email, _auto_pw)
                    if success:
                        st.session_state.supabase = get_supabase_client()
                        st.rerun()
                    else:
                        # Try to register (new user)
                        reg_success, reg_msg = register_user(supabase, _auth_email, _auto_pw, _auth_name)
                        if reg_success:
                            success, message = login_user(supabase, _auth_email, _auto_pw)
                            if success:
                                st.session_state.supabase = get_supabase_client()
                                st.rerun()
                        elif "already" in reg_msg.lower():
                            # Legacy user with old password — need one-time migration
                            st.session_state["_auth_legacy_email"] = _auth_email
                            st.rerun()
                        else:
                            st.error(reg_msg)

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
    if st.button("Logout", width="stretch"):
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
    st.session_state.survey_responses = {"task_a": {}, "task_b": {}}

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

# Uncertainty probe: every N-th draft, probe a rubric criterion the model is uncertain about
if 'probe_draft_counts' not in st.session_state:
    st.session_state.probe_draft_counts = {}  # Per-conversation draft counts: {conv_id: int}
if 'probe_pending' not in st.session_state:
    st.session_state.probe_pending = None  # Dict with probe variants when triggered
if 'probe_results' not in st.session_state:
    st.session_state.probe_results = []  # List of completed probe results

# Evaluation dashboard: grade evaluation and retest history
if 'grade_evaluation_history' not in st.session_state:
    st.session_state.grade_evaluation_history = []
if 'grade_retest_history' not in st.session_state:
    st.session_state.grade_retest_history = []
if 'diagnostic_retest_history' not in st.session_state:
    st.session_state.diagnostic_retest_history = []

# Layer 2: Ranking checkpoint state
if 'ranking_checkpoint_results' not in st.session_state:
    st.session_state.ranking_checkpoint_results = []  # List of completed checkpoint results
if 'ranking_checkpoint_pending' not in st.session_state:
    st.session_state.ranking_checkpoint_pending = None  # {step: 2|3, writing_task, drafts, shuffle_order}
if 'ranking_checkpoint_auto_triggered' not in st.session_state:
    st.session_state.ranking_checkpoint_auto_triggered = False
if 'alignment_check_done' not in st.session_state:
    st.session_state.alignment_check_done = False
if 'alignment_check_skipped' not in st.session_state:
    st.session_state.alignment_check_skipped = False


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
            # Check description change (normalize whitespace to avoid false positives)
            old_desc = ' '.join(old_c.get('description', '').split())
            new_desc = ' '.join(new_c.get('description', '').split())
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
    lines = [f"📋 **(Temporary) Rubric Changes have been made:**"]

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

    # Determine the target value for the selectbox widget
    # Priority: pending delete → pending sync (auto-save) → current selected_conversation
    options = [opt[1] for opt in conversation_options]

    # Keep the selectbox in sync with selected_conversation, BUT skip if the user
    # just interacted with the selector (on_change callback already set the widget value).
    if not st.session_state.get("_user_changed_conversation"):
        _target = st.session_state.get("selected_conversation")
        if _target and _target not in options:
            # Target not in cached list — force reload
            conversations = load_conversations(force_reload=True)
            conversation_options = [("New Conversation", None)]
            for conv in conversations:
                try:
                    dt = datetime.fromisoformat(conv["timestamp"])
                    formatted_time = dt.strftime("%m/%d %H:%M")
                    display = f"{formatted_time} ({conv['messages_count']} msgs)"
                except Exception:
                    display = f"{conv['timestamp']} ({conv['messages_count']} msgs)"
                conversation_options.append((display, conv["filename"]))
            options = [opt[1] for opt in conversation_options]
        if _target and _target in options:
            st.session_state.conversation_selector = _target
        else:
            st.session_state.conversation_selector = None

    def _on_conversation_selector_change():
        """Callback when user explicitly changes the conversation selector."""
        st.session_state._user_changed_conversation = True

    _conv_sel_col, _conv_del_col = st.columns([5, 1])
    with _conv_sel_col:
        selected_file = st.selectbox(
            "💬 Select conversation:",
            options=options,
            format_func=lambda x: next(opt[0] for opt in conversation_options if opt[1] == x) if x is not None else "New Conversation",
            key="conversation_selector",
            on_change=_on_conversation_selector_change,
        )
    with _conv_del_col:
        st.markdown("<div style='height: 1.65rem'></div>", unsafe_allow_html=True)  # align with selectbox
        _del_conv_disabled = selected_file is None  # disable when "New Conversation" selected
        if st.button("🗑️", key="delete_conversation_btn", disabled=_del_conv_disabled, help="Delete this conversation"):
            if selected_file and st.session_state.get("supabase"):
                if delete_conversation(st.session_state.supabase, selected_file):
                    st.session_state.messages = []
                    st.session_state.rubric = None
                    st.session_state.current_analysis = ""
                    st.session_state.selected_conversation = None
                    st.session_state.alignment_check_done = False
                    st.session_state.alignment_check_skipped = False
                    st.session_state.ranking_checkpoint_pending = None
                    st.session_state.ranking_checkpoint_auto_triggered = False
                    # Invalidate conversations cache so the list refreshes
                    _cache_key = f"conversations_{st.session_state.get('current_project_id')}"
                    if _cache_key in st.session_state:
                        del st.session_state[_cache_key]
                    st.toast("Conversation deleted.")
                    st.rerun()
                else:
                    st.error("Failed to delete conversation.")

    # Only react to conversation selector changes when the USER explicitly changed it
    _user_changed = st.session_state.pop("_user_changed_conversation", False)
    # print(f"[SELECTOR] selected_file={selected_file}, selected_conversation={st.session_state.get('selected_conversation')}, user_changed={_user_changed}")

    if _user_changed:
        if selected_file is None and st.session_state.selected_conversation is not None:
            # User switched to "New Conversation"
            # print(f"[SELECTOR] User switched to New Conversation")
            st.session_state.messages = []
            st.session_state.rubric = None
            st.session_state.current_analysis = ""
            st.session_state.selected_conversation = None
            st.session_state.comparison_result = None
            st.session_state.comparison_rubric_version = None
            # probe_draft_counts preserved per-conversation (not reset on switch)
            st.session_state.probe_pending = None
            st.session_state.ranking_checkpoint_pending = None
            st.session_state.ranking_checkpoint_auto_triggered = False
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
            st.session_state.alignment_check_done = False
            st.session_state.alignment_check_skipped = False
            st.rerun()
        elif selected_file and selected_file != st.session_state.selected_conversation:
            # User switched to a different conversation — load it
            # print(f"[SELECTOR] User switched to conversation {selected_file}")
            conv_data = load_conversation_data(selected_file)
            # print(f"[SELECTOR] Loaded conv_data: {bool(conv_data)}, msgs={len(conv_data.get('messages', [])) if conv_data else 0}")
            if conv_data:
                st.session_state.messages = conv_data.get("messages", [])
                st.session_state.rubric = conv_data.get("rubric", None)
                st.session_state.current_analysis = conv_data.get("analysis", "")
                st.session_state.selected_conversation = selected_file
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

    # --- DP inline rendering setup --- [DISABLED: DP extraction + confirmation removed for now]
    import html as _dp_html_lib
    _dp_result = st.session_state.get("infer_decision_points")
    _dp_has_review = any(m.get('is_dp_review') for m in st.session_state.messages)
    _dp_confirmed = st.session_state.get("infer_dp_dimension_confirmed", False)
    # Gate DP display on criteria classification being confirmed (or not applicable)
    _cc_pending = st.session_state.chat_criteria_review_active and not st.session_state.chat_criteria_review_confirmed
    # _dp_active = _dp_has_review and not _cc_pending
    _dp_active = False  # [DISABLED] DP extraction + confirmation removed for now
    _dp_list_all = []
    _dp_by_user_msg = {}  # {user_message_num: [dp, ...]}
    _dp_by_asst_msg = {}  # {assistant_message_num: [dp, ...]}
    _dp_crit_names = []

    if _dp_active and _dp_result:
        _dp_parsed = _dp_result.get("parsed_data", {})
        _dp_list_all = _dp_parsed.get("decision_points", [])
        # Re-number DPs so IDs follow chat display order (sorted by message position)
        _dp_list_all = sorted(_dp_list_all, key=lambda d: (d.get("user_message_num", 0), d.get("id", 0)))
        for _i, _dp in enumerate(_dp_list_all, start=1):
            _dp["id"] = _i
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

    _dp_highlighted_ids = set()  # Track which DP IDs actually had a quote matched in the conversation

    def _highlight_dp_quotes(text, dp_list, quote_key, color):
        """Highlight DP quotes in message text. Returns modified text with HTML highlights and visible DP badge."""
        _dp_badge_tpl = ('<sup style="background:#1976D2;color:white;padding:0 4px;border-radius:6px;'
                         'font-size:0.7em;font-weight:bold;margin-right:2px;">DP#{dp_id}</sup>')
        for dp in dp_list:
            quote = (dp.get(quote_key, '') or '').strip()
            if not quote or len(quote) < 5:
                continue
            dp_id = dp.get('id', 0)
            _badge = _dp_badge_tpl.replace('{dp_id}', str(dp_id))
            # Try exact match first, then case-insensitive
            if quote in text:
                highlight = (f'<span style="background:{color};padding:1px 3px;border-radius:3px;">'
                             f'{_badge}{_dp_html_lib.escape(quote)}</span>')
                text = text.replace(quote, highlight, 1)
                _dp_highlighted_ids.add(dp_id)
            elif quote.lower() in text.lower():
                # Case-insensitive: find the position and replace preserving original case
                idx = text.lower().find(quote.lower())
                original = text[idx:idx+len(quote)]
                highlight = (f'<span style="background:{color};padding:1px 3px;border-radius:3px;">'
                             f'{_badge}{_dp_html_lib.escape(original)}</span>')
                text = text[:idx] + highlight + text[idx+len(quote):]
                _dp_highlighted_ids.add(dp_id)
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

    # Auto-map all DPs upfront so jump buttons always reflect correct state
    if _dp_active and not _dp_confirmed:
        for _dp in _dp_list_all:
            _dp_id_str = str(_dp.get('id', 0))
            if not st.session_state.infer_dp_user_mappings.get(_dp_id_str):
                _dp_auto = _chat_auto_match(_dp)
                if _dp_auto:
                    st.session_state.infer_dp_user_mappings[_dp_id_str] = {"criterion": _dp_auto, "not_in_rubric": False}

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

            # Check if this DP actually had a quote highlighted in the conversation
            if dp_id in _dp_highlighted_ids:
                _dp_hl_badge = ' <span style="background:#A5D6A7;color:#1B5E20;padding:1px 6px;border-radius:8px;font-size:0.75em;">highlighted above</span>'
            else:
                _dp_hl_badge = ' <span style="background:#EEE;color:#888;padding:1px 6px;border-radius:8px;font-size:0.75em;">no highlight</span>'

            # Render a colored banner above the expander
            st.markdown(
                f'<div id="dp-card-{dp_id}" style="background:linear-gradient(135deg, #E3F2FD 0%, #F3E5F5 100%);'
                f'border-left:5px solid #FF9800;padding:6px 14px;margin:8px 0 0 0;border-radius:6px 6px 0 0;'
                f'font-size:0.85em;box-shadow:0 1px 3px rgba(0,0,0,0.12);scroll-margin-top:80px;">'
                f'<span style="background:#FF9800;color:white;padding:1px 8px;border-radius:10px;font-size:0.8em;font-weight:bold;margin-right:6px;">DP#{dp_id}</span> '
                f'<b>{_dp_html_lib.escape(dp.get("dimension", "Unknown"))}</b> &rarr; {_dp_html_lib.escape(title_crit)}'
                f'{_dp_hl_badge}'
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
    _chat_msg_num = 0  # Track message number matching _build_conversation_text numbering
    _dp_intro_shown = False  # Show DP explanation once before first card
    for idx, message in enumerate(st.session_state.messages):
        # Increment message counter to stay in sync with _build_conversation_text numbering.
        # _build_conversation_text numbers every message (user/assistant/system) except _synthetic_changelog.
        if not message.get('_synthetic_changelog'):
            _chat_msg_num += 1

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

        # [DISABLED] Probe A/B testing removed for now
        # _probe_pending_data = st.session_state.get("probe_pending")
        # if _probe_pending_data and message.get('message_id') == _probe_pending_data.get('message_id'):
        #     with st.chat_message("assistant"):
        #         st.markdown("*Your draft is ready — please compare the two versions below first.*")
        #     continue

        if message['role'] == 'system':
            # Only show system messages that belong to the currently selected conversation
            msg_conv_id = message.get("conversation_id")
            if msg_conv_id is not None and msg_conv_id != st.session_state.selected_conversation:
                continue
            # Strip machine-readable HTML comments before display
            display_content = re.sub(r'<!--.*?-->', '', message['content']).strip()
            if st.session_state.message_delete_mode:
                col_check_sys, col_msg_sys = st.columns([0.05, 0.95])
                with col_check_sys:
                    is_selected_sys = idx in st.session_state.messages_to_delete
                    if st.checkbox("Select", value=is_selected_sys, key=f"delete_msg_{idx}", label_visibility="collapsed"):
                        st.session_state.messages_to_delete.add(idx)
                    else:
                        st.session_state.messages_to_delete.discard(idx)
                with col_msg_sys:
                    if display_content:
                        st.info(display_content)
            elif display_content:
                st.info(display_content)
            # Show refinement details expander if available
            _ref_detail = message.get("refinement_detail")
            if _ref_detail:
                _ref_explanation = _ref_detail.get("change_explanation", "")
                _ref_old = _ref_detail.get("old_rubric", [])
                _ref_new = _ref_detail.get("new_rubric", [])
                _ref_old_ver = _ref_detail.get("old_version", "?")
                _ref_new_ver = _ref_detail.get("new_version", "?")
                with st.expander("What changed in v" + str(_ref_new_ver) + " from v" + str(_ref_old_ver) + " and why", expanded=False):
                    if _ref_explanation:
                        st.markdown(_ref_explanation)
                    # Build a simple diff of criteria
                    _ref_old_names = {c.get("name", "") for c in _ref_old}
                    _ref_new_names = {c.get("name", "") for c in _ref_new}
                    _ref_added = _ref_new_names - _ref_old_names
                    _ref_removed = _ref_old_names - _ref_new_names
                    _ref_kept = _ref_old_names & _ref_new_names
                    # Check for description changes in kept criteria
                    _ref_old_map = {c.get("name", ""): c.get("description", "") for c in _ref_old}
                    _ref_new_map = {c.get("name", ""): c.get("description", "") for c in _ref_new}
                    _ref_modified = [n for n in _ref_kept if _ref_old_map.get(n) != _ref_new_map.get(n)]
                    if _ref_added or _ref_removed or _ref_modified:
                        st.markdown(f"**Criteria diff** (v{_ref_old_ver} → v{_ref_new_ver}):")
                        for _rn in sorted(_ref_added):
                            st.markdown(f"- **+ {_rn}** (added)")
                        for _rn in sorted(_ref_removed):
                            st.markdown(f"- **− {_rn}** (removed)")
                        for _rn in sorted(_ref_modified):
                            st.markdown(f"- **~ {_rn}** (modified)")
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
                        safe_msg_id = re.sub(r'[^a-zA-Z0-9_-]', '_', str(message_id))
                        if message['role'] == 'user':
                            content_to_display = message['content']
                        else:
                            content_to_display = message.get('display_content', message['content'])
                        # Highlight DP quotes in message text
                        _dp_highlighted = False
                        if _dp_active:
                            if message['role'] == 'user' and _chat_msg_num in _dp_by_user_msg:
                                content_to_display = _highlight_dp_quotes(content_to_display, _dp_by_user_msg[_chat_msg_num], 'after_quote', '#A5D6A7')
                                _dp_highlighted = True
                            elif message['role'] == 'assistant' and _chat_msg_num in _dp_by_asst_msg:
                                content_to_display = _highlight_dp_quotes(content_to_display, _dp_by_asst_msg[_chat_msg_num], 'before_quote', '#FFF59D')
                                _dp_highlighted = True
                        if message['role'] == 'assistant' and message.get('thinking'):
                            with st.expander("🧠 Thinking", expanded=False):
                                st.markdown(message['thinking'])
                        # Draft-updated-by-rubric: show What changed, annotated draft with markers, then editable draft
                        if message['role'] == 'assistant' and message.get('rubric_revision'):
                            rr = message['rubric_revision']
                            ann = rr.get('annotated_changes', [])
                            _num_ann = len(ann)
                            if rr.get('change_summary'):
                                st.markdown(
                                    f"Based on your rubric changes, I made **{_num_ann} edit{'s' if _num_ann != 1 else ''}** to the draft. "
                                    f"Locate each edit with the numbered **[N]** markers below, and expand "
                                    f"**Edits by rubric change** for the full reasoning."
                                )
                                with st.expander("View change details", expanded=False):
                                    st.markdown(rr['change_summary'])
                            annotated = rr.get('revised_draft_annotated') or rr.get('revised_draft', '')
                            if annotated:
                                st.markdown("**Revised draft** (click a marker to jump to the edit):")
                                st.markdown(_safe_annotated_draft_html(annotated, ann, message_id), unsafe_allow_html=True)
                            if ann:
                                with st.expander(f"Edits by rubric change ({_num_ann})", expanded=False):
                                    import html as _html_mod
                                    safe_msg_id = re.sub(r'[^a-zA-Z0-9_-]', '_', str(message_id))
                                    for i, ac in enumerate(ann, 1):
                                        reason = (ac.get('reason', '') or '').strip()
                                        aid = f"rubric-edit-{safe_msg_id}-{i}"
                                        # Parse "CriterionName: explanation" format
                                        _rc_parts = reason.split(':', 1)
                                        if len(_rc_parts) == 2:
                                            _rc_crit = _rc_parts[0].strip()
                                            _rc_expl = _rc_parts[1].strip()
                                            # Remove parenthetical from criterion name
                                            _rc_paren = _rc_crit.find('(')
                                            if _rc_paren > 0:
                                                _rc_crit = _rc_crit[:_rc_paren].strip()
                                        else:
                                            _rc_crit = ""
                                            _rc_expl = reason
                                        # Split explanation on semicolons into bullet points
                                        _rc_bullets = [s.strip() for s in _rc_expl.split(';') if s.strip()] if _rc_expl else []
                                        _rc_header = f'<div id="{aid}" style="margin:12px 0 4px 0;scroll-margin-top:20vh;"><strong>[{i}]</strong> <strong>{_html_mod.escape(_rc_crit)}</strong>{":" if _rc_crit else ""}</div>'
                                        if len(_rc_bullets) > 1:
                                            _rc_bullet_html = ''.join(f'<div style="margin:2px 0 2px 20px;">- {_html_mod.escape(b)}</div>' for b in _rc_bullets)
                                            st.markdown(_rc_header + _rc_bullet_html, unsafe_allow_html=True)
                                        else:
                                            _rc_single = _html_mod.escape(_rc_expl) if _rc_expl else _html_mod.escape(reason)
                                            st.markdown(f'{_rc_header}<div style="margin:2px 0 2px 20px;">- {_rc_single}</div>', unsafe_allow_html=True)
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
                                            # Also regenerate the draft preview so user sees both at once
                                            _sg_last_draft, _ = get_last_draft_from_messages()
                                            _sg_preview_draft = None
                                            if _sg_last_draft and active_rubric_list:
                                                _sg_conv_parts = []
                                                for _cm in st.session_state.messages:
                                                    _cm_role = _cm.get("role", "")
                                                    _cm_content = _cm.get("display_content") or _cm.get("content", "")
                                                    if _cm_role in ("user", "assistant") and _cm_content:
                                                        _sg_conv_parts.append(f"[{_cm_role.upper()}]: {_cm_content[:2000]}")
                                                _sg_conv_history = "\n\n".join(_sg_conv_parts[-20:]) if _sg_conv_parts else None
                                                _sg_edit_fb = None
                                                _fb_parts = []
                                                for _fb_ac in edits_with_feedback:
                                                    _fb_text = _fb_ac.get("user_feedback", "")
                                                    if _fb_text and _fb_text.strip():
                                                        _fb_parts.append(f"- Edit: \"{_fb_ac.get('original_text', '')}\" → \"{_fb_ac.get('new_text', '')}\"\n  User feedback: {_fb_text}")
                                                if _fb_parts:
                                                    _sg_edit_fb = "\n".join(_fb_parts)
                                                _sg_regen = regenerate_draft_from_rubric_changes(
                                                    active_rubric_list, modified_rubric, _sg_last_draft,
                                                    conversation_history=_sg_conv_history,
                                                    rubric_suggestion_text=suggestion_text,
                                                    user_edit_feedback=_sg_edit_fb,
                                                )
                                                if _sg_regen and _sg_regen.get("revised_draft") and not _sg_regen.get("error"):
                                                    _sg_preview_draft = _sg_regen
                                            message["rubric_suggestion"] = {
                                                "suggestion_text": suggestion_text,
                                                "modified_rubric": modified_rubric,
                                                "edited_rubric": edited_rubric_list,
                                                "preview_draft": _sg_preview_draft,
                                                "original_draft": rr.get("original_draft", ""),
                                            }
                                            # Save suggestion + feedback to database immediately
                                            _sg_pid = st.session_state.get("current_project_id")
                                            if _sg_pid:
                                                _active_ver, _, _ = get_active_rubric()
                                                save_project_data(supabase, _sg_pid, "rubric_edit_suggestion", {
                                                    "timestamp": datetime.now().isoformat(),
                                                    "source": "chat_edit_feedback",
                                                    "rubric_version": _active_ver.get("version", "") if _active_ver else "",
                                                    "conversation_id": st.session_state.get("selected_conversation", ""),
                                                    "message_id": message_id,
                                                    "user_edits_with_feedback": [
                                                        {k: v for k, v in ef.items() if k != "_diff"}
                                                        for ef in edits_with_feedback
                                                    ],
                                                    "suggestion_text": suggestion_text,
                                                    "suggested_rubric": _rubric_list_for_json(modified_rubric),
                                                    "current_rubric": _rubric_list_for_json(edited_rubric_list),
                                                    "applied": False,
                                                })
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
                                            st.caption("Review the suggested rubric changes and draft preview below, then apply all at once.")
                                            # Show draft diff between suggestion text and rubric changes
                                            _sg_preview = suggestion_data.get("preview_draft")
                                            _sg_orig_for_diff = suggestion_data.get("original_draft", "") or rr.get("original_draft", "")
                                            if _sg_preview and _sg_preview.get("revised_draft") and _sg_orig_for_diff:
                                                st.markdown("**Draft preview:**")
                                                _sg_diff_html = _word_level_diff(_sg_orig_for_diff, _sg_preview["revised_draft"])
                                                st.markdown(f'<div style="padding:8px 12px;border:1px solid #444;border-radius:6px;line-height:1.8;">{_sg_diff_html}</div>', unsafe_allow_html=True)
                                                st.markdown("---")
                                            elif _sg_preview and _sg_preview.get("revised_draft"):
                                                st.markdown("**Draft preview:**")
                                                st.markdown(_sg_preview["revised_draft"])
                                                st.markdown("---")
                                            display_rubric_comparison(
                                                suggestion_data.get("edited_rubric", []),
                                                suggestion_data.get("modified_rubric", []),
                                                apply_context={"safe_msg_id": safe_msg_id, "message": message, "message_id": message_id},
                                            )
                                            # Revert button (reverts both draft and rubric)
                                            if st.button("↩️ Revert to Original", key=f"rr_revert_{safe_msg_id}", width="stretch"):
                                                _rr_orig_draft = rr.get('original_draft', '')
                                                rr['_decision'] = 'reverted'
                                                if _rr_orig_draft:
                                                    st.session_state.messages.append({
                                                        "role": "assistant",
                                                        "content": f"<draft>{_rr_orig_draft}</draft>\n\n*Reverted to original draft and rubric.*",
                                                        "display_content": f"<draft>{_rr_orig_draft}</draft>\n\n*Reverted to original draft and rubric.*",
                                                        "is_system_generated": True,
                                                        "message_id": f"revert_{int(time.time() * 1000000)}",
                                                    })
                                                _rr_old_rubric = rr.get('old_rubric')
                                                if _rr_old_rubric:
                                                    _rr_old_rubric_copy = copy.deepcopy(_rr_old_rubric)
                                                    st.session_state.rubric = _rr_old_rubric_copy
                                                    st.session_state.editing_criteria = _rr_old_rubric_copy
                                                    st.session_state.editing_criteria_ui_version = st.session_state.get("editing_criteria_ui_version", 0) + 1
                                                    _rr_revert_ver = message.get("rubric_version")
                                                    if _rr_revert_ver is not None:
                                                        _rr_hist = load_rubric_history()
                                                        for _rr_hi, _rr_hentry in enumerate(_rr_hist):
                                                            if _rr_hentry.get("version") == _rr_revert_ver:
                                                                st.session_state.active_rubric_idx = _rr_hi
                                                                st.session_state["rubric_version_selector"] = f"v{_rr_revert_ver}"
                                                                break
                                                _auto_save_conversation()
                                                st.rerun()

                            # --- Accept / Revert for manual Log Changes (no edit feedback) ---
                            if not message.get("rubric_suggestion"):
                                _rr_decision = rr.get('_decision')
                                if not _rr_decision:
                                    _rr_accept_col, _rr_revert_col, _ = st.columns([1, 1, 1])
                                    with _rr_accept_col:
                                        if st.button("✅ Accept Draft", key=f"rr_accept_{safe_msg_id}", width="stretch", type="primary"):
                                            rr['_decision'] = 'accepted'
                                            # Save the rubric edits as a new version
                                            _rr_new_rubric = rr.get('new_rubric')
                                            if _rr_new_rubric:
                                                _rr_new_criteria = copy.deepcopy(_rr_new_rubric)
                                                hist = load_rubric_history()
                                                _rr_new_ver = next_version_number()
                                                hist.append({"version": _rr_new_ver, "rubric": copy.deepcopy(_rr_new_criteria), "source": "log_changes_accepted", "conversation_id": st.session_state.get("selected_conversation")})
                                                save_rubric_history(hist)
                                                st.session_state.active_rubric_idx = len(hist) - 1
                                                st.session_state.rubric = _rr_new_criteria
                                                st.session_state.editing_criteria = _rr_new_criteria
                                                st.session_state.editing_criteria_ui_version = st.session_state.get("editing_criteria_ui_version", 0) + 1
                                                st.session_state["rubric_version_selector"] = f"v{_rr_new_ver}"
                                                message["rubric_version"] = _rr_new_ver
                                            # Append the revised draft as a new assistant message
                                            _rr_accepted_draft = rr.get('revised_draft', '')
                                            if _rr_accepted_draft:
                                                st.session_state.messages.append({
                                                    "role": "assistant",
                                                    "content": f"<draft>{_rr_accepted_draft}</draft>",
                                                    "display_content": f"<draft>{_rr_accepted_draft}</draft>",
                                                    "is_system_generated": True,
                                                    "message_id": f"accepted_draft_{int(time.time() * 1000000)}",
                                                })
                                            _auto_save_conversation()
                                            st.rerun()
                                    with _rr_revert_col:
                                        if st.button("↩️ Revert to Original", key=f"rr_revert_nofb_{safe_msg_id}", width="stretch"):
                                            _rr_orig_draft = rr.get('original_draft', '')
                                            rr['_decision'] = 'reverted'
                                            if _rr_orig_draft:
                                                st.session_state.messages.append({
                                                    "role": "assistant",
                                                    "content": f"<draft>{_rr_orig_draft}</draft>\n\n*Reverted to original draft and rubric.*",
                                                    "display_content": f"<draft>{_rr_orig_draft}</draft>\n\n*Reverted to original draft and rubric.*",
                                                    "is_system_generated": True,
                                                    "message_id": f"revert_{int(time.time() * 1000000)}",
                                                })
                                            _rr_old_rubric = rr.get('old_rubric')
                                            if _rr_old_rubric:
                                                _rr_old_rubric_copy = copy.deepcopy(_rr_old_rubric)
                                                st.session_state.rubric = _rr_old_rubric_copy
                                                st.session_state.editing_criteria = _rr_old_rubric_copy
                                                st.session_state.editing_criteria_ui_version = st.session_state.get("editing_criteria_ui_version", 0) + 1
                                                _rr_revert_ver = message.get("rubric_version")
                                                if _rr_revert_ver is not None:
                                                    _rr_hist = load_rubric_history()
                                                    for _rr_hi, _rr_hentry in enumerate(_rr_hist):
                                                        if _rr_hentry.get("version") == _rr_revert_ver:
                                                            st.session_state.active_rubric_idx = _rr_hi
                                                            st.session_state["rubric_version_selector"] = f"v{_rr_revert_ver}"
                                                            break
                                            _auto_save_conversation()
                                            st.rerun()
                                elif _rr_decision == 'accepted':
                                    st.success("Draft accepted.")
                                elif _rr_decision == 'reverted':
                                    st.info("Reverted to original draft and rubric.")

                        if message.get('is_probe_log'):
                            _pld = message.get("probe_log_data", {})
                            _pld_va = _pld.get("variant_a", "")
                            _pld_vb = _pld.get("variant_b", "")
                            _pld_choice = _pld.get("user_choice", "")
                            _pld_crit_name = _pld.get("criterion_name", "")
                            _pld_reason = _pld.get("reason", "")
                            _pld_dim = _pld.get("dimension_varied", "")
                            _pld_interp_a = _pld.get("interpretation_a", "")
                            _pld_interp_b = _pld.get("interpretation_b", "")

                            # --- Instruction: what this means ---
                            st.markdown(
                                "I detected that the criterion below may be **ambiguous** — it can be interpreted in more than one way, "
                                "which could lead to inconsistent drafts. I generated two draft variants, each following a "
                                "different interpretation, and asked you to pick the one that better matches your intent."
                            )

                            # --- Header: concise summary line ---
                            _pld_summary = message.get("probe_log_summary", "")
                            if _pld_summary:
                                st.info(_pld_summary)
                            else:
                                st.info(content_to_display)

                            # --- Why this criterion was flagged ---
                            if _pld_reason or _pld_dim:
                                with st.expander("Why was this criterion flagged?", expanded=False):
                                    if _pld_reason:
                                        st.markdown(f"**Reason:** {_pld_reason}")
                                    if _pld_dim:
                                        st.markdown(f"**Dimension varied:** {_pld_dim}")
                                    if _pld_interp_a:
                                        st.markdown(f"\n**Interpretation A:** {_pld_interp_a}")
                                    if _pld_interp_b:
                                        st.markdown(f"**Interpretation B:** {_pld_interp_b}")

                            # --- Probe drafts side by side ---
                            if _pld_va or _pld_vb:
                                with st.expander("Compare the two draft variants", expanded=False):
                                    _pld_col_a, _pld_col_b = st.columns(2)
                                    with _pld_col_a:
                                        _pld_a_label = "**Version A** ✅" if _pld_choice == "a" else "**Version A**"
                                        st.markdown(_pld_a_label)
                                        st.markdown(_pld_va)
                                    with _pld_col_b:
                                        _pld_b_label = "**Version B** ✅" if _pld_choice == "b" else "**Version B**"
                                        st.markdown(_pld_b_label)
                                        st.markdown(_pld_vb)

                            # --- Suggested rubric update ---
                            if _pld_choice and _pld_choice != "skip":
                                _pld_src_id = _pld.get("source_message_id")
                                _pld_src_msg = None
                                if _pld_src_id:
                                    for _m in st.session_state.messages:
                                        if _m.get("message_id") == _pld_src_id:
                                            _pld_src_msg = _m
                                            break
                                _pr = _pld_src_msg.get("probe_result", {}) if _pld_src_msg else {}
                                _pr_crit = _pr.get("criterion_name", _pld_crit_name)
                                _pr_applied = _pr.get("applied", False)
                                _pr_updated = _pr.get("updated_criterion")
                                if _pr_updated:
                                    _pr_chosen_label = "Version A" if _pld_choice == "a" else "Version B"
                                    _pr_exp_label = f"Criterion update (applied as v{_pr['applied_version']})" if _pr_applied else f"Suggested update for \"{_pr_crit}\""
                                    with st.expander(_pr_exp_label, expanded=not _pr_applied):
                                        _pr_chosen_interp = _pr.get(f"interpretation_{_pld_choice}", "")
                                        _pr_user_reason = _pr.get("user_reason", "")
                                        if _pr_chosen_interp:
                                            st.markdown(
                                                f"You preferred **{_pr_chosen_label}**, which interprets \"{_pr_crit}\" as:\n\n"
                                                f"> *{_pr_chosen_interp}*"
                                            )
                                            if _pr_user_reason:
                                                st.markdown(f"**Your reason:** {_pr_user_reason}")
                                            st.markdown("\nBased on this, the criterion was refined to better match your preference:")
                                            st.markdown("---")
                                        if _pr_applied:
                                            st.success(f"Applied as rubric v{_pr['applied_version']}")
                                        else:
                                            _pr_rb_dict, _, _ = get_active_rubric()
                                            _pr_current_list = _pr_rb_dict.get("rubric", []) if _pr_rb_dict else []
                                            _pr_current_crit = None
                                            for _c in _pr_current_list:
                                                if _c.get("name", "").lower().strip() == _pr_crit.lower().strip():
                                                    _pr_current_crit = _c
                                                    break
                                            if _pr_current_crit:
                                                st.caption("Apply the refined criterion below. Changes appear in **Rubric Configuration** in the sidebar.")
                                                display_rubric_comparison(
                                                    [_pr_current_crit],
                                                    [_pr_updated],
                                                    apply_context={"safe_msg_id": safe_msg_id, "message": _pld_src_msg, "message_id": _pld_src_id},
                                                )
                                            else:
                                                st.markdown(f"**Updated description:** {_pr_updated.get('description', '')}")
                        elif message.get('is_alignment_diagnostic') or message.get('is_ranking_checkpoint_result') or message.get('is_dp_confirmation_log') or message.get('is_criteria_classification_log'):
                            # Compact summary + collapsed details to reduce information overload
                            if message.get('is_dp_confirmation_log'):
                                _dp_log_data = message.get('dp_data', {})
                                _dp_log_count = len(_dp_log_data.get('decision_points', []))
                                _rb_log_count = len(_dp_log_data.get('rubric', []))
                                st.success(f"Decision Points confirmed: {_dp_log_count} DPs mapped to {_rb_log_count} rubric criteria.")
                                with st.expander("View full DP confirmation details", expanded=False):
                                    st.markdown(content_to_display)
                            elif message.get('is_criteria_classification_log'):
                                _cc_log_data = message.get('classification_data', {})
                                st.success(f"Criteria classified: {_cc_log_data.get('stated_count', 0)} stated, {_cc_log_data.get('real_count', 0)} real, {_cc_log_data.get('hallucinated_count', 0)} hallucinated")
                                with st.expander("View full classification details", expanded=False):
                                    st.markdown(content_to_display)
                            elif message.get('is_alignment_diagnostic'):
                                st.success("Based on your ranking, we scored each draft against your rubric criteria to identify where the rubric is working well and where it could improve.")
                                # Show ranking summary visibly (not buried in expander)
                                _diag_data = message.get("diagnostic_data", {})
                                _diag_ranking_display = _diag_data.get("ranking_display", "")
                                _diag_ranking_takeaway = _diag_data.get("ranking_takeaway", "")
                                if _diag_ranking_display:
                                    _ranking_summary = f"**Your ranking:** {_diag_ranking_display}"
                                    if _diag_ranking_takeaway:
                                        _ranking_summary += f"\n\n{_diag_ranking_takeaway}"
                                    st.markdown(_ranking_summary)
                                with st.expander("View detailed scoring breakdown", expanded=False):
                                    st.markdown(content_to_display)
                            else:
                                st.success(content_to_display)
                            # Diagnostic rubric suggestion (from alignment diagnostic)
                            if message.get('is_alignment_diagnostic'):
                                suggestion_data = message.get("rubric_suggestion")
                                _ac_pending = message.get("_ac_pending_draft", False)
                                if suggestion_data:
                                    _sg_applied = suggestion_data.get("applied", False)
                                    _sg_label = f"Rubric changes (applied as v{suggestion_data['applied_version']})" if _sg_applied else "Suggested rubric improvements"
                                    with st.expander(_sg_label, expanded=(not _sg_applied and _ac_pending)):
                                        if _sg_applied:
                                            st.success(f"Applied as rubric v{suggestion_data['applied_version']}")
                                        else:
                                            st.caption("Based on how you ranked the drafts, we identified improvements to your rubric. Review the changes below — apply what makes sense, skip what doesn't.")
                                            display_rubric_comparison(
                                                suggestion_data.get("current_rubric", suggestion_data.get("edited_rubric", [])),
                                                suggestion_data.get("updated_rubric", suggestion_data.get("modified_rubric", [])),
                                                apply_context={"safe_msg_id": safe_msg_id, "message": message, "message_id": message_id},
                                                criterion_reasons=suggestion_data.get("suggestion_reasons"),
                                            )
                                        # Show draft diff: original rubric draft vs suggested-rubric draft
                                        _sg_new_draft = suggestion_data.get("suggested_draft", "")
                                        _sg_orig_draft = suggestion_data.get("original_rubric_draft", "")
                                        _sg_ann_changes = suggestion_data.get("suggested_annotated_changes", [])
                                        if _sg_new_draft and _sg_orig_draft:
                                            st.markdown("---")
                                            st.markdown("**Draft preview with suggested rubric:**")
                                            if _sg_ann_changes:
                                                # Rich annotated display with programmatic diff + edit markers with hover reasons
                                                _sg_ann_num = len(_sg_ann_changes)
                                                st.caption(f"{_sg_ann_num} edit{'s' if _sg_ann_num != 1 else ''} from original draft — hover over markers for reasoning")
                                                st.markdown(
                                                    _annotated_diff_html(_sg_orig_draft, _sg_new_draft, _sg_ann_changes, safe_msg_id + "_sgdraft"),
                                                    unsafe_allow_html=True
                                                )
                                                with st.expander(f"Edit details ({_sg_ann_num})", expanded=False):
                                                    import html as _html_mod_sg
                                                    for _sg_i, _sg_ac in enumerate(_sg_ann_changes, 1):
                                                        _sg_reason = (_sg_ac.get('reason', '') or '').strip()
                                                        # Extract "CriterionName: explanation" — strip parenthetical rubric change detail
                                                        _sg_reason_parts = _sg_reason.split(':', 1)
                                                        if len(_sg_reason_parts) == 2:
                                                            _sg_crit_part = _sg_reason_parts[0].strip()
                                                            _sg_expl_part = _sg_reason_parts[1].strip()
                                                            # Remove parenthetical from criterion name
                                                            _sg_paren_idx = _sg_crit_part.find('(')
                                                            if _sg_paren_idx > 0:
                                                                _sg_crit_part = _sg_crit_part[:_sg_paren_idx].strip()
                                                        else:
                                                            _sg_crit_part = ""
                                                            _sg_expl_part = _sg_reason
                                                        # Split explanation on semicolons into bullet points
                                                        _sg_bullets = [s.strip() for s in _sg_expl_part.split(';') if s.strip()] if _sg_expl_part else []
                                                        _sg_header = f'<div style="margin:12px 0 4px 0;"><strong>[{_sg_i}]</strong> <strong>{_html_mod_sg.escape(_sg_crit_part)}</strong>{":" if _sg_crit_part else ""}</div>'
                                                        if len(_sg_bullets) > 1:
                                                            _sg_bullet_html = ''.join(f'<div style="margin:2px 0 2px 20px;">- {_html_mod_sg.escape(b)}</div>' for b in _sg_bullets)
                                                            st.markdown(_sg_header + _sg_bullet_html, unsafe_allow_html=True)
                                                        else:
                                                            _sg_single = _html_mod_sg.escape(_sg_expl_part) if _sg_expl_part else _html_mod_sg.escape(_sg_reason)
                                                            st.markdown(f'{_sg_header}<div style="margin:2px 0 2px 20px;">- {_sg_single}</div>', unsafe_allow_html=True)
                                            else:
                                                # Fallback: generic word-level diff
                                                _sg_diff_html = _word_level_diff(_sg_orig_draft, _sg_new_draft)
                                                st.markdown(
                                                    f'<div style="padding:12px;background:#fafafa;border:1px solid #e0e0e0;border-radius:6px;line-height:1.7;">{_sg_diff_html}</div>',
                                                    unsafe_allow_html=True
                                                )
                                                st.caption("Strikethrough = removed from original rubric draft. Green = added by suggested rubric.")
                                # If conversation-start draft is pending user decision, show Skip button
                                if _ac_pending and not _sg_applied:
                                    _fb_source_key = message.get("_ac_fallback_draft_source", "")
                                    _fb_source_labels = {"rubric": "rubric-guided", "generic": "generic (no rubric)", "preference": "preference-based (from your original preferences)"}
                                    _fb_source_label = _fb_source_labels.get(_fb_source_key, _fb_source_key)
                                    if not suggestion_data:
                                        # No suggestions generated — auto-inject the fallback draft
                                        _fb_draft = message.get("_ac_fallback_draft", "")
                                        if _fb_draft:
                                            st.session_state.messages.append({
                                                "role": "assistant",
                                                "content": f"Here is your starting draft — this is the **{_fb_source_label}** version (your top-ranked draft):\n\n<draft>\n{_fb_draft}\n</draft>",
                                                "message_id": f"ac_draft_{int(time.time() * 1000000)}",
                                            })
                                        message["_ac_pending_draft"] = False
                                        _auto_save_conversation()
                                        st.rerun()
                                    else:
                                        st.markdown("---")
                                        _skip_col, _ = st.columns([0.4, 0.6])
                                        with _skip_col:
                                            if st.button("Skip — use my preferred draft instead", key=f"ac_skip_{safe_msg_id}", width="stretch"):
                                                _fb_draft = message.get("_ac_fallback_draft", "")
                                                if _fb_draft:
                                                    st.session_state.messages.append({
                                                        "role": "assistant",
                                                        "content": f"Here is your starting draft — this is the **{_fb_source_label}** version (your top-ranked draft):\n\n<draft>\n{_fb_draft}\n</draft>",
                                                        "message_id": f"ac_draft_{int(time.time() * 1000000)}",
                                                    })
                                                message["_ac_pending_draft"] = False
                                                _auto_save_conversation()
                                                st.rerun()
                        elif message['role'] == 'assistant':
                            if message.get('rubric_revision'):
                                # Skip editable draft for rubric revision messages — revised draft shown in annotated view above
                                _rr_non_draft = re.sub(r'<draft>.*?</draft>', '', content_to_display, flags=re.DOTALL).strip()
                                if _rr_non_draft:
                                    st.markdown(_rr_non_draft, unsafe_allow_html=_dp_highlighted)
                            else:
                                # Always try original content for draft rendering (DP highlighting may corrupt <draft> tags)
                                _draft_source = message.get('content', content_to_display)
                                has_draft = render_message_with_draft(_draft_source, message_id, editable=True)
                                if not has_draft:
                                    st.markdown(content_to_display, unsafe_allow_html=_dp_highlighted)
                        else:
                            st.markdown(content_to_display, unsafe_allow_html=_dp_highlighted)
                        # Show non-preferred A/B draft inline (blind labels)
                        # Backward compat: render old A/B comparison results
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
                                st.info(f"You chose **{_abc_chosen_blind}**. {_abc_other_blind} is below.")
                                with st.expander(f"Show {_abc_other_blind}", expanded=False):
                                    st.markdown(strip_draft_tags_for_streaming(_abc_other))
                        # Probe result: no longer rendered here — moved to is_probe_log message
                        if message['role'] == 'assistant' and message.get('rubric_assessment'):
                            assessment = message['rubric_assessment']
                            draft_text = assessment.get('draft_text')
                            display_rubric_assessment(assessment, message_id, draft_text)
                # Render DPs (outside chat bubble) — _chat_msg_num already set at top of loop
                if message['role'] in ('user', 'assistant'):
                    if _dp_active and not _dp_confirmed and _chat_msg_num in _dp_by_user_msg:
                        if not _dp_intro_shown:
                            _dp_intro_shown = True
                        for _dp_item in _dp_by_user_msg[_chat_msg_num]:
                            _render_dp_card(_dp_item)
            else:
                with st.chat_message(message['role']):
                    message_id = message.get('message_id', f"{message['role']}_{idx}")
                    safe_msg_id = re.sub(r'[^a-zA-Z0-9_-]', '_', str(message_id))
                    if message['role'] == 'user':
                        content_to_display = message['content']
                    else:
                        content_to_display = message.get('display_content', message['content'])
                    # Highlight DP quotes in message text
                    _dp_highlighted = False
                    if _dp_active:
                        if message['role'] == 'user' and _chat_msg_num in _dp_by_user_msg:
                            content_to_display = _highlight_dp_quotes(content_to_display, _dp_by_user_msg[_chat_msg_num], 'after_quote', '#A5D6A7')
                            _dp_highlighted = True
                        elif message['role'] == 'assistant' and _chat_msg_num in _dp_by_asst_msg:
                            content_to_display = _highlight_dp_quotes(content_to_display, _dp_by_asst_msg[_chat_msg_num], 'before_quote', '#FFF59D')
                            _dp_highlighted = True
                    if message['role'] == 'assistant' and message.get('thinking'):
                        with st.expander("🧠 Thinking", expanded=False):
                            st.markdown(message['thinking'])
                    if message['role'] == 'assistant' and message.get('rubric_revision'):
                        rr = message['rubric_revision']
                        ann = rr.get('annotated_changes', [])
                        _num_ann2 = len(ann)
                        if rr.get('change_summary'):
                            st.markdown(
                                f"Based on your rubric changes, I made **{_num_ann2} edit{'s' if _num_ann2 != 1 else ''}** to the draft. "
                                f"Locate each edit with the numbered **[N]** markers below, and expand "
                                f"**Edits by rubric change** for the full reasoning."
                            )
                            with st.expander("View change details", expanded=False):
                                st.markdown(rr['change_summary'])
                        annotated = rr.get('revised_draft_annotated') or rr.get('revised_draft', '')
                        if annotated:
                            st.markdown("**Revised draft** (click a marker to jump to the edit):")
                            st.markdown(_safe_annotated_draft_html(annotated, ann, message_id), unsafe_allow_html=True)
                        if ann:
                            with st.expander(f"Edits by rubric change ({_num_ann2})", expanded=False):
                                import html as _html_mod
                                safe_msg_id = re.sub(r'[^a-zA-Z0-9_-]', '_', str(message_id))
                                for i, ac in enumerate(ann, 1):
                                    reason = (ac.get('reason', '') or '').strip()
                                    aid = f"rubric-edit-{safe_msg_id}-{i}"
                                    # Parse "CriterionName: explanation" format
                                    _rc_parts2 = reason.split(':', 1)
                                    if len(_rc_parts2) == 2:
                                        _rc_crit2 = _rc_parts2[0].strip()
                                        _rc_expl2 = _rc_parts2[1].strip()
                                        # Remove parenthetical from criterion name
                                        _rc_paren2 = _rc_crit2.find('(')
                                        if _rc_paren2 > 0:
                                            _rc_crit2 = _rc_crit2[:_rc_paren2].strip()
                                    else:
                                        _rc_crit2 = ""
                                        _rc_expl2 = reason
                                    # Split explanation on semicolons into bullet points
                                    _rc_bullets2 = [s.strip() for s in _rc_expl2.split(';') if s.strip()] if _rc_expl2 else []
                                    _rc_header2 = f'<div id="{aid}" style="margin:12px 0 4px 0;scroll-margin-top:20vh;"><strong>[{i}]</strong> <strong>{_html_mod.escape(_rc_crit2)}</strong>{":" if _rc_crit2 else ""}</div>'
                                    if len(_rc_bullets2) > 1:
                                        _rc_bullet_html2 = ''.join(f'<div style="margin:2px 0 2px 20px;">- {_html_mod.escape(b)}</div>' for b in _rc_bullets2)
                                        st.markdown(_rc_header2 + _rc_bullet_html2, unsafe_allow_html=True)
                                    else:
                                        _rc_single2 = _html_mod.escape(_rc_expl2) if _rc_expl2 else _html_mod.escape(reason)
                                        st.markdown(f'{_rc_header2}<div style="margin:2px 0 2px 20px;">- {_rc_single2}</div>', unsafe_allow_html=True)
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
                            if st.button("💡 Suggest how to change the rubric", key=f"rubric_suggest_btn_else_{safe_msg_id}", disabled=not _has_feedback):
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
                                        # Also regenerate the draft preview so user sees both at once
                                        _sg_last_draft2, _ = get_last_draft_from_messages()
                                        _sg_preview_draft2 = None
                                        if _sg_last_draft2 and active_rubric_list:
                                            _sg_conv_parts2 = []
                                            for _cm in st.session_state.messages:
                                                _cm_role = _cm.get("role", "")
                                                _cm_content = _cm.get("display_content") or _cm.get("content", "")
                                                if _cm_role in ("user", "assistant") and _cm_content:
                                                    _sg_conv_parts2.append(f"[{_cm_role.upper()}]: {_cm_content[:2000]}")
                                            _sg_conv_history2 = "\n\n".join(_sg_conv_parts2[-20:]) if _sg_conv_parts2 else None
                                            _sg_edit_fb2 = None
                                            _fb_parts2 = []
                                            for _fb_ac in edits_with_feedback:
                                                _fb_text = _fb_ac.get("user_feedback", "")
                                                if _fb_text and _fb_text.strip():
                                                    _fb_parts2.append(f"- Edit: \"{_fb_ac.get('original_text', '')}\" → \"{_fb_ac.get('new_text', '')}\"\n  User feedback: {_fb_text}")
                                            if _fb_parts2:
                                                _sg_edit_fb2 = "\n".join(_fb_parts2)
                                            _sg_regen2 = regenerate_draft_from_rubric_changes(
                                                active_rubric_list, modified_rubric, _sg_last_draft2,
                                                conversation_history=_sg_conv_history2,
                                                rubric_suggestion_text=suggestion_text,
                                                user_edit_feedback=_sg_edit_fb2,
                                            )
                                            if _sg_regen2 and _sg_regen2.get("revised_draft") and not _sg_regen2.get("error"):
                                                _sg_preview_draft2 = _sg_regen2
                                        message["rubric_suggestion"] = {
                                            "suggestion_text": suggestion_text,
                                            "modified_rubric": modified_rubric,
                                            "edited_rubric": edited_rubric_list,
                                            "preview_draft": _sg_preview_draft2,
                                            "original_draft": rr.get("original_draft", ""),
                                        }
                                        # Save suggestion + feedback to database immediately
                                        _sg_pid2 = st.session_state.get("current_project_id")
                                        if _sg_pid2:
                                            _active_ver2, _, _ = get_active_rubric()
                                            save_project_data(supabase, _sg_pid2, "rubric_edit_suggestion", {
                                                "timestamp": datetime.now().isoformat(),
                                                "source": "chat_edit_feedback",
                                                "rubric_version": _active_ver2.get("version", "") if _active_ver2 else "",
                                                "conversation_id": st.session_state.get("selected_conversation", ""),
                                                "message_id": message_id,
                                                "user_edits_with_feedback": [
                                                    {k: v for k, v in ef.items() if k != "_diff"}
                                                    for ef in edits_with_feedback
                                                ],
                                                "suggestion_text": suggestion_text,
                                                "suggested_rubric": _rubric_list_for_json(modified_rubric),
                                                "current_rubric": _rubric_list_for_json(edited_rubric_list),
                                                "applied": False,
                                            })
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
                                        st.caption("Review the suggested rubric changes and draft preview below, then apply all at once.")
                                        # Show draft diff between suggestion text and rubric changes
                                        _sg_preview2 = suggestion_data.get("preview_draft")
                                        _sg_orig_for_diff2 = suggestion_data.get("original_draft", "") or rr.get("original_draft", "")
                                        if _sg_preview2 and _sg_preview2.get("revised_draft") and _sg_orig_for_diff2:
                                            st.markdown("**Draft preview:**")
                                            _sg_diff_html2 = _word_level_diff(_sg_orig_for_diff2, _sg_preview2["revised_draft"])
                                            st.markdown(f'<div style="padding:8px 12px;border:1px solid #444;border-radius:6px;line-height:1.8;">{_sg_diff_html2}</div>', unsafe_allow_html=True)
                                            st.markdown("---")
                                        elif _sg_preview2 and _sg_preview2.get("revised_draft"):
                                            st.markdown("**Draft preview:**")
                                            st.markdown(_sg_preview2["revised_draft"])
                                            st.markdown("---")
                                        display_rubric_comparison(
                                            suggestion_data.get("edited_rubric", []),
                                            suggestion_data.get("modified_rubric", []),
                                            apply_context={"safe_msg_id": safe_msg_id, "message": message, "message_id": message_id},
                                        )
                                        # Revert button (reverts both draft and rubric)
                                        if st.button("↩️ Revert to Original", key=f"rr_revert_else_{safe_msg_id}", width="stretch"):
                                            _rr_orig_draft2 = rr.get('original_draft', '')
                                            rr['_decision'] = 'reverted'
                                            if _rr_orig_draft2:
                                                st.session_state.messages.append({
                                                    "role": "assistant",
                                                    "content": f"<draft>{_rr_orig_draft2}</draft>\n\n*Reverted to original draft and rubric.*",
                                                    "display_content": f"<draft>{_rr_orig_draft2}</draft>\n\n*Reverted to original draft and rubric.*",
                                                    "is_system_generated": True,
                                                    "message_id": f"revert_{int(time.time() * 1000000)}",
                                                })
                                            _rr_old_rubric2 = rr.get('old_rubric')
                                            if _rr_old_rubric2:
                                                _rr_old_rubric_copy2 = copy.deepcopy(_rr_old_rubric2)
                                                st.session_state.rubric = _rr_old_rubric_copy2
                                                st.session_state.editing_criteria = _rr_old_rubric_copy2
                                                st.session_state.editing_criteria_ui_version = st.session_state.get("editing_criteria_ui_version", 0) + 1
                                                _rr_revert_ver2 = message.get("rubric_version")
                                                if _rr_revert_ver2 is not None:
                                                    _rr_hist2 = load_rubric_history()
                                                    for _rr_hi2, _rr_hentry2 in enumerate(_rr_hist2):
                                                        if _rr_hentry2.get("version") == _rr_revert_ver2:
                                                            st.session_state.active_rubric_idx = _rr_hi2
                                                            st.session_state["rubric_version_selector"] = f"v{_rr_revert_ver2}"
                                                            break
                                            _auto_save_conversation()
                                            st.rerun()

                            # --- Accept / Revert for manual Log Changes (no edit feedback) ---
                            if not message.get("rubric_suggestion"):
                                _rr_decision2 = rr.get('_decision')
                                if not _rr_decision2:
                                    _rr_accept_col2, _rr_revert_col2, _2 = st.columns([1, 1, 1])
                                    with _rr_accept_col2:
                                        if st.button("✅ Accept Draft", key=f"rr_accept_else_{safe_msg_id}", width="stretch", type="primary"):
                                            rr['_decision'] = 'accepted'
                                            # Save the rubric edits as a new version
                                            _rr_new_rubric2 = rr.get('new_rubric')
                                            if _rr_new_rubric2:
                                                _rr_new_criteria2 = copy.deepcopy(_rr_new_rubric2)
                                                hist = load_rubric_history()
                                                _rr_new_ver2 = next_version_number()
                                                hist.append({"version": _rr_new_ver2, "rubric": copy.deepcopy(_rr_new_criteria2), "source": "log_changes_accepted", "conversation_id": st.session_state.get("selected_conversation")})
                                                save_rubric_history(hist)
                                                st.session_state.active_rubric_idx = len(hist) - 1
                                                st.session_state.rubric = _rr_new_criteria2
                                                st.session_state.editing_criteria = _rr_new_criteria2
                                                st.session_state.editing_criteria_ui_version = st.session_state.get("editing_criteria_ui_version", 0) + 1
                                                st.session_state["rubric_version_selector"] = f"v{_rr_new_ver2}"
                                                message["rubric_version"] = _rr_new_ver2
                                            # Append the revised draft as a new assistant message
                                            _rr_accepted_draft2 = rr.get('revised_draft', '')
                                            if _rr_accepted_draft2:
                                                st.session_state.messages.append({
                                                    "role": "assistant",
                                                    "content": f"<draft>{_rr_accepted_draft2}</draft>",
                                                    "display_content": f"<draft>{_rr_accepted_draft2}</draft>",
                                                    "is_system_generated": True,
                                                    "message_id": f"accepted_draft_{int(time.time() * 1000000)}",
                                                })
                                            _auto_save_conversation()
                                            st.rerun()
                                    with _rr_revert_col2:
                                        if st.button("↩️ Revert to Original", key=f"rr_revert_nofb_else_{safe_msg_id}", width="stretch"):
                                            _rr_orig_draft2 = rr.get('original_draft', '')
                                            rr['_decision'] = 'reverted'
                                            if _rr_orig_draft2:
                                                st.session_state.messages.append({
                                                    "role": "assistant",
                                                    "content": f"<draft>{_rr_orig_draft2}</draft>\n\n*Reverted to original draft and rubric.*",
                                                    "display_content": f"<draft>{_rr_orig_draft2}</draft>\n\n*Reverted to original draft and rubric.*",
                                                    "is_system_generated": True,
                                                    "message_id": f"revert_{int(time.time() * 1000000)}",
                                                })
                                            _rr_old_rubric2 = rr.get('old_rubric')
                                            if _rr_old_rubric2:
                                                _rr_old_rubric_copy2 = copy.deepcopy(_rr_old_rubric2)
                                                st.session_state.rubric = _rr_old_rubric_copy2
                                                st.session_state.editing_criteria = _rr_old_rubric_copy2
                                                st.session_state.editing_criteria_ui_version = st.session_state.get("editing_criteria_ui_version", 0) + 1
                                                _rr_revert_ver2 = message.get("rubric_version")
                                                if _rr_revert_ver2 is not None:
                                                    _rr_hist2 = load_rubric_history()
                                                    for _rr_hi2, _rr_hentry2 in enumerate(_rr_hist2):
                                                        if _rr_hentry2.get("version") == _rr_revert_ver2:
                                                            st.session_state.active_rubric_idx = _rr_hi2
                                                            st.session_state["rubric_version_selector"] = f"v{_rr_revert_ver2}"
                                                            break
                                            _auto_save_conversation()
                                            st.rerun()
                                elif _rr_decision2 == 'accepted':
                                    st.success("Draft accepted.")
                                elif _rr_decision2 == 'reverted':
                                    st.info("Reverted to original draft and rubric.")

                    if message.get('is_probe_log'):
                        _pld2 = message.get("probe_log_data", {})
                        _pld2_va = _pld2.get("variant_a", "")
                        _pld2_vb = _pld2.get("variant_b", "")
                        _pld2_choice = _pld2.get("user_choice", "")
                        _pld2_crit_name = _pld2.get("criterion_name", "")
                        _pld2_reason = _pld2.get("reason", "")
                        _pld2_dim = _pld2.get("dimension_varied", "")
                        _pld2_interp_a = _pld2.get("interpretation_a", "")
                        _pld2_interp_b = _pld2.get("interpretation_b", "")

                        # --- Instruction: what this means ---
                        st.markdown(
                            "I detected that the criterion above may be **ambiguous** — it can be read in more than one way, "
                            "which could lead to inconsistent drafts. I generated two draft variants, each following a "
                            "different interpretation, and asked you to pick the one that better matches your intent."
                        )

                        # --- Header: concise summary line ---
                        _pld2_summary = message.get("probe_log_summary", "")
                        if _pld2_summary:
                            st.info(_pld2_summary)
                        else:
                            st.info(content_to_display)

                        # --- Why this criterion was flagged ---
                        if _pld2_reason or _pld2_dim:
                            with st.expander("Why was this criterion flagged?", expanded=False):
                                if _pld2_reason:
                                    st.markdown(f"**Reason:** {_pld2_reason}")
                                if _pld2_dim:
                                    st.markdown(f"**Dimension varied:** {_pld2_dim}")
                                if _pld2_interp_a:
                                    st.markdown(f"\n**Interpretation A:** {_pld2_interp_a}")
                                if _pld2_interp_b:
                                    st.markdown(f"**Interpretation B:** {_pld2_interp_b}")

                        # --- Probe drafts side by side ---
                        if _pld2_va or _pld2_vb:
                            with st.expander("Compare the two draft variants", expanded=False):
                                _pld2_col_a, _pld2_col_b = st.columns(2)
                                with _pld2_col_a:
                                    _pld2_a_label = "**Version A** ✅" if _pld2_choice == "a" else "**Version A**"
                                    st.markdown(_pld2_a_label)
                                    st.markdown(_pld2_va)
                                with _pld2_col_b:
                                    _pld2_b_label = "**Version B** ✅" if _pld2_choice == "b" else "**Version B**"
                                    st.markdown(_pld2_b_label)
                                    st.markdown(_pld2_vb)

                        # --- Suggested rubric update ---
                        if _pld2_choice and _pld2_choice != "skip":
                            _pld2_src_id = _pld2.get("source_message_id")
                            _pld2_src_msg = None
                            if _pld2_src_id:
                                for _m in st.session_state.messages:
                                    if _m.get("message_id") == _pld2_src_id:
                                        _pld2_src_msg = _m
                                        break
                            _pr2 = _pld2_src_msg.get("probe_result", {}) if _pld2_src_msg else {}
                            _pr2_crit = _pr2.get("criterion_name", _pld2_crit_name)
                            _pr2_applied = _pr2.get("applied", False)
                            _pr2_updated = _pr2.get("updated_criterion")
                            if _pr2_updated:
                                _pr2_chosen_label = "Version A" if _pld2_choice == "a" else "Version B"
                                _pr2_exp_label = f"Criterion update (applied as v{_pr2['applied_version']})" if _pr2_applied else f"Suggested update for \"{_pr2_crit}\""
                                with st.expander(_pr2_exp_label, expanded=not _pr2_applied):
                                    _pr2_chosen_interp = _pr2.get(f"interpretation_{_pld2_choice}", "")
                                    _pr2_user_reason = _pr2.get("user_reason", "")
                                    if _pr2_chosen_interp:
                                        st.markdown(
                                            f"You preferred **{_pr2_chosen_label}**, which interprets \"{_pr2_crit}\" as:\n\n"
                                            f"> *{_pr2_chosen_interp}*"
                                        )
                                        if _pr2_user_reason:
                                            st.markdown(f"**Your reason:** {_pr2_user_reason}")
                                        st.markdown("\nBased on this, the criterion was refined to better match your preference:")
                                        st.markdown("---")
                                    if _pr2_applied:
                                        st.success(f"Applied as rubric v{_pr2['applied_version']}")
                                    else:
                                        _pr2_rb_dict, _, _ = get_active_rubric()
                                        _pr2_current_list = _pr2_rb_dict.get("rubric", []) if _pr2_rb_dict else []
                                        _pr2_current_crit = None
                                        for _c in _pr2_current_list:
                                            if _c.get("name", "").lower().strip() == _pr2_crit.lower().strip():
                                                _pr2_current_crit = _c
                                                break
                                        if _pr2_current_crit:
                                            st.caption("Apply the refined criterion below. Changes appear in **Rubric Configuration** in the sidebar.")
                                            display_rubric_comparison(
                                                [_pr2_current_crit],
                                                [_pr2_updated],
                                                apply_context={"safe_msg_id": safe_msg_id, "message": _pld2_src_msg, "message_id": _pld2_src_id},
                                            )
                                        else:
                                            st.markdown(f"**Updated description:** {_pr2_updated.get('description', '')}")
                    elif message.get('is_alignment_diagnostic') or message.get('is_ranking_checkpoint_result') or message.get('is_dp_confirmation_log') or message.get('is_criteria_classification_log'):
                        # Compact summary + collapsed details to reduce information overload
                        if message.get('is_dp_confirmation_log'):
                            _dp_log_data2 = message.get('dp_data', {})
                            _dp_log_count2 = len(_dp_log_data2.get('decision_points', []))
                            _rb_log_count2 = len(_dp_log_data2.get('rubric', []))
                            st.success(f"Decision Points confirmed: {_dp_log_count2} DPs mapped to {_rb_log_count2} rubric criteria.")
                            with st.expander("View full DP confirmation details", expanded=False):
                                st.markdown(content_to_display)
                        elif message.get('is_criteria_classification_log'):
                            _cc_log_data2 = message.get('classification_data', {})
                            st.success(f"Criteria classified: {_cc_log_data2.get('stated_count', 0)} stated, {_cc_log_data2.get('real_count', 0)} real, {_cc_log_data2.get('hallucinated_count', 0)} hallucinated")
                            with st.expander("View full classification details", expanded=False):
                                st.markdown(content_to_display)
                        elif message.get('is_alignment_diagnostic'):
                            st.success("Based on your ranking, we scored each draft against your rubric criteria to identify where the rubric is working well and where it could improve.")
                            # Show ranking summary visibly (not buried in expander)
                            _diag_data2 = message.get("diagnostic_data", {})
                            _diag_ranking_display2 = _diag_data2.get("ranking_display", "")
                            _diag_ranking_takeaway2 = _diag_data2.get("ranking_takeaway", "")
                            if _diag_ranking_display2:
                                _ranking_summary2 = f"**Your ranking:** {_diag_ranking_display2}"
                                if _diag_ranking_takeaway2:
                                    _ranking_summary2 += f"\n\n{_diag_ranking_takeaway2}"
                                st.markdown(_ranking_summary2)
                            with st.expander("View detailed scoring breakdown", expanded=False):
                                st.markdown(content_to_display)
                        else:
                            st.success(content_to_display)
                        # Diagnostic rubric suggestion (from alignment diagnostic)
                        if message.get('is_alignment_diagnostic'):
                            suggestion_data = message.get("rubric_suggestion")
                            _ac_pending = message.get("_ac_pending_draft", False)
                            if suggestion_data:
                                _sg_applied = suggestion_data.get("applied", False)
                                _sg_label = f"Rubric changes (applied as v{suggestion_data['applied_version']})" if _sg_applied else "Suggested rubric improvements"
                                with st.expander(_sg_label, expanded=(not _sg_applied and _ac_pending)):
                                    if _sg_applied:
                                        st.success(f"Applied as rubric v{suggestion_data['applied_version']}")
                                    else:
                                        st.caption("Based on how you ranked the drafts, we identified improvements to your rubric. Review the changes below — apply what makes sense, skip what doesn't.")
                                        display_rubric_comparison(
                                            suggestion_data.get("current_rubric", suggestion_data.get("edited_rubric", [])),
                                            suggestion_data.get("updated_rubric", suggestion_data.get("modified_rubric", [])),
                                            apply_context={"safe_msg_id": safe_msg_id, "message": message, "message_id": message_id},
                                            criterion_reasons=suggestion_data.get("suggestion_reasons"),
                                        )
                                    # Show draft diff: original rubric draft vs suggested-rubric draft
                                    _sg_new_draft = suggestion_data.get("suggested_draft", "")
                                    _sg_orig_draft = suggestion_data.get("original_rubric_draft", "")
                                    _sg_ann_changes = suggestion_data.get("suggested_annotated_changes", [])
                                    if _sg_new_draft and _sg_orig_draft:
                                        st.markdown("---")
                                        st.markdown("**Draft preview with suggested rubric:**")
                                        if _sg_ann_changes:
                                            # Rich annotated display with programmatic diff + edit markers with hover reasons
                                            _sg_ann_num = len(_sg_ann_changes)
                                            st.caption(f"{_sg_ann_num} edit{'s' if _sg_ann_num != 1 else ''} from original draft — hover over markers for reasoning")
                                            st.markdown(
                                                _annotated_diff_html(_sg_orig_draft, _sg_new_draft, _sg_ann_changes, safe_msg_id + "_sgdraft"),
                                                unsafe_allow_html=True
                                            )
                                            with st.expander(f"Edit details ({_sg_ann_num})", expanded=False):
                                                import html as _html_mod_sg2
                                                for _sg_i2, _sg_ac2 in enumerate(_sg_ann_changes, 1):
                                                    _sg_reason2 = (_sg_ac2.get('reason', '') or '').strip()
                                                    # Extract "CriterionName: explanation" — strip parenthetical rubric change detail
                                                    _sg_reason_parts2 = _sg_reason2.split(':', 1)
                                                    if len(_sg_reason_parts2) == 2:
                                                        _sg_crit_part2 = _sg_reason_parts2[0].strip()
                                                        _sg_expl_part2 = _sg_reason_parts2[1].strip()
                                                        # Remove parenthetical from criterion name
                                                        _sg_paren_idx2 = _sg_crit_part2.find('(')
                                                        if _sg_paren_idx2 > 0:
                                                            _sg_crit_part2 = _sg_crit_part2[:_sg_paren_idx2].strip()
                                                    else:
                                                        _sg_crit_part2 = ""
                                                        _sg_expl_part2 = _sg_reason2
                                                    # Split explanation on semicolons into bullet points
                                                    _sg_bullets2 = [s.strip() for s in _sg_expl_part2.split(';') if s.strip()] if _sg_expl_part2 else []
                                                    _sg_header2 = f'<div style="margin:12px 0 4px 0;"><strong>[{_sg_i2}]</strong> <strong>{_html_mod_sg2.escape(_sg_crit_part2)}</strong>{":" if _sg_crit_part2 else ""}</div>'
                                                    if len(_sg_bullets2) > 1:
                                                        _sg_bullet_html2 = ''.join(f'<div style="margin:2px 0 2px 20px;">- {_html_mod_sg2.escape(b)}</div>' for b in _sg_bullets2)
                                                        st.markdown(_sg_header2 + _sg_bullet_html2, unsafe_allow_html=True)
                                                    else:
                                                        _sg_single2 = _html_mod_sg2.escape(_sg_expl_part2) if _sg_expl_part2 else _html_mod_sg2.escape(_sg_reason2)
                                                        st.markdown(f'{_sg_header2}<div style="margin:2px 0 2px 20px;">- {_sg_single2}</div>', unsafe_allow_html=True)
                                        else:
                                            # Fallback: generic word-level diff
                                            _sg_diff_html = _word_level_diff(_sg_orig_draft, _sg_new_draft)
                                            st.markdown(
                                                f'<div style="padding:12px;background:#fafafa;border:1px solid #e0e0e0;border-radius:6px;line-height:1.7;">{_sg_diff_html}</div>',
                                                unsafe_allow_html=True
                                            )
                                            st.caption("Strikethrough = removed from original rubric draft. Green = added by suggested rubric.")
                            # If conversation-start draft is pending user decision, show Skip button
                            if _ac_pending and not (suggestion_data and suggestion_data.get("applied", False)):
                                _fb_source_key = message.get("_ac_fallback_draft_source", "")
                                _fb_source_labels = {"rubric": "rubric-guided", "generic": "generic (no rubric)", "preference": "preference-based (from your original preferences)"}
                                _fb_source_label = _fb_source_labels.get(_fb_source_key, _fb_source_key)
                                if not suggestion_data:
                                    # No suggestions generated — auto-inject the fallback draft
                                    _fb_draft = message.get("_ac_fallback_draft", "")
                                    if _fb_draft:
                                        st.session_state.messages.append({
                                            "role": "assistant",
                                            "content": f"Here is your starting draft — this is the **{_fb_source_label}** version (your top-ranked draft):\n\n<draft>\n{_fb_draft}\n</draft>",
                                            "message_id": f"ac_draft_{int(time.time() * 1000000)}",
                                        })
                                    message["_ac_pending_draft"] = False
                                    _auto_save_conversation()
                                    st.rerun()
                                else:
                                    st.markdown("---")
                                    _skip_col, _ = st.columns([0.4, 0.6])
                                    with _skip_col:
                                        if st.button("Skip — use my preferred draft instead", key=f"ac_skip_{safe_msg_id}", width="stretch"):
                                            _fb_draft = message.get("_ac_fallback_draft", "")
                                            if _fb_draft:
                                                st.session_state.messages.append({
                                                    "role": "assistant",
                                                    "content": f"Here is your starting draft — this is the **{_fb_source_label}** version (your top-ranked draft):\n\n<draft>\n{_fb_draft}\n</draft>",
                                                    "message_id": f"ac_draft_{int(time.time() * 1000000)}",
                                                })
                                            message["_ac_pending_draft"] = False
                                            _auto_save_conversation()
                                            st.rerun()
                    elif message['role'] == 'assistant':
                        if message.get('rubric_revision'):
                            # Skip editable draft for rubric revision messages — revised draft shown in annotated view above
                            _rr_non_draft2 = re.sub(r'<draft>.*?</draft>', '', content_to_display, flags=re.DOTALL).strip()
                            if _rr_non_draft2:
                                st.markdown(_rr_non_draft2, unsafe_allow_html=_dp_highlighted)
                        else:
                            # Always try original content for draft rendering (DP highlighting may corrupt <draft> tags)
                            _draft_source2 = message.get('content', content_to_display)
                            has_draft = render_message_with_draft(_draft_source2, message_id, editable=True)
                            if not has_draft:
                                st.markdown(content_to_display, unsafe_allow_html=_dp_highlighted)
                    else:
                        st.markdown(content_to_display, unsafe_allow_html=_dp_highlighted)
                    # Backward compat: render old A/B comparison results
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
                    # Probe result: no longer rendered here — moved to is_probe_log message
                    if message['role'] == 'assistant' and message.get('rubric_assessment'):
                        assessment = message['rubric_assessment']
                        draft_text = assessment.get('draft_text')
                        display_rubric_assessment(assessment, message_id, draft_text)
                # Render DPs (outside chat bubble) — _chat_msg_num already set at top of loop
                if message['role'] in ('user', 'assistant'):
                    if _dp_active and not _dp_confirmed and _chat_msg_num in _dp_by_user_msg:
                        if not _dp_intro_shown:
                            _dp_intro_shown = True
                        for _dp_item in _dp_by_user_msg[_chat_msg_num]:
                            _render_dp_card(_dp_item)

    # --- Rubric Criteria Classification (shown before DP review) ---
    if _dp_has_review and _cc_pending:
        _cc_llm = st.session_state.chat_criteria_llm_classification
        _cc_user = st.session_state.chat_criteria_user_classifications
        if _cc_llm and _cc_user:
            st.divider()

            # st.markdown("**Rubric Criteria Classification**")

            st.markdown(
                "With your new rubric, let's make sure we're aligned on what it should capture. "
                "For each criterion below, classify whether it's **Stated** (you mentioned it), **Real** (you care about it but didn't mention it), "
                "or **Hallucinated** (the model made it up). \n\nFor non-hallucinated criteria, use the **Importance** number to rank how much you care about each one — "
                "assign **1** to the criterion that matters most to your writing, **2** to the next, and so on. Each rank must be unique."
            )

            # Show user's cold-start preferences prominently for reference
            _cs_ref_text = st.session_state.get("infer_coldstart_text", "").strip()
            if _cs_ref_text:
                st.info(f"**Your writing preferences** (what you described at the start):\n\n{_cs_ref_text}")

            # Build a single list of all criteria, sorted by rubric priority
            _cc_all_names = list(_cc_user.keys())

            st.markdown(
                "We compared each rubric criterion against the writing preferences you described above — please review these below."
            )

            # Build lookup for LLM reasoning per criterion
            _cc_reasoning = {}
            for _cc_comp in _cc_llm.get("criteria_comparison", []):
                _cc_reasoning[_cc_comp.get("criterion_name", "")] = _cc_comp.get("match_reasoning", "")

            # Initialize hallucination reasons in session state if needed
            if 'chat_criteria_hallucination_reasons' not in st.session_state:
                st.session_state.chat_criteria_hallucination_reasons = {}

            _cc_total_criteria = len(_cc_all_names)

            # Build rubric priority lookup from current rubric
            _cc_rubric_priorities = {}
            for _rc in (st.session_state.get("rubric") or []):
                _rc_name = _rc.get("name", "")
                _rc_pri = _rc.get("priority")
                if _rc_name and _rc_pri is not None:
                    _cc_rubric_priorities[_rc_name] = int(_rc_pri)

            # Always reset importance ranks from rubric priorities (normalized)
            # User edits are preserved via widget keys, but the initial values
            # must always reflect the current rubric's normalized priorities.
            _cc_existing_ranks = {}
            _cc_used_ranks = set()
            for _cn in _cc_all_names:
                if _cn in _cc_rubric_priorities:
                    _candidate_rank = _cc_rubric_priorities[_cn]
                    if _candidate_rank not in _cc_used_ranks:
                        _cc_existing_ranks[_cn] = _candidate_rank
                        _cc_used_ranks.add(_candidate_rank)
            # Fill any remaining unranked criteria with next available ranks
            _cc_next_rank = 1
            for _cn in _cc_all_names:
                if _cn not in _cc_existing_ranks:
                    while _cc_next_rank in _cc_used_ranks:
                        _cc_next_rank += 1
                    _cc_existing_ranks[_cn] = _cc_next_rank
                    _cc_used_ranks.add(_cc_next_rank)
                    _cc_next_rank += 1
            st.session_state.chat_criteria_importance_ranks = _cc_existing_ranks

            # Sort all criteria by their importance rank (priority order)
            _cc_sorted_names = sorted(_cc_all_names, key=lambda n: _cc_existing_ranks.get(n, 999))

            for _cr_name in _cc_sorted_names:
                _cr_current = _cc_user.get(_cr_name, "real")
                _cr_default_idx = {"stated": 0, "real": 1, "hallucinated": 2}.get(_cr_current, 1)
                _cr_col_class, _cr_col_rank = st.columns([3, 1])
                with _cr_col_class:
                    _cr_choice = st.selectbox(
                        _cr_name,
                        ["Stated", "Real", "Hallucinated"],
                        index=_cr_default_idx,
                        key=f"cc_chip_{_cr_name}",
                        help="Stated = you mentioned this | Real = you care but didn't mention | Hallucinated = doesn't reflect your preferences"
                    )
                with _cr_col_rank:
                    if _cr_choice != "Hallucinated":
                        _cr_cur_rank = _cc_existing_ranks.get(_cr_name, 1)
                        _cr_new_rank = st.number_input(
                            "Importance",
                            min_value=1,
                            max_value=_cc_total_criteria,
                            value=min(_cr_cur_rank, _cc_total_criteria),
                            step=1,
                            key=f"cc_rank_{_cr_name}",
                        )
                        st.session_state.chat_criteria_importance_ranks[_cr_name] = _cr_new_rank
                _cc_user[_cr_name] = _cr_choice.lower()
                if _cr_choice == "Hallucinated":
                    _cr_reason_text = _cc_reasoning.get(_cr_name, "")
                    if _cr_reason_text:
                        st.caption(f"*Why we inferred this:* {_cr_reason_text}")
                    _cr_existing_reason = st.session_state.get("chat_criteria_hallucination_reasons", {}).get(_cr_name, "")
                    _cr_halluc_reason = st.text_input(
                        f"Why doesn't \"{_cr_name}\" reflect your preferences?",
                        value=_cr_existing_reason,
                        key=f"cc_halluc_reason_{_cr_name}",
                        placeholder="e.g., I never cared about this, the model assumed it from context"
                    )
                    st.session_state.chat_criteria_hallucination_reasons[_cr_name] = _cr_halluc_reason

            st.session_state.chat_criteria_user_classifications = _cc_user

            # Check for duplicate importance ranks
            _cc_all_ranks = st.session_state.get("chat_criteria_importance_ranks", {})
            _cc_non_halluc_ranks = {name: rank for name, rank in _cc_all_ranks.items() if _cc_user.get(name) != "hallucinated"}
            _cc_rank_values = list(_cc_non_halluc_ranks.values())
            _cc_has_duplicate_ranks = len(_cc_rank_values) != len(set(_cc_rank_values))

            if _cc_has_duplicate_ranks:
                # Find which ranks are duplicated
                from collections import Counter
                _cc_rank_counts = Counter(_cc_rank_values)
                _cc_dup_ranks = sorted([r for r, c in _cc_rank_counts.items() if c > 1])
                st.warning(f"Importance ranks must be unique. Duplicate rank(s): {', '.join(str(r) for r in _cc_dup_ranks)}")

            # Confirm button
            if st.button("Confirm Criteria Classifications", type="primary", width="stretch", key="chat_confirm_criteria_pre", disabled=_cc_has_duplicate_ranks):
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

                # Get importance ranking from inline rank inputs
                _cc_inline_ranks = st.session_state.get("chat_criteria_importance_ranks", {})
                # Only include non-hallucinated criteria, sorted by rank
                _cc_rank_map = {name: _cc_inline_ranks.get(name, 999) for name, cat in _cc_final.items() if cat != "hallucinated"}
                _cc_importance = sorted(_cc_rank_map.keys(), key=lambda n: _cc_rank_map[n])

                # Build LLM original classification map for before/after comparison
                _cc_llm_orig = {}
                for _cc_comp in _cc_llm_data.get("criteria_comparison", []):
                    _cc_orig_name = _cc_comp.get("criterion_name", "")
                    _cc_orig_status = _cc_comp.get("status", "unstated")
                    # Normalize: LLM uses "stated"/"unstated", map unstated → "real" for comparison
                    _cc_llm_orig[_cc_orig_name] = "stated" if _cc_orig_status == "stated" else "real"

                # Detect which classifications the user changed
                _cc_user_changes = {}
                for _cc_cname, _cc_user_cat in _cc_final.items():
                    _cc_llm_cat = _cc_llm_orig.get(_cc_cname, "real")
                    if _cc_user_cat != _cc_llm_cat:
                        _cc_user_changes[_cc_cname] = {"from": _cc_llm_cat, "to": _cc_user_cat}

                # Build log message
                _cc_halluc_reasons = st.session_state.get("chat_criteria_hallucination_reasons", {})
                _cc_log_lines = [
                    f"**Criteria classifications confirmed**: {_cc_stated_count} stated, {_cc_real_count} real, {_cc_hallucinated_count} hallucinated.",
                    f"Rubric precision: {_cc_precision:.0%}",
                    ""
                ]
                if _cc_user_changes:
                    _cc_log_lines.append(f"**You changed {len(_cc_user_changes)} classification(s):**")
                    for _cc_ch_name, _cc_ch in _cc_user_changes.items():
                        _from_label = {"stated": "Stated", "real": "Real", "hallucinated": "Hallucinated"}.get(_cc_ch["from"], _cc_ch["from"])
                        _to_label = {"stated": "Stated", "real": "Real", "hallucinated": "Hallucinated"}.get(_cc_ch["to"], _cc_ch["to"])
                        _cc_log_lines.append(f"- **{_cc_ch_name}**: {_from_label} → {_to_label}")
                    _cc_log_lines.append("")
                # Show criteria ordered by importance rank
                _cc_ordered = sorted(_cc_final.items(), key=lambda x: _cc_rank_map.get(x[0], 999))
                for _cc_cname, _cc_cat in _cc_ordered:
                    _cc_icon = {"stated": "✓", "real": "◉", "hallucinated": "✗"}.get(_cc_cat, "?")
                    _cc_label = {"stated": "Stated", "real": "Real", "hallucinated": "Hallucinated"}.get(_cc_cat, _cc_cat)
                    _cc_rank = _cc_rank_map.get(_cc_cname)
                    _cc_rank_str = f"#{_cc_rank}" if _cc_rank else "—"
                    _cc_line = f"- {_cc_rank_str} {_cc_icon} **{_cc_cname}**: {_cc_label}"
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
                        "llm_original_classifications": copy.deepcopy(_cc_llm_orig),
                        "user_changes": copy.deepcopy(_cc_user_changes),
                        "hallucination_reasons": copy.deepcopy(_cc_halluc_reasons),
                        "importance_ranking": list(_cc_importance),
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
                _auto_save_conversation()

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
                            "iteration": _cc_rb_ver if isinstance(_cc_rb_ver, int) else 1,
                            "classifications": copy.deepcopy(_cc_final),
                            "hallucination_reasons": copy.deepcopy(_cc_halluc_reasons),
                            "importance_ranking": list(_cc_importance),
                            "llm_classification_summary": _cc_llm_data.get("summary", {}),
                            "llm_user_agreement": _cc_agreement_rate,
                            "stated_count": _cc_stated_count,
                            "real_count": _cc_real_count,
                            "hallucinated_count": _cc_hallucinated_count,
                            "n_stated": _cc_stated_count,
                            "n_real": _cc_real_count,
                            "n_hallucinated": _cc_hallucinated_count,
                            "n_criteria": _cc_stated_count + _cc_real_count + _cc_hallucinated_count,
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
                    "importance_ranking": list(_cc_importance),
                }
                st.session_state.chat_classification_feedback = _cc_feedback_for_dps

                # Check if rubric actually needs updating:
                # 1. Hallucinated criteria removed, OR
                # 2. User changed the priority/importance ordering
                _cleanup_rb_dict, _, _ = get_active_rubric()
                _cleanup_criteria = list(st.session_state.editing_criteria or [])
                _cleanup_halluc_names = {name.lower().strip() for name, cat in _cc_final.items() if cat == "hallucinated"}

                # Filter out hallucinated criteria
                _cleaned_criteria = [c for c in _cleanup_criteria if c.get("name", "").lower().strip() not in _cleanup_halluc_names]

                # Check if priority order changed
                _old_priority_order = [c.get("name", "").lower().strip() for c in _cleaned_criteria]
                if _cc_importance:
                    _importance_order = {name.lower().strip(): idx for idx, name in enumerate(_cc_importance)}
                    _cleaned_criteria.sort(key=lambda c: _importance_order.get(c.get("name", "").lower().strip(), 999))
                    for _pi, _pc in enumerate(_cleaned_criteria, 1):
                        _pc["priority"] = _pi
                _new_priority_order = [c.get("name", "").lower().strip() for c in _cleaned_criteria]
                _priority_changed = _old_priority_order != _new_priority_order

                _has_rubric_changes = _cc_hallucinated_count > 0 or _priority_changed

                if _has_rubric_changes:
                    # Save as new rubric version
                    _cleanup_hist = load_rubric_history()
                    _cleanup_new_ver = next_version_number()
                    _cleanup_hist.append({
                        "version": _cleanup_new_ver,
                        "rubric": copy.deepcopy(_cleaned_criteria),
                        "source": "criteria_classification",
                        "conversation_id": st.session_state.get("selected_conversation"),
                    })
                    save_rubric_history(_cleanup_hist)
                    st.session_state.active_rubric_idx = len(_cleanup_hist) - 1
                    st.session_state.rubric = _cleaned_criteria
                    st.session_state.editing_criteria = _cleaned_criteria
                    st.session_state.editing_criteria_ui_version = st.session_state.get("editing_criteria_ui_version", 0) + 1
                    st.session_state["rubric_version_selector"] = f"v{_cleanup_new_ver}"

                    # Update the classification log message with the version info
                    for _clm in reversed(st.session_state.messages):
                        if _clm.get("is_criteria_classification_log") and _clm.get("classification_data"):
                            _clm["classification_data"]["rubric_version"] = _cleanup_new_ver
                            break

                    # Log what happened
                    _cleanup_removed_names = [name for name, cat in _cc_final.items() if cat == "hallucinated"]
                    _cleanup_summary_parts = []
                    if _cleanup_removed_names:
                        _cleanup_summary_parts.append(f"Removed {len(_cleanup_removed_names)} hallucinated criteria: {', '.join(_cleanup_removed_names)}")
                    if _priority_changed:
                        _cleanup_summary_parts.append("Re-ordered criteria by your importance ranking")
                    _cleanup_summary_parts.append(f"Rubric saved as v{_cleanup_new_ver} with {len(_cleaned_criteria)} criteria")
                    st.session_state.messages.append({
                        "role": "system",
                        "content": " | ".join(_cleanup_summary_parts),
                        "conversation_id": st.session_state.get("selected_conversation"),
                    })

                    # Save cleanup event to DB
                    _cleanup_sb = st.session_state.get('supabase')
                    _cleanup_pid = st.session_state.get('current_project_id')
                    if _cleanup_sb and _cleanup_pid:
                        try:
                            save_project_data(_cleanup_sb, _cleanup_pid, "rubric_classification_cleanup", {
                                "timestamp": datetime.now().isoformat(),
                                "removed_criteria": _cleanup_removed_names,
                                "importance_ranking": list(_cc_importance),
                                "priority_changed": _priority_changed,
                                "old_version": _cleanup_rb_dict.get("version", "?") if _cleanup_rb_dict else "?",
                                "new_version": _cleanup_new_ver,
                                "n_remaining": len(_cleaned_criteria),
                            })
                        except Exception:
                            pass

                # [DISABLED] Step 3: DP extraction + confirmation removed for now
                # _cc_conv_msgs = st.session_state.get("infer_dp_messages", [])
                # if _cc_conv_msgs:
                #     _dp_rb_dict, _, _ = get_active_rubric()
                #     _dp_rubric_json = json.dumps(_dp_rb_dict.get("rubric", []), ensure_ascii=False, indent=2) if _dp_rb_dict else "[]"
                #     with st.spinner("Extracting decision points with classification context..."):
                #         dp_result = extract_decision_points(
                #             _cc_conv_msgs, _dp_rubric_json, _cc_feedback_for_dps
                #         )
                #         if dp_result:
                #             for dp in dp_result.get("parsed_data", {}).get("decision_points", []):
                #                 if "related_rubric_criterion" in dp and "suggested_criterion_name" not in dp:
                #                     dp["suggested_criterion_name"] = dp["related_rubric_criterion"]
                #             _dp_result_store = {
                #                 "thinking": "", "raw_response": "",
                #                 "parsed_data": dp_result.get("parsed_data", dp_result),
                #                 "conversation_file": "__from_infer_rubric__"
                #             }
                #             st.session_state.infer_decision_points = _dp_result_store
                #             st.session_state.infer_dp_dimension_confirmed = False
                #             st.session_state.infer_dp_user_mappings = {}
                #             _dp_list_new = dp_result.get("parsed_data", {}).get("decision_points", [])
                #             if _dp_list_new:
                #                 st.session_state.infer_expanded_dp = _dp_list_new[0].get("id")
                #             for msg in st.session_state.messages:
                #                 if msg.get("is_dp_review"):
                #                     msg["dp_data"]["decision_points"] = copy.deepcopy(_dp_list_new)
                #                     _rb_v = msg["dp_data"].get("rubric_version", "?")
                #                     msg["content"] = f"**Rubric v{_rb_v} inferred** — {len(_dp_list_new)} decision points extracted. Review each DP below."
                #                     msg["display_content"] = msg["content"]
                #                     if _dp_rb_dict:
                #                         msg["dp_data"]["rubric"] = copy.deepcopy(_dp_rb_dict.get("rubric", []))
                #                     break

                st.rerun()

    # --- DP Confirm button (after all messages) --- [DISABLED: commenting out DP UI]
    if False and _dp_active and _dp_list_all and not _dp_confirmed:
        st.divider()

        # DP instruction + jump navigation
        import streamlit.components.v1 as _dp_components
        st.markdown(
            "**Decision Points** are moments in the conversation where your writing choices reveal preferences. "
            "Reviewing them helps us build a rubric that truly reflects how you write — not just what you said you want, "
            "but what you actually chose when it mattered. Please confirm the rubric criterion each decision point maps to, or correct it."
        )
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
        # Check if user reordered importance ranking vs current rubric priority
        _had_reranking = False
        _rerank_list = st.session_state.get("chat_classification_feedback", {}).get("importance_ranking", [])
        if _rerank_list:
            _rerank_rb, _, _ = get_active_rubric()
            if _rerank_rb:
                _rerank_current = [c.get("name", "") for c in sorted(_rerank_rb.get("rubric", []), key=lambda c: c.get("priority", 99))]
                if _rerank_list != _rerank_current:
                    _had_reranking = True
        _needs_final_rubric = _has_corrections or _had_hallucinated or _had_reranking
        if _needs_final_rubric:
            _confirm_label = "Confirm DPs & Infer Final Rubric"
        else:
            _confirm_label = "Confirm Decision Points"
        if st.button(_confirm_label, type="primary", width="stretch", key="chat_confirm_dp_and_infer"):
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
                # If there were corrections, ask model to explain how they were incorporated
                _dp_corrected = [dp for dp in decision_points if dp.get("user_action") in ("incorrect", "not_in_rubric")]
                _dp_correction_reasoning = ""
                if _dp_corrected and _needs_final_rubric and _cur_rb_log:
                    try:
                        _corr_parts = []
                        for _cdp in _dp_corrected:
                            _cdp_id = _cdp.get("id", "?")
                            _cdp_dim = _cdp.get("dimension", "")
                            _cdp_action = _cdp.get("user_action", "")
                            _cdp_orig = _cdp.get("original_suggestion", "")
                            _cdp_new = _cdp.get("confirmed_criterion", "")
                            _cdp_reason = _cdp.get("user_correction_reason", "")
                            if _cdp_action == "incorrect":
                                _corr_parts.append(f"- DP#{_cdp_id} ({_cdp_dim}): User remapped from '{_cdp_orig}' to '{_cdp_new}'. Reason: {_cdp_reason or 'not given'}")
                            elif _cdp_action == "not_in_rubric":
                                _corr_parts.append(f"- DP#{_cdp_id} ({_cdp_dim}): User said this is not in the rubric. Reason: {_cdp_reason or 'not given'}")
                        _corr_summary = "\n".join(_corr_parts)
                        _corr_rubric_json = json.dumps(_cur_rb_log.get("rubric", []), ensure_ascii=False, indent=2)
                        _corr_prompt = (
                            f"The user just reviewed decision points extracted from their writing conversation. "
                            f"Some decision points were corrected or marked as not belonging to the rubric:\n\n"
                            f"{_corr_summary}\n\n"
                            f"The final inferred rubric (v{_result_rubric_version}) is:\n{_corr_rubric_json}\n\n"
                            f"For each corrected/not-in-rubric DP above, briefly explain (1-2 sentences each) how "
                            f"this user feedback was incorporated into the final rubric. Did it cause a criterion to "
                            f"be added, removed, merged, or refined? Be specific about which criterion was affected. "
                            f"Do NOT include any heading or title — just start with the explanation directly."
                        )
                        _corr_response = _api_call_with_retry(
                            model=MODEL_PRIMARY,
                            max_tokens=2000,
                            messages=[{"role": "user", "content": _corr_prompt}],
                        )
                        if _corr_response and _corr_response.content:
                            _dp_correction_reasoning = _corr_response.content[0].text.strip()
                            # Strip any leading header the model may have added despite instructions
                            import re as _re_corr
                            _dp_correction_reasoning = _re_corr.sub(r'^(?:\*{0,2})\s*(?:How\s+(?:your\s+)?(?:user\s+)?feedback\s+was\s+incorporated|Incorporation\s+of\s+feedback)\s*:?\s*(?:\*{0,2})\s*\n*', '', _dp_correction_reasoning, flags=_re_corr.IGNORECASE).strip()
                    except Exception as _corr_e:
                        # print(f"[DEBUG] Error getting correction reasoning: {_corr_e}")

                        pass
                if _dp_correction_reasoning:
                    _dp_log_lines.append(f"\n**How your feedback was incorporated:**\n{_dp_correction_reasoning}")

                _dp_log_content = "\n".join(_dp_log_lines)

                # DP confirmation log message first
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
                        "correction_reasoning": _dp_correction_reasoning,
                    },
                    "message_id": f"dp_confirm_{int(time.time() * 1000000)}"
                })

                # System message announcing outcome
                if _needs_final_rubric and _source_rubric_version != _result_rubric_version:
                    _refine_data = st.session_state.get("dp_refinement_result") or {}
                    st.session_state.messages.append({
                        "role": "system",
                        "content": f"Rubric **v{_result_rubric_version}** inferred from your conversation and criteria classification / decision point feedback.",
                        "conversation_id": st.session_state.get("selected_conversation"),
                        "refinement_detail": {
                            "change_explanation": _refine_data.get("change_explanation", ""),
                            "refinement_summary": _refine_data.get("refinement_summary", ""),
                            "old_rubric": _refine_data.get("old_rubric", []),
                            "new_rubric": _refine_data.get("new_rubric", []),
                            "old_version": _refine_data.get("old_version", "?"),
                            "new_version": _refine_data.get("new_version", "?"),
                        },
                    })
                else:
                    st.session_state.messages.append({
                        "role": "system",
                        "content": f"Decision points confirmed. Rubric **v{_source_rubric_version}** unchanged.",
                        "conversation_id": st.session_state.get("selected_conversation"),
                    })
                _auto_save_conversation()

                st.rerun()

    elif _dp_active and _dp_confirmed:
        pass  # DP confirmation and classification messages are now shown as conversation messages

    # ---- Rubric Alignment Diagnostic UI ----
    _rcp = st.session_state.get("ranking_checkpoint_pending")
    if _rcp is not None:
        # st.divider()
        st.markdown("### Rubric Alignment Check")
        _rcp_coldstart_avail = bool(st.session_state.get("infer_coldstart_text", "").strip())
        if _rcp_coldstart_avail:
            st.markdown("Let's check how well your rubric is working. We'll generate three drafts — one following your rubric, one from your original preferences, and one generic — and analyze which criteria are making a difference.")
        else:
            st.markdown("Let's check how well your rubric is working. We'll generate two drafts — one following your rubric, one without it — and analyze which criteria are making a difference.")
        _rcp_step = _rcp.get("step", 1)

        # Show which rubric version is being used
        _rcp_rb_dict, _, _ = get_active_rubric()
        _rcp_rb_ver = _rcp_rb_dict.get("version", "?") if _rcp_rb_dict else "?"
        st.caption(f"Using rubric v{_rcp_rb_ver}")

        if _rcp_step == 2:
            # Step 2: Generate 3 drafts (rubric-guided + generic + preference-based)
            _rcp_wt_text = _rcp.get('writing_task', '')
            _rcp_wt_lines = max(1, _rcp_wt_text.count("\n") + 1, len(_rcp_wt_text) // 80)
            _rcp_wt_h = min(300, max(80, _rcp_wt_lines * 28 + 40))
            st.markdown("**Writing task:**")
            with st.container(height=_rcp_wt_h):
                st.markdown(_rcp_wt_text)
            _rcp_rubric_dict, _, _ = get_active_rubric()
            _rcp_rubric_json = json.dumps(
                _rubric_to_json_serializable(_rcp_rubric_dict), indent=2
            ) if _rcp_rubric_dict else ""
            _rcp_task = _rcp["writing_task"]
            _rcp_drafts = {}
            _rcp_ok = True
            _rcp_coldstart_text = st.session_state.get("infer_coldstart_text", "").strip()
            _rcp_has_3_drafts = bool(_rcp_coldstart_text)
            # Build pipeline conversation context: each draft prompt+response is tracked
            # so the diagnostic later has full context of what was generated and how
            _rcp_pipeline_msgs = []

            if _rcp_has_3_drafts:
                st.info("Writing three versions: one following your rubric, one from your original preferences, and one generic...")
            else:
                st.info("Writing two versions: one following your rubric, one without it...")

            with st.spinner("Generating rubric-guided draft..."):
                try:
                    _rcp_pr = GRADING_generate_draft_from_rubric_prompt(_rcp_task, _rcp_rubric_json)
                    _rcp_pipeline_msgs.append({"role": "user", "content": _rcp_pr})
                    _rcp_resp_r = _api_call_with_retry(
                        model=MODEL_LIGHT, max_tokens=1000,
                        messages=_rcp_pipeline_msgs
                    )
                    _rcp_rubric_draft_text = "".join(b.text for b in _rcp_resp_r.content if b.type == "text").strip()
                    _rcp_pipeline_msgs.append({"role": "assistant", "content": _rcp_rubric_draft_text})
                    _rcp_drafts["rubric"] = _rcp_rubric_draft_text
                except Exception as _rcp_e2:
                    st.error(f"Failed to generate rubric draft: {_rcp_e2}")
                    _rcp_ok = False

            if _rcp_ok:
                with st.spinner("Generating generic draft..."):
                    try:
                        _rcp_pg = GRADING_generate_draft_generic_prompt(_rcp_task)
                        _rcp_pipeline_msgs.append({"role": "user", "content": _rcp_pg})
                        _rcp_resp_g = _api_call_with_retry(
                            model=MODEL_LIGHT, max_tokens=1000,
                            messages=_rcp_pipeline_msgs
                        )
                        _rcp_generic_draft_text = "".join(b.text for b in _rcp_resp_g.content if b.type == "text").strip()
                        _rcp_pipeline_msgs.append({"role": "assistant", "content": _rcp_generic_draft_text})
                        _rcp_drafts["generic"] = _rcp_generic_draft_text
                    except Exception as _rcp_e3:
                        st.error(f"Failed to generate generic draft: {_rcp_e3}")
                        _rcp_ok = False

            if _rcp_ok and _rcp_has_3_drafts:
                with st.spinner("Generating preference-based draft..."):
                    try:
                        _rcp_pp = GRADING_generate_draft_from_coldstart_prompt(_rcp_task, _rcp_coldstart_text)
                        _rcp_pipeline_msgs.append({"role": "user", "content": _rcp_pp})
                        _rcp_resp_p = _api_call_with_retry(
                            model=MODEL_LIGHT, max_tokens=1000,
                            messages=_rcp_pipeline_msgs
                        )
                        _rcp_pref_draft_text = "".join(b.text for b in _rcp_resp_p.content if b.type == "text").strip()
                        _rcp_pipeline_msgs.append({"role": "assistant", "content": _rcp_pref_draft_text})
                        _rcp_drafts["preference"] = _rcp_pref_draft_text
                    except Exception as _rcp_e4:
                        st.error(f"Failed to generate preference draft: {_rcp_e4}")
                        # Fall back to 2-draft mode
                        _rcp_has_3_drafts = False

            _rcp_min_drafts_ok = _rcp_ok and _rcp_drafts.get("rubric") and _rcp_drafts.get("generic")
            if _rcp_min_drafts_ok:
                # Randomly assign blind labels to sources
                if _rcp_has_3_drafts and _rcp_drafts.get("preference"):
                    _rcp_sources = ["rubric", "generic", "preference"]
                    random.shuffle(_rcp_sources)
                    _rcp_shuffle_order = list(zip(["A", "B", "C"], _rcp_sources))
                else:
                    _rcp_left_is_rubric = random.choice([True, False])
                    if _rcp_left_is_rubric:
                        _rcp_shuffle_order = [("A", "rubric"), ("B", "generic")]
                    else:
                        _rcp_shuffle_order = [("A", "generic"), ("B", "rubric")]
                _rcp_step3_dict = {
                    "step": 3,
                    "writing_task": _rcp_task,
                    "drafts": _rcp_drafts,
                    "shuffle_order": _rcp_shuffle_order,
                    "rubric_version": _rcp.get("rubric_version", _rcp_rb_ver),
                    "pipeline_messages": _rcp_pipeline_msgs,
                }
                if _rcp.get("is_conversation_start"):
                    _rcp_step3_dict["is_conversation_start"] = True
                st.session_state.ranking_checkpoint_pending = _rcp_step3_dict
                st.rerun()
            elif not _rcp_ok:
                if st.button("Cancel", key="rcp_cancel_s2"):
                    st.session_state.ranking_checkpoint_pending = None
                    st.rerun()

        elif _rcp_step == 3:
            # Step 3: User ranks drafts (3-draft or 2-draft mode)
            _rcp_wt_text3 = _rcp.get('writing_task', '')
            _rcp_wt_lines3 = max(1, _rcp_wt_text3.count("\n") + 1, len(_rcp_wt_text3) // 80)
            _rcp_wt_h3 = min(300, max(80, _rcp_wt_lines3 * 28 + 40))
            st.markdown("**Writing task:**")
            with st.container(height=_rcp_wt_h3):
                st.markdown(_rcp_wt_text3)
            _rcp_shuffle = _rcp.get("shuffle_order", [])
            _rcp_drafts_3 = _rcp.get("drafts", {})
            _rcp_is_3draft = len(_rcp_shuffle) == 3

            if _rcp_is_3draft:
                st.info("We generated three drafts of your writing task using different approaches — one guided by your rubric, one using your original preferences, and one generic. **Your ranking helps us check whether your rubric is capturing what you actually want.** If a non-rubric draft ranks higher, it means the rubric may need adjustments.")
                st.markdown("Read all three drafts and **rank them from best to worst**.")

                # Display 3 drafts side by side
                _rcp_col_a, _rcp_col_b, _rcp_col_c = st.columns(3)
                with _rcp_col_a:
                    st.markdown("**Draft A**")
                    with st.container(height=300):
                        st.markdown(_rcp_drafts_3.get(_rcp_shuffle[0][1], ""))
                with _rcp_col_b:
                    st.markdown("**Draft B**")
                    with st.container(height=300):
                        st.markdown(_rcp_drafts_3.get(_rcp_shuffle[1][1], ""))
                with _rcp_col_c:
                    st.markdown("**Draft C**")
                    with st.container(height=300):
                        st.markdown(_rcp_drafts_3.get(_rcp_shuffle[2][1], ""))

                # Optional reason text input
                _rcp_reason = st.text_input("What influenced your ranking? (optional)", key="rcp_reason_input", placeholder="e.g. 'Draft B felt more natural; Draft A was too formal'")

                # Ranking dropdowns
                _draft_options = ["Draft A", "Draft B", "Draft C"]
                st.markdown("**Rank the drafts:**")
                _rank_cols = st.columns(3)
                with _rank_cols[0]:
                    _rank_1st = st.selectbox("1st (Best)", _draft_options, index=0, key="rcp_rank_1st")
                with _rank_cols[1]:
                    _remaining_2nd = [d for d in _draft_options if d != _rank_1st]
                    _rank_2nd = st.selectbox("2nd", _remaining_2nd, index=0, key="rcp_rank_2nd")
                with _rank_cols[2]:
                    _remaining_3rd = [d for d in _draft_options if d != _rank_1st and d != _rank_2nd]
                    _rank_3rd = st.selectbox("3rd (Worst)", _remaining_3rd, index=0, key="rcp_rank_3rd")

                # Submit / Try Different Prompt / Cancel
                _rcp_btn_cols = st.columns([1, 1, 1])
                _rcp_submitted = False
                with _rcp_btn_cols[0]:
                    if st.button("Submit Ranking", key="rcp_submit_ranking", type="primary", width="stretch"):
                        _rcp_submitted = True
                with _rcp_btn_cols[1]:
                    if st.button("Try Different Prompt", key="rcp_retry_s3", width="stretch"):
                        st.session_state.ranking_checkpoint_pending = None
                        st.rerun()
                with _rcp_btn_cols[2]:
                    if st.button("Cancel", key="rcp_cancel_s3", width="stretch"):
                        st.session_state.ranking_checkpoint_pending = None
                        st.session_state.alignment_check_skipped = True
                        st.rerun()

                if _rcp_submitted:
                    # Convert ranking labels to source keys
                    _label_to_source = {lab: src for lab, src in _rcp_shuffle}
                    _ranking_labels = [_rank_1st, _rank_2nd, _rank_3rd]
                    _user_ranking = [_label_to_source[lab.replace("Draft ", "")] for lab in _ranking_labels]

                    # Reveal blind labels
                    _source_names = {"rubric": "Rubric-guided", "generic": "Generic", "preference": "Your original preferences"}
                    _source_to_label = {src: lab for lab, src in _rcp_shuffle}
                    _reveal_parts = [f"Draft {_source_to_label[src]} = {_source_names[src]}" for _, src in _rcp_shuffle]
                    _ranking_display = " > ".join(
                        f"**{_source_names.get(s, s)}**" for s in _user_ranking
                    )
                    st.info(f"Your ranking: {_ranking_display}  |  {' | '.join(_reveal_parts)}")

                    # Run diagnostic analysis
                    _rcp_result = None
                    _diag_status_placeholder = st.empty()
                    try:
                        _diag_status_placeholder.info("⏳ Scoring each draft against your rubric criteria...")
                        def _diag_status_cb(msg):
                            _diag_status_placeholder.info(f"⏳ {msg}")
                        _rcp_result = _process_alignment_diagnostic(
                            _rcp, _user_ranking, _rcp_reason,
                            status_callback=_diag_status_cb,
                            pipeline_messages=_rcp.get("pipeline_messages")
                        )
                    except Exception as _rcp_err:
                        st.error(f"Error during diagnostic: {_rcp_err}")
                    _diag_status_placeholder.empty()

                    if _rcp_result:
                        # Build ranking takeaway
                        if _user_ranking[0] == "rubric":
                            _ranking_takeaway = "Your rubric is guiding drafts in the right direction!!"
                        elif _user_ranking[0] == "preference":
                            _ranking_takeaway = "Your original preferences still resonate more — the rubric may have drifted."
                        elif _user_ranking[0] == "generic":
                            _ranking_takeaway = "The generic draft won — your rubric may need significant adjustments."
                        else:
                            _ranking_takeaway = ""

                        # Build diagnostic report message
                        # _diag_parts = full content (for API context, includes ranking)
                        # _diag_scoring_parts = scoring only (for expander display, excludes ranking)
                        _diag_parts = ["**Rubric Alignment Diagnostic**\n"]
                        _diag_parts.append(f"Your ranking: {_ranking_display}\n")
                        if _ranking_takeaway:
                            _diag_parts.append(f"{_ranking_takeaway}\n")

                        _diag_scoring_parts = []

                        # Per-criterion analysis
                        _diag_criteria = _rcp_result.get("criteria_analysis", [])
                        if _diag_criteria:
                            _diag_parts.append("---\n\n**Per-Criterion Scores:**\n")
                            _diag_scoring_parts.append("**Per-Criterion Scores:**\n")
                            for _dc in _diag_criteria:
                                _dc_class = _dc["classification"]
                                _dc_icon = {"DIFFERENTIATING": "[+]", "REDUNDANT": "[=]", "UNDERPERFORMING": "[-]", "PREFERENCE_GAP": "[~]"}.get(_dc_class, "[?]")
                                _dc_gap = _dc.get("gap", 0)
                                _gap_str = f"+{_dc_gap}" if _dc_gap > 0 else str(_dc_gap)
                                _pref_score_str = f" | Preference: {_dc['preference_score']}/5" if "preference_score" in _dc else ""
                                _score_line = (
                                    f"\n**{_dc_icon} {_dc['name']}** (priority {_dc.get('priority', '?')}) — {_dc_class}\n"
                                    f"> Rubric: {_dc['rubric_score']}/5 | Generic: {_dc['generic_score']}/5{_pref_score_str} | Gap (R-G): {_gap_str}\n"
                                    f"> *{_dc['reasoning']}*\n"
                                )
                                _diag_parts.append(_score_line)
                                _diag_scoring_parts.append(_score_line)

                        _diag_content = "\n".join(_diag_parts)
                        _diag_display_content = "\n".join(_diag_scoring_parts)

                        # Add ranking display info to result for UI rendering
                        _rcp_result["ranking_display"] = _ranking_display
                        _rcp_result["ranking_takeaway"] = _ranking_takeaway

                        # Store message with diagnostic data + rubric suggestion
                        _diag_msg = {
                            "role": "assistant",
                            "content": _diag_content,
                            "display_content": _diag_display_content,
                            "is_system_generated": True,
                            "is_alignment_diagnostic": True,
                            "diagnostic_data": _rcp_result,
                            "message_id": f"diag_result_{int(time.time() * 1000000)}",
                        }
                        # Attach rubric suggestion if we have one
                        if _rcp_result.get("suggested_rubric"):
                            _diag_msg["rubric_suggestion"] = {
                                "current_rubric": _rcp_rb_dict.get("rubric", []) if _rcp_rb_dict else [],
                                "updated_rubric": _rcp_result["suggested_rubric"],
                                "suggestion_text": _rcp_result.get("suggestion_text", ""),
                                "suggestion_reasons": _rcp_result.get("suggestion_reasons", {}),
                                "suggested_draft": _rcp_result.get("suggested_draft", ""),
                                "original_rubric_draft": _rcp.get("drafts", {}).get("rubric", ""),
                                "suggested_annotated_changes": _rcp_result.get("suggested_annotated_changes", []),
                            }
                        if _rcp.get("is_conversation_start"):
                            _ac_writing_task = _rcp.get("writing_task", "")
                            _ac_drafts = _rcp.get("drafts", {})

                            # Inject user message (writing task)
                            st.session_state.messages.append({
                                "role": "user",
                                "content": _ac_writing_task,
                            })
                            # Inject diagnostic report (includes suggestion expander with draft diff)
                            # Do NOT inject the conversation-start draft yet — wait for user to Apply or Skip
                            _diag_msg["_ac_pending_draft"] = True
                            _diag_msg["_ac_suggested_draft"] = _rcp_result.get("suggested_draft", "")
                            _diag_msg["_ac_fallback_draft"] = _ac_drafts.get(_user_ranking[0], "")
                            _diag_msg["_ac_fallback_draft_source"] = _user_ranking[0]
                            st.session_state.messages.append(_diag_msg)
                            # Mark alignment check as done but draft pending
                            st.session_state.alignment_check_done = True
                        else:
                            st.session_state.messages.append(_diag_msg)
                        st.session_state.ranking_checkpoint_pending = None
                        _auto_save_conversation()
                        st.rerun()

            else:
                # 2-draft fallback (no coldstart preferences available)
                st.info("We generated two drafts — one guided by your rubric and one generic. **Your preference helps us check whether your rubric is working well for this task.** If the generic draft feels better, the rubric may need adjustments.")
                st.markdown("Read both drafts and pick the one closer to what you'd actually want.")

                # Display 2 drafts side by side
                _rcp_col_a, _rcp_col_b = st.columns(2)
                with _rcp_col_a:
                    st.markdown("**Draft A**")
                    with st.container(height=300):
                        st.markdown(_rcp_drafts_3.get(_rcp_shuffle[0][1], ""))
                with _rcp_col_b:
                    st.markdown("**Draft B**")
                    with st.container(height=300):
                        st.markdown(_rcp_drafts_3.get(_rcp_shuffle[1][1], ""))

                # Optional reason text input
                _rcp_reason = st.text_input("What made you prefer it? (optional)", key="rcp_reason_input", placeholder="e.g. 'Draft A felt more concise and direct'")

                # Preference buttons
                _rcp_btn_cols = st.columns([1, 1, 1, 1, 1])
                _rcp_submitted = False
                _rcp_user_pref = None
                with _rcp_btn_cols[0]:
                    if st.button("A is better", key="rcp_pref_a", type="primary"):
                        _rcp_user_pref = _rcp_shuffle[0][1]
                        _rcp_submitted = True
                with _rcp_btn_cols[1]:
                    if st.button("B is better", key="rcp_pref_b", type="primary"):
                        _rcp_user_pref = _rcp_shuffle[1][1]
                        _rcp_submitted = True
                with _rcp_btn_cols[2]:
                    if st.button("About the same", key="rcp_pref_tie"):
                        _rcp_user_pref = "tie"
                        _rcp_submitted = True
                with _rcp_btn_cols[3]:
                    if st.button("Try Different Prompt", key="rcp_retry_s3_2d"):
                        st.session_state.ranking_checkpoint_pending = None
                        st.rerun()
                with _rcp_btn_cols[4]:
                    if st.button("Cancel", key="rcp_cancel_s3"):
                        st.session_state.ranking_checkpoint_pending = None
                        st.session_state.alignment_check_skipped = True
                        st.rerun()

                if _rcp_submitted:
                    # Convert 2-draft preference to ranking format for consistency
                    if _rcp_user_pref == "tie":
                        _user_ranking = ["rubric", "generic"]  # Default to rubric first on tie
                    elif _rcp_user_pref == "rubric":
                        _user_ranking = ["rubric", "generic"]
                    else:
                        _user_ranking = ["generic", "rubric"]

                    # Reveal blind labels
                    _rcp_label_map = {src: lab for lab, src in _rcp_shuffle}
                    _source_names = {"rubric": "Rubric-guided", "generic": "Generic"}
                    _pref_display = " > ".join(f"**{_source_names[s]}**" for s in _user_ranking) if _rcp_user_pref != "tie" else "**About the same**"
                    st.info(f"You preferred: {_pref_display}  |  Draft {_rcp_label_map['rubric']} = Rubric-guided  |  Draft {_rcp_label_map['generic']} = Generic")

                    # Run diagnostic analysis
                    _rcp_result = None
                    _diag_status_placeholder = st.empty()
                    try:
                        _diag_status_placeholder.info("⏳ Scoring each draft against your rubric criteria...")
                        def _diag_status_cb(msg):
                            _diag_status_placeholder.info(f"⏳ {msg}")
                        _rcp_result = _process_alignment_diagnostic(
                            _rcp, _user_ranking, _rcp_reason,
                            status_callback=_diag_status_cb,
                            pipeline_messages=_rcp.get("pipeline_messages")
                        )
                    except Exception as _rcp_err:
                        st.error(f"Error during diagnostic: {_rcp_err}")
                    _diag_status_placeholder.empty()

                    if _rcp_result:
                        # Build ranking takeaway
                        if _rcp_user_pref == "rubric":
                            _ranking_takeaway = "You preferred the **rubric-guided** draft — your rubric is pointing the LLM in the right direction."
                            _ranking_display_2d = _pref_display
                        elif _rcp_user_pref == "generic":
                            _ranking_takeaway = "You preferred the **generic** draft — your rubric may need adjustments."
                            _ranking_display_2d = _pref_display
                        else:
                            _ranking_takeaway = "You found them **about the same** — your rubric may not be adding much differentiation."
                            _ranking_display_2d = _pref_display

                        # Build diagnostic report message
                        # _diag_parts = full content (for API context, includes ranking)
                        # _diag_scoring_parts = scoring only (for expander display, excludes ranking)
                        _diag_parts = ["**Rubric Alignment Diagnostic**\n"]
                        _diag_parts.append(f"Your preference: {_pref_display}\n")
                        if _ranking_takeaway:
                            _diag_parts.append(f"{_ranking_takeaway}\n")

                        _diag_scoring_parts = []

                        # Per-criterion analysis
                        _diag_criteria = _rcp_result.get("criteria_analysis", [])
                        if _diag_criteria:
                            _diag_parts.append("---\n\n**Your Rubric Criteria:**\n")
                            _diag_scoring_parts.append("**Your Rubric Criteria:**\n")
                            for _dc in _diag_criteria:
                                _dc_class = _dc["classification"]
                                _dc_icon = {"DIFFERENTIATING": "[+]", "REDUNDANT": "[=]", "UNDERPERFORMING": "[-]"}.get(_dc_class, "[?]")
                                _dc_gap = _dc.get("gap", 0)
                                _gap_str = f"+{_dc_gap}" if _dc_gap > 0 else str(_dc_gap)
                                _score_line = (
                                    f"\n**{_dc_icon} {_dc['name']}** (priority {_dc.get('priority', '?')}) — {_dc_class}\n"
                                    f"> Rubric: {_dc['rubric_score']}/5 | Generic: {_dc['generic_score']}/5 | Gap: {_gap_str}\n"
                                    f"> *{_dc['reasoning']}*\n"
                                )
                                _diag_parts.append(_score_line)
                                _diag_scoring_parts.append(_score_line)

                        _diag_content = "\n".join(_diag_parts)
                        _diag_display_content = "\n".join(_diag_scoring_parts)

                        # Add ranking display info to result for UI rendering
                        _rcp_result["ranking_display"] = _ranking_display_2d
                        _rcp_result["ranking_takeaway"] = _ranking_takeaway

                        # Store message with diagnostic data + rubric suggestion
                        _diag_msg = {
                            "role": "assistant",
                            "content": _diag_content,
                            "display_content": _diag_display_content,
                            "is_system_generated": True,
                            "is_alignment_diagnostic": True,
                            "diagnostic_data": _rcp_result,
                            "message_id": f"diag_result_{int(time.time() * 1000000)}",
                        }
                        # Attach rubric suggestion if we have one
                        if _rcp_result.get("suggested_rubric"):
                            _diag_msg["rubric_suggestion"] = {
                                "current_rubric": _rcp_rb_dict.get("rubric", []) if _rcp_rb_dict else [],
                                "updated_rubric": _rcp_result["suggested_rubric"],
                                "suggestion_text": _rcp_result.get("suggestion_text", ""),
                                "suggestion_reasons": _rcp_result.get("suggestion_reasons", {}),
                                "suggested_draft": _rcp_result.get("suggested_draft", ""),
                                "original_rubric_draft": _rcp.get("drafts", {}).get("rubric", ""),
                                "suggested_annotated_changes": _rcp_result.get("suggested_annotated_changes", []),
                            }
                        if _rcp.get("is_conversation_start"):
                            _ac_writing_task = _rcp.get("writing_task", "")
                            _ac_drafts = _rcp.get("drafts", {})

                            st.session_state.messages.append({
                                "role": "user",
                                "content": _ac_writing_task,
                            })
                            _diag_msg["_ac_pending_draft"] = True
                            _diag_msg["_ac_suggested_draft"] = _rcp_result.get("suggested_draft", "")
                            _diag_msg["_ac_fallback_draft"] = _ac_drafts.get(_user_ranking[0], "")
                            _diag_msg["_ac_fallback_draft_source"] = _user_ranking[0]
                            st.session_state.messages.append(_diag_msg)
                            st.session_state.alignment_check_done = True
                        else:
                            st.session_state.messages.append(_diag_msg)
                        st.session_state.ranking_checkpoint_pending = None
                        _auto_save_conversation()
                        st.rerun()

    # Delete mode confirmation bar (shown at the bottom when in delete mode)
    if st.session_state.message_delete_mode and st.session_state.messages_to_delete:
        st.warning(f"🗑️ **{len(st.session_state.messages_to_delete)} message(s) selected for deletion**")
        del_col1, del_col2 = st.columns(2)
        with del_col1:
            if st.button("✅ Confirm Delete", width="stretch", type="primary"):
                # Delete selected messages (in reverse order to preserve indices)
                for idx in sorted(st.session_state.messages_to_delete, reverse=True):
                    if idx < len(st.session_state.messages):
                        del st.session_state.messages[idx]
                st.session_state.messages_to_delete = set()
                st.session_state.message_delete_mode = False
                _auto_save_conversation()
                st.success("Messages deleted successfully!")
                st.rerun()
        with del_col2:
            if st.button("❌ Cancel", width="stretch"):
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
            if st.button("✖️ Close", key="close_comparison", width="stretch"):
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

    # --- Uncertainty Probe UI ---
    if False and st.session_state.get("probe_pending"):  # [DISABLED] Probe A/B testing removed for now
        _prb = st.session_state.probe_pending
        _prb_crit = _prb.get("criterion_name", "")
        _prb_reason = _prb.get("uncertainty_reason", "")
        _prb_dim = _prb.get("dimension_varied", "")

        with st.chat_message("assistant"):
            st.markdown(f"**I'm not sure how to apply \"{_prb_crit}\"**")
            st.caption(_prb_reason)
            if _prb_dim:
                st.caption(f"These two versions differ in: *{_prb_dim}*")

            _prb_col_a, _prb_col_b = st.columns(2)
            with _prb_col_a:
                st.markdown("**Version A**")
                with st.container(height=300):
                    st.markdown(_prb.get("variant_a", ""))
            with _prb_col_b:
                st.markdown("**Version B**")
                with st.container(height=300):
                    st.markdown(_prb.get("variant_b", ""))

            _prb_reason_input = st.text_input(
                "What made you prefer it? (optional)",
                key="probe_reason_input",
                placeholder="e.g., 'I want it more conversational, not bullet points'"
            )

            _prb_btn_a, _prb_btn_b, _prb_btn_skip = st.columns(3)
            with _prb_btn_a:
                if st.button("Prefer A", key="probe_prefer_a", type="primary", width="stretch"):
                    _probe_commit_choice(_prb, "a", _prb_reason_input)
                    # Wait for background refinement to finish so updated_criterion is available
                    _prb_evt = st.session_state.get("_probe_refine_done_event")
                    if _prb_evt:
                        with st.spinner("Refining criterion..."):
                            _prb_evt.wait(timeout=30)
                        # Re-save conversation now that background thread has written suggested_update
                        _auto_save_conversation()
                    st.rerun()
            with _prb_btn_b:
                if st.button("Prefer B", key="probe_prefer_b", type="primary", width="stretch"):
                    _probe_commit_choice(_prb, "b", _prb_reason_input)
                    # Wait for background refinement to finish so updated_criterion is available
                    _prb_evt = st.session_state.get("_probe_refine_done_event")
                    if _prb_evt:
                        with st.spinner("Refining criterion..."):
                            _prb_evt.wait(timeout=30)
                        # Re-save conversation now that background thread has written suggested_update
                        _auto_save_conversation()
                    st.rerun()
            with _prb_btn_skip:
                if st.button("Skip", key="probe_skip", width="stretch"):
                    _probe_commit_choice(_prb, "skip")
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
        if st.button("Start Writing", type="primary", width="stretch", key="chat_pref_submit"):
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

    # --- Alignment check gate: rubric alignment check before first message when rubric exists ---
    _alignment_check_needed = (
        _pref_has_project
        and _pref_has_rubric
        and not _pref_has_messages
        and not st.session_state.get("alignment_check_done", False)
        and not st.session_state.get("alignment_check_skipped", False)
        and st.session_state.get("ranking_checkpoint_pending") is None
        and not _pref_blocked
    )

    if _alignment_check_needed:
        _ac_rubric_dict, _, _ = get_active_rubric()
        _ac_rubric_ver = _ac_rubric_dict.get("version", "?") if _ac_rubric_dict else "?"

        st.markdown("### What are you writing today?")
        st.success(
            f"Your rubric **v{_ac_rubric_ver}** is ready. "
            "Describe your writing task below and we'll generate a starting draft to see how well your rubric works for this task."
        )
        # Auto-resize: start small, grow with content
        _ac_prev_text = st.session_state.get("alignment_check_task_input", "")
        _ac_line_count = max(1, _ac_prev_text.count("\n") + 1)
        _ac_char_lines = max(1, len(_ac_prev_text) // 80)  # rough wrap estimate
        _ac_est_lines = max(_ac_line_count, _ac_char_lines)
        _ac_height = min(300, max(68, _ac_est_lines * 30 + 38))
        _ac_task_input = st.text_area(
            "Describe your writing task",
            placeholder="e.g., Write a professional email declining a meeting invitation while maintaining a good relationship.",
            height=_ac_height,
            key="alignment_check_task_input",
            label_visibility="collapsed"
        )
        _ac_col_start, _ac_col_skip = st.columns(2)
        with _ac_col_start:
            if st.button("Generate Starting Draft", type="primary", width="stretch", key="ac_start_btn"):
                if _ac_task_input.strip():
                    st.session_state.ranking_checkpoint_pending = {
                        "step": 2,
                        "writing_task": _ac_task_input.strip(),
                        "rubric_version": _ac_rubric_ver,
                        "is_conversation_start": True,
                    }
                    st.rerun()
                else:
                    st.warning("Please describe your writing task first.")
        with _ac_col_skip:
            if st.button("Skip, just start chatting", width="stretch", key="ac_skip_btn"):
                st.session_state.alignment_check_skipped = True
                st.rerun()

    # Create a container for streaming responses BEFORE chat_input
    # This ensures streaming content appears above the input, not below
    streaming_container = st.container()

    # User input (chat input and buttons) — hidden until preferences are provided or alignment check is done
    _no_project = not bool(st.session_state.get("current_project_id"))
    _ac_draft_pending = any(m.get("_ac_pending_draft") for m in st.session_state.get("messages", []))
    _rcp_active = st.session_state.get("ranking_checkpoint_pending") is not None
    _cc_review_active = st.session_state.get("chat_criteria_review_active", False) and not st.session_state.get("chat_criteria_review_confirmed", False)
    # [DISABLED] DP review no longer blocks chat — DP extraction removed for now
    _dp_review_pending = False
    # _dp_review_pending = (
    #     st.session_state.get("infer_decision_points") is not None
    #     and any(m.get('is_dp_review') for m in st.session_state.get("messages", []))
    #     and not st.session_state.get("infer_dp_dimension_confirmed", False)
    # )
    _chat_blocked = _no_project or _pref_blocked or _alignment_check_needed or _ac_draft_pending or _rcp_active or _cc_review_active or _dp_review_pending
    if _no_project:
        st.info("Create a project first to start writing. Use the **sidebar** to create a new project.")
    prompt = st.chat_input("Type your message here...", disabled=_chat_blocked)
    if prompt and not _chat_blocked:
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

        # Inject alignment pipeline conversation as foundation context
        # This gives the model full context of the draft generation, scoring,
        # rubric improvements, and verification that happened during alignment check.
        _alignment_pipeline = st.session_state.get("alignment_pipeline_messages", [])
        if _alignment_pipeline:
            for _pm in _alignment_pipeline:
                api_messages.append({"role": _pm["role"], "content": _pm["content"]})

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

                        # --- Parse probe signal from response (piggybacked on main API call) ---
                        _probe_signal_match = re.search(r'<probe_signal>(.*?)</probe_signal>', main_content, re.DOTALL)
                        _probe_signal_data = None
                        if _probe_signal_match:
                            try:
                                _probe_signal_data = json.loads(_probe_signal_match.group(1))
                            except json.JSONDecodeError:
                                # print(f"[PROBE] Failed to parse probe_signal JSON: {_probe_signal_match.group(1)}")
                                pass
                            # Strip the probe signal from display content
                            main_content = re.sub(r'\s*<probe_signal>.*?</probe_signal>\s*', '', main_content, flags=re.DOTALL).strip()

                        # --- Uncertainty Probe: decide whether to trigger --- [DISABLED for now]
                        has_draft_tag = bool(re.search(r'<draft>.*?</draft>', main_content, re.DOTALL))
                        trigger_probe = False
                        trigger_via_signal = False
                        if False and has_draft_tag and active_rubric_list:
                            _pdc_conv_id = st.session_state.get('selected_conversation', '_default')
                            _pdc_counts = st.session_state.probe_draft_counts
                            _pdc_counts[_pdc_conv_id] = _pdc_counts.get(_pdc_conv_id, 0) + 1
                            _drafts_since = _pdc_counts[_pdc_conv_id]
                            if _probe_signal_data and _probe_signal_data.get("criterion_name"):
                                # Model flagged uncertainty — trigger probe using the signal
                                trigger_probe = True
                                trigger_via_signal = True
                            elif _drafts_since >= PROBE_FALLBACK_INTERVAL:
                                # Fallback: model hasn't signaled uncertainty in N drafts, force a check
                                trigger_probe = True
                        # Count total drafts in conversation (excluding probes)
                        _total_drafts = sum(
                            1 for m in st.session_state.messages
                            if m.get('role') == 'assistant'
                            and not m.get('is_probe_log')
                            and not m.get('is_inline_rephrase')
                            and re.search(r'<draft>.*?</draft>', m.get('content', ''), re.DOTALL)
                        )
                        # Add 1 for the current draft about to be appended
                        if has_draft_tag:
                            _total_drafts += 1
                        # print(f"[DRAFT COUNT DEBUG] Total drafts in conversation (excl. probes): {_total_drafts}")
                        _pdc_conv_id_dbg = st.session_state.get('selected_conversation', '_default')
                        # print(f"[PROBE DEBUG] has_draft={has_draft_tag}, trigger_probe={trigger_probe}, via_signal={trigger_via_signal}, drafts_since_probe={st.session_state.probe_draft_counts.get(_pdc_conv_id_dbg, 0)}")

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

                        # --- Uncertainty Probe flow (runs AFTER draft is committed) ---
                        if trigger_probe:
                            try:
                                _probe_rubric_json = json.dumps(
                                    _rubric_to_json_serializable(active_rubric_dict), indent=2
                                ) if active_rubric_dict else ""
                                _probe_conv_text = _build_conversation_text(st.session_state.get("messages", []))
                                _probe_conv_text = _probe_conv_text[-3000:] if len(_probe_conv_text) > 3000 else _probe_conv_text

                                _probe_id_data = None

                                if trigger_via_signal:
                                    # Use the probe signal piggybacked on the main response
                                    _probe_id_data = {
                                        "criterion_name": _probe_signal_data.get("criterion_name", ""),
                                        "criterion_index": _probe_signal_data.get("criterion_index", -1),
                                        "interpretation_a": "",
                                        "interpretation_b": "",
                                        "uncertainty_reason": _probe_signal_data.get("uncertainty_reason", ""),
                                        "all_confident": False,
                                    }
                                    # print(f"[PROBE] Using piggybacked signal: {_probe_id_data['criterion_name']}")
                                else:
                                    # Fallback: separate API call for uncertainty identification
                                    response_placeholder.markdown("*Analyzing your rubric for ambiguous criteria — you may be asked to compare two draft variants to help clarify...*")

                                    # Build diagnostic priority guidance for probe
                                    _probe_diagnostic_guidance = ""
                                    _rk_results = st.session_state.get("ranking_checkpoint_results", [])
                                    if _rk_results:
                                        _latest_diag = _rk_results[-1]
                                        _diag_criteria = _latest_diag.get("criteria_analysis", [])
                                        if _diag_criteria:
                                            _already_probed = set()
                                            for _pr in st.session_state.get("probe_results", []):
                                                _pr_name = _pr.get("criterion_name", "").lower().strip()
                                                if _pr_name:
                                                    _already_probed.add(_pr_name)
                                            _priority_lines = []
                                            for _dc in _diag_criteria:
                                                _dc_name = _dc.get("name", "")
                                                _dc_class = _dc.get("classification", "")
                                                if _dc_name.lower().strip() in _already_probed:
                                                    continue
                                                if _dc_class == "UNDERPERFORMING":
                                                    _priority_lines.append(
                                                        f'- HIGH PRIORITY: "{_dc_name}" — UNDERPERFORMING '
                                                        f'(generic draft scored better by {abs(_dc.get("gap", 0))} points). '
                                                        f'Reason: {_dc.get("reasoning", "N/A")}'
                                                    )
                                                elif _dc_class == "REDUNDANT":
                                                    _priority_lines.append(
                                                        f'- MEDIUM PRIORITY: "{_dc_name}" — REDUNDANT '
                                                        f'(no score difference). '
                                                        f'Reason: {_dc.get("reasoning", "N/A")}'
                                                    )
                                            if _priority_lines:
                                                _probe_diagnostic_guidance = "\n".join(_priority_lines)

                                    _probe_id_prompt = PROBE_identify_uncertainty_prompt(
                                        _probe_rubric_json, _probe_conv_text, _probe_diagnostic_guidance
                                    )
                                    _probe_id_resp = _api_call_with_retry(
                                        model=MODEL_PRIMARY, max_tokens=800,
                                        messages=[{"role": "user", "content": _probe_id_prompt}]
                                    )
                                    _probe_id_text = "".join(b.text for b in _probe_id_resp.content if b.type == "text")
                                    _probe_id_match = re.search(r'\{[\s\S]*\}', _probe_id_text)
                                    _probe_id_data = json.loads(_probe_id_match.group()) if _probe_id_match else None
                                    # print(f"[PROBE] Fallback API call result: {_probe_id_data}")

                                if _probe_id_data and not _probe_id_data.get("all_confident", False):
                                    _probe_crit_name = _probe_id_data.get("criterion_name", "")
                                    _probe_interp_a = _probe_id_data.get("interpretation_a", "")
                                    _probe_interp_b = _probe_id_data.get("interpretation_b", "")
                                    _probe_reason = _probe_id_data.get("uncertainty_reason", "")
                                    _probe_crit_idx = _probe_id_data.get("criterion_index", -1)

                                    if _probe_crit_name and _probe_reason:
                                        # Step 2: Generate ONE alternative draft with a contrasting interpretation
                                        response_placeholder.markdown("*Generating comparison variant...*")
                                        # Extract the draft text from the assistant's response
                                        _probe_draft_match = re.search(r'<draft>(.*?)</draft>', main_content, re.DOTALL)
                                        _probe_current_draft = _probe_draft_match.group(1).strip() if _probe_draft_match else main_content[:2000]
                                        # Let the model read draft A + rubric + criterion and figure out
                                        # how A interpreted it, then generate B with a clearly different take
                                        _probe_var_prompt = PROBE_generate_variant_draft_prompt(
                                            _probe_conv_text, _probe_current_draft, _probe_rubric_json,
                                            _probe_crit_name, _probe_reason
                                        )
                                        _probe_var_resp = _api_call_with_retry(
                                            model=MODEL_PRIMARY, max_tokens=4000,
                                            messages=[{"role": "user", "content": _probe_var_prompt}]
                                        )
                                        _probe_var_text = "".join(b.text for b in _probe_var_resp.content if b.type == "text")
                                        _probe_var_match = re.search(r'\{[\s\S]*\}', _probe_var_text)
                                        _probe_var_data = json.loads(_probe_var_match.group()) if _probe_var_match else None

                                        if _probe_var_data and _probe_var_data.get("variant"):
                                            _alt_draft = _probe_var_data["variant"]
                                            _dim_varied = _probe_var_data.get("dimension_varied", "")
                                            # Use model's own descriptions of how each draft interprets the criterion
                                            _interp_a = _probe_var_data.get("draft_a_interpretation", _probe_interp_a or "")
                                            _interp_b = _probe_var_data.get("draft_b_interpretation", _probe_interp_b or "")
                                            # Randomly assign original vs alternative to A/B for blind comparison
                                            if random.random() < 0.5:
                                                _va, _vb = _probe_current_draft, _alt_draft
                                                _orig_slot = "a"  # original is Version A
                                            else:
                                                _va, _vb = _alt_draft, _probe_current_draft
                                                _orig_slot = "b"  # original is Version B
                                            st.session_state.probe_pending = {
                                                "criterion_name": _probe_crit_name,
                                                "criterion_index": _probe_crit_idx,
                                                "interpretation_a": _interp_a,
                                                "interpretation_b": _interp_b,
                                                "uncertainty_reason": _probe_reason,
                                                "variant_a": _va,
                                                "variant_b": _vb,
                                                "original_slot": _orig_slot,
                                                "original_draft": _probe_current_draft,
                                                "alternative_draft": _alt_draft,
                                                "dimension_varied": _dim_varied,
                                                "message_id": message_id,
                                                "rubric_version": rubric_version,
                                            }
                                            _pdc_conv_id_reset = st.session_state.get('selected_conversation', '_default')
                                            st.session_state.probe_draft_counts[_pdc_conv_id_reset] = 0  # Reset counter on successful probe
                                            # print(f"[PROBE] Probe ready: criterion='{_probe_crit_name}', original_slot='{_orig_slot}'")
                                        else:
                                            # print("[PROBE] Variant generation failed or empty, skipping probe")
                                            pass
                                    else:
                                        # print("[PROBE] Uncertainty identification returned incomplete data, skipping")
                                        pass
                                else:
                                    # print("[PROBE] Model confident about all criteria, skipping probe")
                                    pass
                            except Exception as _probe_err:
                                # print(f"[PROBE] Probe flow failed: {_probe_err}")
                                # Fall through to normal rerun

                                pass
                        _auto_save_conversation()
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
        btn_col1, btn_col2, btn_col3 = st.columns(3)

        with btn_col1:
            infer_button = st.button("🔍 Infer Rubric", width="stretch", disabled=_chat_blocked)
            if infer_button and not _chat_blocked:
                if not st.session_state.messages:
                    st.error("No conversation to infer rubric from!")
                else:
                    # Single call: infer rubric + extract DPs together
                    # Include all visible conversation messages (user, assistant, system)
                    # so the model sees the full picture: drafts, probes, rubric changes,
                    # alignment checks, user feedback, accept/revert decisions, etc.
                    _infer_filtered = [
                        m for m in copy.deepcopy(st.session_state.messages)
                        if m.get('role') in ('user', 'assistant', 'system')
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

                            # System message announcing the inferred rubric
                            st.session_state.messages.append({
                                "role": "system",
                                "content": f"Rubric **v{_rb_ver}** inferred from your conversation.",
                                "conversation_id": st.session_state.get("selected_conversation"),
                            })

                            # Append rubric-inferred marker (no DPs yet — hidden from display)
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
                            _auto_save_conversation()

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
        
        with btn_col2:
            pass  # Assess Draft removed — auto-runs in background after every draft

        with btn_col3:
            # Toggle delete mode button
            if st.session_state.message_delete_mode:
                if st.button("✖️ Cancel Delete", width="stretch", type="secondary"):
                    st.session_state.message_delete_mode = False
                    st.session_state.messages_to_delete = set()
                    st.rerun()
            else:
                if st.button("🗑️ Delete Messages", width="stretch"):
                    st.session_state.message_delete_mode = True
                    st.session_state.messages_to_delete = set()
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
            st.session_state.probe_draft_counts = {}
            st.session_state.probe_pending = None
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
                            st.session_state.survey_responses = {"task_a": {}, "task_b": {}}
                    elif isinstance(_loaded_survey, dict) and "task_a" in _loaded_survey:
                        st.session_state.survey_responses = _loaded_survey
                    else:
                        st.session_state.survey_responses = {"task_a": {}, "task_b": {}}
                else:
                    st.session_state.survey_responses = {"task_a": {}, "task_b": {}}
            else:
                st.session_state.survey_responses = {"task_a": {}, "task_b": {}}
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

            # Load probe results, ranking checkpoint results, and judge validations from database
            st.session_state.probe_results = []
            st.session_state.ranking_checkpoint_results = []
            st.session_state.ranking_checkpoint_pending = None
            st.session_state.ranking_checkpoint_auto_triggered = False
            st.session_state.alignment_check_done = False
            st.session_state.alignment_check_skipped = False
            if _supabase and _new_pid:
                try:
                    _probe_loaded = load_project_data(_supabase, _new_pid, "probe_results")
                    if isinstance(_probe_loaded, list):
                        st.session_state.probe_results = _probe_loaded
                except Exception:
                    pass
                try:
                    _rk_loaded = load_project_data(_supabase, _new_pid, "alignment_diagnostic")
                    if isinstance(_rk_loaded, list):
                        st.session_state.ranking_checkpoint_results = _rk_loaded
                except Exception:
                    pass
                try:
                    _ge_loaded = load_project_data(_supabase, _new_pid, "grade_evaluation")
                    if isinstance(_ge_loaded, list):
                        st.session_state.grade_evaluation_history = _ge_loaded
                except Exception:
                    pass
                try:
                    _rt_loaded = load_project_data(_supabase, _new_pid, "grade_retest")
                    if isinstance(_rt_loaded, list):
                        st.session_state.grade_retest_history = _rt_loaded
                except Exception:
                    pass
                try:
                    _drt_loaded = load_project_data(_supabase, _new_pid, "diagnostic_retest")
                    if isinstance(_drt_loaded, list):
                        st.session_state.diagnostic_retest_history = _drt_loaded
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

        # Load probe results and ranking checkpoint results on startup if not already loaded
        if _startup_pid and _startup_sb and not st.session_state.get('probe_results'):
            try:
                _probe_startup = load_project_data(_startup_sb, _startup_pid, "probe_results")
                if isinstance(_probe_startup, list):
                    st.session_state.probe_results = _probe_startup
            except Exception:
                pass
        if _startup_pid and _startup_sb and not st.session_state.get('ranking_checkpoint_results'):
            try:
                _rk_startup = load_project_data(_startup_sb, _startup_pid, "alignment_diagnostic")
                if isinstance(_rk_startup, list):
                    st.session_state.ranking_checkpoint_results = _rk_startup
            except Exception:
                pass
        if _startup_pid and _startup_sb and not st.session_state.get('grade_evaluation_history'):
            try:
                _ge_startup = load_project_data(_startup_sb, _startup_pid, "grade_evaluation")
                if isinstance(_ge_startup, list):
                    st.session_state.grade_evaluation_history = _ge_startup
            except Exception:
                pass
        if _startup_pid and _startup_sb and not st.session_state.get('grade_retest_history'):
            try:
                _rt_startup = load_project_data(_startup_sb, _startup_pid, "grade_retest")
                if isinstance(_rt_startup, list):
                    st.session_state.grade_retest_history = _rt_startup
            except Exception:
                pass
        if _startup_pid and _startup_sb and not st.session_state.get('diagnostic_retest_history'):
            try:
                _drt_startup = load_project_data(_startup_sb, _startup_pid, "diagnostic_retest")
                if isinstance(_drt_startup, list):
                    st.session_state.diagnostic_retest_history = _drt_startup
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
            if st.button("Delete Project", type="primary", width="stretch", key="delete_project_btn"):
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
                            st.session_state.survey_responses = {"task_a": {}, "task_b": {}}
                            st.session_state.probe_draft_counts = {}
                            st.session_state.probe_pending = None
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

        if st.button("Create Project", width="stretch"):
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
                    st.session_state.probe_draft_counts = {}
                    st.session_state.probe_pending = None
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
        # Only set index when session state hasn't already been set by apply/save actions
        _vs_kwargs = {"key": "rubric_version_selector"}
        if "rubric_version_selector" not in st.session_state:
            _vs_kwargs["index"] = active_idx if active_idx is not None else 0
        selected_version = st.selectbox(
            "Active Rubric Version:",
            options=version_options,
            **_vs_kwargs
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
            # Use criterion name as stable widget ID (not list index which shifts on removal)
            _stable_id = criterion_name.replace(" ", "_").lower()[:40]

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
                    # Description (editable); use stable criterion name in key (not list index)
                    desc_key = f"criterion_desc_{_stable_id}_{version_key}_{ui_ver}"
                    # Seed widget cache only on first render for this key
                    if desc_key not in st.session_state:
                        st.session_state[desc_key] = criterion.get("description", "")
                    description = st.text_area(
                        "Description",
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
                            dim_key = f"criterion_dim_{_stable_id}_{dim_id}_{version_key}_{ui_ver}"
                            # Seed widget cache only on first render for this key
                            if dim_key not in st.session_state:
                                st.session_state[dim_key] = dim.get("label", "")
                            dim_label = st.text_input(
                                f"Dimension {dim_idx + 1}",
                                key=dim_key,
                                label_visibility="collapsed",
                                placeholder=f"Dimension {dim_idx + 1} label..."
                            )
                            dim_labels_updated[dim_idx] = dim_label
                        with col2:
                            remove_key = f"remove_dim_{_stable_id}_{dim_id}_{version_key}_{ui_ver}"
                            if st.button("➖", key=remove_key, help="Remove dimension"):
                                # Mark this dimension ID for deletion
                                dim_to_delete = dim_id

                    # Write dimension labels back — safe because we pre-set widget cache above
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
                    add_dim_key = f"add_dim_{_stable_id}_{version_key}_{ui_ver}"
                    if st.button("➕ Add Dimension", key=add_dim_key):
                        # Generate a unique ID using timestamp
                        new_dim_id = f"dim_{int(time.time() * 1000)}"
                        st.session_state.editing_criteria[original_idx]["dimensions"].append({"id": new_dim_id, "label": ""})
                        st.rerun()

                    # Write description back — safe because we pre-set the widget cache
                    # from editing_criteria above, so any diff is a real user edit
                    st.session_state.editing_criteria[original_idx]["description"] = description

                    # Remove criterion button
                    if st.button("🗑️ Remove Criterion", key=f"remove_{_stable_id}"):
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

            # Compare by name (not list position) to handle reordering
            orig_map = {c.get("name", "").lower().strip(): c for c in original_rubric}
            edit_map = {c.get("name", "").lower().strip(): c for c in editing}

            if set(orig_map.keys()) != set(edit_map.keys()):
                return True

            for key, orig in orig_map.items():
                edit = edit_map[key]
                # Check description (normalize whitespace to avoid false positives from text_area)
                if ' '.join(orig.get("description", "").split()) != ' '.join(edit.get("description", "").split()):
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
                    if ' '.join(od.get("label", "").split()) != ' '.join(ed.get("label", "").split()):
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
            if st.button("📝 Log Changes", width="stretch", disabled=not has_changes):
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
                                "original_draft": last_draft,
                                "old_rubric": copy.deepcopy(old_rubric_for_diff),
                                "new_rubric": copy.deepcopy(list(st.session_state.editing_criteria)),
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
                            _auto_save_conversation()
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
                                _auto_save_conversation()
                                st.toast(f"Changes logged. Could not generate draft: {str(e)}")
                                st.rerun()

                    _auto_save_conversation()
                    st.toast("Changes logged & draft updated!" if draft_appended else "Changes logged. No draft in conversation to update — send a message in Chat and get a draft first.")
                    st.rerun()
        with save_col:
            if st.button("💾 Save Version", width="stretch", type="primary", disabled=not has_changes):
                # Save as a NEW version in rubric history
                if rubric_history is not None:
                    saved_criteria = copy.deepcopy(st.session_state.editing_criteria)

                    new_version = next_version_number()
                    new_rubric_entry = {
                        "version": new_version,
                        "rubric": saved_criteria,
                        "source": "edited",
                        "conversation_id": st.session_state.get("selected_conversation"),
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

                    # Persist log_changes as project data for analysis
                    _lc_pid = st.session_state.get('current_project_id')
                    _lc_sb = st.session_state.get('supabase')
                    if _lc_sb and _lc_pid:
                        try:
                            save_project_data(_lc_sb, _lc_pid, "log_changes", {
                                "timestamp": datetime.now().isoformat(),
                                "source": "rubric_editor_save",
                                "edit_classification": edit_class,
                                "rubric_version": new_version,
                                "old_version": old_version_num,
                                "draft_regenerated": False,
                            })
                        except Exception:
                            pass

                    # Update session state to point to new version
                    st.session_state.active_rubric_idx = len(hist) - 1
                    st.session_state.rubric = saved_criteria
                    st.session_state.editing_criteria = copy.deepcopy(saved_criteria)

                    _auto_save_conversation()
                    st.toast(f"Saved as v{new_version}!")
                    st.rerun()
        with reset_col:
            if st.button("↩️ Revert", width="stretch", type="secondary", disabled=not has_changes):
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
                    _auto_save_conversation()
                    st.toast("Rubric reset to saved version!")
                    st.rerun()

        # Delete Version button
        if st.button("🗑️ Delete Version", width="stretch", type="secondary"):
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
                        # Clear the version selector so it picks up active_rubric_idx on rerun
                        if "rubric_version_selector" in st.session_state:
                            del st.session_state["rubric_version_selector"]
                    else:
                        # Deleted the last version — clear rubric state
                        st.session_state.active_rubric_idx = 0
                        st.session_state.rubric = []
                        st.session_state.editing_criteria = []
                        if "rubric_version_selector" in st.session_state:
                            del st.session_state["rubric_version_selector"]

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
            pass  # Source label removed — version number is sufficient

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
        rubric_options = [f"v{r.get('version', 1)}" for r in rubric_history]
        col1, col2 = st.columns(2)

        with col1:
            rubric_a_idx = st.selectbox("Rubric A:", options=list(range(len(rubric_history))),
                                        format_func=lambda x: rubric_options[x], key="tab2_rubric_a_select")
        with col2:
            rubric_b_idx = st.selectbox("Rubric B:", options=list(range(len(rubric_history))),
                                        format_func=lambda x: rubric_options[x], key="tab2_rubric_b_select")

        # Display selected rubrics with mapping summary
        st.markdown("---")

        # Build criteria lists and matching info
        rubric_a_version = rubric_history[rubric_a_idx].get('version', 1)
        rubric_b_version = rubric_history[rubric_b_idx].get('version', 1)
        _src_a = rubric_history[rubric_a_idx].get("source", "")
        _src_b = rubric_history[rubric_b_idx].get("source", "")
        _crit_a = rubric_history[rubric_a_idx].get("rubric", [])
        _crit_b = rubric_history[rubric_b_idx].get("rubric", [])

        # Build name-to-index maps for matching
        _names_a = [(i, c.get("name", "").lower().strip(), c.get("name", "")) for i, c in enumerate(_crit_a)]
        _names_b = {c.get("name", "").lower().strip(): (i, c.get("name", "")) for i, c in enumerate(_crit_b)}
        _names_b_set = set(_names_b.keys())
        _names_a_set = set(n for _, n, _ in _names_a)

        # Build mapping rows for the summary table
        _mapping_rows = []
        _n_unchanged = 0
        _n_modified = 0
        _n_removed = 0
        _n_new = 0
        for _a_i, a_key, a_name in _names_a:
            if a_key in _names_b:
                _b_i, b_name = _names_b[a_key]
                _a_crit = _crit_a[_a_i]
                _b_crit = _crit_b[_b_i]
                _changes = []
                if _a_crit.get("description", "") != _b_crit.get("description", ""):
                    _changes.append("description")
                if _a_crit.get("priority", 0) != _b_crit.get("priority", 0):
                    _changes.append("priority")
                if [d.get("label", "") for d in _a_crit.get("dimensions", [])] != [d.get("label", "") for d in _b_crit.get("dimensions", [])]:
                    _changes.append("dimensions")
                if _changes:
                    _n_modified += 1
                    _mapping_rows.append({
                        f"v{rubric_a_version} Criterion": a_name,
                        "Status": "Modified",
                        f"v{rubric_b_version} Criterion": b_name,
                        "Changes": ", ".join(_changes)
                    })
                else:
                    _n_unchanged += 1
                    _mapping_rows.append({
                        f"v{rubric_a_version} Criterion": a_name,
                        "Status": "Unchanged",
                        f"v{rubric_b_version} Criterion": b_name,
                        "Changes": ""
                    })
            else:
                _n_removed += 1
                _mapping_rows.append({
                    f"v{rubric_a_version} Criterion": a_name,
                    "Status": "Removed",
                    f"v{rubric_b_version} Criterion": "---",
                    "Changes": ""
                })
        for b_key in _names_b:
            if b_key not in _names_a_set:
                _, b_name = _names_b[b_key]
                _n_new += 1
                _mapping_rows.append({
                    f"v{rubric_a_version} Criterion": "---",
                    "Status": "New",
                    f"v{rubric_b_version} Criterion": b_name,
                    "Changes": ""
                })

        # Show summary stats
        _stats_parts = []
        if _n_unchanged:
            _stats_parts.append(f"**{_n_unchanged}** unchanged")
        if _n_modified:
            _stats_parts.append(f"**{_n_modified}** modified")
        if _n_removed:
            _stats_parts.append(f"**{_n_removed}** removed")
        if _n_new:
            _stats_parts.append(f"**{_n_new}** new")
        st.markdown(f"**Criteria Mapping:** {' | '.join(_stats_parts)}")

        # Show mapping table
        if _mapping_rows:
            import pandas as pd
            _map_df = pd.DataFrame(_mapping_rows)

            def _color_status(val):
                colors = {
                    "Unchanged": "background-color: #E8F5E9; color: #2E7D32",
                    "Modified": "background-color: #FFF3E0; color: #E65100",
                    "Removed": "background-color: #FFEBEE; color: #C62828",
                    "New": "background-color: #E3F2FD; color: #1565C0",
                }
                return colors.get(val, "")

            try:
                _styled_df = _map_df.style.map(_color_status, subset=["Status"]).set_properties(**{"text-align": "left"})
            except AttributeError:
                _styled_df = _map_df.style.applymap(_color_status, subset=["Status"]).set_properties(**{"text-align": "left"})
            st.dataframe(_styled_df, width="stretch", hide_index=True)

        # Render 2-column rubric display
        col_rubric_a, col_rubric_b = st.columns(2)

        with col_rubric_a:
            st.markdown(f"### Rubric v{rubric_a_version}")
            display_rubric_criteria(rubric_history[rubric_a_idx], st, comparison_rubric_data=rubric_history[rubric_b_idx])

        with col_rubric_b:
            st.markdown(f"### Rubric v{rubric_b_version}")
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

                    # Save comparison results to database
                    _cmp_project_id = st.session_state.get("current_project_id")
                    if _cmp_project_id:
                        save_project_data(supabase, _cmp_project_id, "rubric_comparison", {
                            "timestamp": datetime.now().isoformat(),
                            "writing_task": compare_input.strip(),
                            "rubric_a": {
                                "version": rubric_a_version,
                                "source": _src_a,
                                "criteria_count": len(_crit_a),
                                "criteria_names": [c.get("name", "") for c in _crit_a],
                            },
                            "rubric_b": {
                                "version": rubric_b_version,
                                "source": _src_b,
                                "criteria_count": len(_crit_b),
                                "criteria_names": [c.get("name", "") for c in _crit_b],
                            },
                            "criteria_mapping": {
                                "unchanged": _n_unchanged,
                                "modified": _n_modified,
                                "removed": _n_removed,
                                "new": _n_new,
                                "details": _mapping_rows,
                            },
                            "comparison_output": {
                                "base_draft": result.get("base_txt", ""),
                                "rubric_a_revision": result.get("a_txt", ""),
                                "rubric_b_revision": result.get("b_txt", ""),
                                "key_differences": result.get("key_diffs", ""),
                                "summary": result.get("summary", ""),
                            },
                        })
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
            if st.button("🔄 Reset All", width="stretch", key="infer_reset"):
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
                if st.button("🔄 Reset All", width="stretch", key="build_reset"):
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
                    if st.button("🔍 Classify Edits", type="primary", width="stretch", key="build_classify_btn"):
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
                        if st.button("✍️ Generate Drafts from Both Rubrics", type="primary", width="stretch", key="build_generate_btn"):
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
                        if st.button("💾 Save Evaluation", type="primary", width="stretch", key="build_save_pref"):
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
                        if st.button("🔬 Run LLM Judge", type="primary", width="stretch", key="build_judge_btn"):
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
    
                    if st.button("💾 Save Build Evaluation", type="primary", width="stretch", key="save_build_eval"):
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
                    if st.button("🔄 Reset All", width="stretch", key="grade_reset"):
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
                    if st.button("✍️ Generate Draft Pair", type="primary", width="stretch", key="grade_generate_btn"):
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
                        if st.button("💾 Save Ratings", type="primary", width="stretch", key="grade_save_ratings"):
                            st.session_state.grade_user_overall_pref = grade_overall_pref
                            st.session_state.grade_user_dim_ratings = grade_dim_ratings
                            st.rerun()
    
                    # ==================== STEP 3: LLM JUDGE EVALUATION ====================
                    if st.session_state.grade_user_overall_pref:
                        st.divider()
                        st.subheader("Step 3: LLM Judge Evaluation")
                        st.markdown("Two LLM judges evaluate the same drafts: one using your rubric + conversation context, one using only generic criteria.")
    
                        if not st.session_state.grade_rubric_judge_result or not st.session_state.grade_generic_judge_result:
                            if st.button("🔬 Run LLM Judges", type="primary", width="stretch", key="grade_judge_btn"):
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

                                # Generic tau: map generic judge avg scores to user per-criterion ranks
                                for _crit_name_g in rubric_dim_agreements:
                                    generic_ranks_list.append(generic_a_avg)
                                    generic_ranks_list.append(generic_b_avg)
                                generic_tau, generic_p = (None, None)
                                if len(user_ranks_list) >= 4 and len(generic_ranks_list) >= 4:
                                    generic_tau, generic_p = kendalltau(user_ranks_list, generic_ranks_list)

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
                                    "generic_tau": generic_tau,
                                    "generic_p": generic_p,
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
    
                            st.markdown("### Per-Dimension Agreement")
                            col_tau_r, col_tau_g, col_rate = st.columns(3)
                            with col_tau_r:
                                tau_val = agreement.get("rubric_tau")
                                if tau_val is not None:
                                    interpretation = "Strong" if abs(tau_val) > 0.6 else ("Moderate" if abs(tau_val) > 0.3 else "Weak")
                                    st.metric("Rubric Judge τ", f"{tau_val:.3f}", delta=interpretation, delta_color="off")
                                else:
                                    st.metric("Rubric Judge τ", "N/A")
                            with col_tau_g:
                                g_tau_val = agreement.get("generic_tau")
                                if g_tau_val is not None:
                                    g_interp = "Strong" if abs(g_tau_val) > 0.6 else ("Moderate" if abs(g_tau_val) > 0.3 else "Weak")
                                    st.metric("Generic Judge τ", f"{g_tau_val:.3f}", delta=g_interp, delta_color="off")
                                else:
                                    st.metric("Generic Judge τ", "N/A")
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
                                    width="stretch",
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
                                if st.button("💾 Save Grade Evaluation", type="primary", width="stretch", key="grade_save_btn"):
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
                                            # Also append to history for dashboard
                                            if 'grade_evaluation_history' not in st.session_state:
                                                st.session_state.grade_evaluation_history = []
                                            st.session_state.grade_evaluation_history.append(export_data)
                                            # Launch background retest for reliability measurement
                                            import threading as _rt_threading
                                            # Rebuild conv_context and rubric_criteria_json safely
                                            _rt_conv_msgs = [m for m in st.session_state.messages if m.get("role") != "system" or "<!--" not in m.get("content", "")]
                                            _rt_recent = _rt_conv_msgs[-10:] if len(_rt_conv_msgs) > 10 else _rt_conv_msgs
                                            _rt_conv_ctx = "\n".join([f"[{m.get('role', 'unknown')}]: {m.get('content', '')[:500]}" for m in _rt_recent])
                                            _rt_rubric_json = json.dumps(grade_rubric_list, indent=2)
                                            _rt_args = {
                                                "draft_good": st.session_state.grade_draft_good,
                                                "draft_degraded": st.session_state.grade_draft_degraded,
                                                "rubric_criteria_json": _rt_rubric_json,
                                                "conv_context": _rt_conv_ctx,
                                                "original_rubric_result": st.session_state.grade_rubric_judge_result,
                                                "original_generic_result": st.session_state.grade_generic_judge_result,
                                                "supabase": supabase,
                                                "project_id": project_id,
                                                "grade_eval_timestamp": export_data["timestamp"],
                                                "results_list_ref": st.session_state.get("grade_retest_history", []),
                                            }
                                            _rt_threading.Thread(
                                                target=_run_grade_retest_bg,
                                                args=(_rt_args,),
                                                daemon=True
                                            ).start()
                                            st.success("✅ Grade evaluation saved successfully!")
                                        else:
                                            st.error("Failed to save evaluation.")
                            else:
                                st.success("✅ Grade evaluation has been saved.")
    
# ============ EVALUATE: GRADING TAB ============
with tab_grading:
    st.header("📊 Evaluate: Grading")

    # --- Check for user preference description (coldstart text) ---
    _gr_coldstart = st.session_state.get("infer_coldstart_text", "").strip()
    _gr_tab_blocked = False
    if not _gr_coldstart:
        st.warning("No writing preferences found. Please describe your writing preferences in the project setup before continuing.")
        _gr_tab_blocked = True

    # Get active rubric and full history
    _gr_rubric_dict, _gr_rubric_idx, _ = get_active_rubric()
    _gr_hist = load_rubric_history()

    if _gr_tab_blocked:
        pass  # Warning already shown above
    elif not _gr_rubric_dict:
        st.warning("No active rubric found. Please create a rubric first in the **Evaluate: Infer** tab.")
    else:
        # ============ SECTION 1: Rubric Probe History ============ [DISABLED]
        # st.subheader("Section 1: Rubric Probe History")
        # (Probe Results UI commented out — not used in current evaluation)

        st.divider()

        # ============ SECTION 2: Rubric Alignment Diagnostic Results ============
        st.subheader("Section 2: Rubric Alignment Diagnostics")
        _rk_results = st.session_state.get("ranking_checkpoint_results", [])
        if not _rk_results:
            st.info("No alignment diagnostic data yet. Complete a diagnostic in the Chat tab to see results here.")
        else:
            _class_icons = {"DIFFERENTIATING": "[+]", "REDUNDANT": "[=]", "UNDERPERFORMING": "[-]", "PREFERENCE_GAP": "[~]"}
            _class_colors = {"DIFFERENTIATING": "#E8F5E9", "REDUNDANT": "#FFF3E0", "UNDERPERFORMING": "#FFEBEE", "PREFERENCE_GAP": "#E3F2FD"}

            for _rki, _rkr in enumerate(reversed(_rk_results)):
                _rk_ts = _rkr.get("timestamp", "")[:16].replace("T", " ")
                _rk_ver = _rkr.get("rubric_version", "?")
                _rk_expanded = _rki == 0
                _rk_ranking = _rkr.get("user_ranking", [])
                _rk_source_names = {"rubric": "Rubric-guided", "generic": "Generic", "preference": "Preferences"}
                if _rk_ranking:
                    _pref_label = " > ".join(_rk_source_names.get(s, s) for s in _rk_ranking)
                else:
                    _pref_label = {"rubric": "Rubric-guided", "generic": "Generic", "tie": "About the same"}.get(
                        _rkr.get("user_preference", ""), _rkr.get("user_preference", "?")
                    )
                with st.expander(f"Diagnostic #{len(_rk_results) - _rki} ({_rk_ts}, rubric v{_rk_ver})", expanded=_rk_expanded):
                    st.markdown(f"**Writing task:** {_rkr.get('writing_task', '')}")
                    st.markdown(f"**User ranking:** {_pref_label}")
                    if _rkr.get("user_reason"):
                        st.markdown(f"**Reason:** {_rkr['user_reason']}")

                    # Per-criterion scores table
                    _ca = _rkr.get("criteria_analysis", [])
                    if _ca:
                        _has_pref_scores = any("preference_score" in _c for _c in _ca)
                        st.markdown("**Per-criterion analysis:**")
                        _pref_header = '<th style="text-align:center;padding:8px 14px;">Pref. Draft</th>' if _has_pref_scores else ""
                        _ca_table = f'''<table style="width:100%;border-collapse:separate;border-spacing:0 4px;margin:12px 0;">
<tr style="background:#f8f9fa;">
  <th style="text-align:left;padding:8px 14px;">Criterion</th>
  <th style="text-align:center;padding:8px 14px;">Rubric Draft</th>
  <th style="text-align:center;padding:8px 14px;">Generic Draft</th>
  {_pref_header}
  <th style="text-align:center;padding:8px 14px;">Gap (R-G)</th>
  <th style="text-align:center;padding:8px 14px;">Classification</th>
</tr>'''
                        for _c in _ca:
                            _cls = _c.get("classification", "REDUNDANT")
                            _icon = _class_icons.get(_cls, "")
                            _bg = _class_colors.get(_cls, "#f5f5f5")
                            _gap = _c.get("gap", 0)
                            _gap_str = f"+{_gap}" if _gap > 0 else str(_gap)
                            _pref_cell = f'<td style="text-align:center;padding:8px 14px;">{_c.get("preference_score", "?")}/5</td>' if _has_pref_scores else ""
                            _ca_table += f'''<tr style="background:white;">
  <td style="padding:8px 14px;font-weight:600;">{_c.get("name", "?")}</td>
  <td style="text-align:center;padding:8px 14px;">{_c.get("rubric_score", "?")}/5</td>
  <td style="text-align:center;padding:8px 14px;">{_c.get("generic_score", "?")}/5</td>
  {_pref_cell}
  <td style="text-align:center;padding:8px 14px;font-weight:600;">{_gap_str}</td>
  <td style="text-align:center;padding:8px 14px;background:{_bg};border-radius:4px;font-weight:500;">{_icon} {_cls}</td>
</tr>'''
                        _ca_table += '</table>'
                        st.markdown(_ca_table, unsafe_allow_html=True)

                        # Show reasoning in sub-expander
                        with st.expander("Show reasoning per criterion", expanded=False):
                            for _c in _ca:
                                if _c.get("reasoning"):
                                    st.markdown(f"**{_c.get('name', '?')}:** {_c['reasoning']}")

                    # Suggestion text
                    if _rkr.get("suggestion_text"):
                        with st.expander("Suggested changes", expanded=False):
                            st.markdown(_rkr["suggestion_text"])

            st.divider()
            st.markdown("**Understanding the classifications:**")
            st.markdown(
                "- **[+] DIFFERENTIATING** — The rubric-guided draft scores highest on this criterion. "
                "Your rubric is effectively guiding the LLM here.\n"
                "- **[~] PREFERENCE_GAP** — The preference-based draft scores higher than the rubric draft. "
                "Your original preferences capture something the rubric hasn't yet.\n"
                "- **[=] REDUNDANT** — All drafts score equally. The criterion may need sharpening or lower priority.\n"
                "- **[-] UNDERPERFORMING** — The generic draft scores highest. "
                "This criterion's description may need revision to be clearer for the LLM."
            )

        # ── Section 3: Evaluation Summary Dashboard ──
        st.divider()
        st.subheader("Section 3: Evaluation Summary")

        _eval_probe_results = st.session_state.get("probe_results", [])
        _eval_diag_results = st.session_state.get("ranking_checkpoint_results", [])
        _eval_retest_history = st.session_state.get("grade_retest_history", [])
        _eval_diag_retest_history = st.session_state.get("diagnostic_retest_history", [])
        _eval_all_retests = _eval_retest_history + _eval_diag_retest_history
        _eval_rubric_hist = load_rubric_history() if st.session_state.get("current_project_id") else []

        _has_any_data = bool(_eval_probe_results or _eval_diag_results or _eval_all_retests)

        if not _has_any_data:
            st.info("No evaluation data yet. Data appears automatically as you chat and complete diagnostics.")
        else:
            # [DISABLED: Grading Reliability tab removed — not used in current evaluation]
            # _ev_tab_rel, _ev_tab_imp, _ev_tab_pref, _ev_tab_eff = st.tabs([
            #     "Grading Reliability", "Rubric Improvement", "User Preference", "Rubric vs Generic"
            # ])
            _ev_tab_imp, _ev_tab_pref, _ev_tab_eff = st.tabs([
                "Rubric Improvement", "User Preference", "Rubric vs Generic"
            ])

            # ════════════════════════════════════════
            # TAB 1: Grading Reliability [DISABLED — not used in current evaluation]
            # ════════════════════════════════════════
            # (Grading Reliability / Kendall's τ retest UI commented out)

            # ════════════════════════════════════════
            # TAB 2: Rubric Improvement
            # ════════════════════════════════════════
            with _ev_tab_imp:
                st.markdown("Tracks how your rubric evolves through probes and diagnostics.")
                st.markdown("")

                _imp_versions = len(_eval_rubric_hist)
                _imp_probes_answered = sum(1 for p in _eval_probe_results if p.get("user_choice") != "skip")
                _imp_probes_applied = sum(1 for p in _eval_probe_results if p.get("rubric_updated"))

                # Classification shift
                _imp_shift_count = 0
                if len(_eval_diag_results) >= 2:
                    _earliest_diag = _eval_diag_results[0].get("criteria_analysis", [])
                    _latest_diag = _eval_diag_results[-1].get("criteria_analysis", [])
                    _earliest_class = {c.get("name", "").lower(): c.get("classification", "") for c in _earliest_diag}
                    for _lc in _latest_diag:
                        _lc_name = _lc.get("name", "").lower()
                        _lc_class = _lc.get("classification", "")
                        _ec_class = _earliest_class.get(_lc_name, "")
                        if _ec_class and _ec_class != "DIFFERENTIATING" and _lc_class == "DIFFERENTIATING":
                            _imp_shift_count += 1

                _imp_col1, _imp_col2, _imp_col3 = st.columns(3)
                with _imp_col1:
                    st.metric("Rubric Versions", _imp_versions)
                with _imp_col2:
                    if _imp_probes_answered > 0:
                        _imp_update_rate = (_imp_probes_applied / _imp_probes_answered * 100)
                        st.metric("Probe → Update", f"{_imp_probes_applied}/{_imp_probes_answered} ({_imp_update_rate:.0f}%)")
                    else:
                        st.metric("Probe → Update", "—", help="Answer probes and apply suggestions to see this")
                with _imp_col3:
                    if len(_eval_diag_results) >= 2:
                        st.metric("Criteria Improved", _imp_shift_count, help="Shifted to DIFFERENTIATING between first and latest diagnostic")
                    else:
                        st.metric("Criteria Improved", "—", help="Need 2+ diagnostics to compare")

                # Per-criterion trajectory
                if len(_eval_diag_results) >= 2:
                    st.markdown("")
                    st.caption("How each criterion's rubric-vs-generic gap changed between your first and latest diagnostic. "
                               "Positive gap = rubric draft scored higher.")
                    _imp_trajectories = {}
                    for _di, _dr in enumerate(_eval_diag_results):
                        _dr_ver = _dr.get("rubric_version", f"d{_di+1}")
                        for _ca in _dr.get("criteria_analysis", []):
                            _ca_name = _ca.get("name", "")
                            if _ca_name not in _imp_trajectories:
                                _imp_trajectories[_ca_name] = []
                            _imp_trajectories[_ca_name].append({
                                "version": _dr_ver,
                                "rubric_score": _ca.get("rubric_score", 0),
                                "generic_score": _ca.get("generic_score", 0),
                                "gap": _ca.get("gap", 0),
                                "classification": _ca.get("classification", "")
                            })
                    if _imp_trajectories:
                        _traj_rows = []
                        for _t_name, _t_entries in sorted(_imp_trajectories.items()):
                            if len(_t_entries) >= 2:
                                _first = _t_entries[0]
                                _last = _t_entries[-1]
                                _gap_change = _last["gap"] - _first["gap"]
                                _trend = "Improved" if _gap_change > 0 else ("Declined" if _gap_change < 0 else "Stable")
                                _traj_rows.append({
                                    "Criterion": _t_name,
                                    "First Diag": f"R:{_first['rubric_score']} G:{_first['generic_score']} (gap {_first['gap']:+d})",
                                    "Latest Diag": f"R:{_last['rubric_score']} G:{_last['generic_score']} (gap {_last['gap']:+d})",
                                    "Status": f"{_first['classification'][:5]} → {_last['classification'][:5]}",
                                    "Trend": _trend,
                                })
                        if _traj_rows:
                            st.dataframe(_traj_rows, width="stretch", hide_index=True)

            # ════════════════════════════════════════
            # TAB 3: User Preference
            # ════════════════════════════════════════
            with _ev_tab_pref:
                st.markdown("In each diagnostic, you rank drafts in a blind comparison "
                            "(without knowing which used your rubric, preferences, or no guidance).")
                st.markdown("")

                # Count how often each source was ranked 1st
                _sat_diag_rubric_1st = 0
                _sat_diag_generic_1st = 0
                _sat_diag_pref_1st = 0
                for _d in _eval_diag_results:
                    _d_ranking = _d.get("user_ranking", [])
                    if _d_ranking:
                        _first = _d_ranking[0]
                    else:
                        _first = _d.get("user_preference", "")
                    if _first == "rubric":
                        _sat_diag_rubric_1st += 1
                    elif _first == "generic":
                        _sat_diag_generic_1st += 1
                    elif _first == "preference":
                        _sat_diag_pref_1st += 1
                _sat_diag_total = len(_eval_diag_results)

                if _sat_diag_total == 0:
                    st.info("No diagnostics completed yet. Preference data appears after you complete an alignment diagnostic.")
                else:
                    _has_any_pref = _sat_diag_pref_1st > 0
                    if _has_any_pref:
                        _pref_col1, _pref_col2, _pref_col3 = st.columns(3)
                    else:
                        _pref_col1, _pref_col2 = st.columns(2)
                    with _pref_col1:
                        st.metric("Rubric Ranked 1st", str(_sat_diag_rubric_1st), help="Times the rubric-guided draft was ranked best")
                    with _pref_col2:
                        st.metric("Generic Ranked 1st", str(_sat_diag_generic_1st), help="Times the generic draft was ranked best")
                    if _has_any_pref:
                        with _pref_col3:
                            st.metric("Preferences Ranked 1st", str(_sat_diag_pref_1st), help="Times your original preferences draft was ranked best")

                    # Per-diagnostic breakdown
                    if _sat_diag_total > 0:
                        st.markdown("")
                        st.markdown("**Per-Diagnostic Breakdown**")
                        _pref_rows = []
                        _src_names = {"rubric": "Rubric", "generic": "Generic", "preference": "Preferences"}
                        for _di, _dr in enumerate(_eval_diag_results):
                            _dr_ranking = _dr.get("user_ranking", [])
                            if _dr_ranking:
                                _dr_pref = " > ".join(_src_names.get(s, s) for s in _dr_ranking)
                            else:
                                _dr_pref = _src_names.get(_dr.get("user_preference", ""), "?")
                            _dr_ts = _dr.get("timestamp", "")[:16].replace("T", " ") if _dr.get("timestamp") else ""
                            _dr_reason = _dr.get("user_reason", "")
                            _pref_rows.append({
                                "Diagnostic": _di + 1,
                                "Rubric Version": _dr.get("rubric_version", "?"),
                                "Ranking": _dr_pref,
                                "Reason": _dr_reason[:80] if _dr_reason else "—",
                                "Time": _dr_ts,
                            })
                        st.dataframe(_pref_rows, width="stretch", hide_index=True)

            # ════════════════════════════════════════
            # TAB 4: Rubric vs Generic vs Preference Effectiveness
            # ════════════════════════════════════════
            with _ev_tab_eff:
                st.markdown("All drafts are scored by the **rubric judge** on your criteria. "
                            "Higher score for the rubric-guided draft means your rubric is steering the LLM in the right direction.")
                st.markdown("")

                _rnr_rubric_scores = []
                _rnr_generic_scores = []
                _rnr_pref_scores = []
                _rnr_differentiating = 0
                _rnr_underperforming = 0
                _rnr_redundant = 0
                _rnr_pref_gap = 0
                for _dr in _eval_diag_results:
                    for _ca in _dr.get("criteria_analysis", []):
                        _rnr_rubric_scores.append(_ca.get("rubric_score", 0))
                        _rnr_generic_scores.append(_ca.get("generic_score", 0))
                        if "preference_score" in _ca:
                            _rnr_pref_scores.append(_ca["preference_score"])
                        _cls = _ca.get("classification", "")
                        if _cls == "DIFFERENTIATING":
                            _rnr_differentiating += 1
                        elif _cls == "UNDERPERFORMING":
                            _rnr_underperforming += 1
                        elif _cls == "REDUNDANT":
                            _rnr_redundant += 1
                        elif _cls == "PREFERENCE_GAP":
                            _rnr_pref_gap += 1

                if not _rnr_rubric_scores:
                    st.info("No diagnostic data yet. Run an alignment diagnostic to see rubric vs generic comparison.")
                else:
                    _avg_rs = sum(_rnr_rubric_scores) / len(_rnr_rubric_scores)
                    _avg_gs = sum(_rnr_generic_scores) / len(_rnr_generic_scores)
                    _avg_ps = sum(_rnr_pref_scores) / len(_rnr_pref_scores) if _rnr_pref_scores else None
                    _diff = _avg_rs - _avg_gs
                    _rnr_total_cls = _rnr_differentiating + _rnr_underperforming + _rnr_redundant + _rnr_pref_gap

                    if _avg_ps is not None:
                        _rnr_col1, _rnr_col2, _rnr_col3, _rnr_col4 = st.columns(4)
                    else:
                        _rnr_col1, _rnr_col2, _rnr_col3 = st.columns(3)
                    with _rnr_col1:
                        st.metric("Rubric Draft", f"{_avg_rs:.1f}/5",
                                  help="Avg score of the rubric-guided draft on your criteria")
                    with _rnr_col2:
                        st.metric("Generic Draft", f"{_avg_gs:.1f}/5",
                                  help="Avg score of the generic draft on your criteria")
                    if _avg_ps is not None:
                        with _rnr_col3:
                            st.metric("Preference Draft", f"{_avg_ps:.1f}/5",
                                      help="Avg score of the preference-based draft on your criteria")
                        with _rnr_col4:
                            _diff_label = f"+{_diff:.2f}" if _diff > 0 else f"{_diff:.2f}"
                            st.metric("Rubric - Generic", _diff_label,
                                      help="Positive = rubric-guided draft scores higher")
                    else:
                        with _rnr_col3:
                            _diff_label = f"+{_diff:.2f}" if _diff > 0 else f"{_diff:.2f}"
                            st.metric("Rubric - Generic", _diff_label,
                                      help="Positive = rubric-guided draft scores higher")

                    # Gap reliability
                    _rnr_retest_taus = [r.get("metrics", {}).get("retest_tau") for r in _eval_diag_retest_history if r.get("metrics", {}).get("retest_tau") is not None]
                    if _rnr_retest_taus:
                        _avg_gap_tau = sum(_rnr_retest_taus) / len(_rnr_retest_taus)
                        _gap_interp = "Strong" if _avg_gap_tau > 0.6 else ("Moderate" if _avg_gap_tau > 0.3 else "Weak")
                        st.caption(f"Gap Reliability: τ {_avg_gap_tau:.3f} ({_gap_interp})")

                    # Criteria classification summary
                    if _rnr_total_cls > 0:
                        st.markdown("")
                        st.markdown("**Criteria Classification**")
                        _has_pref_gap_cls = _rnr_pref_gap > 0
                        if _has_pref_gap_cls:
                            _cls_col1, _cls_col2, _cls_col3, _cls_col4 = st.columns(4)
                        else:
                            _cls_col1, _cls_col2, _cls_col3 = st.columns(3)
                        with _cls_col1:
                            st.metric("Differentiating", _rnr_differentiating,
                                      help="Rubric draft scored highest")
                        with _cls_col2:
                            st.metric("Redundant", _rnr_redundant,
                                      help="All drafts scored similarly")
                        with _cls_col3:
                            st.metric("Underperforming", _rnr_underperforming,
                                      help="Generic draft scored highest — rubric may hurt here")
                        if _has_pref_gap_cls:
                            with _cls_col4:
                                st.metric("Preference Gap", _rnr_pref_gap,
                                          help="Original preferences draft scored higher than rubric")

                    # Per-diagnostic table
                    if _eval_diag_results:
                        st.markdown("")
                        st.markdown("**Per-Diagnostic Comparison**")
                        _rnr_rows = []
                        _src_names = {"rubric": "Rubric", "generic": "Generic", "preference": "Preferences"}
                        for _di, _dr in enumerate(_eval_diag_results):
                            _dr_ca = _dr.get("criteria_analysis", [])
                            _dr_r_scores = [c.get("rubric_score", 0) for c in _dr_ca]
                            _dr_g_scores = [c.get("generic_score", 0) for c in _dr_ca]
                            _dr_p_scores = [c["preference_score"] for c in _dr_ca if "preference_score" in c]
                            _dr_r_avg = sum(_dr_r_scores) / len(_dr_r_scores) if _dr_r_scores else 0
                            _dr_g_avg = sum(_dr_g_scores) / len(_dr_g_scores) if _dr_g_scores else 0
                            _dr_p_avg = sum(_dr_p_scores) / len(_dr_p_scores) if _dr_p_scores else None
                            _dr_diff = sum(1 for c in _dr_ca if c.get("classification") == "DIFFERENTIATING")
                            _dr_under = sum(1 for c in _dr_ca if c.get("classification") == "UNDERPERFORMING")
                            _dr_ranking = _dr.get("user_ranking", [])
                            if _dr_ranking:
                                _dr_pref = " > ".join(_src_names.get(s, s) for s in _dr_ranking)
                            else:
                                _dr_pref = _src_names.get(_dr.get("user_preference", ""), "?")
                            _dr_ts = _dr.get("timestamp", "")[:16].replace("T", " ") if _dr.get("timestamp") else ""
                            _row = {
                                "#": _di + 1,
                                "Version": _dr.get("rubric_version", "?"),
                                "Rubric Avg": f"{_dr_r_avg:.1f}",
                                "Generic Avg": f"{_dr_g_avg:.1f}",
                            }
                            if _dr_p_avg is not None:
                                _row["Pref. Avg"] = f"{_dr_p_avg:.1f}"
                            _row["Diff."] = _dr_diff
                            _row["Under."] = _dr_under
                            _row["User Ranking"] = _dr_pref
                            _row["Time"] = _dr_ts
                            _rnr_rows.append(_row)
                        st.dataframe(_rnr_rows, width="stretch", hide_index=True)



# ============ EVALUATE: SURVEY TAB ============
with tab_survey:
    st.header("📋 Evaluate: Survey")
    st.caption("Complete surveys after each task to track your experience")

    # Initialize survey session state
    if "survey_responses" not in st.session_state:
        st.session_state.survey_responses = {
            "task_a": {},
            "task_b": {},
            "final_review": {},
        }

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
                        # Also save as separate data type for analyze_results.py
                        save_project_data(supabase, project_id, "survey_task_a", {
                            "q1": task_a.get("q1"),
                            "q2": task_a.get("q2"),
                            "q3": task_a.get("q3", ""),
                            "iteration": 1,
                            "timestamp": task_a["timestamp"],
                        })
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
                        # Also save as separate data type for analyze_results.py
                        _rb_dict_b, _, _ = get_active_rubric()
                        _rb_ver_b = _rb_dict_b.get("version", 1) if _rb_dict_b else 1
                        save_project_data(supabase, project_id, "survey_task_b", {
                            "q1": task_b.get("q1"),
                            "q2": task_b.get("q2"),
                            "q3": task_b.get("q3", ""),
                            "q4": task_b.get("q4", ""),
                            "q5": task_b.get("q5", ""),
                            "q6": task_b.get("q6", ""),
                            "iteration": _rb_ver_b,
                            "timestamp": task_b["timestamp"],
                        })
                        st.toast("Task B survey saved!")
                    except Exception as e:
                        st.error(f"Failed to save: {e}")
                else:
                    st.toast("Task B survey saved locally!")
            else:
                st.toast("Task B survey saved locally!")
            st.rerun()

    # ============ FINAL REVIEW SURVEY ============
    elif survey_task == "Final Review":
        st.subheader("Final Review: Rubric Accuracy")
        st.markdown("*Complete this after all tasks are done. Rate how accurately the rubric captures your preferences.*")

        # Get active rubric
        _fr_rubric_dict, _, _ = get_active_rubric()
        _fr_criteria = (_fr_rubric_dict.get("rubric", []) or []) if _fr_rubric_dict else []

        if not _fr_criteria:
            st.warning("No rubric found. Complete at least one task with rubric assistance first.")
        else:
            final_review = st.session_state.survey_responses.setdefault("final_review", {})
            criteria_ratings = final_review.setdefault("criteria_ratings", {})

            st.markdown("**Q1: For each rubric criterion, rate how accurately it captures your preferences.**")

            for i, criterion in enumerate(_fr_criteria):
                crit_name = criterion.get("name", f"Criterion {i+1}")
                crit_desc = criterion.get("description", "No description")
                dims = criterion.get("dimensions", [])
                dim_text = ", ".join(d.get("label", "") for d in dims) if dims else "none"

                with st.expander(f"**{crit_name}** — {crit_desc[:80]}{'...' if len(crit_desc) > 80 else ''}", expanded=True):
                    st.caption(f"Dimensions: {dim_text}")

                    rating = criteria_ratings.setdefault(crit_name, {})
                    accuracy_options = ["Accurate", "Partially right", "Inaccurate"]
                    rating["accuracy"] = st.radio(
                        "Accuracy",
                        accuracy_options,
                        index=accuracy_options.index(rating.get("accuracy", "Partially right")) if rating.get("accuracy") in accuracy_options else 1,
                        key=f"fr_accuracy_{i}",
                        horizontal=True,
                        label_visibility="collapsed",
                    )
                    rating["explanation"] = st.text_input(
                        "Brief explanation",
                        value=rating.get("explanation", ""),
                        placeholder="What's right or wrong about this criterion?",
                        key=f"fr_explanation_{i}",
                        label_visibility="collapsed",
                    )

            st.markdown("---")
            st.markdown("**Q2: Is there anything here you wouldn't have thought to mention yourself?**")
            st.caption("Did the rubric capture preferences you have but might not have articulated if asked directly?")
            final_review["q2"] = st.text_area(
                "Unexpected insights",
                value=final_review.get("q2", ""),
                placeholder="Any preferences the system correctly inferred that surprised you...",
                key="fr_q2",
                label_visibility="collapsed",
                height=100,
            )

            st.markdown("---")
            st.markdown("**Q3: Are any of your preferences missing from the rubric?**")
            st.caption("Things you care about in your writing that the rubric does NOT cover at all.")
            final_review["q3"] = st.text_area(
                "Missing preferences",
                value=final_review.get("q3", ""),
                placeholder="List any preferences the rubric missed, or 'None' if fully covered...",
                key="fr_q3",
                label_visibility="collapsed",
                height=100,
            )

            st.markdown("---")
            st.markdown("**Q4: Did the drafts improve over the course of your conversations?**")
            _q4_options = ["1 - No improvement", "2", "3 - Some improvement", "4", "5 - Dramatically better"]
            _q4_current = final_review.get("q4", "3 - Some improvement")
            final_review["q4"] = st.radio(
                "Draft improvement",
                _q4_options,
                index=_q4_options.index(_q4_current) if _q4_current in _q4_options else 2,
                key="fr_q4",
                horizontal=True,
                label_visibility="collapsed",
            )

            st.markdown("**Q5: Did editing the rubric help you get better drafts?**")
            _q5_options = ["1 - Not at all", "2", "3 - Somewhat", "4", "5 - Very much"]
            _q5_current = final_review.get("q5", "3 - Somewhat")
            final_review["q5"] = st.radio(
                "Rubric editing helped",
                _q5_options,
                index=_q5_options.index(_q5_current) if _q5_current in _q5_options else 2,
                key="fr_q5",
                horizontal=True,
                label_visibility="collapsed",
            )

            st.markdown("**Q6: Would you use a system like this again for future writing?**")
            _q6_options = ["1 - Definitely not", "2", "3 - Maybe", "4", "5 - Definitely yes"]
            _q6_current = final_review.get("q6", "3 - Maybe")
            final_review["q6"] = st.radio(
                "Would use again",
                _q6_options,
                index=_q6_options.index(_q6_current) if _q6_current in _q6_options else 2,
                key="fr_q6",
                horizontal=True,
                label_visibility="collapsed",
            )

            if st.button("Save Final Review", type="primary", key="save_final_review"):
                final_review["completed"] = True
                final_review["timestamp"] = datetime.now().isoformat()
                final_review["rubric_version"] = _fr_rubric_dict.get("version", 0) if _fr_rubric_dict else 0
                project_id = st.session_state.get('current_project_id')
                if project_id:
                    supabase = st.session_state.get('supabase')
                    if supabase:
                        try:
                            save_project_data(supabase, project_id, "survey_responses", st.session_state.survey_responses)
                            save_project_data(supabase, project_id, "survey_final_review", {
                                "criteria_ratings": criteria_ratings,
                                "q2": final_review.get("q2", ""),
                                "q3": final_review.get("q3", ""),
                                "q4": final_review.get("q4", ""),
                                "q5": final_review.get("q5", ""),
                                "q6": final_review.get("q6", ""),
                                "rubric_version": final_review["rubric_version"],
                                "timestamp": final_review["timestamp"],
                            })
                            st.toast("Final Review saved!")
                        except Exception as e:
                            st.error(f"Failed to save: {e}")
                    else:
                        st.toast("Final Review saved locally!")
                else:
                    st.toast("Final Review saved locally!")
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
        fr_done = st.session_state.survey_responses.get("final_review", {}).get("completed", False)
        st.markdown(f"**Final Review:** {'✅ Complete' if fr_done else '⬜ Incomplete'}")

    # Save to database option
    if any([a_done, b_done, fr_done]):
        project_id = st.session_state.get('current_project_id')
        if project_id:
            supabase = st.session_state.get('supabase')
            if supabase:
                if st.button("Save All Surveys to Database", key="save_all_surveys"):
                    try:
                        save_project_data(supabase, project_id, "survey_responses", st.session_state.survey_responses)
                        # Also save as separate data types for analyze_results.py
                        _all_sr = st.session_state.survey_responses
                        _rb_dict_all, _, _ = get_active_rubric()
                        _rb_ver_all = _rb_dict_all.get("version", 0) if _rb_dict_all else 0
                        if _all_sr.get("task_a", {}).get("completed"):
                            _ta = _all_sr["task_a"]
                            save_project_data(supabase, project_id, "survey_task_a", {
                                "q1": _ta.get("q1"), "q2": _ta.get("q2"), "q3": _ta.get("q3", ""),
                                "iteration": 1, "timestamp": _ta.get("timestamp", datetime.now().isoformat()),
                            })
                        if _all_sr.get("task_b", {}).get("completed"):
                            _tb = _all_sr["task_b"]
                            save_project_data(supabase, project_id, "survey_task_b", {
                                "q1": _tb.get("q1"), "q2": _tb.get("q2"), "q3": _tb.get("q3", ""),
                                "q4": _tb.get("q4", ""), "q5": _tb.get("q5", ""), "q6": _tb.get("q6", ""),
                                "iteration": _rb_ver_all, "timestamp": _tb.get("timestamp", datetime.now().isoformat()),
                            })
                        if _all_sr.get("final_review", {}).get("completed"):
                            _fr = _all_sr["final_review"]
                            save_project_data(supabase, project_id, "survey_final_review", {
                                "criteria_ratings": _fr.get("criteria_ratings", {}),
                                "q2": _fr.get("q2", ""),
                                "q3": _fr.get("q3", ""),
                                "q4": _fr.get("q4", ""),
                                "q5": _fr.get("q5", ""),
                                "q6": _fr.get("q6", ""),
                                "rubric_version": _fr.get("rubric_version", _rb_ver_all),
                                "timestamp": _fr.get("timestamp", datetime.now().isoformat()),
                            })
                        st.success("All survey responses saved to database!")
                    except Exception as e:
                        st.error(f"Failed to save: {e}")
