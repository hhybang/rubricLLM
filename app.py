import streamlit as st
import os
import anthropic
from textwrap import dedent
import json
import re
import time
import uuid
import zipfile
import io
import shutil
from datetime import datetime
from pathlib import Path
from prompts import (
    COMPARE_WRITE_EDIT_PROMPT,
    RUBRIC_INFERENCE_SYSTEM_PROMPT,
    get_rubric_inference_user_prompt,
    build_system_instruction,
    DRAFT_EDIT_RUBRIC_UPDATE_PROMPT,
    get_draft_edit_rubric_update_prompt,
    extract_decision_pts,
    generate_reflection_questions_prompt,
    extract_preference_dimensions_prompt,
    generate_test_comparisons_prompt,
    format_user_tests_prompt,
    score_tests_with_rubric_prompt
)

def display_rubric_criteria(rubric_data, container, comparison_rubric_data=None):
    """
    Display rubric criteria in a user-friendly format with headings, descriptions,
    priority icons, and expandable evidence sections. Criteria are grouped by category.

    If comparison_rubric_data is provided, highlights criteria that are added (new) or
    removed (missing in comparison) with different colors.
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

    # Group criteria by category
    from collections import defaultdict
    categories = defaultdict(list)
    for criterion in rubric_list:
        category = criterion.get('category', 'Uncategorized')
        # Add a flag to indicate if this is new or changed compared to comparison rubric
        if comparison_rubric_data:
            criterion_name = criterion.get('name', '').lower().strip()

            if criterion_name not in comparison_map:
                # Completely new criterion
                criterion['is_new'] = True
            else:
                # Check if description or weight changed
                old_criterion = comparison_map[criterion_name]
                description_changed = criterion.get('description', '') != old_criterion.get('description', '')
                weight_changed = criterion.get('weight', 0) != old_criterion.get('weight', 0)
                criterion['is_new'] = description_changed or weight_changed
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
            # Determine if this criterion is new (for highlighting)
            is_new = criterion.get('is_new', False)

            # Choose label based on whether it's new
            criterion_label = f"{criterion.get('name', 'Unnamed')}"
            if is_new:
                criterion_label = f"{'‼️'}  {criterion.get('name', 'Unnamed')}"

            with container.expander(criterion_label, expanded=False):
                # Description
                description = criterion.get('description', 'No description provided')
                st.markdown(description)

                # Weight display
                weight = criterion.get('weight', 0)
                if weight > 0:
                    st.markdown(f"*Weight: {weight}%*")
                    st.markdown("")

                # Achievement Levels section (expandable)
                with st.expander("📊 Achievement Levels", expanded=False):
                    exemplary = criterion.get('exemplary', 'Not specified')
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

def render_message_with_draft(content: str, message_id: str, is_branch: bool = False):
    """
    Render a message that may contain <draft> tags.
    Draft sections are rendered as editable text areas.
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
            if edit_key not in st.session_state[draft_key]:
                st.session_state[draft_key][edit_key] = draft_content

            # Store the original draft for comparison
            if orig_key not in st.session_state[original_key]:
                st.session_state[original_key][orig_key] = draft_content

            # Initialize reset counter (used to force new widget key on reset)
            if reset_counter_key not in st.session_state:
                st.session_state[reset_counter_key] = 0

            # Create a container for the draft with visual styling
            with st.container():
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

                            if is_branch:
                                st.session_state.branch['messages'].append(new_message)
                            else:
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
                            update_rubric_from_draft_edit(original_draft, edited_draft, is_branch)

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


def update_rubric_from_draft_edit(original_draft: str, edited_draft: str, is_branch: bool = False):
    """
    Call the LLM to analyze draft edits and suggest rubric updates.
    Shows analysis in a chat-like format for better readability.
    Uses branch rubric if in branch mode and branch rubric exists.
    """
    # Get the appropriate rubric based on mode
    if is_branch and st.session_state.branch.get('rubric') is not None:
        # Use branch rubric
        active_rubric_list = st.session_state.branch['rubric']
        st.session_state.rubric_update_is_branch = True
    else:
        # Use main rubric
        active_rubric_dict, _, _ = get_active_rubric()

        if not active_rubric_dict:
            st.warning("No active rubric to update. Please create or select a rubric first.")
            return

        active_rubric_list = active_rubric_dict.get("rubric", [])
        st.session_state.rubric_update_is_branch = False

    if not active_rubric_list:
        st.warning("Active rubric has no criteria to update.")
        return

    with st.spinner("Analyzing your edits to suggest rubric updates..."):
        try:
            client = anthropic.Anthropic()

            # Build the prompt
            user_prompt = get_draft_edit_rubric_update_prompt(
                active_rubric_list,
                original_draft,
                edited_draft
            )

            # Make API call
            response = client.messages.create(
                max_tokens=16000,
                system=DRAFT_EDIT_RUBRIC_UPDATE_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                model="claude-opus-4-5-20251101",
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


def regenerate_draft_from_rubric_changes(original_rubric: list, updated_rubric: list, current_draft: str):
    """
    Call the LLM to regenerate the draft based on rubric changes.
    Returns the regenerated draft result or None on error.
    """
    from prompts import REGENERATE_DRAFT_PROMPT, get_regenerate_draft_prompt

    with st.spinner("Regenerating draft based on rubric changes..."):
        try:
            client = anthropic.Anthropic()

            # Build the prompt
            user_prompt = get_regenerate_draft_prompt(
                original_rubric,
                updated_rubric,
                current_draft
            )

            # Make API call
            response = client.messages.create(
                max_tokens=16000,
                system=REGENERATE_DRAFT_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
                model="claude-opus-4-5-20251101",
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
                return result
            else:
                st.error("Could not parse regenerated draft. Please try again.")
                return None

        except json.JSONDecodeError as e:
            st.error(f"Error parsing response: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Error regenerating draft: {str(e)}")
            return None


def get_last_draft_from_messages():
    """
    Find the last message with a <draft></draft> block and return the draft content.
    Returns tuple of (draft_content, message_index) or (None, None) if not found.
    """
    pattern = r'<draft>(.*?)</draft>'

    # Search from most recent to oldest
    for idx in range(len(st.session_state.messages) - 1, -1, -1):
        msg = st.session_state.messages[idx]
        if msg.get('role') == 'assistant':
            content = msg.get('content', '')
            match = re.search(pattern, content, re.DOTALL)
            if match:
                return match.group(1).strip(), idx

    return None, None


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


def display_rubric_comparison(current_rubric: list, updated_rubric: list):
    """
    Display a comparison of current and updated rubric using collapsible sections.
    Each criterion is shown as an expander with status badge visible when collapsed.
    Word-level diffing highlights specific changes.
    """
    # Build a map of current criteria by name for comparison
    current_map = {c.get('name', '').lower().strip(): c for c in current_rubric}
    updated_map = {c.get('name', '').lower().strip(): c for c in updated_rubric}

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
    </style>
    """, unsafe_allow_html=True)

    st.markdown("#### 📋 Rubric Changes")

    # Show criteria in the updated rubric (maintains order), then removed ones
    for criterion in updated_rubric:
        name = criterion.get('name', 'Unnamed')
        name_key = name.lower().strip()
        weight = criterion.get('weight', 0)

        if name_key not in current_map:
            # NEW criterion
            status_html = '<span class="status-badge status-new">NEW</span>'
            expander_label = f"✅ {name} ({weight}%)"

            with st.expander(expander_label, expanded=False):
                st.markdown(status_html, unsafe_allow_html=True)
                st.markdown(f"**Weight:** {weight}%")

                for field in ['description', 'exemplary', 'proficient', 'developing', 'beginning']:
                    value = criterion.get(field, '')
                    st.markdown(f"""
                    <div class="diff-field">
                        <div class="diff-field-label">{field.title()}</div>
                        <div class="diff-field-content"><span class="text-added">{value if value else 'Not specified'}</span></div>
                    </div>
                    """, unsafe_allow_html=True)

        elif _criterion_changed(current_map[name_key], criterion):
            # MODIFIED criterion
            old = current_map[name_key]
            old_weight = old.get('weight', 0)

            status_html = '<span class="status-badge status-modified">MODIFIED</span>'
            expander_label = f"🔄 {name} ({weight}%)"

            with st.expander(expander_label, expanded=False):
                st.markdown(status_html, unsafe_allow_html=True)

                # Show weight change if applicable
                if old_weight != weight:
                    st.markdown(f"""
                    <div class="weight-change">
                        <strong>Weight:</strong> <span class="text-removed">{old_weight}%</span> → <span class="text-added">{weight}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"**Weight:** {weight}%")

                # Show field-by-field diff
                for field in ['description', 'exemplary', 'proficient', 'developing', 'beginning']:
                    old_val = old.get(field, '')
                    new_val = criterion.get(field, '')

                    if old_val != new_val:
                        diff_html = _word_level_diff(old_val, new_val)
                        field_class = "diff-field diff-field-changed"
                    else:
                        diff_html = new_val if new_val else '<span class="no-change-badge">Not specified</span>'
                        field_class = "diff-field"

                    st.markdown(f"""
                    <div class="{field_class}">
                        <div class="diff-field-label">{field.title()}</div>
                        <div class="diff-field-content">{diff_html}</div>
                    </div>
                    """, unsafe_allow_html=True)

        else:
            # UNCHANGED criterion
            status_html = '<span class="status-badge status-unchanged">UNCHANGED</span>'
            expander_label = f"⚪ {name} ({weight}%)"

            with st.expander(expander_label, expanded=False):
                st.markdown(status_html, unsafe_allow_html=True)
                st.markdown(f"**Weight:** {weight}%")

                for field in ['description', 'exemplary', 'proficient', 'developing', 'beginning']:
                    value = criterion.get(field, '')
                    st.markdown(f"""
                    <div class="diff-field">
                        <div class="diff-field-label">{field.title()}</div>
                        <div class="diff-field-content">{value if value else '<span class="no-change-badge">Not specified</span>'}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # Show removed criteria
    for criterion in current_rubric:
        name = criterion.get('name', 'Unnamed')
        name_key = name.lower().strip()

        if name_key not in updated_map:
            weight = criterion.get('weight', 0)
            status_html = '<span class="status-badge status-removed">REMOVED</span>'
            expander_label = f"❌ {name} ({weight}%) - REMOVED"

            with st.expander(expander_label, expanded=False):
                st.markdown(status_html, unsafe_allow_html=True)
                st.markdown(f"**Weight:** ~~{weight}%~~")

                for field in ['description', 'exemplary', 'proficient', 'developing', 'beginning']:
                    value = criterion.get(field, '')
                    st.markdown(f"""
                    <div class="diff-field">
                        <div class="diff-field-label">{field.title()}</div>
                        <div class="diff-field-content"><span class="text-removed">{value if value else 'Not specified'}</span></div>
                    </div>
                    """, unsafe_allow_html=True)


def _criterion_changed(old: dict, new: dict) -> bool:
    """Check if a criterion has changed between old and new versions."""
    fields_to_compare = ['description', 'weight', 'exemplary', 'proficient', 'developing', 'beginning', 'category']
    for field in fields_to_compare:
        if old.get(field) != new.get(field):
            return True
    return False


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

    # Check if this update is for branch rubric
    is_branch_update = st.session_state.get('rubric_update_is_branch', False)

    # Display in a chat message style container
    with st.chat_message("assistant"):
        # Show thinking if available
        if thinking_text:
            with st.expander("🧠 Thinking", expanded=False):
                st.markdown(thinking_text)

        if rubric_updates.get("has_updates"):
            st.markdown("### ✨ Suggested Rubric Updates")
            if is_branch_update:
                st.caption("🌿 Updating Branch Rubric")
            st.markdown(f"**Rationale:** {rubric_updates.get('rationale', '')}")

            modified_rubric = rubric_updates.get("modified_rubric", [])

            if modified_rubric:
                # Get current rubric for comparison (branch or main)
                if is_branch_update and st.session_state.branch.get('rubric') is not None:
                    current_rubric = st.session_state.branch['rubric']
                else:
                    active_rubric_dict, _, _ = get_active_rubric()
                    current_rubric = active_rubric_dict.get("rubric", []) if active_rubric_dict else []

                # Show side-by-side comparison
                display_rubric_comparison(current_rubric, modified_rubric)

                # Action buttons
                col1, col2, col3 = st.columns([1, 1, 3])
                with col1:
                    if st.button("✅ Apply Updates", key="apply_rubric_updates", type="primary"):
                        import copy
                        if is_branch_update:
                            # Update branch rubric directly
                            st.session_state.branch['rubric'] = modified_rubric
                            # Increment counter to force widget refresh
                            st.session_state.branch_rubric_edit_counter += 1

                            # Clear the result
                            st.session_state.rubric_update_result = None
                            st.session_state.rubric_update_is_branch = False

                            st.success("✓ Branch rubric updated!")
                            st.rerun()
                        else:
                            # Save as new rubric version (main rubric)
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

                            # Update active rubric
                            st.session_state.active_rubric_idx = len(hist) - 1
                            st.session_state.rubric = modified_rubric
                            # Also update editing_criteria so Rubric Configuration shows the new version
                            st.session_state.editing_criteria = copy.deepcopy(modified_rubric)

                            # Clear the result
                            st.session_state.rubric_update_result = None
                            st.session_state.rubric_update_is_branch = False

                            st.success(f"✓ Rubric updated to version {new_version}!")
                            st.rerun()

                with col2:
                    if st.button("❌ Dismiss", key="dismiss_rubric_updates"):
                        st.session_state.rubric_update_result = None
                        st.session_state.rubric_update_is_branch = False
                        st.rerun()
        else:
            st.info(f"**No rubric updates needed.** {rubric_updates.get('rationale', 'Your edits are already well-captured by the current rubric.')}")

            if st.button("OK", key="dismiss_no_updates"):
                st.session_state.rubric_update_result = None
                st.session_state.rubric_update_is_branch = False
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
    """Compare two rubrics using the COMPARE_WRITE_EDIT_PROMPT"""
    prompt = COMPARE_WRITE_EDIT_PROMPT.format(
        task=task,
        rubric_a=json.dumps(rubric_a, indent=2),
        rubric_b=json.dumps(rubric_b, indent=2)
    )

    # Call Claude API directly
    message = client.messages.create(
        model="claude-opus-4-5-20251101",
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
                'weight': None,
                'achievement_level': None,
                'level_percentage': None,
                'weighted_score': None,
                'evidence': [],
                'rationale': None,
                'to_reach_next_level': None
            }

            current_field = None
            content_buffer = []

            for line in lines[1:]:
                line = line.strip()

                if line.startswith('**Weight**:'):
                    criterion_data['weight'] = line.replace('**Weight**:', '').strip()
                elif line.startswith('**Achievement Level**:'):
                    level_text = line.replace('**Achievement Level**:', '').strip()
                    criterion_data['achievement_level'] = level_text
                    # Extract percentage if present
                    pct_match = re.search(r'\((\d+)%\)', level_text)
                    if pct_match:
                        criterion_data['level_percentage'] = int(pct_match.group(1))
                elif line.startswith('**Weighted Score**:'):
                    criterion_data['weighted_score'] = line.replace('**Weighted Score**:', '').strip()
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

def process_rubric_edit_request(user_request, current_rubric):
    """Process a conversational rubric edit request using Claude API

    Args:
        user_request: String describing the changes the user wants
        current_rubric: List of criterion dicts representing current rubric

    Returns:
        Dict with either:
        - {'message': str, 'modified_rubric': list, 'changes_summary': str} for successful edits
        - {'message': str} for clarifying questions
    """
    import json
    from prompts import RUBRIC_EDIT_PROMPT

    # Build conversation history including main conversation context
    # This gives the AI context about why changes might be needed

    # Include recent main conversation messages (last 10 messages for context)
    main_messages = st.session_state.messages[-10:] if len(st.session_state.messages) > 10 else st.session_state.messages

    # Add context header
    context_summary = "Recent conversation context (for understanding why rubric changes might be needed):\n\n"
    for msg in main_messages:
        if not msg.get('is_assessment_message', False):  # Skip assessment messages for brevity
            role = msg['role']
            content = msg.get('content', '')[:200]  # Truncate long messages
            context_summary += f"{role.upper()}: {content}...\n\n"

    # Add the edit conversation history
    edit_history = ""
    if st.session_state.rubric_editing['edit_messages']:
        edit_history = "\n\nPrevious edit conversation:\n"
        for msg in st.session_state.rubric_editing['edit_messages']:
            role = "USER" if msg['role'] == 'user' else "ASSISTANT"
            edit_history += f"{role}: {msg['content']}\n\n"

    # Format the prompt with current rubric and user request
    prompt_text = RUBRIC_EDIT_PROMPT.format(
        current_rubric=json.dumps(current_rubric, indent=2),
        user_request=user_request
    )

    # Combine everything
    full_prompt = f"{context_summary}\n\n---\n\n{edit_history}\n\n---\n\n{prompt_text}"

    # Call Claude API
    try:
        response = client.messages.create(
            max_tokens=16000,
            messages=[{
                "role": "user",
                "content": full_prompt
            }],
            model="claude-opus-4-5-20251101",
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
        response_text = response_text.strip()

        # Try to parse as JSON
        try:
            # Extract JSON if wrapped in code blocks
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                # Try to find JSON object directly
                json_text = response_text

            parsed_response = json.loads(json_text)

            # Validate that we have the expected structure
            if 'modified_rubric' in parsed_response and 'changes_summary' in parsed_response:
                # Validate rubric structure
                modified_rubric = parsed_response['modified_rubric']

                # Check weight totals
                total_weight = sum(c.get('weight', 0) for c in modified_rubric)
                if abs(total_weight - 100.0) > 0.1:
                    return {
                        'message': f"Error: Total weight is {total_weight}%, but must equal 100%. Please adjust criterion weights."
                    }

                # Check dimension importance for each criterion
                for criterion in modified_rubric:
                    dimensions = criterion.get('dimensions', [])
                    if dimensions:
                        total_importance = sum(d.get('importance', 0) for d in dimensions)
                        if abs(total_importance - 1.0) > 0.01:
                            return {
                                'message': f"Error: Dimension importance in '{criterion.get('name')}' totals {total_importance:.2f}, but must equal 1.0. Please adjust dimension importance values."
                            }

                # Return successful response
                return {
                    'message': parsed_response['changes_summary'],
                    'modified_rubric': modified_rubric,
                    'changes_summary': parsed_response['changes_summary'],
                    'thinking': thinking_text
                }
            else:
                # Response is a clarifying question, not JSON
                return {
                    'message': response_text,
                    'thinking': thinking_text
                }

        except json.JSONDecodeError:
            # Response is not JSON, treat as clarifying question or message
            return {
                'message': response_text,
                'thinking': thinking_text
            }

    except Exception as e:
        return {
            'message': f"Error calling API: {str(e)}"
        }

def display_rubric_assessment(assessment_data, message_id=None):
    """Display rubric assessment in a card-based layout with collapsible details"""
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

    # Get overall score if available from JSON summary
    overall_score = None
    score_interpretation = None
    overall_assessment = None

    if assessment_data.get('json_summary'):
        json_data = assessment_data['json_summary']
        overall_score = json_data.get('overall_score')
        score_interpretation = json_data.get('score_interpretation')
        overall_assessment = json_data.get('overall_assessment')

    # Display overall score header
    with st.expander("📊 Rubric Assessment", expanded=False):
        # Show thinking if available
        if assessment_data.get('thinking'):
            with st.expander("🧠 Thinking", expanded=False):
                st.markdown(assessment_data['thinking'])

        if overall_score is not None:
            # Color code the score
            if overall_score >= 85:
                score_color = "#4CAF50"  # Green
            elif overall_score >= 70:
                score_color = "#2196F3"  # Blue
            elif overall_score >= 50:
                score_color = "#FF9800"  # Orange
            else:
                score_color = "#F44336"  # Red

            st.markdown(f"""
                <div style="background: linear-gradient(135deg, {score_color}15 0%, {score_color}05 100%);
                            border-left: 4px solid {score_color};
                            padding: 16px;
                            border-radius: 8px;
                            margin-bottom: 20px;">
                    <div style="font-size: 28px; font-weight: 700; color: {score_color};">
                        {overall_score}/100
                    </div>
                    <div style="font-size: 16px; font-weight: 600; color: #555; margin-top: 4px;">
                        {score_interpretation or 'Overall Score'}
                    </div>
                    {f'<div style="font-size: 14px; color: #666; margin-top: 12px; line-height: 1.6;">{overall_assessment}</div>' if overall_assessment else ''}
                </div>
            """, unsafe_allow_html=True)

        # Display individual criteria cards
        criteria_details = assessment_data.get('criteria_details', {})

        for criterion_name, data in criteria_details.items():
            weight = data.get('weight', 'N/A')
            achievement_level = data.get('achievement_level', 'N/A')
            level_percentage = data.get('level_percentage')
            weighted_score = data.get('weighted_score', 'N/A')
            evidence = data.get('evidence', [])
            rationale = data.get('rationale', '')
            to_reach_next_level = data.get('to_reach_next_level', '')

            # Color code based on achievement level
            if 'Exemplary' in achievement_level or level_percentage == 100:
                level_color = "#4CAF50"
                level_emoji = "🌟"
            elif 'Proficient' in achievement_level or level_percentage == 75:
                level_color = "#2196F3"
                level_emoji = "✅"
            elif 'Developing' in achievement_level or level_percentage == 50:
                level_color = "#FF9800"
                level_emoji = "📈"
            else:
                level_color = "#F44336"
                level_emoji = "🎯"

            # Card header (always visible)
            expander_label = f"{level_emoji} **{criterion_name}** — {achievement_level} (Weight: {weight})"

            with st.expander(expander_label, expanded=False):
                # Display weighted score prominently
                st.markdown(f"""
                    <div style="background: {level_color}10;
                                border-radius: 6px;
                                padding: 12px;
                                margin-bottom: 16px;">
                        <div style="color: {level_color}; font-weight: 600; font-size: 14px;">
                            Achievement Level: {achievement_level}
                        </div>
                        <div style="color: #666; font-size: 13px; margin-top: 4px;">
                            Weighted Score: {weighted_score}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                # Evidence section
                if evidence:
                    st.markdown("**Evidence from draft:**")
                    for ev in evidence:
                        st.markdown(f"- {ev}")
                    st.markdown("")

                # Rationale section
                if rationale:
                    st.markdown("**Rationale:**")
                    st.markdown(rationale)
                    st.markdown("")

                # To reach next level section
                if to_reach_next_level and 'Exemplary' not in achievement_level:
                    st.markdown("**To reach next level:**")
                    st.markdown(to_reach_next_level)
                    st.markdown("")

                # Feedback input for this criterion
                st.markdown("---")
                st.markdown("**Your feedback on this assessment:**")
                feedback_text = st.text_area(
                    "Share your thoughts (optional)",
                    value=st.session_state.assessment_feedback[assessment_key].get(criterion_name, ""),
                    key=f"feedback_{assessment_key}_{criterion_name}",
                    placeholder="Disagree with the score? Have additional context? Share it here...",
                    height=80,
                    label_visibility="collapsed"
                )

                # Store feedback in session state
                if feedback_text:
                    st.session_state.assessment_feedback[assessment_key][criterion_name] = feedback_text
                elif criterion_name in st.session_state.assessment_feedback[assessment_key]:
                    # Remove if emptied
                    del st.session_state.assessment_feedback[assessment_key][criterion_name]

        # Submit feedback button at the bottom (outside the loop, inside the main expander)
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

    With extended thinking enabled, this also captures thinking content.
    """
    full_response = ""
    thinking_content = ""

    # Handle streaming with extended thinking
    for event in stream:
        # Handle thinking events
        if hasattr(event, 'type'):
            if event.type == 'content_block_start':
                if hasattr(event, 'content_block') and event.content_block.type == 'thinking':
                    if thinking_placeholder:
                        thinking_placeholder.markdown("🧠 *Thinking...*")
            elif event.type == 'content_block_delta':
                if hasattr(event, 'delta'):
                    if event.delta.type == 'thinking_delta':
                        thinking_content += event.delta.thinking
                        if thinking_placeholder:
                            thinking_placeholder.markdown(f"🧠 *Thinking...*\n\n{thinking_content[:500]}..." if len(thinking_content) > 500 else f"🧠 *Thinking...*\n\n{thinking_content}")
                    elif event.delta.type == 'text_delta':
                        full_response += event.delta.text

                        # Stop streaming if we hit rubric_assessment tag (strip it out)
                        if '<rubric_assessment>' in full_response:
                            # Get content before rubric_assessment tag
                            content_before_assessment = full_response.split('<rubric_assessment>')[0]
                            _, main_content = parse_analysis_and_content(content_before_assessment)
                            response_placeholder.markdown(main_content)
                            continue

                        # Parse the current accumulated response to filter out analysis
                        _, main_content = parse_analysis_and_content(full_response)
                        response_placeholder.markdown(main_content + "▌")

    # Final parse to ensure clean output
    analysis_content, main_content = parse_analysis_and_content(full_response)

    # Strip any rubric_assessment content from main_content
    # (model should not be generating assessments during normal chat)
    if '<rubric_assessment>' in main_content:
        main_content = main_content.split('<rubric_assessment>')[0].strip()

    # Display the final content (without assessment)
    response_placeholder.markdown(main_content)

    # Return the main content (without analysis or assessment), analysis, None for assessment, and thinking
    return main_content, analysis_content, None, thinking_content

def save_message_log(messages, rubric, analysis=None):
    """Save all messages to a log file"""
    # Create logs directory if it doesn't exist
    project_name = st.session_state.get('current_project')
    if not project_name:
        raise ValueError("No project selected. Please select a project first.")
    log_dir = Path("project") / project_name / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f"conversation-{timestamp}.json"
    
    # Prepare log data
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "rubric": rubric,
        "messages": messages,
        "analysis": analysis if analysis else ""
    }
    
    # Save to file
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    
    return log_file

# ========================
# Rubric Management Functions
# ========================
def get_rubric_file():
    """Get the current rubric file path based on active project"""
    project_name = st.session_state.get('current_project')
    if not project_name:
        return None
    return Path("project") / project_name / "rubric_history.json"

def load_rubric_history():
    """Load rubric history from file"""
    try:
        rubric_file = get_rubric_file()
        if rubric_file is None:
            return []
        if rubric_file.exists():
            with open(rubric_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    except Exception as e:
        st.error(f"Error loading rubric history: {e}")
        return []

def save_rubric_history(history):
    """Save rubric history to file"""
    try:
        rubric_file = get_rubric_file()
        if rubric_file is None:
            st.error("No project selected. Please select a project first.")
            return
        # Ensure parent directory exists
        rubric_file.parent.mkdir(parents=True, exist_ok=True)
        with open(rubric_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Error saving rubric history: {e}")

def next_version_number():
    """Get the next version number for a new rubric"""
    hist = load_rubric_history()
    if not hist:
        return 1
    return max(r.get("version", 1) for r in hist) + 1

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

def infer_rubric_from_conversation(messages):
    """Infer a rubric from the conversation history using Claude"""
    # Build the conversation text for the prompt
    conversation_text = ""
    for msg in messages:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        if role == 'user':
            conversation_text += f"\n\nUSER: {content}"
        elif role == 'assistant':
            conversation_text += f"\n\nASSISTANT: {content}"
    
    # Get the current active rubric to build upon
    active_rubric_dict, _, _ = get_active_rubric()
    previous_rubric = active_rubric_dict.get("rubric", []) if active_rubric_dict else []
    previous_rubric_version = active_rubric_dict.get("version", 1) if active_rubric_dict else 1
    
    # Debug: Show if previous rubric is being used
    if previous_rubric:
        st.info(f"🔍 DEBUG: Using previous rubric v{previous_rubric_version}")
        previous_rubric_json = json.dumps(previous_rubric, ensure_ascii=False, indent=2)
        # previous_rubric_block = f"\n\nPREVIOUS RUBRIC (use this as a starting point and refine it based on the conversation):\n{previous_rubric_json}\n\nBuild upon this rubric by keeping what's relevant and adding/modifying criteria based on the conversation."
    else:
        st.info("🔍 DEBUG: No previous rubric found - creating new rubric from scratch")
        previous_rubric_json = ""
        # previous_rubric_block = "\n\nThis is a new rubric - create it from scratch based on the conversation."
    
    system_prompt = RUBRIC_INFERENCE_SYSTEM_PROMPT
    user_prompt = get_rubric_inference_user_prompt(conversation_text, previous_rubric_json)

    try:
        response = client.messages.create(
            model="claude-opus-4-5-20251101",
            max_tokens=32000,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            thinking={
                "type": "enabled",
                "budget_tokens": 10000
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
        response_text = response_text.strip()

        # Show thinking in an expander
        if thinking_text:
            with st.expander("🧠 Thinking", expanded=False):
                st.markdown(thinking_text)
        
        # Try to parse JSON from the response
        import re
        json_match = re.search(r'\{.*"rubric".*\}', response_text, re.DOTALL)
        if json_match:
            rubric_data = json.loads(json_match.group())
            # Add version number
            rubric_data["version"] = next_version_number()
            
            # Save to history
            hist = load_rubric_history()
            hist.append(rubric_data)
            save_rubric_history(hist)
            st.session_state.active_rubric_idx = len(hist) - 1
            
            return rubric_data
        else:
            rubric_data = json.loads(response_text)
            rubric_data["version"] = next_version_number()
            hist = load_rubric_history()
            hist.append(rubric_data)
            save_rubric_history(hist)
            st.session_state.active_rubric_idx = len(hist) - 1
            return rubric_data
    except Exception as e:
        st.error(f"Error inferring rubric: {str(e)}")
        return None

# Page configuration
st.set_page_config(
    page_title="AI Co-Writer",
    page_icon="✍️",
    layout="wide"
)

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
    # Inline logic to get available projects (function defined later)
    projects_dir = Path("project")
    if projects_dir.exists():
        available_projects = sorted([item.name for item in projects_dir.iterdir() if item.is_dir() and not item.name.startswith('.')])
    else:
        available_projects = []
    st.session_state.current_project = available_projects[0] if available_projects else None

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

# Branch mode for temporary experimentation
if 'branch' not in st.session_state:
    st.session_state.branch = {
        'active': False,           # Whether branch is currently active
        'messages': [],            # Temporary messages in the branch
        'rubric': None,            # Temporary rubric modifications (None = use main rubric)
        'branch_point': None,      # Index in main conversation where branch started
        'branch_name': None        # Optional name for the branch
    }

# Rubric editing mode for conversational rubric content changes
if 'rubric_editing' not in st.session_state:
    st.session_state.rubric_editing = {
        'active': False,           # Whether editing mode is active
        'edit_messages': [],       # Conversation about edits
        'proposed_changes': None,  # Parsed changes from AI
        'original_rubric': None    # Snapshot before editing
    }

# Regenerated draft from rubric changes
if 'regenerated_draft' not in st.session_state:
    st.session_state.regenerated_draft = None  # Stores {revised_draft, changes_made, original_rubric, updated_rubric}

# Branch rubric chat for AI suggestions
if 'branch_rubric_chat' not in st.session_state:
    st.session_state.branch_rubric_chat = []  # List of {role, content} messages

# Evaluate tab state
if 'evaluate_selected_conversation' not in st.session_state:
    st.session_state.evaluate_selected_conversation = None
if 'evaluate_decision_points' not in st.session_state:
    st.session_state.evaluate_decision_points = None  # Stores the extracted decision points result (parsed JSON)
if 'evaluate_selected_decision_point' not in st.session_state:
    st.session_state.evaluate_selected_decision_point = None  # Index of currently selected decision point
if 'evaluate_expanded_dp' not in st.session_state:
    st.session_state.evaluate_expanded_dp = None  # ID of currently expanded decision point (for highlighting)
if 'evaluate_reflection_items' not in st.session_state:
    st.session_state.evaluate_reflection_items = None  # Stores the generated reflection items for all decision points
if 'evaluate_user_responses' not in st.session_state:
    st.session_state.evaluate_user_responses = {}  # Dict mapping decision point id to user's reflection responses
if 'evaluate_preference_dimensions' not in st.session_state:
    st.session_state.evaluate_preference_dimensions = None  # Extracted preference dimensions from reflections
if 'evaluate_test_comparisons' not in st.session_state:
    st.session_state.evaluate_test_comparisons = None  # Generated test comparisons
if 'evaluate_user_tests' not in st.session_state:
    st.session_state.evaluate_user_tests = None  # Formatted user-facing tests with answer key
if 'evaluate_test_responses' not in st.session_state:
    st.session_state.evaluate_test_responses = {}  # User's responses to tests
if 'evaluate_rubric_scores' not in st.session_state:
    st.session_state.evaluate_rubric_scores = None  # Rubric-based scores for test comparisons

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

def load_conversations():
    """Load all conversation files from logs directory"""
    project_name = st.session_state.get('current_project')
    if not project_name:
        return []
    log_dir = Path("project") / project_name / "logs"
    if not log_dir.exists():
        return []
    
    conversations = []
    for file in sorted(log_dir.glob("conversation-*.json"), reverse=True):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Extract readable date from filename
                filename = file.stem
                date_str = filename.replace("conversation-", "")
                conversations.append({
                    "filename": file.name,
                    "timestamp": data.get("timestamp", date_str),
                    "messages_count": len(data.get("messages", [])),
                    "data": data
                })
        except Exception as e:
            continue
    return conversations

def load_conversation_data(filepath):
    """Load conversation data from file"""
    project_name = st.session_state.get('current_project')
    if not project_name:
        return None
    log_dir = Path("project") / project_name / "logs"
    file = log_dir / filepath
    if file.exists():
        with open(file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def get_available_projects():
    """Get list of available project folders from the 'project' directory"""
    projects_dir = Path("project")
    if not projects_dir.exists():
        projects_dir.mkdir(exist_ok=True)
        return []

    projects = []
    for item in projects_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            projects.append(item.name)
    return sorted(projects)

def create_new_project(project_name):
    """Create a new project folder with necessary structure inside the 'project' directory"""
    # Validate project name (alphanumeric, hyphens, underscores only)
    import string
    valid_chars = string.ascii_letters + string.digits + '-_'
    if not all(c in valid_chars for c in project_name):
        return False, "Project name can only contain letters, numbers, hyphens, and underscores"

    # Ensure project directory exists
    projects_dir = Path("project")
    projects_dir.mkdir(exist_ok=True)

    project_path = projects_dir / project_name
    if project_path.exists():
        return False, "Project folder already exists"

    try:
        # Create project folder
        project_path.mkdir(exist_ok=True)

        # Create logs subfolder
        logs_path = project_path / "logs"
        logs_path.mkdir(exist_ok=True)

        # Create empty rubric_history.json
        rubric_file = project_path / "rubric_history.json"
        with open(rubric_file, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2)

        return True, f"Project '{project_name}' created successfully!"
    except Exception as e:
        return False, f"Error creating project: {str(e)}"

# Title at the top
st.title("✍️ AI-Rubric Writer")
st.markdown("Collaborate with AI to improve your writing!")

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["💬 Chat", "📋 View Rubric", "🔍 Compare Rubrics", "📊 Evaluate: Coverage", "🎯 Evaluate: Alignment", "⚡ Evaluate: Utility"])

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
                st.rerun()
    
    st.divider()

    # Display chat messages
    for idx, message in enumerate(st.session_state.messages):
        # Skip assessment messages (ASSESS_RUBRIC_PROMPT and evaluation response) from display
        # They're in conversation history for context but shown only as cards
        if message.get('is_assessment_message'):
            continue

        if message['role'] == 'system':
            st.info(message['content'])
        else:
            with st.chat_message(message['role']):
                # Get message_id - prefer stored one, otherwise generate
                message_id = message.get('message_id', f"{message['role']}_{idx}")

                # For user messages, always show the full content (which includes feedback)
                # For assistant messages, use display_content (clean version without annotations)
                if message['role'] == 'user':
                    content_to_display = message['content']
                else:
                    content_to_display = message.get('display_content', message['content'])

                # Show thinking if available (for assistant messages)
                if message['role'] == 'assistant' and message.get('thinking'):
                    with st.expander("🧠 Thinking", expanded=False):
                        st.markdown(message['thinking'])

                # Check if content contains <draft> tags and render accordingly
                # For assistant messages, make drafts editable
                if message['role'] == 'assistant':
                    has_draft = render_message_with_draft(content_to_display, message_id, is_branch=False)
                    if not has_draft:
                        st.markdown(content_to_display)
                else:
                    st.markdown(content_to_display)

                # Display rubric assessment if available (for assistant messages)
                if message['role'] == 'assistant' and message.get('rubric_assessment'):
                    display_rubric_assessment(message['rubric_assessment'], message_id)

    # Display branch messages if in branch mode
    if st.session_state.branch['active'] and st.session_state.branch['messages']:
        st.markdown("---")
        st.markdown("### 🌿 **Branch Messages** (Temporary)")
        st.caption("These messages will not be saved to the main conversation unless you click 'Merge Branch'")

        for idx, message in enumerate(st.session_state.branch['messages']):
            # Skip assessment messages from display
            if message.get('is_assessment_message'):
                continue

            with st.chat_message(message['role']):
                message_id = message.get('message_id', f"branch_{message['role']}_{idx}")

                if message['role'] == 'user':
                    content_to_display = message['content']
                else:
                    content_to_display = message.get('display_content', message['content'])

                # Show thinking if available (for assistant messages)
                if message['role'] == 'assistant' and message.get('thinking'):
                    with st.expander("🧠 Thinking", expanded=False):
                        st.markdown(message['thinking'])

                # Check if content contains <draft> tags and render accordingly
                # For assistant messages, make drafts editable
                if message['role'] == 'assistant':
                    has_draft = render_message_with_draft(content_to_display, message_id, is_branch=True)
                    if not has_draft:
                        st.markdown(content_to_display)
                else:
                    st.markdown(content_to_display)

                # Display rubric assessment if available
                if message['role'] == 'assistant' and message.get('rubric_assessment'):
                    display_rubric_assessment(message['rubric_assessment'], message_id)

        # Branch control buttons
        st.markdown("---")
        branch_col1, branch_col2 = st.columns(2)

        with branch_col1:
            if st.button("❌ Discard Branch", use_container_width=True, type="secondary"):
                # Discard all branch messages and exit branch mode
                st.session_state.branch = {
                    'active': False,
                    'messages': [],
                    'rubric': None,
                    'branch_point': None,
                    'branch_name': None
                }
                # Reset editing state
                st.session_state.rubric_editing = {
                    'active': False,
                    'edit_messages': [],
                    'proposed_changes': None,
                    'original_rubric': None
                }
                # Clear branch rubric chat
                st.session_state.branch_rubric_chat = []
                st.success("Branch discarded")
                st.rerun()

        with branch_col2:
            if st.button("✅ Merge Branch to Main", use_container_width=True, type="primary"):
                import copy

                # Capture the branch rubric BEFORE any modifications
                # The sidebar code should have already updated this on this rerun
                branch_rubric_to_save = copy.deepcopy(st.session_state.branch['rubric']) if st.session_state.branch['rubric'] else None

                # Merge branch messages into main conversation
                st.session_state.messages.extend(st.session_state.branch['messages'])

                # Always save the branch rubric if it exists (it was modified in branch mode)
                rubric_modified = branch_rubric_to_save is not None

                # If branch rubric was modified, save it as a new version
                if rubric_modified:
                    # Get current rubric info for metadata
                    active_rubric_dict, _, _ = get_active_rubric()

                    # Create new rubric version
                    new_version = next_version_number()
                    new_rubric_entry = {
                        "version": new_version,
                        "rubric": branch_rubric_to_save,
                        "writing_type": active_rubric_dict.get("writing_type", "Unknown") if active_rubric_dict else "Unknown",
                        "user_goals_summary": active_rubric_dict.get("user_goals_summary", "Modified from branch") if active_rubric_dict else "Modified from branch",
                        "weighting_rationale": f"Weights adjusted in branch mode from v{active_rubric_dict.get('version', '?') if active_rubric_dict else '?'}",
                        "coaching_notes": "Rubric weights were modified in branch mode experimentation"
                    }

                    # Add to history
                    hist = load_rubric_history()
                    hist.append(new_rubric_entry)
                    save_rubric_history(hist)

                    # Update active rubric index to the new version
                    st.session_state.active_rubric_idx = len(hist) - 1
                    st.session_state.rubric = branch_rubric_to_save
                    # Also update editing_criteria so Rubric Configuration shows the new version
                    st.session_state.editing_criteria = copy.deepcopy(branch_rubric_to_save)

                # Clear branch and exit branch mode
                st.session_state.branch = {
                    'active': False,
                    'messages': [],
                    'rubric': None,
                    'branch_point': None,
                    'branch_name': None
                }
                # Reset editing state
                st.session_state.rubric_editing = {
                    'active': False,
                    'edit_messages': [],
                    'proposed_changes': None,
                    'original_rubric': None
                }
                # Clear regenerated draft state and branch rubric chat
                st.session_state.regenerated_draft = None
                st.session_state.branch_rubric_chat = []

                if rubric_modified:
                    st.success("✓ Branch merged! New rubric version created.")
                else:
                    st.success("✓ Branch merged successfully!")
                st.rerun()

    # Display rubric update analysis result if pending
    display_rubric_update_result()

    # Display regenerated draft if available (from branch mode rubric changes)
    if st.session_state.regenerated_draft:
        st.divider()
        st.markdown("### ✨ Regenerated Draft")
        st.caption("Draft revised based on your rubric changes")

        regen = st.session_state.regenerated_draft

        # Show thinking if available
        if regen.get('thinking'):
            with st.expander("🧠 Thinking", expanded=False):
                st.markdown(regen['thinking'])

        # Show revision strategy
        if regen.get('revision_strategy'):
            st.info(f"**Strategy:** {regen['revision_strategy']}")

        # CSS for diff highlighting
        st.markdown("""
        <style>
        .draft-diff-container {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 16px;
            margin: 10px 0;
            line-height: 1.8;
        }
        .draft-text-removed {
            background-color: #ffcccb;
            color: #721c24;
            text-decoration: line-through;
            padding: 2px 4px;
            border-radius: 3px;
        }
        .draft-text-added {
            background-color: #d4edda;
            color: #155724;
            padding: 2px 4px;
            border-radius: 3px;
            font-weight: 500;
        }
        </style>
        """, unsafe_allow_html=True)

        # Generate word-level diff between original and revised draft
        original_draft = regen.get('original_draft', '')
        revised_draft = regen.get('revised_draft', '')

        if original_draft and revised_draft:
            import difflib

            # Split into paragraphs first, then process each paragraph
            # This preserves paragraph structure while doing word-level diff
            old_paragraphs = original_draft.split('\n\n')
            new_paragraphs = revised_draft.split('\n\n')

            # Process paragraph by paragraph
            all_diff_html = []

            # Use paragraph-level matching first
            para_matcher = difflib.SequenceMatcher(None, old_paragraphs, new_paragraphs)

            for tag, i1, i2, j1, j2 in para_matcher.get_opcodes():
                if tag == 'equal':
                    # Paragraphs are the same, but still do word-level diff for minor changes
                    for para in old_paragraphs[i1:i2]:
                        # Escape HTML in paragraph
                        para_escaped = para.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                        all_diff_html.append(f'<p>{para_escaped}</p>')
                elif tag == 'replace':
                    # Paragraphs changed - do word-level diff within each matched pair
                    old_paras = old_paragraphs[i1:i2]
                    new_paras = new_paragraphs[j1:j2]

                    # Match paragraphs by position where possible
                    max_len = max(len(old_paras), len(new_paras))
                    for idx in range(max_len):
                        old_para = old_paras[idx] if idx < len(old_paras) else ''
                        new_para = new_paras[idx] if idx < len(new_paras) else ''

                        if old_para and new_para:
                            # Do word-level diff
                            old_words = old_para.split()
                            new_words = new_para.split()
                            word_matcher = difflib.SequenceMatcher(None, old_words, new_words)
                            para_diff = []

                            for wtag, wi1, wi2, wj1, wj2 in word_matcher.get_opcodes():
                                if wtag == 'equal':
                                    text = ' '.join(old_words[wi1:wi2])
                                    text_escaped = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                                    para_diff.append(text_escaped)
                                elif wtag == 'replace':
                                    if wi1 < wi2:
                                        text = ' '.join(old_words[wi1:wi2])
                                        text_escaped = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                                        para_diff.append(f'<span class="draft-text-removed">{text_escaped}</span>')
                                    if wj1 < wj2:
                                        text = ' '.join(new_words[wj1:wj2])
                                        text_escaped = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                                        para_diff.append(f'<span class="draft-text-added">{text_escaped}</span>')
                                elif wtag == 'delete':
                                    text = ' '.join(old_words[wi1:wi2])
                                    text_escaped = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                                    para_diff.append(f'<span class="draft-text-removed">{text_escaped}</span>')
                                elif wtag == 'insert':
                                    text = ' '.join(new_words[wj1:wj2])
                                    text_escaped = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                                    para_diff.append(f'<span class="draft-text-added">{text_escaped}</span>')

                            all_diff_html.append(f'<p>{" ".join(para_diff)}</p>')
                        elif old_para:
                            # Entire paragraph removed
                            text_escaped = old_para.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                            all_diff_html.append(f'<p><span class="draft-text-removed">{text_escaped}</span></p>')
                        elif new_para:
                            # Entire paragraph added
                            text_escaped = new_para.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                            all_diff_html.append(f'<p><span class="draft-text-added">{text_escaped}</span></p>')

                elif tag == 'delete':
                    # Paragraphs removed
                    for para in old_paragraphs[i1:i2]:
                        text_escaped = para.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                        all_diff_html.append(f'<p><span class="draft-text-removed">{text_escaped}</span></p>')
                elif tag == 'insert':
                    # Paragraphs added
                    for para in new_paragraphs[j1:j2]:
                        text_escaped = para.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                        all_diff_html.append(f'<p><span class="draft-text-added">{text_escaped}</span></p>')

            diff_html = ''.join(all_diff_html)
            st.markdown(f'<div class="draft-diff-container">{diff_html}</div>', unsafe_allow_html=True)
        else:
            # Fallback to just showing the revised draft with paragraph breaks
            revised_html = revised_draft.replace('\n\n', '</p><p>').replace('\n', '<br>')
            st.markdown(f'<div class="draft-diff-container"><p>{revised_html}</p></div>', unsafe_allow_html=True)

        # Show changes made
        if regen.get('changes_made'):
            with st.expander("📝 Changes Made", expanded=False):
                for change in regen['changes_made']:
                    st.markdown(f"• {change}")

        # Action buttons for the regenerated draft
        st.markdown("---")
        action_col1, action_col2 = st.columns(2)

        with action_col1:
            if st.button("✅ Accept Draft", use_container_width=True, key="accept_regen_draft_chat", type="primary"):
                # Add a new assistant message with the revised draft
                new_draft = regen['revised_draft']

                # Create a new message with the revised draft
                import uuid
                new_message_id = f"assistant_{uuid.uuid4().hex[:8]}"

                # Build message content with the draft wrapped in tags
                new_message_content = f"Here's the revised draft based on your rubric changes:\n\n<draft>{new_draft}</draft>"

                # Add changes made as context
                if regen.get('changes_made'):
                    new_message_content += "\n\n**Changes made:**\n"
                    for change in regen['changes_made']:
                        new_message_content += f"- {change}\n"

                new_message = {
                    'role': 'assistant',
                    'content': new_message_content,
                    'message_id': new_message_id
                }

                # Add to messages (main or branch depending on mode)
                if st.session_state.branch['active']:
                    st.session_state.branch['messages'].append(new_message)
                else:
                    st.session_state.messages.append(new_message)

                # Clear the regenerated draft state
                st.session_state.regenerated_draft = None

                st.success("✓ Draft added to chat!")
                st.rerun()

        with action_col2:
            if st.button("❌ Discard", use_container_width=True, key="discard_regen_draft_chat"):
                # Clear the regenerated draft state
                st.session_state.regenerated_draft = None
                st.rerun()

        st.caption("💡 Merge the branch to save your rubric changes as a new version.")

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
                display_rubric_assessment(comparison_assessment)

    # User input (chat input and buttons)
    # Change prompt based on branch mode
    chat_placeholder = "🌿 Branch message (temporary)..." if st.session_state.branch['active'] else "Type your message here..."

    if prompt := st.chat_input(chat_placeholder):
        # Clear comparison when starting a new message
        st.session_state.comparison_result = None
        st.session_state.comparison_rubric_version = None

        # Check if there's feedback to incorporate
        feedback_context = format_feedback_for_context()

        # Determine which message list to append to
        target_messages = st.session_state.branch['messages'] if st.session_state.branch['active'] else st.session_state.messages

        # Prepend feedback to user's message if available
        if feedback_context:
            full_message = feedback_context + prompt
            # Store both the full message (for API) and display version (for UI)
            target_messages.append({
                "role": "user",
                "content": full_message,  # Full message with feedback for API
                "display_content": prompt  # Just the user's prompt for display
            })

            # Clear the feedback after incorporating it
            st.session_state.assessment_feedback = {}
        else:
            # No feedback
            full_message = prompt
            target_messages.append({"role": "user", "content": full_message})

        # Display user message (show the full message with feedback if present)
        with st.chat_message("user"):
            if feedback_context:
                # Show the full message with feedback context
                st.markdown(full_message)
            else:
                st.markdown(prompt)
        
        # Prepare message history for API
        # If in branch mode, combine main messages + branch messages
        api_messages = []

        # Always include main conversation messages
        for msg in st.session_state.messages:
            content_to_send = msg.get('content', msg.get('display_content', ''))
            api_messages.append({
                "role": msg['role'],
                "content": content_to_send
            })

        # If in branch mode, append branch messages
        if st.session_state.branch['active']:
            for msg in st.session_state.branch['messages']:
                content_to_send = msg.get('content', msg.get('display_content', ''))
                api_messages.append({
                    "role": msg['role'],
                    "content": content_to_send
                })

        # Show assistant response with streaming
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_placeholder = st.empty()

                # Generate unique message ID for this response (use timestamp to ensure uniqueness)
                import time
                message_id = f"assistant_{int(time.time() * 1000000)}"

                # Get rubric to use (branch rubric if set, otherwise main rubric)
                if st.session_state.branch['active'] and st.session_state.branch['rubric'] is not None:
                    # Use branch rubric
                    active_rubric_list = st.session_state.branch['rubric']
                else:
                    # Use main rubric
                    active_rubric_dict, _, _ = get_active_rubric()
                    if active_rubric_dict and isinstance(active_rubric_dict, dict):
                        active_rubric_list = active_rubric_dict.get("rubric", [])
                    else:
                        active_rubric_list = []

                # Build system instruction (without assessment requirements)
                system_instruction = build_system_instruction(active_rubric_list)

                # Debug: Show which rubric is being used
                if active_rubric_list:
                    st.caption(f"🔍 Using rubric with {len(active_rubric_list)} criteria")
                else:
                    st.caption("🔍 No rubric active - system instruction will not include rubric")

                # Create placeholder for thinking display
                thinking_placeholder = st.empty()

                # Stream the response
                try:
                    with client.messages.stream(
                        max_tokens=32000,
                        system=system_instruction,
                        messages=api_messages,
                        model="claude-opus-4-5-20251101",
                        thinking={
                            "type": "enabled",
                            "budget_tokens": 10000
                        }
                    ) as stream:
                        # Stream and filter out analysis tags in real-time
                        # Returns: (main_content without analysis, analysis_content, None for rubric_assessment, thinking_content)
                        main_content, analysis_content, _, thinking_content = stream_without_analysis(stream, response_placeholder, message_id, thinking_placeholder)

                    # Clear the thinking placeholder and show final thinking in expander
                    thinking_placeholder.empty()

                    # Get the currently active rubric version to store with the message
                    active_rubric_dict, active_idx, _ = get_active_rubric()
                    rubric_version = active_rubric_dict.get('version', 1) if active_rubric_dict else None

                    # Store message in appropriate location (branch or main)
                    message_data = {
                        "role": "assistant",
                        "content": main_content,
                        "display_content": main_content,
                        "message_id": message_id,
                        "rubric_version": rubric_version,
                        "rubric_assessment": None,  # Will be filled when user clicks assessment button
                        "thinking": thinking_content  # Store thinking for display
                    }

                    if st.session_state.branch['active']:
                        st.session_state.branch['messages'].append(message_data)
                    else:
                        st.session_state.messages.append(message_data)

                    # Update analysis in session state and rerun to show in sidebar
                    st.session_state.current_analysis = analysis_content

                    st.rerun()
                except Exception as e:
                    st.error(f"Error occurred: {str(e)}")
    
    # Buttons below chat input
    if st.session_state.messages:
        btn_col1, btn_col2, btn_col3, btn_col4, btn_col5 = st.columns(5)

        with btn_col1:
            save_button = st.button("💾 Save Conversation", use_container_width=True)
            if save_button:
                try:
                    # Save the conversation to a file
                    log_file = save_message_log(st.session_state.messages, st.session_state.rubric, st.session_state.current_analysis)
                    
                    # Store a flag to indicate we just saved (don't reload)
                    st.session_state.just_saved = log_file.name
                    
                    # Update selected_conversation to the saved file
                    st.session_state.selected_conversation = log_file.name
                    st.success(f"✓ Log saved to {log_file.name}")
                    st.balloons()  # Show celebration
                    st.rerun()  # Rerun to update the selector
                except Exception as e:
                    st.error(f"Error saving log: {str(e)}")
        
        with btn_col2:
            infer_button = st.button("🔍 Infer Rubric", use_container_width=True)
            if infer_button:
                if not st.session_state.messages:
                    st.error("No conversation to infer rubric from!")
                else:
                    # Store key before inference
                    preserve_state = {
                        'messages': st.session_state.messages.copy(),
                        'selected_conversation': st.session_state.selected_conversation,
                        'current_analysis': st.session_state.current_analysis
                    }
                    
                    with st.spinner("Inferring rubric from conversation..."):
                        inferred_rubric_data = infer_rubric_from_conversation(preserve_state['messages'])
                        if inferred_rubric_data:
                            # Restore preserved state
                            st.session_state.messages = preserve_state['messages']
                            st.session_state.selected_conversation = preserve_state['selected_conversation']
                            st.session_state.current_analysis = preserve_state['current_analysis']
                            
                            # Update rubric and editing criteria
                            st.session_state.rubric = inferred_rubric_data.get("rubric")
                            
                            # Get the newly set active rubric (which should now be the inferred one)
                            active_rubric_dict, active_idx, _ = get_active_rubric()
                            if active_rubric_dict:
                                # active_rubric_dict is a dict with "rubric" and "version" fields
                                rubric_list = active_rubric_dict.get("rubric", [])
                                st.session_state.editing_criteria = rubric_list.copy()
                            
                            st.success(f"✓ Rubric inferred (v{inferred_rubric_data.get('version', '?')})")
                            st.rerun()  # Rerun to update the sidebar
                        else:
                            # Restore preserved state on failure
                            st.session_state.messages = preserve_state['messages']
                            st.session_state.selected_conversation = preserve_state['selected_conversation']
                            st.session_state.current_analysis = preserve_state['current_analysis']
                            st.error("Failed to infer rubric")
        
        with btn_col3:
            # Only show assess button if there's an assistant message and active rubric
            active_rubric_dict, _, _ = get_active_rubric()

            # Check for assistant messages in branch or main
            # In branch mode, check both main messages (always available) and branch messages
            if st.session_state.branch['active']:
                has_assistant_message = (
                    any(msg['role'] == 'assistant' for msg in st.session_state.messages) or
                    any(msg['role'] == 'assistant' for msg in st.session_state.branch['messages'])
                )
            else:
                has_assistant_message = any(msg['role'] == 'assistant' for msg in st.session_state.messages)

            if has_assistant_message and (active_rubric_dict or st.session_state.branch['rubric']):
                assess_button = st.button("📊 Assess Draft", use_container_width=True)
                if assess_button:
                    # Get the last assistant message with a draft (from branch if in branch mode)
                    last_assistant_msg = None
                    last_assistant_idx = None
                    is_branch_message = False
                    draft_content = None

                    # Pattern to extract draft content
                    draft_pattern = r'<draft>(.*?)</draft>'

                    if st.session_state.branch['active']:
                        # Look in branch messages first (from most recent to oldest)
                        for i in range(len(st.session_state.branch['messages']) - 1, -1, -1):
                            if st.session_state.branch['messages'][i]['role'] == 'assistant':
                                msg_content = st.session_state.branch['messages'][i].get('display_content', st.session_state.branch['messages'][i].get('content', ''))
                                match = re.search(draft_pattern, msg_content, re.DOTALL)
                                if match:
                                    last_assistant_msg = st.session_state.branch['messages'][i]
                                    last_assistant_idx = i
                                    is_branch_message = True
                                    draft_content = match.group(1).strip()
                                    break

                        # If no draft found in branch, fall back to main messages
                        if draft_content is None:
                            for i in range(len(st.session_state.messages) - 1, -1, -1):
                                if st.session_state.messages[i]['role'] == 'assistant':
                                    msg_content = st.session_state.messages[i].get('display_content', st.session_state.messages[i].get('content', ''))
                                    match = re.search(draft_pattern, msg_content, re.DOTALL)
                                    if match:
                                        last_assistant_msg = st.session_state.messages[i]
                                        last_assistant_idx = i
                                        is_branch_message = False
                                        draft_content = match.group(1).strip()
                                        break
                    else:
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
                                from prompts import ASSESS_RUBRIC_PROMPT

                                # Build the assessment prompt with the draft content
                                assessment_prompt = f"""{ASSESS_RUBRIC_PROMPT}
                                    ## Draft to Assess

                                    <draft>
                                    {draft_content}
                                    </draft>
                                    """

                                # Build the conversation history with assessment prompt as the last message
                                assessment_messages = []

                                # Always include main messages
                                for msg in st.session_state.messages:
                                    assessment_messages.append({
                                        "role": msg['role'],
                                        "content": msg['content']
                                    })

                                # If in branch mode, add branch messages
                                if st.session_state.branch['active']:
                                    for msg in st.session_state.branch['messages']:
                                        assessment_messages.append({
                                            "role": msg['role'],
                                            "content": msg['content']
                                        })

                                # Add the assessment prompt as the last user message
                                assessment_messages.append({
                                    "role": "user",
                                    "content": assessment_prompt
                                })

                                # Get rubric to use (branch rubric if in branch mode and modified, otherwise main)
                                if st.session_state.branch['active'] and st.session_state.branch['rubric'] is not None:
                                    active_rubric_list = st.session_state.branch['rubric']
                                else:
                                    if active_rubric_dict and isinstance(active_rubric_dict, dict):
                                        active_rubric_list = active_rubric_dict.get("rubric", [])
                                    else:
                                        active_rubric_list = []

                                system_instruction = build_system_instruction(active_rubric_list)

                                # Make API call with full conversation context
                                assessment_response = client.messages.create(
                                    max_tokens=16000,
                                    system=system_instruction,
                                    messages=assessment_messages,
                                    model="claude-opus-4-5-20251101",
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
                                # Store thinking in the assessment
                                if rubric_assessment:
                                    rubric_assessment['thinking'] = thinking_text

                                # Update the last assistant message with the assessment (for UI display)
                                # Use correct message list based on whether it's a branch message
                                if is_branch_message:
                                    st.session_state.branch['messages'][last_assistant_idx]['rubric_assessment'] = rubric_assessment
                                    target_messages = st.session_state.branch['messages']
                                else:
                                    st.session_state.messages[last_assistant_idx]['rubric_assessment'] = rubric_assessment
                                    target_messages = st.session_state.messages

                                # Add the assessment to conversation history (in appropriate location)
                                # First add the assessment prompt as a user message (hidden from display)
                                target_messages.append({
                                    "role": "user",
                                    "content": assessment_prompt,
                                    "is_assessment_message": True  # Mark for hiding from display
                                })

                                # Then add the full assessment text as an assistant message (hidden from display)
                                target_messages.append({
                                    "role": "assistant",
                                    "content": assessment_text,
                                    "display_content": assessment_text,
                                    "message_id": str(uuid.uuid4()),
                                    "rubric_version": active_rubric_dict.get("version") if active_rubric_dict else None,
                                    "rubric_assessment": rubric_assessment,
                                    "is_assessment_message": True  # Mark for hiding from display
                                })

                                st.success("✓ Draft assessed successfully!")
                                st.rerun()

                            except Exception as e:
                                st.error(f"Error during assessment: {str(e)}")

        with btn_col4:
            # Show different button based on branch mode
            if st.session_state.branch['active']:
                exit_branch_button = st.button("🌿 Exit Branch", use_container_width=True, type="secondary")
                if exit_branch_button:
                    # Just switch view without discarding - user can discard from branch view
                    st.session_state.branch['active'] = False
                    # Reset editing state when exiting branch view
                    st.session_state.rubric_editing = {
                        'active': False,
                        'edit_messages': [],
                        'proposed_changes': None,
                        'original_rubric': None
                    }
                    st.rerun()
            else:
                branch_button = st.button("🌿 Create Branch", use_container_width=True)
                if branch_button:
                    # Create a branch from the current conversation state
                    st.session_state.branch = {
                        'active': True,
                        'messages': [],
                        'rubric': None,  # Start with main rubric, user can modify
                        'branch_point': len(st.session_state.messages),
                        'branch_name': f"Branch from message #{len(st.session_state.messages)}"
                    }
                    st.rerun()

        with btn_col5:
            clear_button = st.button("🗑️ Clear Conversation", use_container_width=True)
            if clear_button:
                st.session_state.messages = []
                st.session_state.current_analysis = ""
                st.session_state.selected_conversation = None
                st.session_state.comparison_result = None
                st.session_state.comparison_rubric_version = None
                # Also clear branch if active
                st.session_state.branch = {
                    'active': False,
                    'messages': [],
                    'rubric': None,
                    'branch_point': None,
                    'branch_name': None
                }
                st.rerun()
                
    # Comparison mode UI (only show if there are messages and an assistant response)
# Sidebar for rubric input
with st.sidebar:
    # Branch mode indicator at the very top
    if st.session_state.branch['active']:
        st.info(f"🌿 **Branch Mode Active**: {st.session_state.branch['branch_name']}")
        st.caption("Changes are temporary until merged")
        st.divider()

    # Project Selector at the top
    st.header("📁 Project")

    # Get available projects
    available_projects = get_available_projects()

    # Project selector
    if available_projects:
        selected_project = st.selectbox(
            "Select Project:",
            options=available_projects,
            index=available_projects.index(st.session_state.current_project) if st.session_state.current_project in available_projects else 0,
            key="project_selector"
        )

        # Update current_project if selection changed
        if selected_project != st.session_state.current_project:
            st.session_state.current_project = selected_project
            # Reset session state
            st.session_state.messages = []
            st.session_state.selected_conversation = None
            if 'active_rubric_idx' in st.session_state:
                del st.session_state.active_rubric_idx

            # Load the new project's rubric
            active_rubric_dict, active_idx, _ = get_active_rubric()
            if active_rubric_dict and active_rubric_dict.get("rubric"):
                rubric_list = active_rubric_dict.get("rubric", [])
                st.session_state.rubric = rubric_list
                st.session_state.editing_criteria = rubric_list.copy()
            else:
                st.session_state.rubric = []
                st.session_state.editing_criteria = []

            st.rerun()
    else:
        st.info("No projects found. Create one below!")

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
                success, message = create_new_project(new_project_name.strip())
                if success:
                    st.success(message)
                    # Update current project
                    st.session_state.current_project = new_project_name.strip()
                    # Reset session state
                    st.session_state.rubric = []
                    st.session_state.editing_criteria = []
                    st.session_state.messages = []
                    st.session_state.selected_conversation = None
                    # Clear the text input and collapse expander
                    if 'new_project_name' in st.session_state:
                        del st.session_state.new_project_name
                    st.session_state.create_project_expanded = False
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.error("Please enter a project name")

    # Export/Import Project section
    with st.expander("📦 Export / Import Project"):
        st.markdown("**Export** your project data to save it locally, or **Import** a previously exported project.")

        # Export button
        if st.session_state.current_project:
            project_dir = Path("project") / st.session_state.current_project

            if project_dir.exists():
                # Create zip file in memory
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for file_path in project_dir.rglob('*'):
                        if file_path.is_file():
                            arcname = file_path.relative_to(project_dir.parent)
                            zip_file.write(file_path, arcname)

                zip_buffer.seek(0)

                st.download_button(
                    label=f"📥 Export '{st.session_state.current_project}'",
                    data=zip_buffer.getvalue(),
                    file_name=f"{st.session_state.current_project}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    use_container_width=True
                )

        st.markdown("---")

        # Import project
        uploaded_file = st.file_uploader(
            "Import Project (ZIP file)",
            type=['zip'],
            key="project_import_uploader"
        )

        if uploaded_file is not None:
            if st.button("📤 Import Project", use_container_width=True):
                try:
                    # Extract zip file
                    with zipfile.ZipFile(io.BytesIO(uploaded_file.read()), 'r') as zip_file:
                        # Get the project name from the zip structure
                        namelist = zip_file.namelist()
                        if namelist:
                            # The first part of the path should be the project name
                            project_name = namelist[0].split('/')[0]

                            # Check if project already exists
                            target_dir = Path("project") / project_name
                            if target_dir.exists():
                                # Add timestamp to make it unique
                                project_name = f"{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                target_dir = Path("project") / project_name

                            # Extract to project directory
                            zip_file.extractall(Path("project"))

                            # If we renamed it, we need to rename the extracted folder
                            original_name = namelist[0].split('/')[0]
                            if project_name != original_name:
                                original_dir = Path("project") / original_name
                                if original_dir.exists():
                                    shutil.move(str(original_dir), str(target_dir))

                            st.success(f"Project '{project_name}' imported successfully!")
                            st.session_state.current_project = project_name
                            st.session_state.messages = []
                            st.session_state.selected_conversation = None
                            if 'active_rubric_idx' in st.session_state:
                                del st.session_state.active_rubric_idx
                            st.rerun()
                except Exception as e:
                    st.error(f"Error importing project: {str(e)}")

    st.divider()

    # Hide Rubric Configuration when in branch mode (users edit rubric via Branch Rubric section instead)
    if not st.session_state.branch['active']:
        st.header("📋 Rubric Configuration")

        # Get active rubric
        active_rubric_dict, active_idx, rubric_history = get_active_rubric()

        # Initialize editing criteria if needed
        if "editing_criteria" not in st.session_state:
            if active_rubric_dict:
                rubric_list = active_rubric_dict.get("rubric", [])
                st.session_state.editing_criteria = rubric_list.copy() if rubric_list else []
            else:
                st.session_state.editing_criteria = []

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
                    # Reset editing criteria
                    if active_rubric_dict:
                        st.session_state.editing_criteria = rubric_list.copy()
                    st.rerun()

        # Display current rubric
        if st.session_state.editing_criteria:
            st.markdown("### Current Criteria")

            for i, criterion in enumerate(st.session_state.editing_criteria):
                criterion_id = str(i + 1)

                # Simple layout with just the expander
                with st.expander(f"📌 {criterion.get('name', 'Unnamed Criterion')}", expanded=False):
                    desc_key = f"criterion_desc_{i}_{st.session_state.active_rubric_idx if st.session_state.active_rubric_idx is not None else 0}"
                    description = st.text_area(
                        "Description",
                        value=criterion.get("description", ""),
                        key=desc_key,
                        placeholder="User-specific description using conversation language...",
                        height=100
                    )

                    weight_key = f"criterion_weight_{i}_{st.session_state.active_rubric_idx if st.session_state.active_rubric_idx is not None else 0}"
                    weight = st.number_input(
                        "Weight (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(criterion.get("weight", 0)),
                        key=weight_key,
                        step=0.1,
                        format="%.1f",
                        help="Percentage weight (all weights should sum to 100%)"
                    )

                    exemplary_key = f"criterion_exemplary_{i}_{st.session_state.active_rubric_idx if st.session_state.active_rubric_idx is not None else 0}"
                    exemplary = st.text_area(
                        "Exemplary",
                        value=criterion.get("exemplary", ""),
                        key=exemplary_key,
                        placeholder="Concrete descriptors for highest achievement...",
                        height=80
                    )

                    # Update the criterion in the session state
                    if description:
                        st.session_state.editing_criteria[i]["description"] = description
                        st.session_state.editing_criteria[i]["weight"] = weight
                        st.session_state.editing_criteria[i]["exemplary"] = exemplary

                    if st.button("🗑️ Remove", key=f"remove_{i}"):
                        st.session_state.editing_criteria.pop(i)
                        st.rerun()

            # Delete Version button
            if st.button("🗑️ Delete Version", use_container_width=True, type="secondary"):
                if len(rubric_history) > 1:
                    # Delete the current version
                    deleted_version = rubric_history[active_idx].get("version", "?")
                    rubric_history.pop(active_idx)

                    # Update active rubric index
                    if active_idx >= len(rubric_history):
                        st.session_state.active_rubric_idx = len(rubric_history) - 1
                    elif active_idx > 0:
                        st.session_state.active_rubric_idx = active_idx - 1

                    # Save updated history
                    save_rubric_history(rubric_history)

                    # Update current rubric
                    new_active_rubric_dict, new_active_idx, _ = get_active_rubric()
                    rubric_list = new_active_rubric_dict.get("rubric", []) if new_active_rubric_dict else []
                    st.session_state.rubric = rubric_list
                    st.session_state.active_rubric_idx = new_active_idx

                    # Update editing criteria
                    if new_active_rubric_dict:
                        st.session_state.editing_criteria = rubric_list.copy()

                    st.success(f"✓ Rubric version {deleted_version} deleted")
                    st.rerun()
                else:
                    st.error("Cannot delete the last rubric version")

            # Fallback for when Update button condition is false
            if active_rubric_dict and active_idx is None:
                rubric_list = active_rubric_dict.get("rubric", [])
                st.session_state.rubric = rubric_list
        else:
            st.info("No rubric loaded. Use 'Infer Rubric' to create one from your conversation.")

        st.divider()

    # Rubric editing interface (only show if editing mode is active in branch mode)
    if st.session_state.branch['active'] and st.session_state.rubric_editing['active']:
        with st.expander("✏️ Edit Rubric Content", expanded=True):
            st.caption("Describe the changes you want to make to your rubric. The AI will interpret your request and show you proposed changes.")

            # Display current branch rubric
            st.markdown("**Current Branch Rubric:**")
            current_rubric = st.session_state.rubric_editing['original_rubric']
            if current_rubric:
                # Create a compact display
                rubric_display = '<div style="max-height: 200px; overflow-y: auto; padding: 10px; background-color: #f5f5f5; border-radius: 5px; margin-bottom: 10px; font-size: 0.9em;">'
                for i, criterion in enumerate(current_rubric):
                    rubric_display += f'<div style="margin-bottom: 8px;">'
                    rubric_display += f'<strong>{i+1}. {criterion.get("name", "Unnamed")}</strong><br>'
                    rubric_display += f'<span style="color: #666;">{criterion.get("description", "No description")[:150]}...</span>'
                    rubric_display += f'</div>'
                rubric_display += '</div>'
                st.markdown(rubric_display, unsafe_allow_html=True)
            else:
                st.caption("No rubric loaded")

            st.divider()

            # Chat interface for editing
            st.markdown("**Edit Conversation:**")

            # Display edit messages in a scrollable container
            if st.session_state.rubric_editing['edit_messages']:
                # Create a scrollable container using custom HTML
                messages_html = '<div style="max-height: 300px; overflow-y: auto; padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 10px;">'

                for msg in st.session_state.rubric_editing['edit_messages']:
                    if msg['role'] == 'user':
                        messages_html += f'<p style="margin-bottom: 10px;"><strong>You:</strong> {msg["content"]}</p>'
                    else:
                        messages_html += f'<p style="margin-bottom: 10px; color: #0066cc;"><strong>AI:</strong> {msg["content"]}</p>'

                messages_html += '</div>'
                st.markdown(messages_html, unsafe_allow_html=True)
            else:
                st.caption("No conversation yet. Describe your changes below to get started.")

            # Input for new edit request
            edit_input = st.text_area(
                "Describe your changes:",
                placeholder="E.g., 'Change the Academic Register description to emphasize formality' or 'Add a new criterion for citation quality'",
                key="rubric_edit_input",
                height=100
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("💬 Send Request", use_container_width=True, key="send_edit_request", type="primary"):
                    if edit_input.strip():
                        # Add user message to edit conversation
                        st.session_state.rubric_editing['edit_messages'].append({
                            'role': 'user',
                            'content': edit_input
                        })

                        # Process the edit request (will implement this function next)
                        with st.spinner("Understanding your request..."):
                            try:
                                response = process_rubric_edit_request(
                                    edit_input,
                                    st.session_state.rubric_editing['original_rubric']
                                )

                                # Add AI response to conversation
                                st.session_state.rubric_editing['edit_messages'].append({
                                    'role': 'assistant',
                                    'content': response.get('message', '')
                                })

                                # Store proposed changes if present
                                if 'modified_rubric' in response:
                                    st.session_state.rubric_editing['proposed_changes'] = response

                                st.rerun()
                            except Exception as e:
                                st.error(f"Error processing request: {str(e)}")
                    else:
                        st.warning("Please enter a description of your changes.")

            with col2:
                if st.button("❌ Cancel Editing", use_container_width=True, key="cancel_edit_request"):
                    # Exit editing mode without applying changes
                    st.session_state.rubric_editing['active'] = False
                    st.session_state.rubric_editing['edit_messages'] = []
                    st.session_state.rubric_editing['proposed_changes'] = None
                    st.session_state.rubric_editing['original_rubric'] = None
                    st.rerun()

            # Show proposed changes if available
            if st.session_state.rubric_editing['proposed_changes']:
                st.divider()
                st.markdown("**Proposed Changes:**")

                proposed = st.session_state.rubric_editing['proposed_changes']

                # Show thinking if available
                if proposed.get('thinking'):
                    with st.expander("🧠 Thinking", expanded=False):
                        st.markdown(proposed['thinking'])

                st.info(f"**Summary:** {proposed.get('changes_summary', 'Changes proposed')}")

                # Display diff with detailed comparison
                modified_rubric = proposed.get('modified_rubric', [])
                original_rubric = st.session_state.rubric_editing['original_rubric']

                # Create mapping of original criteria by name for comparison
                original_by_name = {c.get('name'): c for c in original_rubric}
                modified_names = {c.get('name') for c in modified_rubric}
                original_names = {c.get('name') for c in original_rubric}

                # Show changes by category
                added_names = modified_names - original_names
                removed_names = original_names - modified_names
                common_names = modified_names & original_names

                # Show added criteria
                if added_names:
                    st.success(f"**Added {len(added_names)} criterion/criteria:**")
                    for criterion in modified_rubric:
                        if criterion.get('name') in added_names:
                            st.markdown(f"+ **{criterion.get('name')}** ({criterion.get('weight', 0)}%)")
                            st.caption(criterion.get('description', 'No description'))

                # Show removed criteria
                if removed_names:
                    st.error(f"**Removed {len(removed_names)} criterion/criteria:**")
                    for name in removed_names:
                        original_c = original_by_name[name]
                        st.markdown(f"- **{name}** ({original_c.get('weight', 0)}%)")

                # Show modified criteria with detailed before/after
                modified_count = 0
                for criterion in modified_rubric:
                    name = criterion.get('name')
                    if name in common_names:
                        original = original_by_name[name]
                        has_changes = False

                        # Collect all changes for this criterion
                        change_details = []

                        # Check for weight change
                        if criterion.get('weight') != original.get('weight'):
                            has_changes = True
                            change_details.append({
                                'field': 'Weight',
                                'before': f"{original.get('weight')}%",
                                'after': f"{criterion.get('weight')}%"
                            })

                        # Check for description change
                        if criterion.get('description') != original.get('description'):
                            has_changes = True
                            change_details.append({
                                'field': 'Description',
                                'before': original.get('description', ''),
                                'after': criterion.get('description', '')
                            })

                        # Check for achievement level changes
                        for level in ['exemplary', 'proficient', 'developing', 'beginning']:
                            if criterion.get(level) != original.get(level):
                                has_changes = True
                                change_details.append({
                                    'field': f'{level.capitalize()} level',
                                    'before': original.get(level, ''),
                                    'after': criterion.get(level, '')
                                })

                        # Check for dimension changes
                        orig_dims = {d.get('id'): d for d in original.get('dimensions', [])}
                        mod_dims = {d.get('id'): d for d in criterion.get('dimensions', [])}

                        # Added dimensions
                        for dim_id, mod_dim in mod_dims.items():
                            if dim_id not in orig_dims:
                                has_changes = True
                                change_details.append({
                                    'field': f"Dimension '{mod_dim.get('label')}'",
                                    'before': '(not present)',
                                    'after': f"importance: {mod_dim.get('importance')}"
                                })

                        # Removed dimensions
                        for dim_id, orig_dim in orig_dims.items():
                            if dim_id not in mod_dims:
                                has_changes = True
                                change_details.append({
                                    'field': f"Dimension '{orig_dim.get('label')}'",
                                    'before': f"importance: {orig_dim.get('importance')}",
                                    'after': '(removed)'
                                })

                        # Modified dimensions
                        for dim_id, mod_dim in mod_dims.items():
                            if dim_id in orig_dims:
                                orig_dim = orig_dims[dim_id]
                                if mod_dim.get('importance') != orig_dim.get('importance'):
                                    has_changes = True
                                    change_details.append({
                                        'field': f"Dimension '{mod_dim.get('label')}' importance",
                                        'before': str(orig_dim.get('importance')),
                                        'after': str(mod_dim.get('importance'))
                                    })

                        if has_changes:
                            modified_count += 1
                            st.warning(f"**Modified: {name}**")

                            # Display each change with before/after
                            for change in change_details:
                                with st.expander(f"📝 {change['field']}", expanded=False):
                                    st.markdown("**Before:**")
                                    st.caption(change['before'][:500] + ('...' if len(change['before']) > 500 else ''))
                                    st.markdown("**After:**")
                                    st.caption(change['after'][:500] + ('...' if len(change['after']) > 500 else ''))

                if modified_count == 0 and not added_names and not removed_names:
                    st.info("No changes detected")
                elif modified_count > 0:
                    st.info(f"Modified {modified_count} existing criterion/criteria")

                # Apply or discard buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Apply Changes", use_container_width=True, key="apply_edit_changes", type="primary"):
                        # Apply changes to branch rubric
                        st.session_state.branch['rubric'] = modified_rubric

                        # Add confirmation message
                        st.session_state.rubric_editing['edit_messages'].append({
                            'role': 'assistant',
                            'content': '✓ Changes applied to branch rubric! You can continue editing or merge the branch.'
                        })

                        # Clear proposed changes but keep editing mode active
                        st.session_state.rubric_editing['proposed_changes'] = None
                        st.session_state.rubric_editing['original_rubric'] = modified_rubric  # Update original to new state

                        st.success("✓ Rubric updated successfully!")
                        st.rerun()

                with col2:
                    if st.button("🔄 Discard Changes", use_container_width=True, key="discard_edit_changes"):
                        # Clear proposed changes
                        st.session_state.rubric_editing['proposed_changes'] = None

                        # Add message
                        st.session_state.rubric_editing['edit_messages'].append({
                            'role': 'assistant',
                            'content': 'Changes discarded. You can make a new request.'
                        })
                        st.rerun()

    # Branch rubric editable view (only show if in branch mode)
    if st.session_state.branch['active']:
        st.markdown("#### 📝 Branch Rubric")
        st.caption("Edit weights, dimensions, and text. Changes are temporary until merged.")

        # Get active rubric (or branch rubric if already modified)
        if st.session_state.branch['rubric'] is not None:
            current_branch_rubric = st.session_state.branch['rubric']
        else:
            active_rubric_dict, _, _ = get_active_rubric()
            current_branch_rubric = active_rubric_dict.get("rubric", []) if active_rubric_dict else []

        # Initialize branch edit state if needed
        if 'branch_rubric_edit_counter' not in st.session_state:
            st.session_state.branch_rubric_edit_counter = 0

        if current_branch_rubric:
            import copy
            modified_rubric = []
            total_weight = 0.0

            for idx, criterion in enumerate(current_branch_rubric):
                criterion_name = criterion.get("name", criterion.get("criterion", "Unknown"))
                current_weight = criterion.get("weight", 0)

                # Determine if weight is percentage or decimal
                if current_weight > 2.0:
                    default_display_weight = current_weight
                    is_percentage = True
                else:
                    default_display_weight = current_weight * 100
                    is_percentage = False

                # Check if there's already a value in session state for this weight
                # This allows the expander title to reflect the current input value
                weight_key = f"branch_weight_{idx}_{st.session_state.branch_rubric_edit_counter}"
                if weight_key in st.session_state:
                    display_weight = st.session_state[weight_key]
                else:
                    display_weight = default_display_weight

                # Create expander for each criterion
                with st.expander(f"📌 {criterion_name} ({display_weight:.0f}%)", expanded=False):
                    modified_criterion = copy.deepcopy(criterion)

                    # Weight input
                    st.markdown("**Weight (%)**")
                    new_weight = st.number_input(
                        "Weight",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(default_display_weight),
                        step=1.0,
                        format="%.1f",
                        key=weight_key,
                        label_visibility="collapsed"
                    )

                    # Store weight back in original format
                    if is_percentage:
                        modified_criterion['weight'] = new_weight
                    else:
                        modified_criterion['weight'] = new_weight / 100.0

                    total_weight += new_weight

                    st.divider()

                    # Description
                    st.markdown("**Description**")
                    new_description = st.text_area(
                        "Description",
                        value=criterion.get("description", ""),
                        height=100,
                        key=f"branch_desc_{idx}_{st.session_state.branch_rubric_edit_counter}",
                        label_visibility="collapsed"
                    )
                    modified_criterion['description'] = new_description

                    st.divider()

                    # Achievement levels
                    st.markdown("**Achievement Levels**")

                    # Exemplary
                    st.markdown("*Exemplary*")
                    new_exemplary = st.text_area(
                        "Exemplary",
                        value=criterion.get("exemplary", ""),
                        height=80,
                        key=f"branch_exemplary_{idx}_{st.session_state.branch_rubric_edit_counter}",
                        label_visibility="collapsed"
                    )
                    modified_criterion['exemplary'] = new_exemplary

                    # Proficient
                    st.markdown("*Proficient*")
                    new_proficient = st.text_area(
                        "Proficient",
                        value=criterion.get("proficient", ""),
                        height=80,
                        key=f"branch_proficient_{idx}_{st.session_state.branch_rubric_edit_counter}",
                        label_visibility="collapsed"
                    )
                    modified_criterion['proficient'] = new_proficient

                    # Developing
                    st.markdown("*Developing*")
                    new_developing = st.text_area(
                        "Developing",
                        value=criterion.get("developing", ""),
                        height=80,
                        key=f"branch_developing_{idx}_{st.session_state.branch_rubric_edit_counter}",
                        label_visibility="collapsed"
                    )
                    modified_criterion['developing'] = new_developing

                    # Beginning
                    st.markdown("*Beginning*")
                    new_beginning = st.text_area(
                        "Beginning",
                        value=criterion.get("beginning", ""),
                        height=80,
                        key=f"branch_beginning_{idx}_{st.session_state.branch_rubric_edit_counter}",
                        label_visibility="collapsed"
                    )
                    modified_criterion['beginning'] = new_beginning

                    # Dimensions (if they exist)
                    dimensions = criterion.get("dimensions", [])
                    if dimensions:
                        st.divider()
                        st.markdown("**Dimensions**")

                        modified_dimensions = []
                        dimension_total = 0.0

                        for dim_idx, dimension in enumerate(dimensions):
                            dim_label = dimension.get("label", "Unknown")
                            dim_importance = dimension.get("importance", 0.5)

                            st.markdown(f"*{dim_label}*")
                            dim_col1, dim_col2 = st.columns([3, 1])

                            with dim_col1:
                                new_dim_label = st.text_input(
                                    "Label",
                                    value=dim_label,
                                    key=f"branch_dim_label_{idx}_{dim_idx}_{st.session_state.branch_rubric_edit_counter}",
                                    label_visibility="collapsed"
                                )

                            with dim_col2:
                                new_importance = st.number_input(
                                    "Importance",
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=float(dim_importance),
                                    step=0.05,
                                    format="%.2f",
                                    key=f"branch_dim_imp_{idx}_{dim_idx}_{st.session_state.branch_rubric_edit_counter}",
                                    label_visibility="collapsed"
                                )

                            dimension_total += new_importance

                            modified_dimension = dimension.copy()
                            modified_dimension['label'] = new_dim_label
                            modified_dimension['importance'] = new_importance
                            modified_dimensions.append(modified_dimension)

                        # Show dimension total
                        if abs(dimension_total - 1.0) < 0.01:
                            st.caption(f"✓ Dimensions total: {dimension_total * 100:.0f}%")
                        else:
                            st.caption(f"⚠ Dimensions total: {dimension_total * 100:.0f}% (Should be 100%)")

                        modified_criterion['dimensions'] = modified_dimensions

                modified_rubric.append(modified_criterion)

            # Show total weight validation
            if abs(total_weight - 100.0) < 0.1:
                st.success(f"✓ Total weight: {total_weight:.0f}%")
            else:
                st.error(f"⚠ Total weight: {total_weight:.0f}% (Should be 100%)")

            # Chat interface for rubric suggestions
            st.markdown("---")
            st.markdown("#### 💬 Ask for Rubric Suggestions")
            st.caption("Describe what you want to change and get AI suggestions")

            # Scrollable chat container with CSS
            st.markdown("""
            <style>
            .rubric-chat-container {
                max-height: 300px;
                overflow-y: auto;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 10px;
                margin-bottom: 10px;
                background-color: #f9f9f9;
            }
            .rubric-chat-user {
                background-color: #e3f2fd;
                padding: 8px 12px;
                border-radius: 8px;
                margin-bottom: 8px;
            }
            .rubric-chat-ai {
                background-color: #fff;
                padding: 8px 12px;
                border-radius: 8px;
                margin-bottom: 8px;
                border: 1px solid #e0e0e0;
            }
            .rubric-change-old {
                background-color: #ffcdd2;
                text-decoration: line-through;
                padding: 2px 4px;
                border-radius: 3px;
            }
            .rubric-change-new {
                background-color: #c8e6c9;
                padding: 2px 4px;
                border-radius: 3px;
            }
            .rubric-criterion-name {
                font-weight: bold;
                color: #1976d2;
            }
            </style>
            """, unsafe_allow_html=True)

            # Display chat history in scrollable container
            if st.session_state.branch_rubric_chat:
                chat_html = '<div class="rubric-chat-container">'
                for msg in st.session_state.branch_rubric_chat:
                    if msg['role'] == 'user':
                        chat_html += f'<div class="rubric-chat-user"><strong>You:</strong> {msg["content"]}</div>'
                    else:
                        # AI response - already contains HTML formatting
                        thinking_html = ""
                        if msg.get('thinking'):
                            thinking_html = f'<details><summary>🧠 Thinking</summary><div style="font-size: 12px; color: #666; padding: 8px; background: #f5f5f5; border-radius: 4px; margin-top: 4px;">{msg["thinking"][:500]}{"..." if len(msg.get("thinking", "")) > 500 else ""}</div></details>'
                        chat_html += f'<div class="rubric-chat-ai"><strong>AI:</strong>{thinking_html}<br>{msg["content"]}</div>'
                chat_html += '</div>'
                st.markdown(chat_html, unsafe_allow_html=True)

            # Chat input
            rubric_chat_input = st.text_input(
                "Ask about rubric changes:",
                placeholder="e.g., 'Make clarity more important' or 'Add a criterion for creativity'",
                key="branch_rubric_chat_input"
            )

            col_send, col_clear = st.columns([3, 1])
            with col_send:
                send_clicked = st.button("💡 Get Suggestions", use_container_width=True, key="branch_rubric_chat_send")
            with col_clear:
                if st.session_state.branch_rubric_chat:
                    if st.button("🗑️", use_container_width=True, key="clear_rubric_chat", help="Clear chat"):
                        st.session_state.branch_rubric_chat = []
                        st.rerun()

            if send_clicked:
                if rubric_chat_input.strip():
                    # Add user message to chat
                    st.session_state.branch_rubric_chat.append({
                        'role': 'user',
                        'content': rubric_chat_input
                    })

                    # Call AI for suggestions
                    with st.spinner("Getting suggestions..."):
                        try:
                            # Build prompt with current rubric context
                            rubric_context = json.dumps(modified_rubric, indent=2, ensure_ascii=False)

                            # System prompt with rubric context and formatting instructions
                            system_prompt = f"""You are a rubric editing assistant. The user wants to modify their writing rubric.

Current rubric:
{rubric_context}

Provide specific, actionable suggestions for how to modify the rubric. Format your response as a brief summary followed by the specific criteria changes.

IMPORTANT: Show changes INLINE within the text using word-level diffs, NOT as "from X to Y" format.
- Use <span class="rubric-change-old">removed words</span> for text being removed (will show strikethrough)
- Use <span class="rubric-change-new">added words</span> for text being added (will show highlight)
- Use <span class="rubric-criterion-name">Criterion Name</span> for criterion names

Show the FULL updated text with the changes marked inline. Keep unchanged text as plain text.

Example format:
"To address your request, I recommend updating the following:

<span class="rubric-criterion-name">Clarity</span> (weight: <span class="rubric-change-old">20%</span><span class="rubric-change-new">35%</span>)

<span class="rubric-criterion-name">Structure</span> Description:
<span class="rubric-change-old">The old phrasing that is being removed</span><span class="rubric-change-new">The new phrasing that replaces it</span> while the unchanged middle part stays as plain text <span class="rubric-change-old">and more removed text</span><span class="rubric-change-new">with its replacement</span>."

Keep your response concise. Focus only on the criteria that need to change."""

                            # Build conversation messages from chat history
                            conversation_messages = []
                            for msg in st.session_state.branch_rubric_chat:
                                conversation_messages.append({
                                    "role": msg['role'],
                                    "content": msg['content']
                                })

                            response = client.messages.create(
                                max_tokens=8000,
                                system=system_prompt,
                                messages=conversation_messages,
                                model="claude-opus-4-5-20251101",
                                thinking={
                                    "type": "enabled",
                                    "budget_tokens": 4000
                                }
                            )

                            # Extract thinking and text from response
                            thinking_text = ""
                            ai_response = ""
                            for block in response.content:
                                if block.type == "thinking":
                                    thinking_text = block.thinking
                                elif block.type == "text":
                                    ai_response = block.text

                            # Add AI response to chat (with thinking)
                            st.session_state.branch_rubric_chat.append({
                                'role': 'assistant',
                                'content': ai_response,
                                'thinking': thinking_text
                            })

                            st.rerun()

                        except Exception as e:
                            st.error(f"Error getting suggestions: {str(e)}")
                else:
                    st.warning("Please enter a question or request")

            # Update Rubric with Suggestions button - only show if there are AI suggestions
            has_ai_suggestions = any(msg['role'] == 'assistant' for msg in st.session_state.branch_rubric_chat)
            if has_ai_suggestions:
                if st.button("✨ Update Rubric with Suggestions", use_container_width=True, key="apply_rubric_suggestions", type="primary"):
                    with st.spinner("Applying suggestions to rubric..."):
                        try:
                            # Get the last AI suggestion
                            last_ai_msg = None
                            for msg in reversed(st.session_state.branch_rubric_chat):
                                if msg['role'] == 'assistant':
                                    last_ai_msg = msg['content']
                                    break

                            if last_ai_msg:
                                # Build prompt to get structured rubric updates
                                rubric_context = json.dumps(modified_rubric, indent=2, ensure_ascii=False)
                                apply_prompt = f"""Based on the suggestions you previously provided, generate the updated rubric.

Current rubric:
{rubric_context}

Your previous suggestions:
{last_ai_msg}

IMPORTANT: Return ONLY a valid JSON array representing the updated rubric. The structure must match the current rubric exactly, with only the suggested changes applied.

Return the complete rubric array with all criteria, not just the changed ones. Do not include any explanation, just the JSON array."""

                                response = client.messages.create(
                                    max_tokens=8000,
                                    messages=[{"role": "user", "content": apply_prompt}],
                                    model="claude-opus-4-5-20251101",
                                    thinking={
                                        "type": "enabled",
                                        "budget_tokens": 4000
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
                                response_text = response_text.strip()

                                # Show thinking in an expander
                                if thinking_text:
                                    with st.expander("🧠 Thinking", expanded=False):
                                        st.markdown(thinking_text)

                                # Try to parse JSON from response
                                # Handle case where response might have markdown code blocks
                                if "```" in response_text:
                                    # Extract JSON from code block
                                    import re
                                    # Match ```json or ``` followed by content and closing ```
                                    code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response_text)
                                    if code_block_match:
                                        response_text = code_block_match.group(1).strip()

                                # Also handle case where JSON array is embedded in text
                                if not response_text.startswith('['):
                                    # Try to find JSON array in the response
                                    import re
                                    json_array_match = re.search(r'\[[\s\S]*\]', response_text)
                                    if json_array_match:
                                        response_text = json_array_match.group(0)

                                updated_rubric = json.loads(response_text)

                                # Validate it's a list
                                if isinstance(updated_rubric, list):
                                    # Update the branch rubric
                                    st.session_state.branch['rubric'] = updated_rubric
                                    # Increment counter to force widget refresh
                                    st.session_state.branch_rubric_edit_counter += 1
                                    st.success("✓ Rubric updated with suggestions!")
                                    st.rerun()
                                else:
                                    st.error("Invalid rubric format returned. Expected a list of criteria.")

                        except json.JSONDecodeError as e:
                            st.error(f"Failed to parse rubric update: {str(e)}")
                        except Exception as e:
                            st.error(f"Error applying suggestions: {str(e)}")

            st.markdown("---")

            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Reset", use_container_width=True, key="branch_reset_rubric"):
                    st.session_state.branch['rubric'] = None
                    st.session_state.regenerated_draft = None
                    st.session_state.branch_rubric_chat = []  # Also clear chat
                    st.session_state.branch_rubric_edit_counter += 1
                    st.rerun()

            with col2:
                if st.button("✨ Regenerate Draft", use_container_width=True, key="branch_regenerate_draft", type="primary"):
                    # First, check if there's a draft in the last message
                    current_draft, draft_msg_idx = get_last_draft_from_messages()

                    if current_draft is None:
                        st.warning("⚠️ No draft found in recent messages. The last assistant message must contain text wrapped in <draft></draft> tags.")
                    else:
                        # Get the original rubric (before branch modifications)
                        active_rubric_dict, _, _ = get_active_rubric()
                        original_rubric = active_rubric_dict.get("rubric", []) if active_rubric_dict else []

                        # Save the modified rubric first
                        st.session_state.branch['rubric'] = modified_rubric

                        # Regenerate the draft
                        result = regenerate_draft_from_rubric_changes(
                            original_rubric,
                            modified_rubric,
                            current_draft
                        )

                        if result:
                            # Store the result with additional context
                            st.session_state.regenerated_draft = {
                                'revised_draft': result.get('revised_draft', ''),
                                'changes_made': result.get('changes_made', []),
                                'rubric_changes_identified': result.get('rubric_changes_identified', []),
                                'revision_strategy': result.get('revision_strategy', ''),
                                'original_rubric': original_rubric,
                                'updated_rubric': modified_rubric,
                                'original_draft': current_draft,
                                'draft_msg_idx': draft_msg_idx
                            }
                            st.rerun()

            # Auto-save changes to branch rubric as user edits
            # This ensures the branch rubric is always up-to-date
            if st.session_state.branch['rubric'] is None:
                # Initialize branch rubric on first edit
                st.session_state.branch['rubric'] = copy.deepcopy(current_branch_rubric)
            else:
                # Update branch rubric with current edits
                st.session_state.branch['rubric'] = modified_rubric

            # Show indicator if regenerated draft is available (displayed in main chat area)
            if st.session_state.regenerated_draft:
                st.divider()
                st.success("✨ Regenerated draft is ready! View it in the chat area below.")

        else:
            st.warning("No rubric loaded to edit.")

# These are duplicate sections that were moved into tab1 - keeping the sections inside tab1 only.

# View Rubric Tab
with tab2:
    st.subheader("📋 View Rubric")
    st.markdown("View the active rubric version with all criteria and achievement levels.")

    # Get active rubric
    active_rubric_dict, active_idx, rubric_history = get_active_rubric()

    if not active_rubric_dict or not active_rubric_dict.get("rubric"):
        st.warning("No active rubric available. Please create or select a rubric first.")
    else:
        # Display version, writing type, and user goals at the top
        col_meta1, col_meta2 = st.columns([1, 2])

        with col_meta1:
            version = active_rubric_dict.get("version", 1)
            st.metric("Version", f"v{version}")

        with col_meta2:
            writing_type = active_rubric_dict.get("writing_type", "Not specified")
            st.markdown(f"**Writing Type:** {writing_type}")

        user_goals = active_rubric_dict.get("user_goals_summary", "")
        if user_goals:
            st.markdown("**User Goals:**")
            st.info(user_goals)

        st.markdown("---")

        # Prepare data for pie chart
        rubric_list = active_rubric_dict.get("rubric", [])

        # Create pie chart for weight distribution
        if rubric_list:
            import plotly.graph_objects as go

            criterion_names = [c.get('name', 'Unnamed') for c in rubric_list]
            weights = [c.get('weight', 0) for c in rubric_list]

            # Only show pie chart if weights are defined
            if any(w > 0 for w in weights):
                # Create custom data with criterion indices for click handling
                customdata = list(range(len(criterion_names)))

                fig = go.Figure(data=[go.Pie(
                    labels=criterion_names,
                    values=weights,
                    hole=0.3,  # Makes it a donut chart
                    textinfo='label+percent',
                    textposition='auto',
                    hovertemplate='<b>%{label}</b><br>Weight: %{value}%<extra></extra>',
                    customdata=customdata
                )])

                fig.update_layout(
                    title_text="Criteria Weight Distribution",
                    title_x=0.5,
                    showlegend=True,
                    height=400,
                    margin=dict(t=50, b=0, l=0, r=0)
                )

                # Display chart
                st.plotly_chart(fig, use_container_width=True, key="rubric_pie_chart")
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
                weight = criterion.get('weight', 0)
                description = criterion.get('description', 'No description provided')

                # Display criterion name, weight, and description
                st.markdown(f"**{criterion_name}**")
                st.markdown(f"*Weight: {weight}%*")
                st.markdown(f"{description}")

                # Expander for achievement levels
                with st.expander("📊 Achievement Levels", expanded=False):
                    exemplary = criterion.get('exemplary', 'Not specified')
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
with tab3:
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
        
        # Display selected rubrics
        st.markdown("---")
        col_rubric_a, col_rubric_b = st.columns(2)
        
        with col_rubric_a:
            rubric_a_version = rubric_history[rubric_a_idx].get('version', 1)
            st.markdown(f"### 📋 Rubric v{rubric_a_version}")
            # Pass rubric B as comparison to highlight differences in A
            display_rubric_criteria(rubric_history[rubric_a_idx], st, comparison_rubric_data=rubric_history[rubric_b_idx])

        with col_rubric_b:
            rubric_b_version = rubric_history[rubric_b_idx].get('version', 1)
            st.markdown(f"### 📋 Rubric v{rubric_b_version}")
            # Pass rubric A as comparison to highlight differences in B
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
                    result = compare_rubrics(compare_input, rubric_history[rubric_a_idx], rubric_history[rubric_b_idx])

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

        # Key differences and summary at the top
        col_diff, col_summary = st.columns(2)

        with col_diff:
            st.markdown("### Key Differences")
            st.markdown(f"""
            <div style="
                height: 200px;
                overflow-y: auto;
                border: 1px solid #ccc;
                padding: 10px;
                background-color: #f9f9f9;
                border-radius: 5px;
            ">
            {results["key_diffs"]}
            </div>
            """, unsafe_allow_html=True)

        with col_summary:
            st.markdown("### Summary")
            st.markdown(f"""
            <div style="
                height: 200px;
                overflow-y: auto;
                border: 1px solid #ccc;
                padding: 10px;
                background-color: #f9f9f9;
                border-radius: 5px;
            ">
            {results["summary"]}
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
            st.markdown(f"""
            <div style="
                height: 400px;
                overflow-y: auto;
                border: 1px solid #ccc;
                padding: 10px;
                background-color: #f9f9f9;
                border-radius: 5px;
            ">
            {results["base_txt"]}
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

# Evaluate: Coverage Tab
with tab4:
    st.header("📊 Evaluate: Coverage")
    st.markdown("Test whether your rubric covers your preferences.")

    # Check for active rubric
    coverage_rubric_dict, coverage_rubric_idx, _ = get_active_rubric()

    if coverage_rubric_dict:
        coverage_rubric_list = coverage_rubric_dict.get("rubric", [])
        coverage_rubric_version = coverage_rubric_dict.get("version", coverage_rubric_idx + 1)

        st.success(f"Using rubric: **Version {coverage_rubric_version}** ({len(coverage_rubric_list)} criteria)")

        # Reset button
        col_reset, col_spacer = st.columns([1, 3])
        with col_reset:
            if st.button("🔄 Reset Coverage Test", use_container_width=True, key="coverage_reset"):
                st.session_state.evaluate_selected_conversation = None
                st.session_state.evaluate_decision_points = None
                st.session_state.evaluate_selected_decision_point = None
                st.session_state.evaluate_expanded_dp = None
                st.session_state.evaluate_reflection_items = None
                st.session_state.evaluate_user_responses = {}
                st.session_state.evaluate_preference_dimensions = None
                st.session_state.evaluate_test_comparisons = None
                st.session_state.evaluate_user_tests = None
                st.session_state.evaluate_test_responses = {}
                st.session_state.evaluate_rubric_scores = None
                st.rerun()
    else:
        st.warning("No active rubric selected. Please select a rubric version in the sidebar to enable coverage testing.")

    # Load conversations for selector
    eval_conversations = load_conversations()

    # Create selection with conversation details
    eval_conversation_options = [("Select a conversation...", None)]
    if eval_conversations:
        for conv in eval_conversations:
            # Format timestamp for display
            try:
                dt = datetime.fromisoformat(conv["timestamp"])
                formatted_time = dt.strftime("%m/%d %H:%M")
                display = f"{formatted_time} ({conv['messages_count']} msgs)"
            except:
                display = f"{conv['timestamp']} ({conv['messages_count']} msgs)"
            eval_conversation_options.append((display, conv["filename"]))

    # Create selectbox
    eval_options = [opt[1] for opt in eval_conversation_options]

    # Find index of current selection
    eval_current_index = 0
    if st.session_state.evaluate_selected_conversation:
        try:
            eval_current_index = eval_options.index(st.session_state.evaluate_selected_conversation)
        except ValueError:
            eval_current_index = 0

    eval_selected_file = st.selectbox(
        "📂 Select conversation:",
        options=eval_options,
        format_func=lambda x: next(opt[0] for opt in eval_conversation_options if opt[1] == x) if x is not None else "Select a conversation...",
        index=eval_current_index,
        key="evaluate_conversation_selector"
    )

    # Handle conversation selection change
    if eval_selected_file != st.session_state.evaluate_selected_conversation:
        st.session_state.evaluate_selected_conversation = eval_selected_file
        st.session_state.evaluate_decision_points = None  # Clear previous results
        st.session_state.evaluate_selected_decision_point = None
        st.rerun()

    if eval_selected_file:
        # Load the selected conversation
        eval_conv_data = load_conversation_data(eval_selected_file)

        if eval_conv_data:
            eval_messages = eval_conv_data.get("messages", [])

            # Build message number mapping (only count user/assistant messages)
            msg_num_to_idx = {}  # Maps display message number to actual index in eval_messages
            msg_num = 1
            for idx, msg in enumerate(eval_messages):
                role = msg.get('role', 'unknown')
                if role in ['user', 'assistant']:
                    msg_num_to_idx[msg_num] = idx
                    msg_num += 1

            # Extract Decision Points section
            col_extract, col_clear = st.columns([3, 1])

            with col_extract:
                if st.button("🎯 Extract Decision Points & Generate Reflections", use_container_width=True, type="primary"):
                    # Build conversation text for the prompt with message numbers
                    conversation_text = ""
                    msg_num = 1
                    for msg in eval_messages:
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')
                        if role == 'user':
                            conversation_text += f"\n\n[Message #{msg_num}] USER:\n{content}"
                            msg_num += 1
                        elif role == 'assistant':
                            conversation_text += f"\n\n[Message #{msg_num}] ASSISTANT:\n{content}"
                            msg_num += 1

                    with st.spinner("Step 1/2: Analyzing conversation for decision points..."):
                        try:
                            # Get active rubric to pass to the prompt for prioritizing relevant decision points
                            active_rubric_for_extraction, _, _ = get_active_rubric()
                            rubric_json_for_extraction = json.dumps(active_rubric_for_extraction, indent=2) if active_rubric_for_extraction else None

                            # Build the prompt with rubric context
                            decision_prompt = extract_decision_pts(conversation_text, rubric_json_for_extraction)

                            # Call Claude API with thinking
                            response = client.messages.create(
                                model="claude-opus-4-5-20251101",
                                max_tokens=16000,
                                messages=[{"role": "user", "content": decision_prompt}],
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

                            # Parse JSON from response
                            parsed_data = None
                            try:
                                # Try to extract JSON from response
                                json_match = re.search(r'\{[\s\S]*\}', response_text)
                                if json_match:
                                    parsed_data = json.loads(json_match.group())
                            except json.JSONDecodeError:
                                parsed_data = None

                            # Store the result
                            st.session_state.evaluate_decision_points = {
                                "thinking": thinking_text,
                                "raw_response": response_text,
                                "parsed_data": parsed_data,
                                "conversation_file": eval_selected_file
                            }
                            st.session_state.evaluate_selected_decision_point = None

                        except Exception as e:
                            st.error(f"Error extracting decision points: {str(e)}")
                            st.stop()

                    # Now generate reflection questions
                    if parsed_data and "decision_points" in parsed_data:
                        with st.spinner("Step 2/2: Generating reflection questions..."):
                            try:
                                # Build conversation text for reflection
                                conv_text = ""
                                msg_idx = 1
                                for msg in eval_messages:
                                    role = msg.get('role', 'unknown')
                                    if role in ['user', 'assistant']:
                                        content = msg.get('display_content', msg.get('content', ''))
                                        conv_text += f"[Message #{msg_idx}] {role.upper()}:\n{content}\n\n"
                                        msg_idx += 1

                                # Get decision points JSON
                                dp_json = json.dumps(parsed_data["decision_points"], indent=2)

                                # Generate reflection questions
                                reflection_prompt = generate_reflection_questions_prompt(conv_text, dp_json)

                                response = client.messages.create(
                                    model="claude-opus-4-5-20251101",
                                    max_tokens=8000,
                                    messages=[{"role": "user", "content": reflection_prompt}],
                                    thinking={
                                        "type": "enabled",
                                        "budget_tokens": 6000
                                    }
                                )

                                # Extract response
                                reflection_thinking = ""
                                reflection_response = ""
                                for block in response.content:
                                    if block.type == "thinking":
                                        reflection_thinking = block.thinking
                                    elif block.type == "text":
                                        reflection_response = block.text

                                # Parse JSON
                                json_match = re.search(r'\{[\s\S]*\}', reflection_response)
                                if json_match:
                                    reflection_data = json.loads(json_match.group())
                                    reflection_data['thinking'] = reflection_thinking
                                    st.session_state.evaluate_reflection_items = reflection_data
                                else:
                                    st.warning("Could not parse reflection questions. Decision points extracted successfully.")

                            except Exception as e:
                                st.warning(f"Error generating reflections: {str(e)}. Decision points extracted successfully.")

                    st.rerun()

            with col_clear:
                if st.session_state.evaluate_decision_points:
                    if st.button("🗑️ Clear", use_container_width=True):
                        # Clear Step 1 data
                        st.session_state.evaluate_decision_points = None
                        st.session_state.evaluate_selected_decision_point = None
                        st.session_state.evaluate_reflection_items = None
                        st.session_state.evaluate_expanded_dp = None
                        # Clear user's reflection answers
                        st.session_state.evaluate_user_responses = {}
                        # Clear Step 2 data (preference dimensions, tests)
                        st.session_state.evaluate_preference_dimensions = None
                        st.session_state.evaluate_test_comparisons = None
                        st.session_state.evaluate_user_tests = None
                        st.session_state.evaluate_test_responses = {}
                        # Clear rubric scores
                        st.session_state.evaluate_rubric_scores = None
                        st.rerun()

            # Display decision points and conversation if available
            if st.session_state.evaluate_decision_points:
                result = st.session_state.evaluate_decision_points
                parsed_data = result.get("parsed_data")

                # Show thinking from decision point extraction
                if result.get("thinking"):
                    with st.expander("🧠 Decision Point Extraction Thinking", expanded=False):
                        st.markdown(result["thinking"])

                if parsed_data and "decision_points" in parsed_data:
                    decision_points = parsed_data["decision_points"]
                    reflection_items = []
                    if st.session_state.evaluate_reflection_items:
                        reflection_items = st.session_state.evaluate_reflection_items.get('reflection_items', [])

                    # Overall patterns summary
                    if parsed_data.get("overall_patterns"):
                        st.info(f"**Overall Patterns:** {parsed_data['overall_patterns']}")

                    st.divider()

                    # Build a mapping from decision point ID to message numbers (from original decision points)
                    dp_to_messages = {}
                    for dp in decision_points:
                        dp_id = dp.get('id')
                        messages = set()
                        if dp.get("assistant_message_num"):
                            messages.add(dp["assistant_message_num"])
                        if dp.get("user_message_num"):
                            messages.add(dp["user_message_num"])
                        dp_to_messages[dp_id] = messages

                    # Get highlighted message numbers based on expanded decision point
                    highlighted_messages = set()
                    if st.session_state.evaluate_expanded_dp is not None:
                        highlighted_messages = dp_to_messages.get(st.session_state.evaluate_expanded_dp, set())

                    # CSS for message highlighting
                    st.markdown("""
                    <style>
                    .highlighted-message {
                        background: linear-gradient(90deg, #fff3cd 0%, #fff9e6 100%);
                        border-left: 4px solid #ffc107;
                        padding: 12px 16px;
                        margin: 8px 0;
                        border-radius: 0 8px 8px 0;
                    }
                    .normal-message {
                        background: #f8f9fa;
                        padding: 12px 16px;
                        margin: 8px 0;
                        border-radius: 8px;
                        border-left: 4px solid #dee2e6;
                    }
                    .message-user {
                        border-left-color: #28a745;
                    }
                    .message-assistant {
                        border-left-color: #007bff;
                    }
                    .highlighted-message.message-user {
                        border-left-color: #28a745;
                        background: linear-gradient(90deg, #d4edda 0%, #e8f5e9 100%);
                    }
                    .highlighted-message.message-assistant {
                        border-left-color: #007bff;
                        background: linear-gradient(90deg, #cce5ff 0%, #e3f2fd 100%);
                    }
                    .message-number {
                        font-weight: bold;
                        color: #6c757d;
                        font-size: 0.85em;
                    }
                    .message-role {
                        font-weight: 600;
                        margin-bottom: 4px;
                    }
                    .message-content {
                        color: #333;
                        line-height: 1.6;
                    }
                    </style>
                    """, unsafe_allow_html=True)

                    # Display conversation with highlighting in scrollable container
                    st.markdown("### 📜 Conversation")
                    if st.session_state.evaluate_expanded_dp:
                        st.caption(f"*Highlighting messages for Decision Point #{st.session_state.evaluate_expanded_dp}*")

                    # Use Streamlit's native scrollable container
                    conversation_container = st.container(height=500)

                    with conversation_container:
                        msg_num = 1
                        for idx, msg in enumerate(eval_messages):
                            role = msg.get('role', 'unknown')
                            if role not in ['user', 'assistant']:
                                continue

                            content = msg.get('display_content', msg.get('content', ''))
                            is_highlighted = msg_num in highlighted_messages

                            # Determine CSS classes
                            css_class = "highlighted-message" if is_highlighted else "normal-message"
                            role_class = f"message-{role}"

                            # Role emoji and label
                            role_label = "👤 User" if role == "user" else "🤖 Assistant"

                            # Escape HTML in content
                            escaped_content = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")

                            # Render message with HTML styling
                            st.markdown(f"""
                            <div class="{css_class} {role_class}">
                                <div class="message-number">Message #{msg_num}</div>
                                <div class="message-role">{role_label}</div>
                                <div class="message-content">{escaped_content}</div>
                            </div>
                            """, unsafe_allow_html=True)

                            msg_num += 1

                    # Reflection Questions Section
                    st.markdown("---")
                    st.markdown("### 🔍 Reflection Questions")
                    st.markdown("Click on a decision point to expand it and see the corresponding messages highlighted above.")

                    if st.session_state.evaluate_reflection_items:
                        reflection_data = st.session_state.evaluate_reflection_items

                        # Show thinking from reflection generation
                        if reflection_data.get('thinking'):
                            with st.expander("🧠 Reflection Generation Thinking", expanded=False):
                                st.markdown(reflection_data['thinking'])

                        if reflection_items:
                            st.markdown(f"*Found {len(reflection_items)} decision points to reflect on:*")

                            for idx, item in enumerate(reflection_items):
                                dp_id = item.get('decision_point_id', idx + 1)

                                # Find corresponding decision point from original extraction for message numbers
                                original_dp = next((dp for dp in decision_points if dp.get('id') == dp_id), {})
                                assistant_msg_num = original_dp.get('assistant_message_num', '?')
                                user_msg_num = original_dp.get('user_message_num', '?')

                                # Create expander title with message references
                                expander_title = f"**Decision Point #{dp_id}**: {item.get('dimension', 'Unknown')} (Messages #{assistant_msg_num} → #{user_msg_num})"

                                # Check if this is the currently expanded one
                                is_expanded = (st.session_state.evaluate_expanded_dp == dp_id)

                                # Create a button to toggle expansion and update highlighting
                                col_expand, col_status = st.columns([5, 1])
                                with col_expand:
                                    if st.button(
                                        f"{'📍' if is_expanded else '📌'} Decision Point #{dp_id}: {item.get('dimension', 'Unknown')} (Messages #{assistant_msg_num} → #{user_msg_num})",
                                        key=f"expand_dp_{dp_id}",
                                        use_container_width=True,
                                        type="primary" if is_expanded else "secondary"
                                    ):
                                        if st.session_state.evaluate_expanded_dp == dp_id:
                                            st.session_state.evaluate_expanded_dp = None  # Collapse
                                        else:
                                            st.session_state.evaluate_expanded_dp = dp_id  # Expand
                                        st.rerun()

                                with col_status:
                                    if dp_id in st.session_state.evaluate_user_responses:
                                        st.markdown("✅")
                                    else:
                                        st.markdown("⬜")

                                # Show content if expanded
                                if is_expanded:
                                    with st.container():
                                        st.markdown("---")

                                        # Show before/after from original decision point
                                        col_before, col_after = st.columns(2)

                                        with col_before:
                                            st.markdown(f"**📤 Before (Message #{assistant_msg_num}):**")
                                            st.markdown(f"> {original_dp.get('before_quote', item.get('before_text', 'N/A'))}")

                                        with col_after:
                                            st.markdown(f"**📥 After (Message #{user_msg_num}):**")
                                            st.markdown(f"> {original_dp.get('after_quote', item.get('after_text', 'N/A'))}")

                                        # Show summary and user's stated reason (not collapsed)
                                        if original_dp.get('summary'):
                                            st.markdown(f"**📝 Summary:** {original_dp.get('summary')}")
                                        if original_dp.get('user_reason'):
                                            st.markdown(f"**💬 User's stated reason:** {original_dp.get('user_reason')}")

                                        # Collapsible Reflection Analysis section
                                        with st.expander("🔍 Reflection Analysis", expanded=False):
                                            col_orig_q, col_user_q = st.columns(2)
                                            with col_orig_q:
                                                st.markdown(f"**Original quality:** *{item.get('original_quality', 'N/A')}*")
                                            with col_user_q:
                                                st.markdown(f"**Your quality:** *{item.get('user_quality', 'N/A')}*")

                                            st.markdown(f"**💡 Potential preference:** *{item.get('potential_preference', 'N/A')}*")

                                            # Show rubric impact if available
                                            rubric_criterion = original_dp.get('related_rubric_criterion')
                                            rubric_impact = original_dp.get('rubric_impact', {})
                                            if rubric_criterion or rubric_impact:
                                                # st.markdown("---")
                                                # st.markdown("**📊 Rubric Impact:**")
                                                if rubric_criterion:
                                                    st.markdown(f"**📊 Rubric Impact:** *{rubric_criterion}*")
                                                if rubric_impact:
                                                    # impact_type = rubric_impact.get('type', 'none')
                                                    impact_desc = rubric_impact.get('description', '')
                                                    # Icon based on impact type
                                                    # impact_icons = {
                                                    #     'validates': '✅',
                                                    #     'refines': '🔧',
                                                    #     'contradicts': '⚠️',
                                                    #     'suggests_new': '➕',
                                                    #     'none': '⚪'
                                                    # }
                                                    # impact_icon = impact_icons.get(impact_type, '⚪')
                                                    # impact_labels = {
                                                    #     'validates': 'Validates',
                                                    #     'refines': 'Refines',
                                                    #     'contradicts': 'Contradicts',
                                                    #     'suggests_new': 'Suggests new criterion',
                                                    #     'none': 'No direct impact'
                                                    # }
                                                    # impact_label = impact_labels.get(impact_type, impact_type)
                                                    # st.markdown(f"*Impact:* {impact_icon} **{impact_label}**")
                                                    if impact_desc:
                                                        st.markdown(f"*{impact_desc}*")

                                        st.markdown("---")
                                        st.markdown("### 📝 Your Reflection")

                                        # Get existing responses or initialize
                                        user_response = st.session_state.evaluate_user_responses.get(dp_id, {})

                                        # Question 1: What made you want to change this?
                                        st.markdown("**1. What made you want to change this?**")
                                        q1_response = st.text_area(
                                            "What motivated this change?",
                                            value=user_response.get('motivation', ''),
                                            key=f"q1_{dp_id}",
                                            placeholder="Describe what prompted you to make this change...",
                                            label_visibility="collapsed"
                                        )

                                        # Question 2: What was wrong/better?
                                        st.markdown("**2. What was wrong with the original, or what was better about your version?**")
                                        q2_response = st.text_area(
                                            "What was the issue or improvement?",
                                            value=user_response.get('reasoning', ''),
                                            key=f"q2_{dp_id}",
                                            placeholder="Explain the specific issues with the original or benefits of your version...",
                                            label_visibility="collapsed"
                                        )

                                        # Question 3: General or specific?
                                        st.markdown("**3. Is this something you generally prefer in your writing, or was it specific to this situation?**")
                                        q3_options = ["This is a general preference", "Specific to this situation", "Depends on context"]
                                        q3_response = st.radio(
                                            "Generalization:",
                                            options=q3_options,
                                            index=q3_options.index(user_response.get('generalization', "Depends on context")),
                                            key=f"q3_{dp_id}",
                                            horizontal=True,
                                            label_visibility="collapsed"
                                        )

                                        # Save button for this decision point
                                        if st.button("💾 Save Reflection", key=f"save_ref_{dp_id}", type="primary"):
                                            st.session_state.evaluate_user_responses[dp_id] = {
                                                'decision_point_id': dp_id,
                                                'dimension': item.get('dimension', ''),
                                                'before_text': item.get('before_text', ''),
                                                'after_text': item.get('after_text', ''),
                                                'original_quality': item.get('original_quality', ''),
                                                'user_quality': item.get('user_quality', ''),
                                                'potential_preference': item.get('potential_preference', ''),
                                                'motivation': q1_response,
                                                'reasoning': q2_response,
                                                'generalization': q3_response
                                            }
                                            st.success("✅ Reflection saved!")
                                            st.rerun()

                                        # Show if already saved
                                        if dp_id in st.session_state.evaluate_user_responses:
                                            st.caption("✅ *Reflection saved*")

                                        st.markdown("---")
                    else:
                        st.warning("Reflection questions were not generated. Click 'Clear' and try again.")

                else:
                    # Fallback: show raw response if parsing failed
                    st.warning("Could not parse decision points as structured data. Showing raw response:")
                    st.markdown(result.get("raw_response", "No response available."))

            else:
                # No decision points extracted yet - show conversation preview
                st.divider()
                st.markdown("### 📜 Conversation Preview")

                msg_num = 1
                for idx, msg in enumerate(eval_messages):
                    role = msg.get('role', 'unknown')
                    if role not in ['user', 'assistant']:
                        continue

                    content = msg.get('display_content', msg.get('content', ''))[:500]
                    role_label = "👤 User" if role == "user" else "💻 Assistant"

                    st.markdown(f"**#{msg_num} {role_label}:** {content}{'...' if len(msg.get('content', '')) > 500 else ''}")

                    msg_num += 1

    else:
        st.info("👆 Select a conversation above to begin evaluation.")

    # Summary section at the bottom - show all collected responses
    if st.session_state.evaluate_user_responses:
        st.markdown("---")
        # For export calculations
        responses = list(st.session_state.evaluate_user_responses.values())
        total_responses = len(responses)
        general_count = sum(1 for r in responses if "general" in r.get('generalization', '').lower())
        specific_count = sum(1 for r in responses if "Specific" in r.get('generalization', ''))
        depends_count = sum(1 for r in responses if "Depends" in r.get('generalization', ''))

        for dp_id, response in st.session_state.evaluate_user_responses.items():
            with st.expander(f"**Decision Point {response.get('decision_point_id', dp_id)}**: {response.get('dimension', 'N/A')}", expanded=False):
                col_v1, col_v2 = st.columns(2)

                with col_v1:
                    st.markdown("**Original (Model's suggestion):**")
                    before_text = response.get('before_text', 'N/A')
                    st.caption(before_text[:200] + "..." if len(before_text) > 200 else before_text)
                    st.caption(f"*Quality: {response.get('original_quality', 'N/A')}*")

                with col_v2:
                    st.markdown("**Your Version:**")
                    after_text = response.get('after_text', 'N/A')
                    st.caption(after_text[:200] + "..." if len(after_text) > 200 else after_text)
                    st.caption(f"*Quality: {response.get('user_quality', 'N/A')}*")

                st.markdown("---")
                st.markdown(f"**Potential Preference:** *{response.get('potential_preference', 'N/A')}*")
                st.markdown(f"**Motivation:** {response.get('motivation', 'N/A') or '*Not provided*'}")
                st.markdown(f"**Reasoning:** {response.get('reasoning', 'N/A') or '*Not provided*'}")
                st.markdown(f"**Generalization:** {response.get('generalization', 'N/A')}")

        # Export button
        # st.markdown("---")
        col_export, col_clear = st.columns([3, 1])

        with col_export:
            # Create JSON for download
            export_data = {
                "conversation_file": st.session_state.evaluate_selected_conversation,
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_reflections": total_responses,
                    "general_preferences": general_count,
                    "situation_specific": specific_count,
                    "context_dependent": depends_count
                },
                "reflections": responses
            }

            st.download_button(
                label="📥 Export Reflections as JSON",
                data=json.dumps(export_data, indent=2, ensure_ascii=False),
                file_name=f"reflections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

        with col_clear:
            if st.button("🗑️ Clear All", key="clear_all_responses", type="secondary"):
                st.session_state.evaluate_user_responses = {}
                st.session_state.evaluate_reflection_items = None
                st.session_state.evaluate_preference_dimensions = None
                st.session_state.evaluate_test_comparisons = None
                st.session_state.evaluate_user_tests = None
                st.session_state.evaluate_test_responses = {}
                st.rerun()

        # Choose Preferences
        st.markdown("---")
        st.markdown("## 🧪 Choose Preferences")
        st.markdown("Compare different writing versions side-by-side and select the one you prefer. Your choices help us understand what matters most to you.")

        if not st.session_state.evaluate_user_tests:
            if st.button("✨ Generate Preference Tests", type="primary", use_container_width=True):
                # Step 1: Extract preference dimensions
                with st.spinner("Step 1/3: Extracting preference dimensions..."):
                    try:
                        # Prepare reflections JSON
                        reflections_json = json.dumps(list(st.session_state.evaluate_user_responses.values()), indent=2)

                        # Generate extraction prompt
                        extract_prompt = extract_preference_dimensions_prompt(reflections_json)

                        response = client.messages.create(
                            model="claude-opus-4-5-20251101",
                            max_tokens=8000,
                            messages=[{"role": "user", "content": extract_prompt}],
                            thinking={
                                "type": "enabled",
                                "budget_tokens": 6000
                            }
                        )

                        # Extract response
                        thinking_text = ""
                        response_text = ""
                        for block in response.content:
                            if block.type == "thinking":
                                thinking_text = block.thinking
                            elif block.type == "text":
                                response_text = block.text

                        # Parse JSON
                        json_match = re.search(r'\{[\s\S]*\}', response_text)
                        if json_match:
                            dimensions_data = json.loads(json_match.group())
                            dimensions_data['thinking'] = thinking_text
                            st.session_state.evaluate_preference_dimensions = dimensions_data
                        else:
                            st.error("Could not parse preference dimensions response.")
                            st.stop()

                    except Exception as e:
                        st.error(f"Error extracting preference dimensions: {str(e)}")
                        st.stop()

                # Step 2: Generate test comparisons
                with st.spinner("Step 2/3: Generating test writing samples..."):
                    try:
                        # Prepare preference dimensions JSON
                        dims_json = json.dumps(st.session_state.evaluate_preference_dimensions.get('preference_dimensions', []), indent=2)

                        # Generate test comparisons prompt
                        test_prompt = generate_test_comparisons_prompt(dims_json)

                        response = client.messages.create(
                            model="claude-opus-4-5-20251101",
                            max_tokens=10000,
                            messages=[{"role": "user", "content": test_prompt}],
                            thinking={
                                "type": "enabled",
                                "budget_tokens": 8000
                            }
                        )

                        # Extract response
                        thinking_text = ""
                        response_text = ""
                        for block in response.content:
                            if block.type == "thinking":
                                thinking_text = block.thinking
                            elif block.type == "text":
                                response_text = block.text

                        # Parse JSON
                        json_match = re.search(r'\{[\s\S]*\}', response_text)
                        if json_match:
                            test_data = json.loads(json_match.group())
                            test_data['thinking'] = thinking_text
                            st.session_state.evaluate_test_comparisons = test_data
                        else:
                            st.error("Could not parse test comparisons response.")
                            st.stop()

                    except Exception as e:
                        st.error(f"Error generating test comparisons: {str(e)}")
                        st.stop()

                # Step 3: Format for user presentation
                with st.spinner("Step 3/3: Preparing randomized tests..."):
                    try:
                        # Prepare test comparisons JSON
                        test_json = json.dumps(st.session_state.evaluate_test_comparisons.get('test_comparisons', []), indent=2)

                        # Format user tests prompt
                        format_prompt = format_user_tests_prompt(test_json)

                        response = client.messages.create(
                            model="claude-opus-4-5-20251101",
                            max_tokens=8000,
                            messages=[{"role": "user", "content": format_prompt}],
                            thinking={
                                "type": "enabled",
                                "budget_tokens": 4000
                            }
                        )

                        # Extract response
                        thinking_text = ""
                        response_text = ""
                        for block in response.content:
                            if block.type == "thinking":
                                thinking_text = block.thinking
                            elif block.type == "text":
                                response_text = block.text

                        # Parse JSON
                        json_match = re.search(r'\{[\s\S]*\}', response_text)
                        if json_match:
                            user_tests_data = json.loads(json_match.group())
                            user_tests_data['thinking'] = thinking_text
                            st.session_state.evaluate_user_tests = user_tests_data
                            st.rerun()
                        else:
                            st.error("Could not parse user tests response.")

                    except Exception as e:
                        st.error(f"Error formatting user tests: {str(e)}")
        else:
            user_tests_data = st.session_state.evaluate_user_tests
            user_tests = user_tests_data.get('user_tests', [])
            answer_key = user_tests_data.get('answer_key', [])
            dimensions_data = st.session_state.evaluate_preference_dimensions

            # Show thinking and decision point analyses collapsed
            if dimensions_data:
                if dimensions_data.get('thinking'):
                    with st.expander("🧠 Dimension Extraction Thinking", expanded=False):
                        st.markdown(dimensions_data['thinking'])

                if dimensions_data.get('decision_point_analyses'):
                    with st.expander("📋 Decision Point Analyses", expanded=False):
                        for analysis in dimensions_data['decision_point_analyses']:
                            st.markdown(f"**Decision Point {analysis.get('decision_point_id', '?')}**")
                            st.markdown(f"- Core preference: {analysis.get('core_preference', 'N/A')}")
                            st.markdown(f"- Generalized: {analysis.get('generalized_preference', 'N/A')}")
                            st.markdown(f"- Dimension: {analysis.get('dimension', 'N/A')}")
                            st.markdown("---")

            col_test_info, col_clear_tests = st.columns([4, 1])
            with col_test_info:
                st.markdown(f"*Complete {len(user_tests)} test(s) below:*")
            with col_clear_tests:
                if st.button("🗑️ Clear All", key="clear_tests"):
                    st.session_state.evaluate_preference_dimensions = None
                    st.session_state.evaluate_test_comparisons = None
                    st.session_state.evaluate_user_tests = None
                    st.session_state.evaluate_test_responses = {}
                    st.session_state.evaluate_rubric_scores = None
                    st.rerun()

            # Build mappings from test_id to dimension and preference statement from test_comparisons
            test_comparisons = st.session_state.evaluate_test_comparisons.get('test_comparisons', [])
            test_id_to_info = {
                tc.get('test_id'): {
                    'dimension': tc.get('dimension', 'Unknown'),
                    'preference_statement': tc.get('preference_statement', '')
                }
                for tc in test_comparisons
            }

            # Build mapping from dimension to confidence from preference_dimensions
            preference_dims = dimensions_data.get('preference_dimensions', []) if dimensions_data else []
            dim_to_confidence = {dim.get('dimension', ''): dim.get('confidence', 'low') for dim in preference_dims}

            for test in user_tests:
                test_id = test.get('test_id', 0)
                existing_response = st.session_state.evaluate_test_responses.get(test_id, {})
                test_info = test_id_to_info.get(test_id, {})
                dimension = test_info.get('dimension', 'Unknown')
                preference_statement = test_info.get('preference_statement', '')
                confidence = dim_to_confidence.get(dimension, 'low')
                confidence_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "⚪")

                # Show dimension with confidence in title
                with st.expander(f"**Test {test_id}** - {dimension} {confidence_icon}", expanded=test_id not in st.session_state.evaluate_test_responses):
                    # Show preference statement inside
                    if preference_statement:
                        st.info(f"📋 *{preference_statement}*")
                        # st.markdown("---")

                    col_v1, col_v2 = st.columns(2)

                    with col_v1:
                        st.markdown("**Version 1:**")
                        st.info(test.get('version_1', {}).get('text', 'N/A'))

                    with col_v2:
                        st.markdown("**Version 2:**")
                        st.info(test.get('version_2', {}).get('text', 'N/A'))

                    # Questions
                    st.markdown("---")

                    # Q1: Which version do you prefer?
                    preference = st.radio(
                        "Which version do you prefer?",
                        options=["Version 1", "Version 2", "No preference"],
                        index=["Version 1", "Version 2", "No preference"].index(existing_response.get('preference', "Version 1")),
                        key=f"test_pref_{test_id}",
                        horizontal=True
                    )

                    # Q2: What makes that version better?
                    reasoning = st.text_area(
                        "What makes that version better?",
                        value=existing_response.get('reasoning', ''),
                        key=f"test_reason_{test_id}",
                        placeholder="Explain your preference..."
                    )

                    # Save button
                    if st.button("💾 Save Response", key=f"save_test_{test_id}"):
                        st.session_state.evaluate_test_responses[test_id] = {
                            'test_id': test_id,
                            'context': test.get('context', ''),
                            'preference': preference,
                            'reasoning': reasoning
                        }
                        st.success("✅ Response saved!")
                        st.rerun()

                    if test_id in st.session_state.evaluate_test_responses:
                        st.caption("✅ *Response saved*")

            # Show rubric scoring if all tests completed (outside the for loop)
            if len(st.session_state.evaluate_test_responses) == len(user_tests) and len(user_tests) > 0:
                st.markdown("---")
                st.markdown("### 📊 Validate Rubric")
                st.markdown("Check how well your rubric captures your preferences. The rubric will predict which version you'd prefer for each test—accuracy measures correct predictions, and coverage shows how many preference dimensions your rubric addresses.")

                # Check if there's an active rubric
                active_rubric_dict, active_rubric_name, _ = get_active_rubric()

                if not active_rubric_dict:
                    st.warning("No active rubric selected. Please select a rubric version in the sidebar to enable rubric scoring.")
                elif not st.session_state.evaluate_rubric_scores:
                    if st.button("📏 Test with Rubric", type="primary", use_container_width=True):
                        with st.spinner("Testing versions with the rubric..."):
                            try:
                                # Prepare rubric JSON
                                rubric_json = json.dumps(active_rubric_dict, indent=2)

                                # Prepare test comparisons JSON
                                test_comparisons_data = st.session_state.evaluate_test_comparisons.get('test_comparisons', [])
                                test_json = json.dumps(test_comparisons_data, indent=2)

                                # Generate scoring prompt
                                scoring_prompt = score_tests_with_rubric_prompt(rubric_json, test_json)

                                response = client.messages.create(
                                    model="claude-opus-4-5-20251101",
                                    max_tokens=12000,
                                    messages=[{"role": "user", "content": scoring_prompt}],
                                    thinking={
                                        "type": "enabled",
                                        "budget_tokens": 8000
                                    }
                                )

                                # Extract response
                                thinking_text = ""
                                response_text = ""
                                for block in response.content:
                                    if block.type == "thinking":
                                        thinking_text = block.thinking
                                    elif block.type == "text":
                                        response_text = block.text

                                # Parse JSON
                                json_match = re.search(r'\{[\s\S]*\}', response_text)
                                if json_match:
                                    scores_data = json.loads(json_match.group())
                                    scores_data['thinking'] = thinking_text
                                    scores_data['rubric_name'] = f"Version {active_rubric_dict.get('version', active_rubric_name + 1)}"
                                    st.session_state.evaluate_rubric_scores = scores_data
                                    st.rerun()
                                else:
                                    st.error("Could not parse rubric test response.")

                            except Exception as e:
                                st.error(f"Error testing with rubric: {str(e)}")
                else:
                    scores_data = st.session_state.evaluate_rubric_scores

                    st.success(f"Tested using rubric: **{scores_data.get('rubric_name', 'Unknown')}**")

                    # Calculate quantitative metrics
                    test_scores = scores_data.get('test_scores', [])
                    user_choices = {resp.get('test_id'): resp.get('preference', '') for resp in st.session_state.evaluate_test_responses.values()}
                    aligned_versions = {key.get('test_id'): key.get('aligned_version', '') for key in answer_key}

                    correct_predictions = 0
                    total_with_predictions = 0
                    dimensions_with_matching = 0
                    total_dimensions = len(test_scores)

                    for score in test_scores:
                        test_id = score.get('test_id')
                        rubric_predicts = score.get('rubric_predicts')
                        has_matching = score.get('has_matching_criterion', False)

                        if has_matching:
                            dimensions_with_matching += 1

                        if has_matching and rubric_predicts is not None:
                            total_with_predictions += 1
                            user_chose = user_choices.get(test_id, 'Unknown')
                            aligned_version = aligned_versions.get(test_id, '')

                            if rubric_predicts == "Version A":
                                rubric_predicts_user_version = aligned_version
                            else:
                                rubric_predicts_user_version = "Version 2" if aligned_version == "Version 1" else "Version 1"

                            if rubric_predicts_user_version in user_chose:
                                correct_predictions += 1

                    # Display metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        accuracy_str = f"{correct_predictions}/{total_with_predictions}" if total_with_predictions > 0 else "N/A"
                        accuracy_pct = f" ({correct_predictions/total_with_predictions*100:.0f}%)" if total_with_predictions > 0 else ""
                        st.metric("Prediction Accuracy", f"{accuracy_str}{accuracy_pct}")
                    with col2:
                        coverage_str = f"{dimensions_with_matching}/{total_dimensions}" if total_dimensions > 0 else "N/A"
                        coverage_pct = f" ({dimensions_with_matching/total_dimensions*100:.0f}%)" if total_dimensions > 0 else ""
                        st.metric("Coverage", f"{coverage_str}{coverage_pct}")

                    # Show thinking
                    if scores_data.get('thinking'):
                        with st.expander("🧠 Testing Thinking", expanded=False):
                            st.markdown(scores_data['thinking'])

                    # Show detailed scores for each test
                    test_scores = scores_data.get('test_scores', [])

                    # Build mapping from test_id to user's actual choice
                    user_choices = {resp.get('test_id'): resp.get('preference', '') for resp in st.session_state.evaluate_test_responses.values()}

                    # Build mapping from answer_key to know which version is "aligned"
                    aligned_versions = {key.get('test_id'): key.get('aligned_version', '') for key in answer_key}

                    for score in test_scores:
                        test_id = score.get('test_id')
                        rubric_predicts = score.get('rubric_predicts')  # Can be None if no matching criterion
                        user_chose = user_choices.get(test_id, 'Unknown')
                        aligned_version = aligned_versions.get(test_id, '')
                        has_matching = score.get('has_matching_criterion', False)
                        matching_criterion = score.get('matching_criterion')

                        # Determine match status
                        if not has_matching or rubric_predicts is None:
                            match_icon = "⚪"  # No matching criterion
                            rubric_predicts_user_version = "N/A"
                            rubric_matches_user = None
                        else:
                            # Map rubric prediction to user-facing version number
                            if rubric_predicts == "Version A":
                                rubric_predicts_user_version = aligned_version
                            else:
                                rubric_predicts_user_version = "Version 2" if aligned_version == "Version 1" else "Version 1"

                            rubric_matches_user = rubric_predicts_user_version in user_chose
                            match_icon = "✅" if rubric_matches_user else "❌"

                        with st.expander(f"{match_icon} **Test {test_id}** - {score.get('dimension', 'Unknown')}", expanded=False):
                            st.markdown(f"*Context: {score.get('context', 'N/A')}*")

                            # Show preference statement being tested
                            if score.get('preference_statement'):
                                st.info(f"📋 *Preference: {score.get('preference_statement')}*")

                            st.markdown("---")

                            # Show matching criterion info
                            if has_matching and matching_criterion:
                                st.markdown(f"**Matching Rubric Criterion:** {matching_criterion}")
                                if score.get('matching_criterion_description'):
                                    st.caption(f"*{score.get('matching_criterion_description')}*")
                            else:
                                st.warning("No matching criterion found in rubric for this dimension")

                            st.markdown("---")

                            # Show prediction and user choice
                            col_pred, col_user = st.columns(2)
                            with col_pred:
                                st.markdown(f"**Rubric Predicts:** {rubric_predicts_user_version}")
                            with col_user:
                                st.markdown(f"**User Chose:** {user_chose}")

                            # Show match result
                            if rubric_matches_user is None:
                                st.info("⚪ No prediction (no matching criterion)")
                            elif rubric_matches_user:
                                st.success("✅ Rubric criterion correctly predicted user choice")
                            else:
                                st.error("❌ Rubric criterion did not predict user choice")

                            # Show reasoning
                            if score.get('reasoning'):
                                st.markdown(f"**Reasoning:** {score.get('reasoning')}")

                    # Clear rubric scores button
                    col_clear_scores, _ = st.columns([1, 4])
                    with col_clear_scores:
                        if st.button("🗑️ Clear Rubric Tests", key="clear_rubric_scores"):
                            st.session_state.evaluate_rubric_scores = None
                            st.rerun()

                # Export all results
                st.markdown("---")
                all_results = {
                    "conversation_file": st.session_state.evaluate_selected_conversation,
                    "timestamp": datetime.now().isoformat(),
                    "reflections": list(st.session_state.evaluate_user_responses.values()),
                    "preference_dimensions": st.session_state.evaluate_preference_dimensions,
                    "test_comparisons": st.session_state.evaluate_test_comparisons,
                    "user_test_responses": list(st.session_state.evaluate_test_responses.values()),
                    "rubric_scores": st.session_state.evaluate_rubric_scores
                }

                st.download_button(
                    label="📥 Export Complete Analysis",
                    data=json.dumps(all_results, indent=2, ensure_ascii=False),
                    file_name=f"preference_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )

with tab5:
    st.header("🎯 Evaluate: Alignment")
    st.markdown("Test whether you and the AI interpret rubric criteria the same way. ")

    # Check for active rubric
    align_rubric_dict, align_rubric_idx, _ = get_active_rubric()

    if not align_rubric_dict:
        st.warning("No active rubric selected. Please select a rubric version in the sidebar to enable alignment testing.")
    else:
        align_rubric_list = align_rubric_dict.get("rubric", [])
        align_rubric_version = align_rubric_dict.get("version", align_rubric_idx + 1)

        st.success(f"Using rubric: **Version {align_rubric_version}** ({len(align_rubric_list)} criteria)")

        # Reset button
        col_reset, col_spacer = st.columns([1, 3])
        with col_reset:
            if st.button("🔄 Reset Alignment Test", use_container_width=True, key="alignment_reset"):
                st.session_state.alignment_selected_conversation = None
                st.session_state.alignment_selected_draft_idx = None
                st.session_state.alignment_draft_content = None
                st.session_state.alignment_user_scores = {}
                st.session_state.alignment_llm_scores = None
                st.session_state.alignment_results = None
                st.session_state.alignment_evidence_highlights = []
                st.rerun()

        # Step 1: Select conversation and draft
        st.markdown("---")
        st.markdown("## 📂 Select Draft")
        st.markdown("Choose a conversation and select a draft to evaluate.")

        # Load conversations
        align_conversations = load_conversations()

        # Create conversation selector
        align_conv_options = [("Select a conversation...", None)]
        if align_conversations:
            for conv in align_conversations:
                try:
                    dt = datetime.fromisoformat(conv["timestamp"])
                    formatted_time = dt.strftime("%m/%d %H:%M")
                    display = f"{formatted_time} ({conv['messages_count']} msgs)"
                except:
                    display = f"{conv['timestamp']} ({conv['messages_count']} msgs)"
                align_conv_options.append((display, conv["filename"]))

        align_options = [opt[1] for opt in align_conv_options]

        # Find index of current selection
        align_current_index = 0
        if st.session_state.alignment_selected_conversation:
            try:
                align_current_index = align_options.index(st.session_state.alignment_selected_conversation)
            except ValueError:
                align_current_index = 0

        align_selected_file = st.selectbox(
            "📂 Select conversation:",
            options=align_options,
            format_func=lambda x: next(opt[0] for opt in align_conv_options if opt[1] == x) if x is not None else "Select a conversation...",
            index=align_current_index,
            key="alignment_conversation_selector"
        )

        # Handle conversation selection change
        if align_selected_file != st.session_state.alignment_selected_conversation:
            st.session_state.alignment_selected_conversation = align_selected_file
            st.session_state.alignment_selected_draft_idx = None
            st.session_state.alignment_draft_content = None
            st.session_state.alignment_user_scores = {}
            st.session_state.alignment_llm_scores = None
            st.session_state.alignment_results = None
            st.rerun()

        if align_selected_file:
            # Load the selected conversation
            align_conv_data = load_conversation_data(align_selected_file)

            if align_conv_data:
                align_messages = align_conv_data.get("messages", [])

                # Extract all drafts from the conversation
                drafts = []
                draft_pattern = r'<draft>(.*?)</draft>'
                for msg_idx, msg in enumerate(align_messages):
                    content = msg.get('content', '')
                    matches = re.findall(draft_pattern, content, re.DOTALL)
                    for match_idx, match in enumerate(matches):
                        drafts.append({
                            "msg_idx": msg_idx,
                            "draft_idx": len(drafts),
                            "content": match.strip(),
                            "preview": match.strip()[:100] + "..." if len(match.strip()) > 100 else match.strip()
                        })

                if not drafts:
                    st.warning("No drafts found in this conversation. Drafts must be wrapped in `<draft></draft>` tags.")
                else:
                    st.info(f"Found **{len(drafts)} draft(s)** in this conversation.")

                    # Draft selector
                    draft_options = ["Select a draft..."] + [f"Draft {d['draft_idx'] + 1}: {d['preview']}" for d in drafts]

                    current_draft_idx = 0
                    if st.session_state.alignment_selected_draft_idx is not None:
                        current_draft_idx = st.session_state.alignment_selected_draft_idx + 1

                    selected_draft_option = st.selectbox(
                        "📝 Select draft to evaluate:",
                        options=draft_options,
                        index=current_draft_idx,
                        key="alignment_draft_selector"
                    )

                    if selected_draft_option != "Select a draft...":
                        selected_draft_idx = draft_options.index(selected_draft_option) - 1
                        selected_draft = drafts[selected_draft_idx]

                        if st.session_state.alignment_selected_draft_idx != selected_draft_idx:
                            st.session_state.alignment_selected_draft_idx = selected_draft_idx
                            st.session_state.alignment_draft_content = selected_draft["content"]
                            st.session_state.alignment_user_scores = {}
                            st.session_state.alignment_llm_scores = None
                            st.session_state.alignment_results = None
                            st.rerun()

                        # Show the selected draft
                        st.markdown("### Selected Draft:")
                        with st.expander("View full draft", expanded=False):
                            st.markdown(st.session_state.alignment_draft_content)

                        # Step 2: User scores the draft
                        st.markdown("---")
                        st.markdown("## 📊 Score Draft")
                        st.markdown("Score this draft on each rubric criterion. Use the achievement level descriptions to guide your judgment.")

                        # Track completion
                        total_criteria = len(align_rubric_list)
                        scored_criteria = len(st.session_state.alignment_user_scores)

                        st.progress(scored_criteria / total_criteria if total_criteria > 0 else 0,
                                   text=f"Scored {scored_criteria} of {total_criteria} criteria")

                        # Display each criterion for scoring
                        for crit_idx, criterion in enumerate(align_rubric_list):
                            crit_name = criterion.get("name", f"Criterion {crit_idx + 1}")
                            crit_desc = criterion.get("description", "")

                            # Check if scored
                            is_scored = crit_idx in st.session_state.alignment_user_scores
                            crit_status = "✅" if is_scored else ""

                            with st.expander(f"{crit_status} **{crit_name}**", expanded=(crit_idx == 0 and not is_scored)):
                                st.markdown(f"*{crit_desc}*")

                                # Show achievement levels
                                st.markdown("**Achievement Levels:**")
                                level_cols = st.columns(4)
                                levels = [
                                    ("Exemplary (4)", criterion.get("exemplary", "")),
                                    ("Proficient (3)", criterion.get("proficient", "")),
                                    ("Developing (2)", criterion.get("developing", "")),
                                    ("Beginning (1)", criterion.get("beginning", ""))
                                ]
                                for col, (level_name, level_desc) in zip(level_cols, levels):
                                    with col:
                                        st.markdown(f"**{level_name}**")
                                        with st.container(height=150):
                                            st.caption(level_desc)

                                st.markdown("---")

                                # Get current score if exists
                                current_score = st.session_state.alignment_user_scores.get(crit_idx, None)

                                # Score selection
                                score_options = ["Select score...", "Beginning (1)", "Developing (2)", "Proficient (3)", "Exemplary (4)"]
                                default_idx = 0
                                if current_score is not None:
                                    default_idx = current_score  # 1-4 maps to indices 1-4

                                selected = st.selectbox(
                                    f"Score for {crit_name}",
                                    options=score_options,
                                    index=default_idx,
                                    key=f"align_crit_score_{crit_idx}",
                                    label_visibility="collapsed"
                                )

                                if selected != "Select score...":
                                    score_val = score_options.index(selected)  # 1-4
                                    if st.session_state.alignment_user_scores.get(crit_idx) != score_val:
                                        st.session_state.alignment_user_scores[crit_idx] = score_val
                                        st.rerun()

                        # Step 3: LLM scoring and comparison
                        if scored_criteria == total_criteria and total_criteria > 0:
                            st.markdown("---")
                            st.markdown("## 💻 Compare with AI")
                            st.markdown("Now the AI will score the same draft using the same rubric. We'll compare scores to measure alignment.")

                            if not st.session_state.alignment_llm_scores:
                                if st.button("🎯 Get AI Scores & Compare", type="primary", use_container_width=True):
                                    with st.spinner("AI is scoring the draft..."):
                                        try:
                                            from prompts import ALIGNMENT_SCORING_PROMPT

                                            draft_text = st.session_state.alignment_draft_content
                                            rubric_json = json.dumps(align_rubric_list, indent=2)

                                            # Build the assessment prompt with rubric and draft
                                            assessment_prompt = f"""{ALIGNMENT_SCORING_PROMPT}

## Rubric Criteria

{rubric_json}

## Draft to Assess

{draft_text}
"""

                                            # Use temperature=0 for consistent scoring
                                            response = client.messages.create(
                                                model="claude-opus-4-5-20251101",
                                                max_tokens=16000,
                                                temperature=0,
                                                messages=[{"role": "user", "content": assessment_prompt}]
                                            )

                                            response_text = response.content[0].text

                                            # Parse JSON from response
                                            json_match = re.search(r'\{[\s\S]*\}', response_text)
                                            if json_match:
                                                scores_data = json.loads(json_match.group())

                                                # Map level_percentage to 1-4 scale
                                                level_pct_to_score = {25: 1, 50: 2, 75: 3, 100: 4}
                                                level_name_to_score = {"beginning": 1, "developing": 2, "proficient": 3, "exemplary": 4}

                                                # Map scores by criterion name to index
                                                llm_scores = {}
                                                for score_item in scores_data.get("criteria_scores", []):
                                                    crit_name = score_item.get("name")
                                                    # Find matching criterion index
                                                    for idx, crit in enumerate(align_rubric_list):
                                                        if crit.get("name") == crit_name:
                                                            # Convert level_percentage or achievement_level to 1-4 score
                                                            level_pct = score_item.get("level_percentage")
                                                            level_name = score_item.get("achievement_level", "").lower()

                                                            if level_pct in level_pct_to_score:
                                                                score = level_pct_to_score[level_pct]
                                                            elif level_name in level_name_to_score:
                                                                score = level_name_to_score[level_name]
                                                            else:
                                                                score = 2  # Default to developing if unclear

                                                            llm_scores[idx] = {
                                                                "score": score,
                                                                "level": score_item.get("achievement_level", "Unknown"),
                                                                "rationale": score_item.get("rationale", ""),
                                                                "evidence": score_item.get("evidence", []),
                                                                "evidence_summary": score_item.get("evidence_summary", "")
                                                            }
                                                            break

                                                # Store evidence highlights for display
                                                st.session_state.alignment_evidence_highlights = scores_data.get("evidence_highlights", [])
                                                st.session_state.alignment_llm_scores = llm_scores

                                                # Calculate alignment metrics
                                                from scipy import stats

                                                all_user_scores = []
                                                all_llm_scores = []
                                                exact_matches = 0
                                                total_comparisons = 0
                                                score_differences = []

                                                for crit_idx, user_score in st.session_state.alignment_user_scores.items():
                                                    llm_data = llm_scores.get(crit_idx, {})
                                                    llm_score = llm_data.get("score")

                                                    if llm_score is not None:
                                                        all_user_scores.append(user_score)
                                                        all_llm_scores.append(llm_score)
                                                        total_comparisons += 1

                                                        if user_score == llm_score:
                                                            exact_matches += 1

                                                        score_differences.append(llm_score - user_score)

                                                # Calculate metrics
                                                results = {
                                                    "total_comparisons": total_comparisons,
                                                    "exact_agreement": exact_matches / total_comparisons if total_comparisons > 0 else 0,
                                                    "exact_matches": exact_matches,
                                                    "mean_difference": sum(score_differences) / len(score_differences) if score_differences else 0,
                                                    "mean_absolute_difference": sum(abs(d) for d in score_differences) / len(score_differences) if score_differences else 0,
                                                }

                                                # Spearman correlation
                                                if len(all_user_scores) >= 3:
                                                    try:
                                                        spearman_result = stats.spearmanr(all_user_scores, all_llm_scores)
                                                        results["spearman_rho"] = spearman_result.correlation
                                                        results["spearman_pvalue"] = spearman_result.pvalue
                                                    except:
                                                        results["spearman_rho"] = None
                                                        results["spearman_pvalue"] = None
                                                else:
                                                    results["spearman_rho"] = None
                                                    results["spearman_pvalue"] = None

                                                st.session_state.alignment_results = results
                                                st.rerun()
                                            else:
                                                st.error("Could not parse AI response.")

                                        except Exception as e:
                                            st.error(f"Error getting AI scores: {str(e)}")
                            else:
                                # Display results
                                results = st.session_state.alignment_results
                                llm_scores = st.session_state.alignment_llm_scores

                                st.success("Alignment analysis complete!")

                                # Display metrics
                                st.markdown("### 📈 Alignment Metrics")

                                col1, col2, col3, col4 = st.columns(4)

                                with col1:
                                    exact_pct = results.get("exact_agreement", 0) * 100
                                    st.metric(
                                        "Exact Agreement",
                                        f"{results.get('exact_matches', 0)}/{results.get('total_comparisons', 0)}",
                                        f"{exact_pct:.0f}%"
                                    )

                                with col2:
                                    spearman = results.get("spearman_rho")
                                    if spearman is not None:
                                        st.metric("Rank Correlation (ρ)", f"{spearman:.2f}")
                                    else:
                                        st.metric("Rank Correlation (ρ)", "N/A")

                                with col3:
                                    mad = results.get("mean_absolute_difference", 0)
                                    st.metric("Avg Score Distance", f"{mad:.2f}")

                                with col4:
                                    bias = results.get("mean_difference", 0)
                                    bias_label = "AI scores higher" if bias > 0 else "AI scores lower" if bias < 0 else "No bias"
                                    st.metric("Systematic Bias", f"{bias:+.2f}", bias_label)

                                # Interpretation
                                st.markdown("### 📖 Interpretation")

                                exact_pct = results.get("exact_agreement", 0) * 100
                                if exact_pct >= 70:
                                    st.success(f"**Strong alignment** ({exact_pct:.0f}% exact agreement): You and the AI interpret the rubric criteria similarly.")
                                elif exact_pct >= 50:
                                    st.warning(f"**Moderate alignment** ({exact_pct:.0f}% exact agreement): Some differences in interpretation. Consider clarifying rubric language.")
                                else:
                                    st.error(f"**Low alignment** ({exact_pct:.0f}% exact agreement): Significant interpretation differences. The rubric language may be ambiguous.")

                                # Detailed comparison by criterion
                                st.markdown("### 🔍 Detailed Comparison")

                                level_names = {1: "Beginning", 2: "Developing", 3: "Proficient", 4: "Exemplary"}

                                for crit_idx, criterion in enumerate(align_rubric_list):
                                    crit_name = criterion.get("name", f"Criterion {crit_idx + 1}")

                                    user_score = st.session_state.alignment_user_scores.get(crit_idx)
                                    llm_data = llm_scores.get(crit_idx, {})
                                    llm_score = llm_data.get("score")
                                    llm_rationale = llm_data.get("rationale", "")
                                    llm_evidence = llm_data.get("evidence", [])

                                    match_icon = "✅" if user_score == llm_score else "❌"

                                    with st.expander(f"{match_icon} **{crit_name}**"):
                                        col_user, col_llm = st.columns(2)
                                        with col_user:
                                            st.markdown(f"**Your score:** {level_names.get(user_score, 'N/A')} ({user_score})")
                                        with col_llm:
                                            st.markdown(f"**AI score:** {level_names.get(llm_score, 'N/A')} ({llm_score})")

                                        if user_score != llm_score:
                                            diff = abs(user_score - llm_score) if user_score and llm_score else 0
                                            st.caption(f"*Difference: {diff} level(s)*")

                                        if llm_rationale:
                                            st.markdown("**AI rationale:**")
                                            st.caption(f"*{llm_rationale}*")

                                # Display highlighted draft with evidence
                                st.markdown("### 📄 Draft with Evidence Highlights")
                                st.markdown("Hover over highlighted text to see the criterion and relevance explanation.")

                                draft_text = st.session_state.alignment_draft_content

                                # Build evidence map from all criteria scores
                                # Find actual positions by searching for quotes in the draft
                                all_evidence = []
                                for crit_idx, criterion in enumerate(align_rubric_list):
                                    crit_name = criterion.get("name", f"Criterion {crit_idx + 1}")
                                    llm_data = llm_scores.get(crit_idx, {})
                                    for ev in llm_data.get("evidence", []):
                                        quote = ev.get("quote", "").strip()
                                        if quote:
                                            # Find the actual position by searching for the quote
                                            # Try exact match first
                                            start_idx = draft_text.find(quote)
                                            if start_idx == -1:
                                                # Try with normalized whitespace
                                                normalized_quote = " ".join(quote.split())
                                                normalized_draft = " ".join(draft_text.split())
                                                norm_start = normalized_draft.find(normalized_quote)
                                                if norm_start != -1:
                                                    # Map back to original position (approximate)
                                                    start_idx = draft_text.find(quote[:30]) if len(quote) > 30 else -1

                                            if start_idx != -1:
                                                all_evidence.append({
                                                    "criterion": crit_name,
                                                    "quote": quote,
                                                    "start_index": start_idx,
                                                    "end_index": start_idx + len(quote),
                                                    "relevance": ev.get("relevance", "")
                                                })

                                if all_evidence and draft_text:
                                    # Sort by start_index
                                    sorted_evidence = sorted(all_evidence, key=lambda x: x.get("start_index", 0))

                                    # Remove overlapping highlights (keep first one)
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
                                    for i, crit in enumerate(align_rubric_list):
                                        criterion_colors[crit.get("name", f"Criterion {i+1}")] = colors[i % len(colors)]

                                    # Legend showing criterion colors (above draft)
                                    st.markdown("**Legend:**")
                                    legend_html = "<div style='display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px;'>"
                                    for crit in align_rubric_list:
                                        crit_name = crit.get("name", "")
                                        color = criterion_colors.get(crit_name, "#ffeb3b")
                                        legend_html += f'<span style="background-color: {color}; padding: 2px 8px; border-radius: 3px; font-size: 0.85em;">{crit_name}</span>'
                                    legend_html += "</div>"
                                    st.markdown(legend_html, unsafe_allow_html=True)

                                    # Build highlighted HTML with CSS for tooltips
                                    tooltip_css = """
                                    <style>
                                    .evidence-highlight {
                                        position: relative;
                                        cursor: pointer;
                                        padding: 2px 4px;
                                        border-radius: 3px;
                                        display: inline;
                                    }
                                    .evidence-highlight .tooltip-text {
                                        visibility: hidden;
                                        width: 320px;
                                        background-color: #333;
                                        color: #fff;
                                        text-align: left;
                                        border-radius: 6px;
                                        padding: 10px;
                                        position: absolute;
                                        z-index: 9999;
                                        bottom: calc(100% + 5px);
                                        left: 50%;
                                        transform: translateX(-50%);
                                        font-size: 0.85em;
                                        line-height: 1.4;
                                        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
                                        white-space: normal;
                                    }
                                    .evidence-highlight .tooltip-text::after {
                                        content: "";
                                        position: absolute;
                                        top: 100%;
                                        left: 50%;
                                        margin-left: -5px;
                                        border-width: 5px;
                                        border-style: solid;
                                        border-color: #333 transparent transparent transparent;
                                    }
                                    .evidence-highlight:hover .tooltip-text {
                                        visibility: visible;
                                    }
                                    .tooltip-criterion {
                                        font-weight: bold;
                                        color: #ffc107;
                                        margin-bottom: 5px;
                                    }
                                    </style>
                                    """

                                    highlighted_html = tooltip_css
                                    highlighted_html += '<div style="line-height: 1.8; padding: 10px;">'

                                    last_end = 0
                                    for ev in sorted_evidence:
                                        start = ev.get("start_index", 0)
                                        end = ev.get("end_index", 0)
                                        criterion = ev.get("criterion", "")
                                        relevance = ev.get("relevance", "")
                                        color = criterion_colors.get(criterion, "#ffeb3b")

                                        if start >= last_end and end > start and start < len(draft_text):
                                            # Add unhighlighted text before this highlight
                                            highlighted_html += draft_text[last_end:start].replace("\n", "<br>")

                                            # Add highlighted text with tooltip
                                            quote_text = draft_text[start:end].replace("\n", "<br>")
                                            tooltip_content = f'<div class="tooltip-criterion">{criterion}</div><div>{relevance}</div>'

                                            highlighted_html += f'<span class="evidence-highlight" style="background-color: {color};">{quote_text}<span class="tooltip-text">{tooltip_content}</span></span>'

                                            last_end = end

                                    # Add remaining text
                                    if last_end < len(draft_text):
                                        highlighted_html += draft_text[last_end:].replace("\n", "<br>")

                                    highlighted_html += '</div>'

                                    st.markdown(highlighted_html, unsafe_allow_html=True)
                                else:
                                    st.markdown(draft_text)

with tab6:
    st.header("⚡ Evaluate: Utility")
    st.markdown("Measure how useful your rubric is for improving writing quality through surveys before and after using the rubric.")

    # Reset button
    col_reset, col_spacer = st.columns([1, 3])
    with col_reset:
        if st.button("🔄 Reset Utility Test", use_container_width=True, key="utility_reset"):
            # Clear survey responses
            for key in list(st.session_state.keys()):
                if key.startswith("survey_"):
                    del st.session_state[key]
            st.rerun()

    # Helper function to render cognitive load survey
    def render_cognitive_load_survey(prefix):
        st.subheader("Cognitive Load Survey")
        st.markdown("For each of the six scales, evaluate the task you recently performed by moving the slider bars. Each line has two endpoint descriptors that describe the scale.")

        # 1. Mental Demand
        st.markdown("**1. Mental Demand**")
        st.markdown("*How much mental and perceptual activity was required (e.g., thinking, deciding, calculating, remembering, reading, concentrating, etc.)?*")
        col_left, col_slider, col_right = st.columns([1, 4, 1])
        with col_left:
            st.markdown("<p style='text-align: right; color: gray; font-size: 0.85em; margin-top: 5px;'>Low</p>", unsafe_allow_html=True)
        with col_slider:
            mental_demand = st.slider("Mental Demand", min_value=0, max_value=20, value=10, key=f"{prefix}_mental_demand", label_visibility="collapsed")
        with col_right:
            st.markdown("<p style='text-align: left; color: gray; font-size: 0.85em; margin-top: 5px;'>High</p>", unsafe_allow_html=True)
        st.markdown("")

        # 2. Physical Demand
        st.markdown("**2. Physical Demand**")
        st.markdown("*How much physical activity was required (e.g., typing, staring at the monitor, etc.)?*")
        col_left, col_slider, col_right = st.columns([1, 4, 1])
        with col_left:
            st.markdown("<p style='text-align: right; color: gray; font-size: 0.85em; margin-top: 5px;'>Low</p>", unsafe_allow_html=True)
        with col_slider:
            physical_demand = st.slider("Physical Demand", min_value=0, max_value=20, value=10, key=f"{prefix}_physical_demand", label_visibility="collapsed")
        with col_right:
            st.markdown("<p style='text-align: left; color: gray; font-size: 0.85em; margin-top: 5px;'>High</p>", unsafe_allow_html=True)
        st.markdown("")

        # 3. Temporal Demand
        st.markdown("**3. Temporal Demand**")
        st.markdown("*How much time pressure did you feel due to the rate or pace at which the task elements occurred?*")
        col_left, col_slider, col_right = st.columns([1, 4, 1])
        with col_left:
            st.markdown("<p style='text-align: right; color: gray; font-size: 0.85em; margin-top: 5px;'>Low</p>", unsafe_allow_html=True)
        with col_slider:
            temporal_demand = st.slider("Temporal Demand", min_value=0, max_value=20, value=10, key=f"{prefix}_temporal_demand", label_visibility="collapsed")
        with col_right:
            st.markdown("<p style='text-align: left; color: gray; font-size: 0.85em; margin-top: 5px;'>High</p>", unsafe_allow_html=True)
        st.markdown("")

        # 4. Effort
        st.markdown("**4. Effort**")
        st.markdown("*How hard did you have to work (mentally and physically) to accomplish your level of performance?*")
        col_left, col_slider, col_right = st.columns([1, 4, 1])
        with col_left:
            st.markdown("<p style='text-align: right; color: gray; font-size: 0.85em; margin-top: 5px;'>Low</p>", unsafe_allow_html=True)
        with col_slider:
            effort = st.slider("Effort", min_value=0, max_value=20, value=10, key=f"{prefix}_effort", label_visibility="collapsed")
        with col_right:
            st.markdown("<p style='text-align: left; color: gray; font-size: 0.85em; margin-top: 5px;'>High</p>", unsafe_allow_html=True)
        st.markdown("")

        # 5. Performance
        st.markdown("**5. Performance**")
        st.markdown("*How successful do you think you were in accomplishing the goals of the task set by the experimenter (or yourself)?*")
        col_left, col_slider, col_right = st.columns([1, 4, 1])
        with col_left:
            st.markdown("<p style='text-align: right; color: gray; font-size: 0.85em; margin-top: 5px;'>Good</p>", unsafe_allow_html=True)
        with col_slider:
            performance = st.slider("Performance", min_value=0, max_value=20, value=10, key=f"{prefix}_performance", label_visibility="collapsed")
        with col_right:
            st.markdown("<p style='text-align: left; color: gray; font-size: 0.85em; margin-top: 5px;'>Poor</p>", unsafe_allow_html=True)
        st.markdown("")

        # 6. Frustration
        st.markdown("**6. Frustration**")
        st.markdown("*How insecure, discouraged, irritated, stressed, and annoyed were you during the task?*")
        col_left, col_slider, col_right = st.columns([1, 4, 1])
        with col_left:
            st.markdown("<p style='text-align: right; color: gray; font-size: 0.85em; margin-top: 5px;'>Low</p>", unsafe_allow_html=True)
        with col_slider:
            frustration = st.slider("Frustration", min_value=0, max_value=20, value=10, key=f"{prefix}_frustration", label_visibility="collapsed")
        with col_right:
            st.markdown("<p style='text-align: left; color: gray; font-size: 0.85em; margin-top: 5px;'>High</p>", unsafe_allow_html=True)

        return mental_demand, physical_demand, temporal_demand, effort, performance, frustration

    # Helper function to render writing quality survey
    def render_writing_quality_survey(prefix):
        st.subheader("Writing Quality Survey")
        st.markdown("For each of the six scales, evaluate your writing experience by moving the slider bars.")

        # 1. Content Satisfaction
        st.markdown("**1. Content Satisfaction**")
        st.markdown("*How satisfied are you with the content of your writing, i.e., the final response to the prompt question?*")
        col_left, col_slider, col_right = st.columns([1, 4, 1])
        with col_left:
            st.markdown("<p style='text-align: right; color: gray; font-size: 0.85em; margin-top: 5px;'>Not satisfied</p>", unsafe_allow_html=True)
        with col_slider:
            content_satisfaction = st.slider("Content Satisfaction", min_value=0, max_value=20, value=10, key=f"{prefix}_content_satisfaction", label_visibility="collapsed")
        with col_right:
            st.markdown("<p style='text-align: left; color: gray; font-size: 0.85em; margin-top: 5px;'>Very satisfied</p>", unsafe_allow_html=True)
        st.markdown("")

        # 2. Ownership
        st.markdown("**2. Ownership**")
        st.markdown("*How much ownership do you feel over the content of your writing?*")
        col_left, col_slider, col_right = st.columns([1, 4, 1])
        with col_left:
            st.markdown("<p style='text-align: right; color: gray; font-size: 0.85em; margin-top: 5px;'>No ownership</p>", unsafe_allow_html=True)
        with col_slider:
            ownership = st.slider("Ownership", min_value=0, max_value=20, value=10, key=f"{prefix}_ownership", label_visibility="collapsed")
        with col_right:
            st.markdown("<p style='text-align: left; color: gray; font-size: 0.85em; margin-top: 5px;'>Full ownership</p>", unsafe_allow_html=True)
        st.markdown("")

        # 3. Responsibility
        st.markdown("**3. Responsibility**")
        st.markdown("*How responsible do you feel for the text you wrote?*")
        col_left, col_slider, col_right = st.columns([1, 4, 1])
        with col_left:
            st.markdown("<p style='text-align: right; color: gray; font-size: 0.85em; margin-top: 5px;'>Not responsible</p>", unsafe_allow_html=True)
        with col_slider:
            responsibility = st.slider("Responsibility", min_value=0, max_value=20, value=10, key=f"{prefix}_responsibility", label_visibility="collapsed")
        with col_right:
            st.markdown("<p style='text-align: left; color: gray; font-size: 0.85em; margin-top: 5px;'>Fully responsible</p>", unsafe_allow_html=True)
        st.markdown("")

        # 4. Personal Connection
        st.markdown("**4. Personal Connection**")
        st.markdown("*How much personal connection do you feel to the content of your writing?*")
        col_left, col_slider, col_right = st.columns([1, 4, 1])
        with col_left:
            st.markdown("<p style='text-align: right; color: gray; font-size: 0.85em; margin-top: 5px;'>No connection</p>", unsafe_allow_html=True)
        with col_slider:
            personal_connection = st.slider("Personal Connection", min_value=0, max_value=20, value=10, key=f"{prefix}_personal_connection", label_visibility="collapsed")
        with col_right:
            st.markdown("<p style='text-align: left; color: gray; font-size: 0.85em; margin-top: 5px;'>Full connection</p>", unsafe_allow_html=True)
        st.markdown("")

        # 5. Emotional Connection
        st.markdown("**5. Emotional Connection**")
        st.markdown("*How much emotional connection do you feel to the content of your writing?*")
        col_left, col_slider, col_right = st.columns([1, 4, 1])
        with col_left:
            st.markdown("<p style='text-align: right; color: gray; font-size: 0.85em; margin-top: 5px;'>No connection</p>", unsafe_allow_html=True)
        with col_slider:
            emotional_connection = st.slider("Emotional Connection", min_value=0, max_value=20, value=10, key=f"{prefix}_emotional_connection", label_visibility="collapsed")
        with col_right:
            st.markdown("<p style='text-align: left; color: gray; font-size: 0.85em; margin-top: 5px;'>Full connection</p>", unsafe_allow_html=True)
        st.markdown("")

        # 6. Prompt Difficulty
        st.markdown("**6. Prompt Difficulty**")
        st.markdown("*How difficult was the given prompt to write about?*")
        col_left, col_slider, col_right = st.columns([1, 4, 1])
        with col_left:
            st.markdown("<p style='text-align: right; color: gray; font-size: 0.85em; margin-top: 5px;'>Easy</p>", unsafe_allow_html=True)
        with col_slider:
            prompt_difficulty = st.slider("Prompt Difficulty", min_value=0, max_value=20, value=10, key=f"{prefix}_prompt_difficulty", label_visibility="collapsed")
        with col_right:
            st.markdown("<p style='text-align: left; color: gray; font-size: 0.85em; margin-top: 5px;'>Difficult</p>", unsafe_allow_html=True)

        return content_satisfaction, ownership, responsibility, personal_connection, emotional_connection, prompt_difficulty

    # Create top-level tabs: Without Rubric / With Rubric
    without_rubric_tab, with_rubric_tab = st.tabs(["📝  WITHOUT RUBRIC  ", "📋  WITH RUBRIC  "])

    with without_rubric_tab:
        st.subheader("📝 Without Rubric")
        st.markdown("Complete these surveys after writing **without** using the rubric.")
        st.markdown("")

        # Nested tabs for Cognitive Load and Writing Quality
        wo_cognitive_tab, wo_writing_tab = st.tabs(["🧠 Cognitive Load", "✍️ Writing Quality"])

        with wo_cognitive_tab:
            mental_demand, physical_demand, temporal_demand, effort, performance, frustration = render_cognitive_load_survey("survey_wo_cl")

            st.markdown("---")
            if st.button("Submit Cognitive Load Survey", key="submit_wo_cognitive", type="primary"):
                survey_response = {
                    "timestamp": datetime.now().isoformat(),
                    "condition": "without_rubric",
                    "mental_demand": mental_demand,
                    "physical_demand": physical_demand,
                    "temporal_demand": temporal_demand,
                    "effort": effort,
                    "performance": performance,
                    "frustration": frustration
                }

                project_name = st.session_state.get('current_project')
                if not project_name:
                    st.error("No project selected. Please select a project first.")
                else:
                    project_dir = Path("project") / project_name
                    cognitive_load_file = project_dir / "cognitive_load_responses.json"
                    existing_responses = []
                    if cognitive_load_file.exists():
                        with open(cognitive_load_file, "r") as f:
                            existing_responses = json.load(f)
                    existing_responses.append(survey_response)
                    with open(cognitive_load_file, "w") as f:
                        json.dump(existing_responses, f, indent=2)
                    st.success("Cognitive Load Survey (Without Rubric) submitted and saved successfully!")

        with wo_writing_tab:
            content_satisfaction, ownership, responsibility, personal_connection, emotional_connection, prompt_difficulty = render_writing_quality_survey("survey_wo_wq")

            st.markdown("---")
            if st.button("Submit Writing Quality Survey", key="submit_wo_writing", type="primary"):
                survey_response = {
                    "timestamp": datetime.now().isoformat(),
                    "condition": "without_rubric",
                    "content_satisfaction": content_satisfaction,
                    "ownership": ownership,
                    "responsibility": responsibility,
                    "personal_connection": personal_connection,
                    "emotional_connection": emotional_connection,
                    "prompt_difficulty": prompt_difficulty
                }

                project_name = st.session_state.get('current_project')
                if not project_name:
                    st.error("No project selected. Please select a project first.")
                else:
                    project_dir = Path("project") / project_name
                    writing_quality_file = project_dir / "writing_quality_responses.json"
                    existing_responses = []
                    if writing_quality_file.exists():
                        with open(writing_quality_file, "r") as f:
                            existing_responses = json.load(f)
                    existing_responses.append(survey_response)
                    with open(writing_quality_file, "w") as f:
                        json.dump(existing_responses, f, indent=2)
                    st.success("Writing Quality Survey (Without Rubric) submitted and saved successfully!")

    with with_rubric_tab:
        st.subheader("📋 With Rubric")
        st.markdown("Complete these surveys after writing **with** the rubric.")
        st.markdown("")

        # Nested tabs for Cognitive Load and Writing Quality
        w_cognitive_tab, w_writing_tab = st.tabs(["🧠 Cognitive Load", "✍️ Writing Quality"])

        with w_cognitive_tab:
            mental_demand, physical_demand, temporal_demand, effort, performance, frustration = render_cognitive_load_survey("survey_w_cl")

            st.markdown("---")
            if st.button("Submit Cognitive Load Survey", key="submit_w_cognitive", type="primary"):
                survey_response = {
                    "timestamp": datetime.now().isoformat(),
                    "condition": "with_rubric",
                    "mental_demand": mental_demand,
                    "physical_demand": physical_demand,
                    "temporal_demand": temporal_demand,
                    "effort": effort,
                    "performance": performance,
                    "frustration": frustration
                }

                project_name = st.session_state.get('current_project')
                if not project_name:
                    st.error("No project selected. Please select a project first.")
                else:
                    project_dir = Path("project") / project_name
                    cognitive_load_file = project_dir / "cognitive_load_responses.json"
                    existing_responses = []
                    if cognitive_load_file.exists():
                        with open(cognitive_load_file, "r") as f:
                            existing_responses = json.load(f)
                    existing_responses.append(survey_response)
                    with open(cognitive_load_file, "w") as f:
                        json.dump(existing_responses, f, indent=2)
                    st.success("Cognitive Load Survey (With Rubric) submitted and saved successfully!")

        with w_writing_tab:
            content_satisfaction, ownership, responsibility, personal_connection, emotional_connection, prompt_difficulty = render_writing_quality_survey("survey_w_wq")

            st.markdown("---")
            if st.button("Submit Writing Quality Survey", key="submit_w_writing", type="primary"):
                survey_response = {
                    "timestamp": datetime.now().isoformat(),
                    "condition": "with_rubric",
                    "content_satisfaction": content_satisfaction,
                    "ownership": ownership,
                    "responsibility": responsibility,
                    "personal_connection": personal_connection,
                    "emotional_connection": emotional_connection,
                    "prompt_difficulty": prompt_difficulty
                }

                project_name = st.session_state.get('current_project')
                if not project_name:
                    st.error("No project selected. Please select a project first.")
                else:
                    project_dir = Path("project") / project_name
                    writing_quality_file = project_dir / "writing_quality_responses.json"
                    existing_responses = []
                    if writing_quality_file.exists():
                        with open(writing_quality_file, "r") as f:
                            existing_responses = json.load(f)
                    existing_responses.append(survey_response)
                    with open(writing_quality_file, "w") as f:
                        json.dump(existing_responses, f, indent=2)
                    st.success("Writing Quality Survey (With Rubric) submitted and saved successfully!")
