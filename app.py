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
    COMPARE_WRITE_EDIT_PROMPT,
    RUBRIC_INFERENCE_SYSTEM_PROMPT,
    get_rubric_inference_user_prompt,
    build_system_instruction,
    DRAFT_EDIT_RUBRIC_UPDATE_PROMPT,
    get_draft_edit_rubric_update_prompt,
    extract_decision_pts,
    generate_reflection_questions_prompt,
    generate_novel_alternatives_prompt,
    score_alternatives_with_rubric_prompt,
    score_alternatives_with_freetext_prompt,
    score_alternatives_generic_prompt
)
from scipy.stats import kendalltau
import random

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

def render_message_with_draft(content: str, message_id: str):
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
        priority = criterion.get('priority', criterion.get('weight', 0))

        if name_key not in current_map:
            # NEW criterion
            status_html = '<span class="status-badge status-new">NEW</span>'
            expander_label = f"✅ {name} (Priority #{priority})"

            with st.expander(expander_label, expanded=False):
                st.markdown(status_html, unsafe_allow_html=True)
                st.markdown(f"**Priority:** #{priority}")

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
            expander_label = f"⚪ {name} (Priority #{priority})"

            with st.expander(expander_label, expanded=False):
                st.markdown(status_html, unsafe_allow_html=True)
                st.markdown(f"**Priority:** #{priority}")

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
            priority = criterion.get('priority', criterion.get('weight', 0))
            status_html = '<span class="status-badge status-removed">REMOVED</span>'
            expander_label = f"❌ {name} (Priority #{priority}) - REMOVED"

            with st.expander(expander_label, expanded=False):
                st.markdown(status_html, unsafe_allow_html=True)
                st.markdown(f"**Priority:** ~~#{priority}~~")

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
    fields_to_compare = ['description', 'priority', 'exemplary', 'proficient', 'developing', 'beginning', 'category']
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

    # Clean _diff metadata from rubric before serializing
    clean_current_rubric = copy.deepcopy(current_rubric)
    if isinstance(clean_current_rubric, dict) and 'rubric' in clean_current_rubric:
        for criterion in clean_current_rubric['rubric']:
            if '_diff' in criterion:
                del criterion['_diff']
    elif isinstance(clean_current_rubric, list):
        for criterion in clean_current_rubric:
            if isinstance(criterion, dict) and '_diff' in criterion:
                del criterion['_diff']

    # Format the prompt with current rubric and user request
    prompt_text = RUBRIC_EDIT_PROMPT.format(
        current_rubric=json.dumps(clean_current_rubric, indent=2),
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

                # Check priority rankings (should be unique integers from 1 to N)
                priorities = [c.get('priority', i+1) for i, c in enumerate(modified_rubric)]
                if len(priorities) != len(set(priorities)):
                    return {
                        'message': f"Error: Duplicate priority rankings detected. Each criterion should have a unique rank from 1 to {len(modified_rubric)}."
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

        # Overall assessment narrative at the top
        if overall_assessment:
            st.markdown(f"*{overall_assessment}*")
            st.markdown("")

        # Build a lookup from (criterion, dimension_id) to evidence_highlights quotes
        # This links the highlighted text to specific dimensions
        highlight_lookup = {}
        for ev in evidence_highlights:
            key = (ev.get('criterion', ''), ev.get('dimension_id', ''))
            if key not in highlight_lookup:
                highlight_lookup[key] = []
            highlight_lookup[key].append({
                'quote': ev.get('quote', ''),
                'relevance': ev.get('relevance', '')
            })

        # Criteria with Dimension Checklists (sorted by priority)
        for crit in sorted_criteria:
            crit_name = crit.get('name', 'Unknown')
            priority = crit.get('priority', 'N/A')
            achievement_level = crit.get('achievement_level', 'N/A')
            dimensions_detail = crit.get('dimensions_detail', [])
            dims_met = crit.get('dimensions_met', 0)
            dims_total = crit.get('dimensions_total', 0)
            improvement_explanation = crit.get('improvement_explanation', '')

            level_color, level_emoji = get_level_info(achievement_level)

            # Criterion header with dimension count
            expander_label = f"{priority}. {level_emoji} **{crit_name}** ({dims_met}/{dims_total} dimensions)"

            with st.expander(expander_label, expanded=False):

                for dim in dimensions_detail:
                    dim_id = dim.get('id', '')
                    dim_label = dim.get('label', 'Unknown dimension')
                    dim_met = dim.get('met', False)
                    dim_evidence = dim.get('evidence', '')

                    # Look up any evidence_highlights for this specific dimension
                    linked_highlights = highlight_lookup.get((crit_name, dim_id), [])

                    if dim_met:
                        st.success(f"✅ {dim_label}")
                        # Show linked highlight quote if available, otherwise fall back to evidence field
                        if linked_highlights:
                            for hl in linked_highlights:
                                quote = hl.get('quote', '')
                                if quote:
                                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**Why met:** *\"{quote}\"*")
                        elif dim_evidence:
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**Why met:** *\"{dim_evidence}\"*")
                    else:
                        st.error(f"❌ {dim_label}")
                        # Show linked highlight quote if available, otherwise fall back to evidence field
                        if linked_highlights:
                            for hl in linked_highlights:
                                quote = hl.get('quote', '')
                                if quote:
                                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**Why not met:** *\"{quote}\"*")
                        elif dim_evidence:
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**Why not met:** *\"{dim_evidence}\"*")

                # Improvement explanation (if not Excellent)
                if improvement_explanation and 'excellent' not in achievement_level.lower():
                    st.markdown("")
                    st.markdown("**To improve:**")
                    st.markdown(improvement_explanation)

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
    
    # Show if previous rubric is being used
    if previous_rubric:
        st.info(f"🔍 Using previous rubric v{previous_rubric_version}")
        previous_rubric_json = json.dumps(previous_rubric, ensure_ascii=False, indent=2)
        # previous_rubric_block = f"\n\nPREVIOUS RUBRIC (use this as a starting point and refine it based on the conversation):\n{previous_rubric_json}\n\nBuild upon this rubric by keeping what's relevant and adding/modifying criteria based on the conversation."
    else:
        st.info("🔍 No previous rubric found - creating new rubric from scratch")
        previous_rubric_json = ""
        # previous_rubric_block = "\n\nThis is a new rubric - create it from scratch based on the conversation."
    
    system_prompt = RUBRIC_INFERENCE_SYSTEM_PROMPT
    user_prompt = get_rubric_inference_user_prompt(conversation_text, previous_rubric_json)

    # Retry logic for overloaded errors
    max_retries = 3
    retry_delay = 5  # seconds

    # Create placeholder once outside the loop
    progress_placeholder = st.empty()

    for attempt in range(max_retries):
        try:
            # Use streaming to avoid timeout for long-running requests
            thinking_text = ""
            response_text = ""

            # Show progress indicator
            if attempt > 0:
                progress_placeholder.info(f"Retry attempt {attempt + 1} of {max_retries}...")
            else:
                progress_placeholder.info("Inferring rubric from conversation...")

            with client.messages.stream(
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
            ) as stream:
                for event in stream:
                    if event.type == "content_block_start":
                        if hasattr(event.content_block, 'type'):
                            if event.content_block.type == "thinking":
                                pass  # Thinking block started
                            elif event.content_block.type == "text":
                                pass  # Text block started
                    elif event.type == "content_block_delta":
                        if hasattr(event.delta, 'thinking'):
                            thinking_text += event.delta.thinking
                        elif hasattr(event.delta, 'text'):
                            response_text += event.delta.text

            # Clear progress indicator
            progress_placeholder.empty()

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
                # Add version number and mark as inferred (not a template)
                rubric_data["version"] = next_version_number()
                rubric_data["source"] = "inferred"

                # Save to history
                hist = load_rubric_history()
                hist.append(rubric_data)
                save_rubric_history(hist)
                st.session_state.active_rubric_idx = len(hist) - 1

                return rubric_data
            else:
                rubric_data = json.loads(response_text)
                # Add version number and mark as inferred (not a template)
                rubric_data["version"] = next_version_number()
                rubric_data["source"] = "inferred"
                hist = load_rubric_history()
                hist.append(rubric_data)
                save_rubric_history(hist)
                st.session_state.active_rubric_idx = len(hist) - 1
                return rubric_data

        except Exception as e:
            error_str = str(e)

            # Check if it's an overloaded error
            if 'overloaded' in error_str.lower():
                if attempt < max_retries - 1:
                    # Show countdown with retry message
                    countdown_placeholder = st.empty()
                    for remaining in range(retry_delay, 0, -1):
                        countdown_placeholder.warning(
                            f"Claude's servers are currently experiencing high demand. "
                            f"Retrying in {remaining} seconds... (Attempt {attempt + 1} of {max_retries})"
                        )
                        time.sleep(1)
                    countdown_placeholder.empty()
                    continue  # Retry
                else:
                    # Final attempt failed
                    st.error(
                        "Claude's servers are currently overloaded. "
                        "Please try again in a few minutes."
                    )
                    return None
            else:
                # Not an overloaded error, don't retry
                st.error(f"Error inferring rubric: {error_str}")
                return None

    return None

# Page configuration
st.set_page_config(
    page_title="AI Co-Writer",
    page_icon="✍️",
    layout="wide"
)

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

# Rubric editing mode for conversational rubric content changes
if 'rubric_editing' not in st.session_state:
    st.session_state.rubric_editing = {
        'active': False,           # Whether editing mode is active
        'edit_messages': [],       # Conversation about edits
        'proposed_changes': None,  # Parsed changes from AI
        'original_rubric': None    # Snapshot before editing
    }

# Evaluate: Coverage tab state (9-step workflow)
if 'evaluate_selected_conversation' not in st.session_state:
    st.session_state.evaluate_selected_conversation = None
if 'evaluate_decision_points' not in st.session_state:
    st.session_state.evaluate_decision_points = None  # Step 1: Extracted decision points
if 'evaluate_expanded_dp' not in st.session_state:
    st.session_state.evaluate_expanded_dp = None  # ID of currently expanded decision point
if 'evaluate_user_responses' not in st.session_state:
    st.session_state.evaluate_user_responses = {}  # Step 2: User's cold reflection responses
if 'evaluate_novel_alternatives' not in st.session_state:
    st.session_state.evaluate_novel_alternatives = None  # Step 3: Generated alternatives per decision point
if 'evaluate_user_rankings' not in st.session_state:
    st.session_state.evaluate_user_rankings = {}  # Step 4: User's 1st/2nd/3rd rankings
if 'evaluate_rubric_rankings' not in st.session_state:
    st.session_state.evaluate_rubric_rankings = None  # Step 5: Rubric-based rankings
if 'evaluate_freetext_rankings' not in st.session_state:
    st.session_state.evaluate_freetext_rankings = None  # Step 6: Free-text-based rankings
if 'evaluate_generic_rankings' not in st.session_state:
    st.session_state.evaluate_generic_rankings = None  # Step 7: Generic baseline rankings
if 'evaluate_agreement_scores' not in st.session_state:
    st.session_state.evaluate_agreement_scores = None  # Step 8: Correlation results
if 'evaluate_current_step' not in st.session_state:
    st.session_state.evaluate_current_step = 1  # Track progress (1-9)

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
    """Load all conversations from Supabase database"""
    project_id = st.session_state.get('current_project_id')
    if not project_id:
        return []

    supabase = st.session_state.get('supabase')
    if not supabase:
        return []

    conversations = db_load_conversations(supabase, project_id)
    # Add filename field for compatibility with existing code
    for conv in conversations:
        conv["filename"] = conv["id"]
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

# Create tabs
tab1, tab2, tab5, tab3, tab4 = st.tabs(["💬 Chat", "📋 Evaluate: Survey", "📊 Evaluate: Coverage", "📁 View Rubric", "🔍 Compare Rubrics"])

with tab1:
    # Show save toast if we just saved (after rerun)
    if st.session_state.get('show_save_toast'):
        st.toast("✓ Conversation saved!", icon="✅")
        st.session_state.show_save_toast = False

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
            # In delete mode, show checkbox alongside message
            if st.session_state.message_delete_mode:
                col_check, col_msg = st.columns([0.05, 0.95])
                with col_check:
                    is_selected = idx in st.session_state.messages_to_delete
                    if st.checkbox("", value=is_selected, key=f"delete_msg_{idx}", label_visibility="collapsed"):
                        st.session_state.messages_to_delete.add(idx)
                    else:
                        st.session_state.messages_to_delete.discard(idx)
                with col_msg:
                    with st.chat_message(message['role']):
                        message_id = message.get('message_id', f"{message['role']}_{idx}")
                        if message['role'] == 'user':
                            content_to_display = message['content']
                        else:
                            content_to_display = message.get('display_content', message['content'])
                        if message['role'] == 'assistant' and message.get('thinking'):
                            with st.expander("🧠 Thinking", expanded=False):
                                st.markdown(message['thinking'])
                        if message['role'] == 'assistant':
                            has_draft = render_message_with_draft(content_to_display, message_id)
                            if not has_draft:
                                st.markdown(content_to_display)
                        else:
                            st.markdown(content_to_display)
                        if message['role'] == 'assistant' and message.get('rubric_assessment'):
                            assessment = message['rubric_assessment']
                            draft_text = assessment.get('draft_text')
                            display_rubric_assessment(assessment, message_id, draft_text)
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
                        has_draft = render_message_with_draft(content_to_display, message_id)
                        if not has_draft:
                            st.markdown(content_to_display)
                    else:
                        st.markdown(content_to_display)

                    # Display rubric assessment if available (for assistant messages)
                    if message['role'] == 'assistant' and message.get('rubric_assessment'):
                        assessment = message['rubric_assessment']
                        draft_text = assessment.get('draft_text')
                        display_rubric_assessment(assessment, message_id, draft_text)

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

    # Create a container for streaming responses BEFORE chat_input
    # This ensures streaming content appears above the input, not below
    streaming_container = st.container()

    # User input (chat input and buttons)
    if prompt := st.chat_input("Type your message here..."):
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

        # Include main conversation messages
        for msg in st.session_state.messages:
            content_to_send = msg.get('content', msg.get('display_content', ''))
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
                system_instruction = build_system_instruction(active_rubric_dict if active_rubric_dict else [])

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
                            model="claude-opus-4-5-20251101",
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

                        # SUCCESS - Now add both user and assistant messages to session state
                        # Add user message to conversation history
                        st.session_state.messages.append(user_message_data)

                        # Add assistant message to conversation history
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
                    # Save the conversation to Supabase
                    conv_id = save_message_log(st.session_state.messages, st.session_state.rubric, st.session_state.current_analysis)

                    if conv_id:
                        # Store a flag to indicate we just saved (don't reload the conversation)
                        # This flag is checked in the conversation selector logic
                        st.session_state.just_saved = conv_id

                        # Update selected_conversation to the saved conversation ID
                        st.session_state.selected_conversation = conv_id

                        # Set a flag to show toast after rerun
                        st.session_state.show_save_toast = True

                        # Rerun to update the conversation selector list
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
                                st.session_state.editing_criteria = copy.deepcopy(rubric_list)
                            
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

                                # Include all messages
                                for msg in st.session_state.messages:
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

                                system_instruction = build_system_instruction(rubric_for_system)

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
                                    "is_assessment_message": True  # Mark for hiding from display
                                })

                                # Then add the full assessment text as an assistant message (hidden from display)
                                st.session_state.messages.append({
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

            st.rerun()

        # Ensure current_project_id is set if we have a current project
        if st.session_state.current_project and not st.session_state.current_project_id:
            st.session_state.current_project_id = project_id_map.get(st.session_state.current_project)
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
                    # Description (editable)
                    desc_key = f"criterion_desc_{original_idx}_{version_key}"
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
                            dim_key = f"criterion_dim_{original_idx}_{dim_id}_{version_key}"
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
                            remove_key = f"remove_dim_{original_idx}_{dim_id}_{version_key}"
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
                    add_dim_key = f"add_dim_{original_idx}_{version_key}"
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

        # Save and Reset buttons
        save_col, reset_col = st.columns(2)
        with save_col:
            if st.button("💾 Save Changes", use_container_width=True, type="primary"):
                # Save as a NEW version in rubric history
                if rubric_history is not None:
                    # Deep copy the editing criteria to avoid reference issues
                    saved_criteria = copy.deepcopy(st.session_state.editing_criteria)

                    # Create new version entry
                    new_version = next_version_number()
                    new_rubric_entry = {
                        "version": new_version,
                        "rubric": saved_criteria,
                        "source": "edited"
                    }

                    # Add to history and save
                    hist = load_rubric_history()
                    hist.append(new_rubric_entry)
                    save_rubric_history(hist)

                    # Update session state to point to new version
                    st.session_state.active_rubric_idx = len(hist) - 1
                    st.session_state.rubric = saved_criteria
                    st.session_state.editing_criteria = copy.deepcopy(saved_criteria)

                    st.toast(f"Saved as v{new_version}!")
                    st.rerun()
        with reset_col:
            if st.button("↩️ Reset", use_container_width=True, type="secondary"):
                # Reset editing criteria to the original saved version
                if rubric_history and active_idx is not None:
                    original_rubric = rubric_history[active_idx].get("rubric", [])
                    st.session_state.editing_criteria = copy.deepcopy(original_rubric)
                    st.session_state.rubric = copy.deepcopy(original_rubric)
                    st.toast("Rubric reset to saved version!")
                    st.rerun()

        # Delete Version button
        if st.button("🗑️ Delete Version", use_container_width=True, type="secondary"):
            if len(rubric_history) > 1:
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

                        # Update active rubric index before reloading
                        new_idx = active_idx - 1 if active_idx > 0 else 0
                        st.session_state.active_rubric_idx = new_idx

                        # Reload history from database
                        new_history = load_rubric_history(force_reload=True)

                        # Update session state with new active rubric
                        if new_history and new_idx < len(new_history):
                            rubric_list = new_history[new_idx].get("rubric", [])
                            st.session_state.rubric = rubric_list
                            st.session_state.editing_criteria = copy.deepcopy(rubric_list)

                        st.toast(f"Version {deleted_version} deleted!")
                        st.rerun()
                    else:
                        st.error("Failed to delete rubric version")
                elif not supabase:
                    st.error("Cannot delete: not connected to database")
                elif not rubric_id:
                    st.error("Cannot delete: this version was not saved to database yet")
            else:
                st.error("Cannot delete the last rubric version")

        # Fallback for when Update button condition is false
        if active_rubric_dict and active_idx is None:
            rubric_list = active_rubric_dict.get("rubric", [])
            st.session_state.rubric = rubric_list
    else:
        st.info("No rubric loaded. Use 'Infer Rubric' to create one from your conversation.")

    st.divider()

    # === Rubric Chat Section ===
    st.header("💬 Improve Rubric")
    st.caption("Describe changes you want to make to your rubric and see how they affect your draft.")

    # Initialize rubric chat state
    if "rubric_chat_messages" not in st.session_state:
        st.session_state.rubric_chat_messages = []
    if "rubric_chat_suggestion" not in st.session_state:
        st.session_state.rubric_chat_suggestion = None
    if "rubric_chat_preview_draft" not in st.session_state:
        st.session_state.rubric_chat_preview_draft = None

    # Check if we have a rubric and draft to work with
    if not st.session_state.editing_criteria:
        st.info("Load a rubric first to start improving it.")
    else:
        # Get last draft from conversation
        last_draft, draft_msg_idx = get_last_draft_from_messages()

        if not last_draft:
            st.info("Start a conversation and generate a draft to see how rubric changes affect it.")
        else:
            # Display chat history in scrollable container with white background
            if st.session_state.rubric_chat_messages or st.session_state.rubric_chat_suggestion:
                # Create scrollable white container using HTML/CSS
                chat_html_start = """
                <div style="
                    background-color: white;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    padding: 12px;
                    max-height: 300px;
                    overflow-y: auto;
                    margin-bottom: 10px;
                ">
                """
                chat_content = ""

                def md_to_html(text):
                    # Convert **bold** to <strong>bold</strong>
                    import re
                    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
                    # Convert newlines to <br>
                    text = text.replace('\n', '<br>')
                    return text

                for msg in st.session_state.rubric_chat_messages:
                    content_html = md_to_html(msg["content"])
                    if msg["role"] == "user":
                        chat_content += f'<p><strong>You:</strong> {content_html}</p>'
                    else:
                        chat_content += f'<p><strong>AI:</strong> {content_html}</p>'

                chat_html_end = "</div>"

                st.markdown(chat_html_start + chat_content + chat_html_end, unsafe_allow_html=True)

            # User input for rubric changes
            rubric_change_input = st.text_area(
                "What would you like to change?",
                placeholder="e.g., Make 'Clarity' the top priority, Add a criterion for 'Evidence Quality', Remove the 'Brevity' dimension...",
                key="rubric_chat_input",
                height=80
            )

            col_send, col_clear = st.columns([3, 1])

            with col_send:
                if st.button("💡 Get Suggestions", type="primary", use_container_width=True, disabled=not rubric_change_input.strip()):
                    if rubric_change_input.strip():
                        # Add user message to chat
                        st.session_state.rubric_chat_messages.append({
                            "role": "user",
                            "content": rubric_change_input.strip()
                        })

                        with st.spinner("Analyzing your request..."):
                            try:
                                from prompts import RUBRIC_EDIT_PROMPT

                                client = anthropic.Anthropic()

                                # Build system prompt with current rubric
                                current_rubric_json = json.dumps(st.session_state.editing_criteria, indent=2)
                                system_prompt = RUBRIC_EDIT_PROMPT.format(
                                    current_rubric=current_rubric_json,
                                    user_request="{user_request}"  # Placeholder, actual request in messages
                                )

                                # Build conversation messages from chat history
                                api_messages = []
                                for msg in st.session_state.rubric_chat_messages:
                                    api_messages.append({
                                        "role": msg["role"],
                                        "content": msg["content"]
                                    })

                                response = client.messages.create(
                                    model="claude-sonnet-4-20250514",
                                    max_tokens=8000,
                                    system=system_prompt,
                                    messages=api_messages
                                )

                                response_text = response.content[0].text

                                # Try to parse JSON response
                                json_match = re.search(r'\{[\s\S]*\}', response_text)
                                if json_match:
                                    result = json.loads(json_match.group())
                                    changes_summary = result.get("changes_summary", "Changes applied")
                                    modified_rubric = result.get("modified_rubric", [])

                                    # Store the suggestion
                                    original_rubric = copy.deepcopy(st.session_state.editing_criteria)
                                    st.session_state.rubric_chat_suggestion = {
                                        "summary": changes_summary,
                                        "modified_rubric": modified_rubric,
                                        "original_rubric": original_rubric
                                    }

                                    # Add AI response to chat
                                    st.session_state.rubric_chat_messages.append({
                                        "role": "assistant",
                                        "content": f"✅ **Suggested changes:** {changes_summary}"
                                    })

                                    # Build rubric change preview and add as chat message
                                    original_names = {c.get("name"): c for c in original_rubric}
                                    modified_names = {c.get("name"): c for c in modified_rubric}
                                    added = set(modified_names.keys()) - set(original_names.keys())
                                    removed = set(original_names.keys()) - set(modified_names.keys())
                                    common = set(original_names.keys()) & set(modified_names.keys())

                                    change_lines = []
                                    # Added criteria
                                    for name in added:
                                        crit = modified_names[name]
                                        change_lines.append(f"➕ **NEW: {name}** (priority #{crit.get('priority', '?')})")
                                        if crit.get("description"):
                                            change_lines.append(f"   {crit.get('description')}")

                                    # Removed criteria
                                    for name in removed:
                                        change_lines.append(f"➖ **REMOVED:** ~~{name}~~")

                                    # Modified criteria
                                    for name in common:
                                        orig = original_names[name]
                                        mod = modified_names[name]
                                        orig_p = orig.get("priority", 0)
                                        mod_p = mod.get("priority", 0)
                                        orig_desc = orig.get("description", "")
                                        mod_desc = mod.get("description", "")
                                        orig_dims = [d.get("label", "") for d in orig.get("dimensions", [])]
                                        mod_dims = [d.get("label", "") for d in mod.get("dimensions", [])]

                                        if orig_p != mod_p or orig_desc != mod_desc or orig_dims != mod_dims:
                                            change_lines.append(f"🔄 **{name}**")
                                            if orig_p != mod_p:
                                                change_lines.append(f"   Priority: #{orig_p} → #{mod_p}")
                                            if orig_desc != mod_desc:
                                                change_lines.append(f"   Description:")
                                                change_lines.append(f"   ❌ {orig_desc[:150]}{'...' if len(orig_desc) > 150 else ''}")
                                                change_lines.append(f"   ✅ {mod_desc}")
                                            added_dims = set(mod_dims) - set(orig_dims)
                                            removed_dims = set(orig_dims) - set(mod_dims)
                                            if added_dims or removed_dims:
                                                change_lines.append(f"   Dimensions:")
                                                for d in removed_dims:
                                                    change_lines.append(f"   ➖ ~~{d}~~")
                                                for d in added_dims:
                                                    change_lines.append(f"   ➕ {d}")

                                    if change_lines:
                                        st.session_state.rubric_chat_messages.append({
                                            "role": "assistant",
                                            "content": "\n".join(change_lines)
                                        })

                                else:
                                    # AI asked a clarifying question
                                    st.session_state.rubric_chat_messages.append({
                                        "role": "assistant",
                                        "content": response_text
                                    })

                                st.rerun()

                            except Exception as e:
                                st.error(f"Error: {str(e)}")

            with col_clear:
                if st.button("🗑️", help="Clear chat", use_container_width=True):
                    st.session_state.rubric_chat_messages = []
                    st.session_state.rubric_chat_suggestion = None
                    st.session_state.rubric_chat_preview_draft = None
                    st.rerun()

            # Show Accept/Reject/Preview buttons if there's a pending suggestion
            if st.session_state.rubric_chat_suggestion:
                col_preview, col_accept, col_reject = st.columns(3)

                with col_preview:
                    if st.button("📝 Preview Draft", use_container_width=True):
                        # Regenerate draft with the suggested rubric changes
                        modified_rubric = st.session_state.rubric_chat_suggestion.get("modified_rubric", [])
                        regenerate_result = regenerate_draft_from_rubric_changes(
                            st.session_state.editing_criteria,
                            modified_rubric,
                            last_draft
                        )

                        if regenerate_result:
                            revised_draft = regenerate_result.get("revised_draft", "")
                            st.session_state.rubric_chat_preview_draft = revised_draft
                            # Add draft preview as a chat message so it stays in order
                            st.session_state.rubric_chat_messages.append({
                                "role": "assistant",
                                "content": f"📝 **Updated Draft Preview:**\n\n{revised_draft}"
                            })
                            st.rerun()

                with col_accept:
                    if st.button("✅ Accept", type="primary", use_container_width=True):
                        # Get rubric before and after
                        original_rubric = st.session_state.rubric_chat_suggestion.get("original_rubric", [])
                        modified_rubric = st.session_state.rubric_chat_suggestion.get("modified_rubric", [])

                        # Save as new version
                        hist = load_rubric_history()
                        new_version = next_version_number()
                        new_rubric_entry = {
                            "version": new_version,
                            "rubric": modified_rubric,
                            "source": "chat_edit"
                        }
                        hist.append(new_rubric_entry)
                        save_rubric_history(hist)

                        # Save rubric conversation to database
                        project_id = st.session_state.get('current_project_id')
                        supabase = st.session_state.get('supabase')
                        if project_id and supabase:
                            rubric_conversation_data = {
                                "timestamp": datetime.now().isoformat(),
                                "conversation": st.session_state.rubric_chat_messages.copy(),
                                "rubric_before": original_rubric,
                                "rubric_after": modified_rubric,
                                "new_version": new_version,
                                "draft_preview": st.session_state.rubric_chat_preview_draft
                            }
                            if save_project_data(supabase, project_id, "rubric_conversation", rubric_conversation_data):
                                st.toast("Rubric conversation saved to database")
                            else:
                                st.warning("Failed to save rubric conversation to database")

                        # Update session state
                        st.session_state.active_rubric_idx = len(hist) - 1
                        st.session_state.rubric = modified_rubric
                        st.session_state.editing_criteria = copy.deepcopy(modified_rubric)

                        # Optionally add the new draft to conversation
                        if st.session_state.rubric_chat_preview_draft:
                            new_draft_msg = {
                                "role": "assistant",
                                "content": f"<draft>{st.session_state.rubric_chat_preview_draft}</draft>\n\n*Draft updated based on rubric changes.*"
                            }
                            st.session_state.messages.append(new_draft_msg)

                        # Clear suggestion state
                        st.session_state.rubric_chat_suggestion = None
                        st.session_state.rubric_chat_preview_draft = None
                        st.session_state.rubric_chat_messages.append({
                            "role": "assistant",
                            "content": f"✅ Changes saved as v{new_version}!"
                        })

                        st.toast(f"Rubric saved as v{new_version}!")
                        st.rerun()

                with col_reject:
                    if st.button("❌ Reject", use_container_width=True):
                        st.session_state.rubric_chat_suggestion = None
                        st.session_state.rubric_chat_preview_draft = None
                        st.session_state.rubric_chat_messages.append({
                            "role": "assistant",
                            "content": "Changes rejected. Feel free to describe what you'd like instead."
                        })
                        st.rerun()

    st.divider()
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

        # Achievement level explanation
        st.markdown("**Achievement Levels** *(based on dimensions met)*")
        st.markdown("⭐⭐⭐ **Excellent**: 100% | ⭐⭐ **Good**: 75%+ | ⭐ **Fair**: 50-74% | ☆ **Weak**: <50%")

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

# Evaluate: Coverage Tab - 9-Step Workflow
with tab5:
    st.header("📊 Evaluate: Coverage")
    st.markdown("Test whether your rubric captures your writing preferences through a systematic comparison.")

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
    coverage_rubric_dict, coverage_rubric_idx, _ = get_active_rubric()

    if not coverage_rubric_dict:
        st.warning("No active rubric selected. Please select a rubric version in the sidebar to enable coverage testing.")
    else:
        coverage_rubric_list = coverage_rubric_dict.get("rubric", [])
        coverage_rubric_version = coverage_rubric_dict.get("version", coverage_rubric_idx + 1)

        # Header with rubric info and reset
        col_info, col_reset = st.columns([3, 1])
        with col_info:
            st.success(f"Using rubric: **Version {coverage_rubric_version}** ({len(coverage_rubric_list)} criteria)")
        with col_reset:
            if st.button("🔄 Reset All", use_container_width=True, key="coverage_reset"):
                st.session_state.evaluate_selected_conversation = None
                st.session_state.evaluate_decision_points = None
                st.session_state.evaluate_expanded_dp = None
                st.session_state.evaluate_user_responses = {}
                st.session_state.evaluate_novel_alternatives = None
                st.session_state.evaluate_user_rankings = {}
                st.session_state.evaluate_rubric_rankings = None
                st.session_state.evaluate_freetext_rankings = None
                st.session_state.evaluate_generic_rankings = None
                st.session_state.evaluate_agreement_scores = None
                st.session_state.evaluate_current_step = 1
                st.rerun()

        # Step progress indicator
        steps = [
            "1. Extract Decision Points",
            "2. Reflection",
            "3. Generate Alternatives",
            "4. Rank Alternatives",
            "5. Score & Analyze",
            "6. View Results"
        ]

        # Determine current step based on state
        def get_current_step():
            if not st.session_state.evaluate_decision_points:
                return 1
            if not st.session_state.evaluate_user_responses or len(st.session_state.evaluate_user_responses) == 0:
                return 2
            dp_count = len(st.session_state.evaluate_decision_points.get("parsed_data", {}).get("decision_points", []))
            if len(st.session_state.evaluate_user_responses) < dp_count:
                return 2
            if not st.session_state.evaluate_novel_alternatives:
                return 3
            if not st.session_state.evaluate_user_rankings or len(st.session_state.evaluate_user_rankings) < dp_count:
                return 4
            if not st.session_state.evaluate_agreement_scores:
                return 5
            return 6

        current_step = get_current_step()
        st.progress(current_step / 6)
        st.markdown(f"**Step {current_step}/6: {steps[current_step - 1]}**")

        st.divider()

        # ==================== STEP 1: EXTRACT DECISION POINTS ====================
        st.subheader("Step 1: Select Conversation & Extract Decision Points")

        # Load conversations for selector
        eval_conversations = load_conversations()
        eval_conversation_options = [("Select a conversation...", None)]
        if eval_conversations:
            for conv in eval_conversations:
                try:
                    dt = datetime.fromisoformat(conv["timestamp"])
                    formatted_time = dt.strftime("%m/%d %H:%M")
                    display = f"{formatted_time} ({conv['messages_count']} msgs)"
                except:
                    display = f"{conv['timestamp']} ({conv['messages_count']} msgs)"
                eval_conversation_options.append((display, conv["filename"]))

        eval_options = [opt[1] for opt in eval_conversation_options]
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

        if eval_selected_file != st.session_state.evaluate_selected_conversation:
            st.session_state.evaluate_selected_conversation = eval_selected_file
            st.session_state.evaluate_decision_points = None
            st.session_state.evaluate_user_responses = {}
            st.session_state.evaluate_novel_alternatives = None
            st.session_state.evaluate_user_rankings = {}
            st.session_state.evaluate_rubric_rankings = None
            st.session_state.evaluate_freetext_rankings = None
            st.session_state.evaluate_generic_rankings = None
            st.session_state.evaluate_agreement_scores = None
            st.rerun()

        if eval_selected_file:
            eval_conv_data = load_conversation_data(eval_selected_file)

            if eval_conv_data:
                eval_messages = eval_conv_data.get("messages", [])

                # Build conversation text for display and API calls
                def build_conversation_text(messages):
                    text = ""
                    msg_num = 1
                    for msg in messages:
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')
                        if role == 'user':
                            text += f"\n\n[Message #{msg_num}] USER:\n{content}"
                            msg_num += 1
                        elif role == 'assistant':
                            text += f"\n\n[Message #{msg_num}] ASSISTANT:\n{content}"
                            msg_num += 1
                    return text

                # Extract Decision Points button (only show if not already extracted)
                if not st.session_state.evaluate_decision_points:
                    if st.button("🎯 Extract Decision Points", use_container_width=True, type="primary", key="extract_dp_btn"):
                        conversation_text = build_conversation_text(eval_messages)

                        with st.spinner("Analyzing conversation for decision points..."):
                            try:
                                # Get active rubric for context
                                active_rubric_for_extraction, _, _ = get_active_rubric()
                                if active_rubric_for_extraction:
                                    clean_rubric = copy.deepcopy(active_rubric_for_extraction)
                                    if 'rubric' in clean_rubric:
                                        for criterion in clean_rubric['rubric']:
                                            if '_diff' in criterion:
                                                del criterion['_diff']
                                    rubric_json_for_extraction = json.dumps(clean_rubric, indent=2)
                                else:
                                    rubric_json_for_extraction = None

                                decision_prompt = extract_decision_pts(conversation_text, rubric_json_for_extraction)
                                response = client.messages.create(
                                    model="claude-opus-4-5-20251101",
                                    max_tokens=16000,
                                    messages=[{"role": "user", "content": decision_prompt}],
                                    thinking={"type": "enabled", "budget_tokens": 8000}
                                )

                                thinking_text = ""
                                response_text = ""
                                for block in response.content:
                                    if block.type == "thinking":
                                        thinking_text = block.thinking
                                    elif block.type == "text":
                                        response_text = block.text

                                parsed_data = None
                                try:
                                    json_match = re.search(r'\{[\s\S]*\}', response_text)
                                    if json_match:
                                        parsed_data = json.loads(json_match.group())
                                except json.JSONDecodeError:
                                    pass

                                st.session_state.evaluate_decision_points = {
                                    "thinking": thinking_text,
                                    "raw_response": response_text,
                                    "parsed_data": parsed_data,
                                    "conversation_file": eval_selected_file
                                }
                                st.rerun()

                            except Exception as e:
                                st.error(f"Error extracting decision points: {str(e)}")

                # Display decision points if extracted
                if st.session_state.evaluate_decision_points:
                    result = st.session_state.evaluate_decision_points
                    parsed_data = result.get("parsed_data")

                    if parsed_data and "decision_points" in parsed_data:
                        decision_points = parsed_data["decision_points"]
                        st.success(f"Found {len(decision_points)} decision points")

                        # Show thinking (collapsed)
                        if result.get("thinking"):
                            with st.expander("🧠 Extraction Thinking", expanded=False):
                                st.markdown(result["thinking"])

                        # ==================== STEP 2: COLD REFLECTION ====================
                        st.divider()
                        st.subheader("Step 2: Reflection")
                        st.markdown("For each decision point, reflect on what was wrong with the original or better about your version.**")

                        reflections_complete = len(st.session_state.evaluate_user_responses) >= len(decision_points)
                        st.progress(len(st.session_state.evaluate_user_responses) / max(len(decision_points), 1))
                        st.caption(f"Completed: {len(st.session_state.evaluate_user_responses)} / {len(decision_points)}")

                        # Build mapping from decision point to message numbers
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
                        .message-user { border-left-color: #28a745; }
                        .message-assistant { border-left-color: #007bff; }
                        .highlighted-message.message-user {
                            border-left-color: #28a745;
                            background: linear-gradient(90deg, #d4edda 0%, #e8f5e9 100%);
                        }
                        .highlighted-message.message-assistant {
                            border-left-color: #007bff;
                            background: linear-gradient(90deg, #cce5ff 0%, #e3f2fd 100%);
                        }
                        .message-number { font-weight: bold; color: #6c757d; font-size: 0.85em; }
                        .message-role { font-weight: 600; margin-bottom: 4px; }
                        .message-content { color: #333; line-height: 1.6; }
                        </style>
                        """, unsafe_allow_html=True)

                        # Two-column layout: Conversation on left, Decision points on right
                        col_conv, col_dp = st.columns([1, 1])

                        with col_conv:
                            st.markdown("### 📜 Conversation")
                            if st.session_state.evaluate_expanded_dp:
                                st.caption(f"*Highlighting messages for Decision Point #{st.session_state.evaluate_expanded_dp}*")

                            conversation_container = st.container(height=500)
                            with conversation_container:
                                msg_num = 1
                                for idx, msg in enumerate(eval_messages):
                                    role = msg.get('role', 'unknown')
                                    if role not in ['user', 'assistant']:
                                        continue

                                    content = msg.get('display_content', msg.get('content', ''))
                                    is_highlighted = msg_num in highlighted_messages

                                    css_class = "highlighted-message" if is_highlighted else "normal-message"
                                    role_class = f"message-{role}"
                                    role_label = "👤 User" if role == "user" else "🤖 Assistant"

                                    escaped_content = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")

                                    st.markdown(f"""
                                    <div class="{css_class} {role_class}">
                                        <div class="message-number">Message #{msg_num}</div>
                                        <div class="message-role">{role_label}</div>
                                        <div class="message-content">{escaped_content}</div>
                                    </div>
                                    """, unsafe_allow_html=True)

                                    msg_num += 1

                        with col_dp:
                            st.markdown("### 🔍 Decision Points")
                            st.caption("Click to expand and see highlighted messages")

                            for dp in decision_points:
                                dp_id = dp.get('id', 0)
                                is_expanded = st.session_state.evaluate_expanded_dp == dp_id
                                already_responded = dp_id in st.session_state.evaluate_user_responses

                                # Show message numbers for this decision point
                                asst_msg = dp.get('assistant_message_num', '?')
                                user_msg = dp.get('user_message_num', '?')

                                status_icon = "✅" if already_responded else "⬜"
                                btn_label = f"{status_icon} DP#{dp_id}: {dp.get('dimension', 'Unknown')} (Msgs #{asst_msg}→#{user_msg})"

                                if st.button(btn_label, key=f"dp_btn_{dp_id}", use_container_width=True,
                                            type="primary" if is_expanded else "secondary"):
                                    st.session_state.evaluate_expanded_dp = dp_id if not is_expanded else None
                                    st.rerun()

                                if is_expanded:
                                    with st.container():
                                        col_before, col_after = st.columns(2)
                                        with col_before:
                                            st.markdown(f"**📤 Original (Msg #{asst_msg}):**")
                                            st.info(dp.get('before_quote', 'N/A'))
                                        with col_after:
                                            st.markdown(f"**📥 Your Edit (Msg #{user_msg}):**")
                                            st.success(dp.get('after_quote', 'N/A'))

                                        existing = st.session_state.evaluate_user_responses.get(dp_id, {})

                                        q1 = st.text_area(
                                            "**Q1:** What was wrong with the original, or what was better about your version?",
                                            value=existing.get('reasoning', ''),
                                            key=f"q1_{dp_id}",
                                            placeholder="Explain the specific issues or improvements..."
                                        )

                                        q2_options = ["This is a general preference", "Specific to this situation", "Depends on context"]
                                        q2 = st.radio(
                                            "**Q2:** Is this a general preference or situation-specific?",
                                            options=q2_options,
                                            index=q2_options.index(existing.get('generalization', "Depends on context")),
                                            key=f"q2_{dp_id}",
                                            horizontal=True
                                        )

                                        if st.button("💾 Save Reflection", key=f"save_{dp_id}", type="primary"):
                                            st.session_state.evaluate_user_responses[dp_id] = {
                                                'decision_point_id': dp_id,
                                                'dimension': dp.get('dimension', ''),
                                                'before_text': dp.get('before_quote', ''),
                                                'after_text': dp.get('after_quote', ''),
                                                'reasoning': q1,
                                                'generalization': q2
                                            }
                                            # Collapse the decision point after saving
                                            st.session_state.evaluate_expanded_dp = None
                                            st.rerun()

                                        st.markdown("---")

                        # ==================== STEP 3: GENERATE NOVEL ALTERNATIVES ====================
                        if reflections_complete and not st.session_state.evaluate_novel_alternatives:
                            st.divider()
                            st.subheader("Step 3: Generate Alternatives")
                            st.markdown("Generate 3 new versions for each decision point, varying along the identified dimension.")

                            if st.button("🎲 Generate Alternatives", use_container_width=True, type="primary", key="gen_alts_btn"):
                                with st.spinner("Generating alternatives for all decision points..."):
                                    alternatives_data = {}
                                    # Get writing domain context from rubric
                                    writing_type = coverage_rubric_dict.get('writing_type', '')
                                    user_goals = coverage_rubric_dict.get('user_goals_summary', '')

                                    for dp in decision_points:
                                        dp_id = dp.get('id')
                                        dimension = dp.get('dimension', 'style')

                                        try:
                                            prompt = generate_novel_alternatives_prompt(dp, dimension, writing_type, user_goals)
                                            response = client.messages.create(
                                                model="claude-opus-4-5-20251101",
                                                max_tokens=8000,
                                                messages=[{"role": "user", "content": prompt}],
                                                thinking={"type": "enabled", "budget_tokens": 4000}
                                            )

                                            response_text = ""
                                            for block in response.content:
                                                if block.type == "text":
                                                    response_text = block.text

                                            json_match = re.search(r'\{[\s\S]*\}', response_text)
                                            if json_match:
                                                alt_data = json.loads(json_match.group())
                                                # Randomize order and create display mapping
                                                alts = alt_data.get('alternatives', [])
                                                random.shuffle(alts)
                                                display_mapping = {alts[i]['id']: chr(65 + i) for i in range(len(alts))}  # A, B, C
                                                alternatives_data[dp_id] = {
                                                    'alternatives': alts,
                                                    'display_mapping': display_mapping,
                                                    'dimension': dimension,
                                                    'dimension_description': alt_data.get('dimension_description', ''),
                                                    'content_objective': alt_data.get('content_objective', '')
                                                }
                                        except Exception as e:
                                            st.warning(f"Error generating alternatives for DP#{dp_id}: {str(e)}")

                                    st.session_state.evaluate_novel_alternatives = alternatives_data
                                    st.rerun()

                        # ==================== STEP 4: USER RANKS ALTERNATIVES ====================
                        if st.session_state.evaluate_novel_alternatives:
                            st.divider()
                            st.subheader("Step 4: Rank Alternatives")
                            st.markdown("For each decision point, rank the 3 alternatives from most to least preferred.")

                            rankings_complete = len(st.session_state.evaluate_user_rankings) >= len(decision_points)
                            st.progress(len(st.session_state.evaluate_user_rankings) / max(len(decision_points), 1))
                            st.caption(f"Ranked: {len(st.session_state.evaluate_user_rankings)} / {len(decision_points)}")

                            for dp in decision_points:
                                dp_id = dp.get('id')
                                if dp_id not in st.session_state.evaluate_novel_alternatives:
                                    continue

                                alt_info = st.session_state.evaluate_novel_alternatives[dp_id]
                                alts = alt_info['alternatives']
                                display_map = alt_info['display_mapping']
                                already_ranked = dp_id in st.session_state.evaluate_user_rankings

                                with st.expander(f"{'✅' if already_ranked else '⬜'} Decision Point #{dp_id}: {alt_info['dimension']}", expanded=not already_ranked):
                                    # Show context/objective to help user understand what they're ranking
                                    if alt_info.get('content_objective'):
                                        st.info(f"**Context:** {alt_info['content_objective']}")
                                    st.caption(f"**Dimension being varied:** {alt_info.get('dimension_description', alt_info['dimension'])}")

                                    for alt in alts:
                                        display_letter = display_map[alt['id']]
                                        st.markdown(f"**Version {display_letter}:**")
                                        st.markdown(f"> {alt['text']}")
                                        st.markdown("")

                                    existing_ranking = st.session_state.evaluate_user_rankings.get(dp_id, {})
                                    options = ["Select...", "Version A", "Version B", "Version C"]

                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        rank1 = st.selectbox("1st (Best)", options, key=f"rank1_{dp_id}",
                                                           index=options.index(existing_ranking.get('rank1_display', "Select...")))
                                    with col2:
                                        rank2 = st.selectbox("2nd", options, key=f"rank2_{dp_id}",
                                                           index=options.index(existing_ranking.get('rank2_display', "Select...")))
                                    with col3:
                                        rank3 = st.selectbox("3rd (Worst)", options, key=f"rank3_{dp_id}",
                                                           index=options.index(existing_ranking.get('rank3_display', "Select...")))

                                    if st.button("💾 Save Ranking", key=f"save_rank_{dp_id}", type="primary"):
                                        if rank1 != "Select..." and rank2 != "Select..." and rank3 != "Select...":
                                            if len(set([rank1, rank2, rank3])) == 3:
                                                # Convert display letters back to alt_ids
                                                reverse_map = {v: k for k, v in display_map.items()}
                                                ranking_ids = [
                                                    reverse_map.get(rank1.replace("Version ", "")),
                                                    reverse_map.get(rank2.replace("Version ", "")),
                                                    reverse_map.get(rank3.replace("Version ", ""))
                                                ]
                                                st.session_state.evaluate_user_rankings[dp_id] = {
                                                    'ranking': ranking_ids,
                                                    'rank1_display': rank1,
                                                    'rank2_display': rank2,
                                                    'rank3_display': rank3,
                                                    'dimension': alt_info['dimension']
                                                }
                                                st.success("Ranking saved!")
                                                st.rerun()
                                            else:
                                                st.error("Please select different versions for each rank.")
                                        else:
                                            st.error("Please complete all rankings.")

                        # ==================== STEP 5: SCORE & COMPUTE AGREEMENT ====================
                        rankings_complete = st.session_state.evaluate_novel_alternatives and len(st.session_state.evaluate_user_rankings) >= len(decision_points)

                        if rankings_complete and not st.session_state.evaluate_agreement_scores:
                            st.divider()
                            st.subheader("Step 5: Score & Analyze")
                            st.markdown("Score your ranked alternatives against three baselines: the inferred rubric, your stated preferences (Task A Survey), and generic writing quality.")

                            # Check for free-text preferences
                            freetext_prefs = st.session_state.get('survey_responses', {}).get('task_a', {}).get('q4', '')
                            if not freetext_prefs:
                                st.warning("⚠️ Please complete Task A Survey Q4 (your writing preferences) before proceeding.")
                            elif st.button("🔬 Score All & Compute Results", use_container_width=True, type="primary", key="score_all_btn"):
                                rubric_json = json.dumps(coverage_rubric_dict, indent=2)
                                rubric_rankings = {}
                                freetext_rankings = {}
                                generic_rankings = {}

                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                total_steps = len(st.session_state.evaluate_novel_alternatives) * 3

                                step_count = 0
                                for dp_id, alt_info in st.session_state.evaluate_novel_alternatives.items():
                                    alts_for_scoring = [{"id": a['id'], "text": a['text']} for a in alt_info['alternatives']]

                                    # Score with Rubric
                                    status_text.text(f"Scoring DP#{dp_id} with rubric...")
                                    try:
                                        prompt = score_alternatives_with_rubric_prompt(alts_for_scoring, rubric_json)
                                        response = client.messages.create(
                                            model="claude-opus-4-5-20251101",
                                            max_tokens=4000,
                                            messages=[{"role": "user", "content": prompt}]
                                        )
                                        response_text = response.content[0].text if response.content else ""
                                        json_match = re.search(r'\{[\s\S]*\}', response_text)
                                        if json_match:
                                            rubric_rankings[dp_id] = json.loads(json_match.group())
                                    except Exception as e:
                                        st.warning(f"Error scoring DP#{dp_id} with rubric: {e}")
                                    step_count += 1
                                    progress_bar.progress(step_count / total_steps)

                                    # Score with Free-text
                                    status_text.text(f"Scoring DP#{dp_id} with free-text preferences...")
                                    try:
                                        prompt = score_alternatives_with_freetext_prompt(alts_for_scoring, freetext_prefs)
                                        response = client.messages.create(
                                            model="claude-opus-4-5-20251101",
                                            max_tokens=4000,
                                            messages=[{"role": "user", "content": prompt}]
                                        )
                                        response_text = response.content[0].text if response.content else ""
                                        json_match = re.search(r'\{[\s\S]*\}', response_text)
                                        if json_match:
                                            freetext_rankings[dp_id] = json.loads(json_match.group())
                                    except Exception as e:
                                        st.warning(f"Error scoring DP#{dp_id} with free-text: {e}")
                                    step_count += 1
                                    progress_bar.progress(step_count / total_steps)

                                    # Score with Generic
                                    status_text.text(f"Scoring DP#{dp_id} with generic quality...")
                                    try:
                                        prompt = score_alternatives_generic_prompt(alts_for_scoring)
                                        response = client.messages.create(
                                            model="claude-opus-4-5-20251101",
                                            max_tokens=4000,
                                            messages=[{"role": "user", "content": prompt}]
                                        )
                                        response_text = response.content[0].text if response.content else ""
                                        json_match = re.search(r'\{[\s\S]*\}', response_text)
                                        if json_match:
                                            generic_rankings[dp_id] = json.loads(json_match.group())
                                    except Exception as e:
                                        st.warning(f"Error scoring DP#{dp_id} with generic: {e}")
                                    step_count += 1
                                    progress_bar.progress(step_count / total_steps)

                                # Store all rankings
                                st.session_state.evaluate_rubric_rankings = rubric_rankings
                                st.session_state.evaluate_freetext_rankings = freetext_rankings
                                st.session_state.evaluate_generic_rankings = generic_rankings

                                # Compute agreement scores
                                status_text.text("Computing agreement scores...")
                                rubric_tau, rubric_p = compute_rank_correlation(
                                    st.session_state.evaluate_user_rankings,
                                    rubric_rankings
                                )
                                freetext_tau, freetext_p = compute_rank_correlation(
                                    st.session_state.evaluate_user_rankings,
                                    freetext_rankings
                                )
                                generic_tau, generic_p = compute_rank_correlation(
                                    st.session_state.evaluate_user_rankings,
                                    generic_rankings
                                )

                                # Per-dimension breakdown
                                per_dimension = {}
                                for dp_id, user_data in st.session_state.evaluate_user_rankings.items():
                                    dim = user_data.get('dimension', 'unknown')
                                    if dim not in per_dimension:
                                        per_dimension[dim] = {'user': {}, 'rubric': {}, 'freetext': {}, 'generic': {}}
                                    per_dimension[dim]['user'][dp_id] = user_data
                                    if dp_id in rubric_rankings:
                                        per_dimension[dim]['rubric'][dp_id] = rubric_rankings[dp_id]
                                    if dp_id in freetext_rankings:
                                        per_dimension[dim]['freetext'][dp_id] = freetext_rankings[dp_id]
                                    if dp_id in generic_rankings:
                                        per_dimension[dim]['generic'][dp_id] = generic_rankings[dp_id]

                                st.session_state.evaluate_agreement_scores = {
                                    'rubric': {'tau': rubric_tau, 'p_value': rubric_p},
                                    'freetext': {'tau': freetext_tau, 'p_value': freetext_p},
                                    'generic': {'tau': generic_tau, 'p_value': generic_p},
                                    'per_dimension': per_dimension
                                }

                                progress_bar.progress(1.0)
                                status_text.text("Done!")
                                st.rerun()

                        # ==================== STEP 6: SHOW RESULTS ====================
                        if st.session_state.evaluate_agreement_scores:
                            st.divider()
                            st.subheader("Step 6: Results & Analysis")

                            scores = st.session_state.evaluate_agreement_scores

                            # Main results table
                            st.markdown("### Overall Prediction Accuracy")
                            results_data = []
                            for name, data in [('Rubric', scores['rubric']), ('Free-text', scores['freetext']), ('Generic', scores['generic'])]:
                                tau = data.get('tau')
                                p = data.get('p_value')
                                if tau is not None:
                                    interpretation = "Strong" if abs(tau) > 0.6 else "Moderate" if abs(tau) > 0.3 else "Weak"
                                    results_data.append({
                                        'Preference Source': name,
                                        "Kendall's Tau": f"{tau:.3f}" if tau else "N/A",
                                        'p-value': f"{p:.4f}" if p else "N/A",
                                        'Interpretation': interpretation
                                    })

                            if results_data:
                                st.table(results_data)

                            # Inference surplus analysis
                            st.markdown("### Inference Surplus Analysis")
                            rubric_better_count = 0
                            freetext_better_count = 0
                            ties = 0

                            for dp_id in st.session_state.evaluate_user_rankings:
                                user_rank = st.session_state.evaluate_user_rankings[dp_id].get('ranking', [])
                                rubric_rank = st.session_state.evaluate_rubric_rankings.get(dp_id, {}).get('ranking', [])
                                freetext_rank = st.session_state.evaluate_freetext_rankings.get(dp_id, {}).get('ranking', [])

                                if user_rank and rubric_rank and freetext_rank:
                                    # Check if first choice matches
                                    rubric_match = user_rank[0] == rubric_rank[0] if rubric_rank else False
                                    freetext_match = user_rank[0] == freetext_rank[0] if freetext_rank else False

                                    if rubric_match and not freetext_match:
                                        rubric_better_count += 1
                                    elif freetext_match and not rubric_match:
                                        freetext_better_count += 1
                                    else:
                                        ties += 1

                            st.markdown(f"""
                            - **Inference Surplus** (rubric correct, free-text wrong): **{rubric_better_count}** cases
                            - Free-text better than rubric: **{freetext_better_count}** cases
                            - Both agree or both wrong: **{ties}** cases
                            """)

                            # Detailed breakdown
                            with st.expander("📋 Detailed Scoring Breakdown"):
                                for dp_id in st.session_state.evaluate_user_rankings:
                                    st.markdown(f"**Decision Point #{dp_id}**")
                                    user_rank = st.session_state.evaluate_user_rankings[dp_id].get('ranking', [])
                                    rubric_rank = st.session_state.evaluate_rubric_rankings.get(dp_id, {}).get('ranking', [])
                                    freetext_rank = st.session_state.evaluate_freetext_rankings.get(dp_id, {}).get('ranking', [])
                                    generic_rank = st.session_state.evaluate_generic_rankings.get(dp_id, {}).get('ranking', [])

                                    st.markdown(f"- User ranking: {user_rank}")
                                    st.markdown(f"- Rubric ranking: {rubric_rank}")
                                    st.markdown(f"- Free-text ranking: {freetext_rank}")
                                    st.markdown(f"- Generic ranking: {generic_rank}")
                                    st.markdown("---")

                            # Export
                            st.markdown("### Export Data")
                            export_data = {
                                "timestamp": datetime.now().isoformat(),
                                "conversation_file": eval_selected_file,
                                "rubric_version": coverage_rubric_version,
                                "decision_points": decision_points,
                                "user_reflections": st.session_state.evaluate_user_responses,
                                "novel_alternatives": st.session_state.evaluate_novel_alternatives,
                                "user_rankings": st.session_state.evaluate_user_rankings,
                                "rubric_rankings": st.session_state.evaluate_rubric_rankings,
                                "freetext_rankings": st.session_state.evaluate_freetext_rankings,
                                "generic_rankings": st.session_state.evaluate_generic_rankings,
                                "agreement_scores": st.session_state.evaluate_agreement_scores
                            }

                            if st.button("💾 Save Evaluation", type="primary", use_container_width=True, key="save_coverage_eval"):
                                project_id = st.session_state.get('current_project_id')
                                if not project_id:
                                    st.error("No project selected. Please select a project first.")
                                else:
                                    supabase = st.session_state.get('supabase')
                                    if supabase and save_project_data(supabase, project_id, "coverage_evaluation", export_data):
                                        st.success("✅ Evaluation saved successfully!")
                                    else:
                                        st.error("Failed to save evaluation.")

                    else:
                        st.warning("Could not parse decision points. Raw response:")
                        st.text(result.get("raw_response", "No response"))

        else:
            st.info("👆 Select a conversation above to begin evaluation.")

# ============ EVALUATE: SURVEY TAB ============
with tab2:
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
                        st.success("Task A survey saved!")
                    except Exception as e:
                        st.error(f"Failed to save: {e}")
                else:
                    st.success("Task A survey saved locally!")
            else:
                st.success("Task A survey saved locally!")

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
                        st.success("Task B survey saved!")
                    except Exception as e:
                        st.error(f"Failed to save: {e}")
                else:
                    st.success("Task B survey saved locally!")
            else:
                st.success("Task B survey saved locally!")

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
                            st.success("Final review saved!")
                        except Exception as e:
                            st.error(f"Failed to save: {e}")
                    else:
                        st.success("Final review saved locally!")
                else:
                    st.success("Final review saved locally!")

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
