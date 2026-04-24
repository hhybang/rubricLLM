"""Render chat messages that contain <draft> blocks."""
from rubric_writer.imports import *
from rubric_writer.draft_text import (
    parse_draft_content,
    split_draft_into_sentences,
    replace_sentences_in_draft,
)
from rubric_writer.draft_rubric_llm import regenerate_selected_text
from rubric_writer.persistence import get_active_rubric, _auto_save_conversation
from rubric_writer import draft_grading as _draft_grading


def compute_draft_number(messages: list, target_message_id: str) -> int | None:
    """Return the 1-based draft position of `target_message_id` among graded
    assistant drafts in `messages`, matching the enumeration used by
    render_rubric_scores_panel (which iterates `[m for m in messages if
    m["role"] == "assistant" and m.get("draft_grade")]` and calls the last
    one "draft {len(graded)}"). Returns None if the target isn't a graded
    draft (grading still pending, or not a draft message).

    Keeping the numbering rule in one place ensures the "Draft N" label on
    the conversation panel always matches the "draft N" in the scorecard."""
    n = 0
    for m in messages or []:
        if m.get("role") != "assistant":
            continue
        if not m.get("draft_grade"):
            continue
        n += 1
        if str(m.get("message_id") or "") == str(target_message_id):
            return n
    return None


def render_message_with_draft(content: str, message_id: str, wrap_draft_in_expander: bool = False, editable: bool = True, draft_number: int | None = None):
    """
    Render a message that may contain <draft> tags.
    Draft sections are rendered as editable text areas (when editable=True) or read-only text (when editable=False).
    If wrap_draft_in_expander is True, the draft (editable) section is shown inside a collapsed expander.
    draft_number (1-based) is displayed in the draft label so users can cross-reference with the Rubric Scores
    panel's "Latest draft scorecard (draft N)" caption.
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
            _draft_num_prefix = f"Draft {draft_number} — " if draft_number is not None else ""
            if editable:
                _draft_label = f"📝 **{_draft_num_prefix}Your Draft**"
            else:
                _draft_label = f"📝 **{_draft_num_prefix}Draft Preview**"
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

                # --- Preview / edit mode toggle ---
                # Drafts render as plain prose by default so the conversation
                # reads like a conversation. Clicking "Edit" switches this
                # specific draft into the full editable textarea + sentence-
                # rephrase UI (which is what used to render unconditionally).
                # Per-draft state means editing one draft doesn't expand
                # every draft in the thread.
                edit_mode_key = f"draft_edit_mode_{message_id}_{draft_idx}"
                in_edit_mode = st.session_state.get(edit_mode_key, False)

                if not in_edit_mode:
                    # Prose preview: render the draft as-if it were regular
                    # assistant message text. Streamlit's st.markdown preserves
                    # paragraph breaks, lists, and basic formatting, which is
                    # what a "conversation message" should look like.
                    st.markdown(current_value)
                    if st.button(
                        "✏️ Edit",
                        key=f"enter_edit_{edit_key}",
                        help="Open the editable view to edit the draft directly or rephrase selected sentences.",
                    ):
                        st.session_state[edit_mode_key] = True
                        st.rerun()
                    draft_idx += 1
                    continue

                # In edit mode: show a "Done editing" button at the top so
                # the user can collapse back to the prose view. Edits persist
                # regardless of which mode the draft is in.
                if st.button(
                    "✅ Done editing",
                    key=f"exit_edit_{edit_key}",
                    help="Collapse back to the conversation-style view. Your edits are saved.",
                ):
                    st.session_state[edit_mode_key] = False
                    st.rerun()

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
                                save_project_data(st.session_state.get("supabase"), _reph_pid, "inline_rephrase", {
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
                                "message_id": new_message_id,
                            })
                            st.session_state[draft_key][edit_key] = original_draft
                            st.session_state[reset_counter_key] += 1
                            _rb_g, _, _ = get_active_rubric()
                            if _rb_g and _rb_g.get("rubric"):
                                _draft_grading.schedule_background_grade(
                                    supabase=st.session_state.get("supabase"),
                                    conversation_id=st.session_state.get("selected_conversation"),
                                    message_id=new_message_id,
                                    draft_text=edited_draft,
                                    rubric_version=_rb_g.get("version"),
                                    rubric_dict=_rb_g,
                                    trigger="user_edit",
                                )
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
