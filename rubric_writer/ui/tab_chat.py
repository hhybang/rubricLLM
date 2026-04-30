"""Tab UI — extracted from git HEAD app.py (monolith)."""
from rubric_writer.ui._deps import *
from rubric_writer.persistence import _auto_save_conversation
from rubric_writer.rubric_display import _build_rubric_version_changelog
from rubric_writer.widget_keys import project_scoped_key
from rubric_writer import draft_grading as _draft_grading
from rubric_writer import draft_grading_ui as _draft_grading_ui


def render_chat_panel():
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
    _conv_widget_key = project_scoped_key("conversation_selector")
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
            st.session_state[_conv_widget_key] = _target
        else:
            st.session_state[_conv_widget_key] = None

    def _on_conversation_selector_change():
        """Callback when user explicitly changes the conversation selector."""
        st.session_state._user_changed_conversation = True

    def _conversation_option_label(x):
        if x is None:
            return "New Conversation"
        for _disp, _fid in conversation_options:
            if _fid == x:
                return _disp
        return "(unknown conversation)"

    _conv_sel_col, _conv_del_col = st.columns([5, 1])
    with _conv_sel_col:
        selected_file = st.selectbox(
            "💬 Select conversation:",
            options=options,
            format_func=_conversation_option_label,
            key=_conv_widget_key,
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
                    _draft_grading_ui.clear_rubric_edit_session_state()
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
            st.session_state.messages = []
            st.session_state.rubric = None
            st.session_state.current_analysis = ""
            st.session_state.selected_conversation = None
            st.session_state.comparison_result = None
            st.session_state.comparison_rubric_version = None
            _draft_grading_ui.clear_rubric_edit_session_state()
            st.session_state.ranking_checkpoint_pending = None
            st.session_state.ranking_checkpoint_auto_triggered = False
            st.session_state.alignment_check_done = False
            st.session_state.alignment_check_skipped = False
            st.rerun()
        elif selected_file and selected_file != st.session_state.selected_conversation:
            # User switched to a different conversation — load it
            conv_data = load_conversation_data(selected_file)
            if conv_data:
                st.session_state.messages = conv_data.get("messages", [])
                st.session_state.rubric = conv_data.get("rubric", None)
                st.session_state.current_analysis = conv_data.get("analysis", "")
                st.session_state.selected_conversation = selected_file
                _draft_grading_ui.clear_rubric_edit_session_state()
                st.rerun()

    st.divider()

    _draft_grading.sync_draft_grades_into_session()
    _draft_grading.flush_pending_conversation_save()
    _draft_grading.maybe_schedule_pending_grades()
    if _draft_grading.grade_poll_fragment_enabled() and _draft_grading.count_pending_draft_grades(
        st.session_state.get("messages")
    ) > 0:
        _draft_grading.run_draft_grade_poll_fragment()

    # Anchor at the top of the chat scroll area so the floating "scroll to
    # top" button has somewhere to land.
    st.markdown('<div id="chat-top"></div>', unsafe_allow_html=True)

    # Display chat messages
    _chat_msg_num = 0  # Track message number matching _build_conversation_text numbering
    for idx, message in enumerate(st.session_state.messages):
        # Increment message counter to stay in sync with _build_conversation_text numbering.
        # _build_conversation_text numbers every message (user/assistant/system) except _synthetic_changelog.
        if not message.get('_synthetic_changelog'):
            _chat_msg_num += 1

        # Skip assessment messages (CHAT_ASSESS_DRAFT_PROMPT and evaluation response) from display
        # They're in conversation history for context but shown only as cards
        if message.get('is_assessment_message'):
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
                        st.caption(f"#{_chat_msg_num}")
                        st.info(display_content)
            elif display_content:
                st.caption(f"#{_chat_msg_num}")
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
                        st.caption(f"#{_chat_msg_num}")
                        message_id = message.get('message_id', f"{message['role']}_{idx}")
                        safe_msg_id = re.sub(r'[^a-zA-Z0-9_-]', '_', str(message_id))
                        if message['role'] == 'user':
                            content_to_display = message['content']
                        else:
                            content_to_display = message.get('display_content', message['content'])
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
                                                                st.session_state[project_scoped_key("rubric_version_selector")] = f"v{_rr_revert_ver}"
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
                                                st.session_state[project_scoped_key("rubric_version_selector")] = f"v{_rr_new_ver}"
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
                                                            st.session_state[project_scoped_key("rubric_version_selector")] = f"v{_rr_revert_ver}"
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
                                            display_rubric_comparison(
                                                suggestion_data.get("current_rubric", suggestion_data.get("edited_rubric", [])),
                                                suggestion_data.get("updated_rubric", suggestion_data.get("modified_rubric", [])),
                                                criterion_reasons=suggestion_data.get("suggestion_reasons"),
                                            )
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
                                if _ac_pending and not (suggestion_data and suggestion_data.get("applied", False)):
                                    _fb_source_key = message.get("_ac_fallback_draft_source", "")
                                    _fb_source_labels = {"rubric": "top-ranked", "generic": "top-ranked", "preference": "top-ranked"}
                                    _fb_source_label = _fb_source_labels.get(_fb_source_key, _fb_source_key)
                                    if not suggestion_data:
                                        # No suggestions generated — auto-inject the fallback draft
                                        _fb_draft = message.get("_ac_fallback_draft", "")
                                        if _fb_draft:
                                            st.session_state.messages.append({
                                                "role": "assistant",
                                                "content": f"Here is your starting draft (your top-ranked draft from the alignment check):\n\n<draft>\n{_fb_draft}\n</draft>",
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
                                                        "content": f"Here is your starting draft (your top-ranked draft from the alignment check):\n\n<draft>\n{_fb_draft}\n</draft>",
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
                                    st.markdown(_rr_non_draft, unsafe_allow_html=False)
                            else:
                                # Always try original content for draft rendering (DP highlighting may corrupt <draft> tags)
                                _draft_source = message.get('content', content_to_display)
                                from rubric_writer.draft_render import compute_draft_number as _compute_draft_number
                                _draft_num = _compute_draft_number(st.session_state.get("messages", []), message_id)
                                has_draft = render_message_with_draft(_draft_source, message_id, editable=True, draft_number=_draft_num)
                                if has_draft:
                                    _draft_grading_ui.render_draft_grading_chrome(message)
                                    _draft_grading_ui.render_drift_panel(message, safe_msg_id)
                                if not has_draft:
                                    st.markdown(content_to_display, unsafe_allow_html=False)
                        else:
                            st.markdown(content_to_display, unsafe_allow_html=False)
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
            else:
                with st.chat_message(message['role']):
                    st.caption(f"#{_chat_msg_num}")
                    message_id = message.get('message_id', f"{message['role']}_{idx}")
                    safe_msg_id = re.sub(r'[^a-zA-Z0-9_-]', '_', str(message_id))
                    if message['role'] == 'user':
                        content_to_display = message['content']
                    else:
                        content_to_display = message.get('display_content', message['content'])
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
                                                            st.session_state[project_scoped_key("rubric_version_selector")] = f"v{_rr_revert_ver2}"
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
                                                st.session_state[project_scoped_key("rubric_version_selector")] = f"v{_rr_new_ver2}"
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
                                                            st.session_state[project_scoped_key("rubric_version_selector")] = f"v{_rr_revert_ver2}"
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
                                        display_rubric_comparison(
                                            suggestion_data.get("current_rubric", suggestion_data.get("edited_rubric", [])),
                                            suggestion_data.get("updated_rubric", suggestion_data.get("modified_rubric", [])),
                                            criterion_reasons=suggestion_data.get("suggestion_reasons"),
                                        )
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
                                _fb_source_labels = {"rubric": "top-ranked", "generic": "top-ranked", "preference": "top-ranked"}
                                _fb_source_label = _fb_source_labels.get(_fb_source_key, _fb_source_key)
                                if not suggestion_data:
                                    # No suggestions generated — auto-inject the fallback draft
                                    _fb_draft = message.get("_ac_fallback_draft", "")
                                    if _fb_draft:
                                        st.session_state.messages.append({
                                            "role": "assistant",
                                            "content": f"Here is your starting draft (your top-ranked draft from the alignment check):\n\n<draft>\n{_fb_draft}\n</draft>",
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
                                                    "content": f"Here is your starting draft (your top-ranked draft from the alignment check):\n\n<draft>\n{_fb_draft}\n</draft>",
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
                                st.markdown(_rr_non_draft2, unsafe_allow_html=False)
                        else:
                            # Always try original content for draft rendering (DP highlighting may corrupt <draft> tags)
                            _draft_source2 = message.get('content', content_to_display)
                            from rubric_writer.draft_render import compute_draft_number as _compute_draft_number
                            _draft_num2 = _compute_draft_number(st.session_state.get("messages", []), message_id)
                            has_draft = render_message_with_draft(_draft_source2, message_id, editable=True, draft_number=_draft_num2)
                            if has_draft:
                                _draft_grading_ui.render_draft_grading_chrome(message)
                                _draft_grading_ui.render_drift_panel(message, safe_msg_id)
                            if not has_draft:
                                st.markdown(content_to_display, unsafe_allow_html=False)
                    else:
                        st.markdown(content_to_display, unsafe_allow_html=False)
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
                    if message['role'] == 'assistant' and message.get('rubric_assessment'):
                        assessment = message['rubric_assessment']
                        draft_text = assessment.get('draft_text')
                        display_rubric_assessment(assessment, message_id, draft_text)


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
                st.info("Generating three draft versions for blind comparison...")
            else:
                st.info("Generating two draft versions for blind comparison...")

            # Generate each draft INDEPENDENTLY (no shared conversation context)
            # so the LLM doesn't anchor on previous drafts
            with st.spinner("Generating draft 1 of 3..." if _rcp_has_3_drafts else "Generating draft 1 of 2..."):
                try:
                    _rcp_pr = GRADING_generate_draft_from_rubric_prompt(_rcp_task, _rcp_rubric_json)
                    _rcp_resp_r = _api_call_with_retry(
                        model=MODEL_LIGHT, max_tokens=1000,
                        messages=[{"role": "user", "content": _rcp_pr}]
                    )
                    _rcp_rubric_draft_text = "".join(b.text for b in _rcp_resp_r.content if b.type == "text").strip()
                    _rcp_drafts["rubric"] = _rcp_rubric_draft_text
                except Exception as _rcp_e2:
                    st.error(f"Failed to generate draft 1: {_rcp_e2}")
                    _rcp_ok = False

            if _rcp_ok:
                with st.spinner("Generating draft 2 of 3..." if _rcp_has_3_drafts else "Generating draft 2 of 2..."):
                    try:
                        _rcp_pg = GRADING_generate_draft_generic_prompt(_rcp_task)
                        _rcp_resp_g = _api_call_with_retry(
                            model=MODEL_LIGHT, max_tokens=1000,
                            messages=[{"role": "user", "content": _rcp_pg}]
                        )
                        _rcp_generic_draft_text = "".join(b.text for b in _rcp_resp_g.content if b.type == "text").strip()
                        _rcp_drafts["generic"] = _rcp_generic_draft_text
                    except Exception as _rcp_e3:
                        st.error(f"Failed to generate draft 2: {_rcp_e3}")
                        _rcp_ok = False

            if _rcp_ok and _rcp_has_3_drafts:
                with st.spinner("Generating draft 3 of 3..."):
                    try:
                        _rcp_pp = GRADING_generate_draft_from_coldstart_prompt(_rcp_task, _rcp_coldstart_text)
                        _rcp_resp_p = _api_call_with_retry(
                            model=MODEL_LIGHT, max_tokens=1000,
                            messages=[{"role": "user", "content": _rcp_pp}]
                        )
                        _rcp_pref_draft_text = "".join(b.text for b in _rcp_resp_p.content if b.type == "text").strip()
                        _rcp_drafts["preference"] = _rcp_pref_draft_text
                    except Exception as _rcp_e4:
                        st.error(f"Failed to generate draft 3: {_rcp_e4}")
                        # Fall back to 2-draft mode
                        _rcp_has_3_drafts = False

            # Assemble pipeline context for downstream diagnostic (after all drafts generated)
            if _rcp_drafts.get("rubric"):
                _rcp_pipeline_msgs.append({"role": "user", "content": _rcp_pr})
                _rcp_pipeline_msgs.append({"role": "assistant", "content": _rcp_drafts["rubric"]})
            if _rcp_drafts.get("generic"):
                _rcp_pipeline_msgs.append({"role": "user", "content": _rcp_pg})
                _rcp_pipeline_msgs.append({"role": "assistant", "content": _rcp_drafts["generic"]})
            if _rcp_drafts.get("preference"):
                _rcp_pipeline_msgs.append({"role": "user", "content": _rcp_pp})
                _rcp_pipeline_msgs.append({"role": "assistant", "content": _rcp_drafts["preference"]})

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
                # Debug: log blind label mapping to terminal
                print(f"[ALIGNMENT CHECK] Blind label mapping: {_rcp_shuffle_order}")
                for _dbg_label, _dbg_src in _rcp_shuffle_order:
                    print(f"  Draft {_dbg_label} = {_dbg_src}")

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

            if True:  # Always 3-draft mode (alignment check only at conversation start with coldstart prefs)
                st.info("We generated three drafts of your writing task using different approaches. **Rank them from best to worst** — your ranking helps us check whether your rubric is capturing what you actually want.")
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

                    # Build blind ranking display (Draft A > Draft B > Draft C)
                    _source_to_label = {src: lab for lab, src in _rcp_shuffle}
                    _ranking_display = " > ".join(
                        f"**Draft {_source_to_label[s]}**" for s in _user_ranking
                    )
                    st.info(f"Your ranking: {_ranking_display}")

                    # Run diagnostic analysis (skip suggestions if rubric draft won)
                    _rcp_rubric_won = (_user_ranking[0] == "rubric")
                    _rcp_result = None
                    _diag_status_placeholder = st.empty()
                    try:
                        _diag_status_placeholder.info("⏳ Scoring each draft against your rubric criteria...")
                        def _diag_status_cb(msg):
                            _diag_status_placeholder.info(f"⏳ {msg}")
                        _rcp_result = _process_alignment_diagnostic(
                            _rcp, _user_ranking, _rcp_reason,
                            status_callback=_diag_status_cb,
                            pipeline_messages=_rcp.get("pipeline_messages"),
                            skip_suggestions=_rcp_rubric_won,
                        )
                    except Exception as _rcp_err:
                        st.error(f"Error during diagnostic: {_rcp_err}")
                    _diag_status_placeholder.empty()

                    if _rcp_result:
                        # Build scoring breakdown content
                        _diag_parts = ["**Rubric Alignment Diagnostic**\n"]
                        _diag_parts.append(f"Your ranking: {_ranking_display}\n")
                        _diag_scoring_parts = []

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

                        _rcp_result["ranking_display"] = _ranking_display
                        _rcp_result["ranking_takeaway"] = ""

                        _preferred_source = _user_ranking[0]
                        _preferred_draft_text = _rcp.get("drafts", {}).get(_preferred_source, "")
                        _preferred_blind_label = f"Draft {_source_to_label.get(_preferred_source, '?')}"
                        _ac_writing_task = _rcp.get("writing_task", "")

                        # --- Branch: rubric draft is #1 vs not ---
                        _rubric_won = (_preferred_source == "rubric")
                        _ac_new_version = None

                        if not _rubric_won and _rcp_result.get("suggested_rubric"):
                            # Auto-apply rubric suggestions
                            _ac_suggested = _rcp_result["suggested_rubric"]
                            _ac_hist = load_rubric_history()
                            _ac_next_ver = next_version_number()
                            _ac_hist.append({
                                "version": _ac_next_ver,
                                "rubric": copy.deepcopy(_ac_suggested),
                                "source": "alignment_check_auto",
                                "conversation_id": st.session_state.get("selected_conversation"),
                            })
                            _ac_db_ver = save_rubric_history(_ac_hist)
                            _ac_new_version = _ac_db_ver if _ac_db_ver is not None else _ac_next_ver
                            st.session_state.rubric = copy.deepcopy(_ac_suggested)
                            st.session_state.editing_criteria = copy.deepcopy(_ac_suggested)
                            st.session_state.editing_criteria_ui_version = st.session_state.get("editing_criteria_ui_version", 0) + 1
                            st.session_state[project_scoped_key("rubric_version_selector")] = f"v{_ac_new_version}"

                        # Build diagnostic message
                        _diag_msg = {
                            "role": "assistant",
                            "content": _diag_content,
                            "display_content": _diag_display_content,
                            "is_system_generated": True,
                            "is_alignment_diagnostic": True,
                            "diagnostic_data": _rcp_result,
                            "preferred_draft_text": _preferred_draft_text,
                            "preferred_draft_label": _preferred_blind_label,
                            "message_id": f"diag_result_{int(time.time() * 1000000)}",
                        }

                        # Attach rubric changes info if rubric was auto-updated
                        if not _rubric_won and _rcp_result.get("suggested_rubric"):
                            _diag_msg["rubric_suggestion"] = {
                                "current_rubric": _rcp_rb_dict.get("rubric", []) if _rcp_rb_dict else [],
                                "updated_rubric": _rcp_result["suggested_rubric"],
                                "suggestion_text": _rcp_result.get("suggestion_text", ""),
                                "suggestion_reasons": _rcp_result.get("suggestion_reasons", {}),
                                "applied": True,
                                "applied_version": _ac_new_version,
                            }

                        # Inject messages: user task → diagnostic → draft
                        st.session_state.messages.append({
                            "role": "user",
                            "content": _ac_writing_task,
                        })
                        st.session_state.messages.append(_diag_msg)

                        # Inject the preferred draft as an editable draft message
                        if _rubric_won:
                            _ac_draft_intro = "Here is your starting draft:\n\n"
                        else:
                            _ver_label = f" (to v{_ac_new_version})" if _ac_new_version else ""
                            _ac_draft_intro = f"To better align our rubric with your preferences, we updated your rubric{_ver_label}. Here is your starting draft:\n\n"
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"{_ac_draft_intro}<draft>\n{_preferred_draft_text}\n</draft>",
                            "message_id": f"ac_draft_{int(time.time() * 1000000)}",
                        })

                        st.session_state.alignment_check_done = True
                        st.session_state.ranking_checkpoint_pending = None
                        _auto_save_conversation()
                        st.rerun()

            else:
                pass  # 2-draft fallback removed — alignment check always uses 3 drafts

    # Drain any refiner work queued by drift-panel button clicks. Done AFTER
    # the message loop so spinners/captions the refiner emits appear below
    # the conversation instead of inside the drift expander that triggered
    # them -- which was causing the drift panel to visually "disappear"
    # while the user was still looking at their in-progress feedback.
    _draft_grading_ui.run_deferred_refiner_work()

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

    # --- Preference prompt: require writing preferences before first message if no rubric exists ---
    _pref_has_project = bool(st.session_state.get("current_project_id"))
    _pref_rubric_hist = load_rubric_history() if _pref_has_project else []
    _pref_has_rubric = len(_pref_rubric_hist) > 0
    _pref_has_messages = len(st.session_state.messages) > 0
    _pref_blocked = False  # No upfront preference gate -- system discovers preferences through conversation

    # --- Alignment check gate: DISABLED ---
    _alignment_check_needed = False

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
    _dim_recognition_pending = (
        st.session_state.get("precision_validation_pending", False)
        and not st.session_state.get("dim_recognition_done", False)
    )
    _chat_blocked = _no_project or _pref_blocked or _alignment_check_needed or _ac_draft_pending or _rcp_active or _dim_recognition_pending

    # Render dimension recognition UI if pending (styled as system message)
    if _dim_recognition_pending:
        _recog_rubric = st.session_state.get("precision_validation_rubric")
        if _recog_rubric:
            from rubric_writer.metrics import render_dimension_recognition
            with st.chat_message("assistant", avatar="🔍"):
                render_dimension_recognition(_recog_rubric)
            if st.session_state.get("dim_recognition_done"):
                st.session_state.precision_validation_pending = False
                st.rerun()

    if _no_project:
        st.info("Create a project first to start writing. Use the **sidebar** to create a new project.")

    # Anchor at the bottom of the chat stream (just above the chat input) so
    # the floating "scroll to bottom" button has somewhere to land.
    st.markdown('<div id="chat-bottom"></div>', unsafe_allow_html=True)

    # Floating up/down nav in the lower-right corner. Uses plain anchor links
    # so the browser handles scrolling natively -- no JS, works regardless of
    # whether Streamlit renders in an iframe or the main document.
    st.markdown(
        """
<style>
.chat-nav-floater {
    position: fixed;
    right: 24px;
    bottom: 96px;
    z-index: 999;
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.chat-nav-floater a {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.95);
    border: 1px solid #d0d0d0;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #333;
    font-size: 16px;
    font-weight: bold;
    text-decoration: none;
    transition: background 0.15s, transform 0.1s;
}
.chat-nav-floater a:hover {
    background: #f0f0f0;
    transform: scale(1.05);
    text-decoration: none;
    color: #000;
}
</style>
<div class="chat-nav-floater">
    <a href="#chat-top" title="Scroll to top">↑</a>
    <a href="#chat-bottom" title="Scroll to bottom">↓</a>
</div>
        """,
        unsafe_allow_html=True,
    )

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

        # Pre-compute draft numbering so we can tag each assistant draft with
        # its 1-based draft number. Users frequently say "fix X in draft #3"
        # and the model has to figure out which assistant message that is --
        # with interleaved user feedback, system messages, and assistant
        # responses that aren't drafts, the counting is error-prone. Tagging
        # each draft message with a `[This is Draft #N]` prefix before we
        # send it removes the ambiguity.
        #
        # CRITICAL: the numbering MUST match what `compute_draft_number`
        # returns, which is what the UI displays as "Draft N" on every chat
        # panel and in the scorecard. That function counts assistant
        # messages with a `draft_grade` attached (i.e. grading completed).
        # We use it as the single source of truth so the user's "Draft 3"
        # and the model's "Draft #3" are always the same message.
        from rubric_writer.draft_render import compute_draft_number as _compute_draft_number
        _draft_number_by_mid: dict[str, int] = {}
        for _msg in st.session_state.messages:
            if _msg.get('role') != 'assistant':
                continue
            _mid = str(_msg.get('message_id') or '')
            if not _mid:
                continue
            _dn = _compute_draft_number(st.session_state.messages, _mid)
            if _dn is not None:
                _draft_number_by_mid[_mid] = _dn

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

                # Tag assistant drafts with their 1-based draft number so the
                # model can resolve references like "draft #3" without having
                # to count interleaved messages. The tag sits before the draft
                # body so it's visible to the model but doesn't affect the
                # <draft> extraction regex, which scans for `<draft>...</draft>`
                # inside the content.
                #
                # `_draft_number_by_mid` already holds only the mids for
                # graded drafts (numbering matches `compute_draft_number`,
                # which is what the UI shows), so a hit here implies both
                # (a) this message has a <draft> body and (b) grading is
                # complete. Ungraded drafts get no number -- same as the UI.
                if msg['role'] == 'assistant':
                    _mid_str = str(msg.get('message_id') or '')
                    _dn = _draft_number_by_mid.get(_mid_str)
                    if _dn is not None:
                        content_to_send = (
                            f"[This is Draft #{_dn}.]\n\n" + content_to_send
                        )

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
                            thinking={"type": "adaptive"},
                            tools=[{
                                "type": "web_search_20250305",
                                "name": "web_search",
                                "max_uses": 5,
                            }],
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

                        # --- Strip any `[This is Draft #N.]` tag the model echoed ---
                        # These tags are injected by the system before each past
                        # draft the model sees, so the model can resolve "draft #N"
                        # references. The model occasionally copies the pattern
                        # into its own output, which leaks internal context to the
                        # user. The system prompt already tells the model NOT to
                        # add the tag, but we strip as a belt-and-suspenders guard.
                        main_content = re.sub(
                            r'^\s*\[This is Draft #\d+\.\]\s*\n*',
                            '',
                            main_content,
                        ).strip()

                        has_draft_tag = bool(re.search(r'<draft>.*?</draft>', main_content, re.DOTALL))

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

                        if has_draft_tag and active_rubric_dict and active_rubric_dict.get("rubric"):
                            _grade_draft = _draft_grading.extract_primary_draft_text(main_content)
                            if _grade_draft:
                                _draft_grading.schedule_background_grade(
                                    supabase=st.session_state.get("supabase"),
                                    conversation_id=st.session_state.get("selected_conversation"),
                                    message_id=message_id,
                                    draft_text=_grade_draft,
                                    rubric_version=rubric_version
                                    or active_rubric_dict.get("version"),
                                    rubric_dict=active_rubric_dict,
                                    trigger="user_message",
                                )

                        # Clear the feedback after successful response
                        if feedback_context:
                            st.session_state.assessment_feedback = {}

                        # Update analysis in session state and rerun to show in sidebar
                        st.session_state.current_analysis = analysis_content

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
                    # Include all visible conversation messages (user, assistant, system)
                    # so the model sees the full picture: drafts, rubric changes,
                    # alignment checks, user feedback, accept/revert decisions, etc.
                    _infer_filtered = [
                        m for m in copy.deepcopy(st.session_state.messages)
                        if m.get('role') in ('user', 'assistant', 'system')
                        and not m.get('is_assessment_message')
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

                    with st.spinner("Inferring rubric from conversation..."):
                        rubric_data = infer_rubric_only(conversation_for_infer)
                        if rubric_data:
                            _rb_ver = rubric_data.get("version", "?")

                            # Mark all existing drafts as pre-rubric so they don't get graded
                            for _m in st.session_state.messages:
                                if _m.get("role") == "assistant":
                                    _m["_pre_rubric"] = True

                            st.session_state.messages.append({
                                "role": "system",
                                "content": f"Rubric **v{_rb_ver}** inferred from your conversation.",
                                "conversation_id": st.session_state.get("selected_conversation"),
                            })

                            # Trigger dimension recognition validation (RQ1 metrics)
                            st.session_state.precision_validation_pending = True
                            st.session_state.precision_validation_rubric = rubric_data
                            st.session_state.dim_recognition_done = False
                            st.session_state.dim_recognition_results = {}

                            _auto_save_conversation()
                            st.success(f"Rubric v{_rb_ver} inferred.")
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



def render_chat_sidebar():
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
            _draft_grading_ui.clear_rubric_edit_session_state()
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

            st.session_state.ranking_checkpoint_results = []
            st.session_state.ranking_checkpoint_pending = None
            st.session_state.ranking_checkpoint_auto_triggered = False
            st.session_state.alignment_check_done = False
            st.session_state.alignment_check_skipped = False
            if _supabase and _new_pid:
                try:
                    _rk_loaded = load_project_data(_supabase, _new_pid, "alignment_diagnostic")
                    if isinstance(_rk_loaded, list):
                        st.session_state.ranking_checkpoint_results = _rk_loaded
                except Exception:
                    pass

            st.session_state._pending_clear_widgets_after_project_switch = True
            st.rerun()

        # Ensure current_project_id is set if we have a current project
        if st.session_state.current_project and not st.session_state.current_project_id:
            st.session_state.current_project_id = project_id_map.get(st.session_state.current_project)

        _startup_pid = st.session_state.get('current_project_id')
        _startup_sb = st.session_state.get('supabase')

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

        if _startup_pid and _startup_sb and not st.session_state.get('ranking_checkpoint_results'):
            try:
                _rk_startup = load_project_data(_startup_sb, _startup_pid, "alignment_diagnostic")
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
                            if 'active_rubric_idx' in st.session_state:
                                del st.session_state.active_rubric_idx
                            if 'delete_project_confirm' in st.session_state:
                                del st.session_state.delete_project_confirm
                            st.session_state._pending_clear_widgets_after_project_switch = True
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
                    st.session_state.infer_coldstart_saved = False
                    # Clear the text input and collapse expander
                    if 'new_project_name' in st.session_state:
                        del st.session_state.new_project_name
                    st.session_state.create_project_expanded = False
                    st.session_state._pending_clear_widgets_after_project_switch = True
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.error("Please enter a project name")

    st.divider()

    _draft_grading_ui.render_rubric_edit_suggestions()

    with st.expander("📊 Rubric Scores", expanded=False):
        _draft_grading_ui.render_rubric_scores_panel(st.session_state.get("messages", []))

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
                key=project_scoped_key("general_rubric_selector")
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

    # Version selector. Default = highest-numbered version (active_idx, set by
    # save_rubric_history to len(history)-1 on every save). The widget's
    # persisted value can go stale if a new version is saved by background
    # work (inference, draft-edit refinement) while the user has an older
    # version selected -- in that case the persisted label still points at
    # the old version even though active_rubric_idx has advanced. Drop the
    # widget key when its value disagrees with active_idx so the index= kwarg
    # takes effect and the latest version is shown selected.
    if rubric_history:
        version_options = [f"v{r.get('version', 1)}" for r in rubric_history]
        _rvk = project_scoped_key("rubric_version_selector")
        # Self-healing guard: if active_rubric_idx changed since last render
        # (e.g. background inference, drift apply, or any path that bumped the
        # active version), drop the persisted selector value so the selectbox
        # re-initializes from index=active_idx instead of the stale label.
        # Note: this guard must NOT compare the widget value against active_idx
        # — when the user clicks a different version in the selectbox the new
        # value lands in the widget key BEFORE the handler below has a chance
        # to update active_idx, so a value-vs-active_idx check would
        # mis-classify legitimate user picks as stale and snap them back.
        _last_seen_idx_key = project_scoped_key("rubric_version_selector_last_idx")
        _last_seen_idx = st.session_state.get(_last_seen_idx_key)
        if _last_seen_idx != active_idx:
            st.session_state.pop(_rvk, None)
            st.session_state[_last_seen_idx_key] = active_idx
        # Only drop the widget value if it points at a version that no longer
        # exists in the options (e.g. version was deleted).
        if _rvk in st.session_state and st.session_state[_rvk] not in version_options:
            st.session_state.pop(_rvk, None)
        _vs_kwargs = {"key": _rvk}
        if _rvk not in st.session_state:
            _vs_kwargs["index"] = active_idx if active_idx is not None else len(version_options) - 1
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
            # Include hash suffix to avoid collisions when names truncate identically
            _stable_id = criterion_name.replace(" ", "_").lower()[:40] + f"_{hash(criterion_name) % 10000}"

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
                    if st.button("🗑️ Remove Criterion", key=f"remove_{_stable_id}_{version_key}_{ui_ver}"):
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

        # Warn the user that unsaved edits are NOT applied in chat. The
        # generator only sees the saved version (via get_active_rubric), so
        # typing edits in the Configuration UI without clicking Save Version
        # leaves the chat using the old rubric -- which can be confusing.
        # This keeps the invariant: "the rubric the chat sees is exactly the
        # one the user can see has been saved."
        if has_changes:
            _active_ver_for_warning = (
                rubric_history[active_idx].get("version", active_idx + 1)
                if rubric_history and active_idx is not None else "?"
            )
            st.warning(
                f"⚠️ **Unsaved rubric edits.** The chat is still using the saved "
                f"rubric (**v{_active_ver_for_warning}**) — your edits here won't "
                f"affect draft generation until you click **Save Version**."
            )

        # Save Version, Revert buttons
        save_col, reset_col = st.columns(2)
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
                    db_version = save_rubric_history(hist)
                    if db_version is not None:
                        new_version = db_version

                    # Clear the version-selectbox widget key so the selectbox
                    # re-initializes from `index=active_idx` on the next
                    # render. Setting the key directly raises
                    # StreamlitAPIException because the selectbox was already
                    # instantiated earlier in this run; popping is safe.
                    _rvk_save = project_scoped_key("rubric_version_selector")
                    st.session_state.pop(_rvk_save, None)

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

                    # Update session state (active_rubric_idx already set by save_rubric_history)
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
        if st.button("🗑️ Delete Version", width="stretch", type="secondary", disabled=(active_idx is None)):
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
                        _rvk_del = project_scoped_key("rubric_version_selector")
                        if _rvk_del in st.session_state:
                            del st.session_state[_rvk_del]
                    else:
                        # Deleted the last version — clear rubric state
                        st.session_state.active_rubric_idx = 0
                        st.session_state.rubric = []
                        st.session_state.editing_criteria = []
                        _rvk_del = project_scoped_key("rubric_version_selector")
                        if _rvk_del in st.session_state:
                            del st.session_state[_rvk_del]

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

