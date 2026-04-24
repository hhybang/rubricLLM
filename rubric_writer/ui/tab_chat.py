"""Tab UI — extracted from git HEAD app.py (monolith)."""
from rubric_writer.ui._deps import *
from rubric_writer.persistence import _auto_save_conversation, _rubric_to_json_serializable
from rubric_writer.rubric_display import _build_rubric_version_changelog
from rubric_writer.session_reset import reset_evaluate_tab_workflow_state
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
            # print(f"[SELECTOR] User switched to New Conversation")
            st.session_state.messages = []
            st.session_state.rubric = None
            st.session_state.current_analysis = ""
            st.session_state.selected_conversation = None
            st.session_state.comparison_result = None
            st.session_state.comparison_rubric_version = None
            _draft_grading_ui.clear_rubric_edit_session_state()
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
            if "chat_criteria_importance_ranks" in st.session_state:
                del st.session_state.chat_criteria_importance_ranks
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
                _draft_grading_ui.clear_rubric_edit_session_state()
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

    _draft_grading.sync_draft_grades_into_session()
    _draft_grading.flush_pending_conversation_save()
    _draft_grading.maybe_schedule_pending_grades()
    if _draft_grading.grade_poll_fragment_enabled() and _draft_grading.count_pending_draft_grades(
        st.session_state.get("messages")
    ) > 0:
        _draft_grading.run_draft_grade_poll_fragment()

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

    # Anchor at the top of the chat scroll area so the floating "scroll to
    # top" button has somewhere to land.
    st.markdown('<div id="chat-top"></div>', unsafe_allow_html=True)

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
                                    st.markdown(_rr_non_draft, unsafe_allow_html=_dp_highlighted)
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
                    st.caption(f"#{_chat_msg_num}")
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
                                st.markdown(_rr_non_draft2, unsafe_allow_html=_dp_highlighted)
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
    # HUMAN_LOOP_DISABLED: criteria chips + importance ranks + confirm. Re-enable by removing `False and`.
    if False and _dp_has_review and _cc_pending:
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

            # Initialize importance ranks from rubric priorities only once;
            # after that, user edits via number_input widgets are preserved.
            if "chat_criteria_importance_ranks" not in st.session_state:
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
            _cc_existing_ranks = st.session_state.chat_criteria_importance_ranks

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
                        _cr_rank_key = f"cc_rank_{_cr_name}"
                        # Seed widget cache on first render to ensure it matches initialized ranks
                        if _cr_rank_key not in st.session_state:
                            st.session_state[_cr_rank_key] = min(_cr_cur_rank, _cc_total_criteria)
                        _cr_new_rank = st.number_input(
                            "Importance",
                            min_value=1,
                            max_value=_cc_total_criteria,
                            step=1,
                            key=_cr_rank_key,
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
            _cc_non_halluc_ranks = {name: rank for name, rank in _cc_all_ranks.items() if name in _cc_user and _cc_user.get(name) != "hallucinated"}
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
                    st.session_state[project_scoped_key("rubric_version_selector")] = f"v{_cleanup_new_ver}"

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
    _cc_review_active = st.session_state.get("chat_criteria_review_active", False) and not st.session_state.get("chat_criteria_review_confirmed", False)
    # [DISABLED] DP review no longer blocks chat — DP extraction removed for now
    _dp_review_pending = False
    # _dp_review_pending = (
    #     st.session_state.get("infer_decision_points") is not None
    #     and any(m.get('is_dp_review') for m in st.session_state.get("messages", []))
    #     and not st.session_state.get("infer_dp_dimension_confirmed", False)
    # )
    _dim_recognition_pending = (
        st.session_state.get("precision_validation_pending", False)
        and not st.session_state.get("dim_recognition_done", False)
    )
    _chat_blocked = _no_project or _pref_blocked or _alignment_check_needed or _ac_draft_pending or _rcp_active or _cc_review_active or _dp_review_pending or _dim_recognition_pending

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
                            thinking={"type": "adaptive"}
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
                            if "chat_criteria_importance_ranks" in st.session_state:
                                del st.session_state.chat_criteria_importance_ranks

                            # Remove existing DP review messages
                            st.session_state.messages = [m for m in st.session_state.messages if not m.get('is_dp_review')]

                            # System message announcing the inferred rubric
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

                            # --- HUMAN_LOOP_DISABLED: `is_dp_review` bubble + Step 1b LLM classification ---
                            # Re-enable by uncommenting and restoring `chat_criteria_review_active = True` below.
                            #
                            # _rb_criteria_log = rubric_data.get("rubric", [])
                            # _rb_review_content = f"**Rubric v{_rb_ver} inferred.** Review criteria classifications below."
                            # st.session_state.messages.append({
                            #     "role": "assistant",
                            #     "content": _rb_review_content,
                            #     "display_content": _rb_review_content,
                            #     "is_dp_review": True,
                            #     "is_system_generated": True,
                            #     "message_id": f"dp_review_{int(time.time() * 1000000)}",
                            #     "dp_data": {
                            #         "decision_points": [],
                            #         "rubric_version": _rb_ver,
                            #         "rubric": copy.deepcopy(_rb_criteria_log),
                            #     },
                            # })
                            #
                            # _cs_text_infer = st.session_state.get("infer_coldstart_text", "").strip()
                            # if _cs_text_infer and rubric_data.get("rubric"):
                            #     _class_conv_text = _build_conversation_text(conversation_for_infer)
                            #     with st.spinner("Classifying criteria against your writing preferences..."):
                            #         try:
                            #             _class_rubric_json = json.dumps(rubric_data.get("rubric", []), ensure_ascii=False, indent=2)
                            #             _class_prompt = RUBRIC_compare_to_coldstart_prompt(_class_rubric_json, _cs_text_infer, _class_conv_text)
                            #             _class_response = _api_call_with_retry(
                            #                 model=MODEL_PRIMARY,
                            #                 max_tokens=16000,
                            #                 messages=[{"role": "user", "content": _class_prompt}],
                            #                 thinking={"type": "adaptive"},
                            #             )
                            #             _class_text = ""
                            #             for block in _class_response.content:
                            #                 if block.type == "text":
                            #                     _class_text += block.text
                            #             _class_json_match = re.search(r"\{[\s\S]*\}", _class_text)
                            #             if _class_json_match:
                            #                 _class_parsed = json.loads(_class_json_match.group())
                            #                 st.session_state.chat_criteria_llm_classification = _class_parsed
                            #                 _class_user = {}
                            #                 for _cc in _class_parsed.get("criteria_comparison", []):
                            #                     _cc_name = _cc.get("criterion_name", "")
                            #                     _cc_status = _cc.get("status", "unstated")
                            #                     _class_user[_cc_name] = "real" if _cc_status == "unstated" else _cc_status
                            #                 st.session_state.chat_criteria_user_classifications = _class_user
                            #                 st.session_state.chat_criteria_review_active = True
                            #                 st.session_state.chat_criteria_review_confirmed = False
                            #         except Exception:
                            #             pass

                            _auto_save_conversation()

                            # Synthetic “classification complete” so chat input stays unblocked and Evaluate: Infer has data.
                            _crit_names_infer = [c.get("name", "") for c in (rubric_data.get("rubric") or []) if c.get("name")]
                            st.session_state.chat_criteria_review_active = False
                            st.session_state.chat_criteria_review_confirmed = True
                            st.session_state.chat_criteria_llm_classification = None
                            st.session_state.chat_criteria_user_classifications = {n: "real" for n in _crit_names_infer}
                            st.session_state.infer_user_categorizations = copy.deepcopy(
                                st.session_state.chat_criteria_user_classifications
                            )
                            st.session_state.infer_categorizations_complete = True
                            _by_pri = sorted(
                                rubric_data.get("rubric") or [],
                                key=lambda c: (c.get("priority", 99), str(c.get("name", ""))),
                            )
                            _imp_rank = [c.get("name") for c in _by_pri if c.get("name")]
                            st.session_state.chat_classification_feedback = {
                                "classifications": {n: "real" for n in _crit_names_infer},
                                "stated_count": 0,
                                "real_count": len(_crit_names_infer),
                                "hallucinated_count": 0,
                                "hallucination_reasons": {},
                                "importance_ranking": _imp_rank,
                            }

                            _rb_after, _, _ = get_active_rubric()
                            _rv_after = _rb_after.get("version", _rb_ver) if _rb_after else _rb_ver
                            _infer_sess = {
                                "messages": copy.deepcopy(st.session_state.get("infer_dp_messages", [])),
                                "decision_points": None,
                                "timestamp": datetime.now().isoformat(),
                                "source_rubric_version": _rv_after,
                                "result_rubric_version": _rv_after,
                                "had_corrections": False,
                                "had_hallucinated": False,
                                "num_messages": len(st.session_state.get("infer_dp_messages", [])),
                                "conversation_id": st.session_state.get("selected_conversation", ""),
                                "classification_feedback": copy.deepcopy(
                                    st.session_state.get("chat_classification_feedback", {})
                                ),
                                "user_categorizations": copy.deepcopy(
                                    st.session_state.get("infer_user_categorizations", {})
                                ),
                            }
                            if "infer_all_conversations" not in st.session_state:
                                st.session_state.infer_all_conversations = []
                            st.session_state.infer_all_conversations.append(_infer_sess)
                            _infer_sb = st.session_state.get("supabase")
                            _infer_pid = st.session_state.get("current_project_id")
                            if _infer_sb and _infer_pid:
                                try:
                                    _infer_sb.table("project_data").delete().eq(
                                        "project_id", _infer_pid
                                    ).eq("data_type", "infer_conversation").execute()
                                    _infer_sb.table("project_data").insert({
                                        "project_id": _infer_pid,
                                        "data_type": "infer_conversation",
                                        "data": json.dumps(st.session_state.infer_all_conversations),
                                        "created_at": datetime.now().isoformat(),
                                    }).execute()
                                except Exception:
                                    pass

                            st.success(
                                f"Rubric v{_rb_ver} inferred. You can keep chatting or open **Evaluate: Infer**."
                            )
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
            st.session_state.probe_draft_counts = {}
            st.session_state.probe_pending = None
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

            reset_evaluate_tab_workflow_state(clear_infer_session=False)
            st.session_state._pending_clear_widgets_after_project_switch = True
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
                            reset_evaluate_tab_workflow_state()
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
                    st.session_state.probe_draft_counts = {}
                    st.session_state.probe_pending = None
                    st.session_state.infer_coldstart_saved = False
                    st.session_state.chat_criteria_llm_classification = None
                    st.session_state.chat_criteria_user_classifications = {}
                    st.session_state.chat_criteria_review_active = False
                    st.session_state.chat_criteria_review_confirmed = False
                    st.session_state.chat_classification_feedback = {}
                    st.session_state.chat_criteria_hallucination_reasons = {}
                    if "chat_criteria_importance_ranks" in st.session_state:
                        del st.session_state.chat_criteria_importance_ranks
                    # Clear the text input and collapse expander
                    if 'new_project_name' in st.session_state:
                        del st.session_state.new_project_name
                    st.session_state.create_project_expanded = False
                    reset_evaluate_tab_workflow_state()
                    st.session_state._pending_clear_widgets_after_project_switch = True
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

    # Version selector
    if rubric_history:
        version_options = [f"v{r.get('version', 1)}" for r in rubric_history]
        _rvk = project_scoped_key("rubric_version_selector")
        if _rvk in st.session_state:
            _rv_sel = st.session_state[_rvk]
            if _rv_sel not in version_options:
                st.session_state.pop(_rvk, None)
        # Only set index when session state hasn't already been set by apply/save actions
        _vs_kwargs = {"key": _rvk}
        if _rvk not in st.session_state:
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
                        # Build conversation history for context
                        _lc_conv_parts = []
                        for _lc_m in st.session_state.messages:
                            _lc_role = _lc_m.get("role", "")
                            _lc_content = _lc_m.get("display_content") or _lc_m.get("content", "")
                            if _lc_role in ("user", "assistant") and _lc_content:
                                _lc_conv_parts.append(f"[{_lc_role.upper()}]: {_lc_content[:2000]}")
                        _lc_conv_history = "\n\n".join(_lc_conv_parts[-20:]) if _lc_conv_parts else None

                        regenerate_result = regenerate_draft_from_rubric_changes(
                            old_rubric_for_diff,
                            st.session_state.editing_criteria,
                            last_draft,
                            conversation_history=_lc_conv_history,
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

