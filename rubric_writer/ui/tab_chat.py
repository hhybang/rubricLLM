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
                        if message['role'] == 'assistant':
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
                    if message['role'] == 'assistant':
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

    # Create a container for streaming responses BEFORE chat_input
    # This ensures streaming content appears above the input, not below
    streaming_container = st.container()

    # User input (chat input and buttons) — hidden until preferences are provided
    _no_project = not bool(st.session_state.get("current_project_id"))
    _dim_recognition_pending = (
        st.session_state.get("precision_validation_pending", False)
        and not st.session_state.get("dim_recognition_done", False)
    )
    _chat_blocked = _no_project or _pref_blocked or _dim_recognition_pending

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

