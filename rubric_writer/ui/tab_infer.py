"""Tab UI — extracted from git HEAD app.py (monolith)."""
from rubric_writer.ui._deps import *
from rubric_writer.widget_keys import project_scoped_key


def render_infer_tab():
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
                if "chat_criteria_importance_ranks" in st.session_state:
                    del st.session_state.chat_criteria_importance_ranks
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
                _n_infer_opts = len(_infer_session_options)
                _isk = project_scoped_key("infer_session_selector")
                if _isk in st.session_state:
                    _prev_ik = st.session_state[_isk]
                    if _prev_ik not in range(_n_infer_opts):
                        st.session_state.pop(_isk, None)

                _selected_session_key = st.selectbox(
                    "Select inference session:",
                    options=range(_n_infer_opts),
                    format_func=lambda i: _infer_session_options[i][0],
                    index=_infer_default_idx,
                    key=_isk,
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
