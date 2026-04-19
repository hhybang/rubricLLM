"""Tab UI — extracted from git HEAD app.py (monolith)."""
from rubric_writer.ui._deps import *


def render_grading_dashboard_tab():
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
