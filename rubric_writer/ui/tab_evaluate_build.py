"""Tab UI — extracted from git HEAD app.py (monolith)."""
from rubric_writer.ui._deps import *


def render_evaluate_build_tab():
    st.header("🔨 Evaluate: Build")
    st.markdown("Measure whether user rubric editing improves outcomes compared to inference-only rubrics.")
    
    # Load rubric history
    build_rubric_history = load_rubric_history()
    
    if not build_rubric_history or len(build_rubric_history) < 2:
        st.warning("You need at least 2 rubric versions to use this tab. Edit your rubric to create additional versions.")
    else:
        _n_br = len(build_rubric_history)
        for _attr in ("build_rubric_a_idx", "build_rubric_b_idx"):
            _v = st.session_state.get(_attr)
            if _v is not None and (not isinstance(_v, int) or _v < 0 or _v >= _n_br):
                st.session_state[_attr] = None

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
                                        thinking={"type": "adaptive"}
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
                                            thinking={"type": "adaptive"}
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
                                    thinking={"type": "adaptive"}
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
