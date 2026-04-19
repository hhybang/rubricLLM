"""Tab UI — extracted from git HEAD app.py (monolith)."""
from rubric_writer.ui._deps import *


def render_evaluate_grade_tab():
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
                                    thinking={"type": "adaptive"}
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
                                        thinking={"type": "adaptive"}
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
                                        thinking={"type": "adaptive"}
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
                                            thinking={"type": "adaptive"}
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
