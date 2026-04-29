"""Tab UI — extracted from git HEAD app.py (monolith)."""
from rubric_writer.ui._deps import *
from rubric_writer.widget_keys import project_scoped_key


def render_survey_tab():
    st.header("📋 Evaluate: Survey")
    st.caption("Complete surveys after each task to track your experience")

    # Initialize survey session state
    if "survey_responses" not in st.session_state:
        st.session_state.survey_responses = {
            "task_a": {},
            "task_b": {},
            "final_review": {},
        }

    # Task selection
    survey_task = st.radio(
        "Select which survey to complete:",
        ["Task A (without rubric)", "Task B (with rubric)", "Final Review"],
        horizontal=True,
        key=project_scoped_key("survey_task_select")
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
            key=project_scoped_key("task_a_q1"),
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
            key=project_scoped_key("task_a_q2"),
            label_visibility="collapsed",
            horizontal=True
        )

        # Q3
        st.markdown("**Q3: Is there anything the model kept getting wrong?**")
        task_a["q3"] = st.text_area(
            "What went wrong",
            value=task_a.get("q3", ""),
            placeholder="Describe any recurring issues or misunderstandings...",
            key=project_scoped_key("task_a_q3"),
            label_visibility="collapsed",
            height=100
        )

        if st.button("Save Task A Survey", type="primary", key=project_scoped_key("save_task_a")):
            task_a["completed"] = True
            task_a["timestamp"] = datetime.now().isoformat()
            # Save to database
            project_id = st.session_state.get('current_project_id')
            if project_id:
                supabase = st.session_state.get('supabase')
                if supabase:
                    try:
                        save_project_data(supabase, project_id, "survey_responses", st.session_state.survey_responses)
                        # Also save as separate data type for analyze_results.py
                        save_project_data(supabase, project_id, "survey_task_a", {
                            "q1": task_a.get("q1"),
                            "q2": task_a.get("q2"),
                            "q3": task_a.get("q3", ""),
                            "iteration": 1,
                            "timestamp": task_a["timestamp"],
                        })
                        st.toast("Task A survey saved!")
                    except Exception as e:
                        st.error(f"Failed to save: {e}")
                else:
                    st.toast("Task A survey saved locally!")
            else:
                st.toast("Task A survey saved locally!")
            st.rerun()

    # ============ TASK B SURVEY ============
    elif survey_task == "Task B (with rubric)":
        st.subheader("Task B: With Rubric")
        st.markdown("*Complete this after working on a task where you could see and edit the rubric.*")

        task_b = st.session_state.survey_responses["task_b"]

        # Q1
        st.markdown("**Q1: How well did the model understand what you wanted from the start?**")
        q1_options = ["1 - Not at all", "2", "3", "4", "5 - Perfectly"]
        task_b["q1"] = st.radio(
            "Understanding rating",
            q1_options,
            index=q1_options.index(task_b.get("q1", "3")) if task_b.get("q1") in q1_options else 2,
            key=project_scoped_key("task_b_q1"),
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
            key=project_scoped_key("task_b_q2"),
            label_visibility="collapsed",
            horizontal=True
        )

        # Q3
        st.markdown("**Q3: Is there anything the model kept getting wrong?**")
        task_b["q3"] = st.text_area(
            "What went wrong",
            value=task_b.get("q3", ""),
            placeholder="Describe any recurring issues or misunderstandings...",
            key=project_scoped_key("task_b_q3"),
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
            key=project_scoped_key("task_b_q4"),
            label_visibility="collapsed"
        )

        # Q5
        st.markdown("**Q5: Was having the rubric in the interaction useful?**")
        task_b["q5"] = st.text_area(
            "Rubric usefulness",
            value=task_b.get("q5", ""),
            placeholder="Describe whether and how the rubric was useful...",
            key=project_scoped_key("task_b_q5"),
            label_visibility="collapsed",
            height=100
        )

        if st.button("Save Task B Survey", type="primary", key=project_scoped_key("save_task_b")):
            task_b["completed"] = True
            task_b["timestamp"] = datetime.now().isoformat()
            # Save to database
            project_id = st.session_state.get('current_project_id')
            if project_id:
                supabase = st.session_state.get('supabase')
                if supabase:
                    try:
                        save_project_data(supabase, project_id, "survey_responses", st.session_state.survey_responses)
                        # Also save as separate data type for analyze_results.py
                        _rb_dict_b, _, _ = get_active_rubric()
                        _rb_ver_b = _rb_dict_b.get("version", 1) if _rb_dict_b else 1
                        save_project_data(supabase, project_id, "survey_task_b", {
                            "q1": task_b.get("q1"),
                            "q2": task_b.get("q2"),
                            "q3": task_b.get("q3", ""),
                            "q4": task_b.get("q4", ""),
                            "q5": task_b.get("q5", ""),
                            "iteration": _rb_ver_b,
                            "timestamp": task_b["timestamp"],
                        })
                        st.toast("Task B survey saved!")
                    except Exception as e:
                        st.error(f"Failed to save: {e}")
                else:
                    st.toast("Task B survey saved locally!")
            else:
                st.toast("Task B survey saved locally!")
            st.rerun()

    # ============ FINAL REVIEW SURVEY ============
    elif survey_task == "Final Review":
        st.subheader("Final Review: Rubric Accuracy")
        st.markdown("*Complete this after all tasks are done. Rate how accurately the rubric captures your preferences.*")

        # Get active rubric
        _fr_rubric_dict, _, _ = get_active_rubric()
        _fr_criteria = (_fr_rubric_dict.get("rubric", []) or []) if _fr_rubric_dict else []

        if not _fr_criteria:
            st.warning("No rubric found. Complete at least one task with rubric assistance first.")
        else:
            final_review = st.session_state.survey_responses.setdefault("final_review", {})
            criteria_ratings = final_review.setdefault("criteria_ratings", {})

            st.markdown("**Q1: For each rubric criterion, rate how accurately it captures your preferences.**")

            for i, criterion in enumerate(_fr_criteria):
                crit_name = criterion.get("name", f"Criterion {i+1}")
                crit_desc = criterion.get("description", "No description")

                with st.expander(f"**{crit_name}**", expanded=True):
                    st.caption(crit_desc)

                    rating = criteria_ratings.setdefault(crit_name, {})
                    accuracy_options = ["Accurate", "Partially right", "Inaccurate"]
                    rating["accuracy"] = st.radio(
                        "Accuracy",
                        accuracy_options,
                        index=accuracy_options.index(rating.get("accuracy", "Partially right")) if rating.get("accuracy") in accuracy_options else 1,
                        key=project_scoped_key(f"fr_accuracy_{i}"),
                        horizontal=True,
                        label_visibility="collapsed",
                    )
                    rating["explanation"] = st.text_input(
                        "Brief explanation",
                        value=rating.get("explanation", ""),
                        placeholder="What's right or wrong about this criterion?",
                        key=project_scoped_key(f"fr_explanation_{i}"),
                        label_visibility="collapsed",
                    )

            st.markdown("**Q2: Looking at the full rubric — what got captured well, and what's missing?**")
            st.caption("Call out any preferences the rubric correctly named (especially ones you wouldn't have thought to articulate yourself) AND any preferences you care about that the rubric doesn't cover.")
            final_review["q2"] = st.text_area(
                "What got captured / what's missing",
                value=final_review.get("q2", ""),
                placeholder="What the rubric got right, what it missed...",
                key=project_scoped_key("fr_q2"),
                label_visibility="collapsed",
                height=120,
            )

            st.markdown("**Q3: Did working with the rubric help you get better drafts?**")
            _q3_options = ["1 - Not at all", "2", "3 - Somewhat", "4", "5 - Very much"]
            _q3_current = final_review.get("q3", "3 - Somewhat")
            final_review["q3"] = st.radio(
                "Rubric editing helped",
                _q3_options,
                index=_q3_options.index(_q3_current) if _q3_current in _q3_options else 2,
                key=project_scoped_key("fr_q3"),
                horizontal=True,
                label_visibility="collapsed",
            )

            st.markdown("**Q4: Would you use a system like this again for future writing?**")
            _q4_options = ["1 - Definitely not", "2", "3 - Maybe", "4", "5 - Definitely yes"]
            _q4_current = final_review.get("q4", "3 - Maybe")
            final_review["q4"] = st.radio(
                "Would use again",
                _q4_options,
                index=_q4_options.index(_q4_current) if _q4_current in _q4_options else 2,
                key=project_scoped_key("fr_q4"),
                horizontal=True,
                label_visibility="collapsed",
            )

            if st.button("Save Final Review", type="primary", key=project_scoped_key("save_final_review")):
                final_review["completed"] = True
                final_review["timestamp"] = datetime.now().isoformat()
                final_review["rubric_version"] = _fr_rubric_dict.get("version", 0) if _fr_rubric_dict else 0
                project_id = st.session_state.get('current_project_id')
                if project_id:
                    supabase = st.session_state.get('supabase')
                    if supabase:
                        try:
                            save_project_data(supabase, project_id, "survey_responses", st.session_state.survey_responses)
                            save_project_data(supabase, project_id, "survey_final_review", {
                                "criteria_ratings": criteria_ratings,
                                "q2": final_review.get("q2", ""),
                                "q3": final_review.get("q3", ""),
                                "q4": final_review.get("q4", ""),
                                "rubric_version": final_review["rubric_version"],
                                "timestamp": final_review["timestamp"],
                            })
                            st.toast("Final Review saved!")
                        except Exception as e:
                            st.error(f"Failed to save: {e}")
                    else:
                        st.toast("Final Review saved locally!")
                else:
                    st.toast("Final Review saved locally!")
                st.rerun()

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
        fr_done = st.session_state.survey_responses.get("final_review", {}).get("completed", False)
        st.markdown(f"**Final Review:** {'✅ Complete' if fr_done else '⬜ Incomplete'}")

    # Save to database option
    if any([a_done, b_done, fr_done]):
        project_id = st.session_state.get('current_project_id')
        if project_id:
            supabase = st.session_state.get('supabase')
            if supabase:
                if st.button("Save All Surveys to Database", key=project_scoped_key("save_all_surveys")):
                    try:
                        save_project_data(supabase, project_id, "survey_responses", st.session_state.survey_responses)
                        # Also save as separate data types for analyze_results.py
                        _all_sr = st.session_state.survey_responses
                        _rb_dict_all, _, _ = get_active_rubric()
                        _rb_ver_all = _rb_dict_all.get("version", 0) if _rb_dict_all else 0
                        if _all_sr.get("task_a", {}).get("completed"):
                            _ta = _all_sr["task_a"]
                            save_project_data(supabase, project_id, "survey_task_a", {
                                "q1": _ta.get("q1"), "q2": _ta.get("q2"), "q3": _ta.get("q3", ""),
                                "iteration": 1, "timestamp": _ta.get("timestamp", datetime.now().isoformat()),
                            })
                        if _all_sr.get("task_b", {}).get("completed"):
                            _tb = _all_sr["task_b"]
                            save_project_data(supabase, project_id, "survey_task_b", {
                                "q1": _tb.get("q1"), "q2": _tb.get("q2"), "q3": _tb.get("q3", ""),
                                "q4": _tb.get("q4", ""), "q5": _tb.get("q5", ""), "q6": _tb.get("q6", ""),
                                "iteration": _rb_ver_all, "timestamp": _tb.get("timestamp", datetime.now().isoformat()),
                            })
                        if _all_sr.get("final_review", {}).get("completed"):
                            _fr = _all_sr["final_review"]
                            save_project_data(supabase, project_id, "survey_final_review", {
                                "criteria_ratings": _fr.get("criteria_ratings", {}),
                                "q2": _fr.get("q2", ""),
                                "q3": _fr.get("q3", ""),
                                "q4": _fr.get("q4", ""),
                                "rubric_version": _fr.get("rubric_version", _rb_ver_all),
                                "timestamp": _fr.get("timestamp", datetime.now().isoformat()),
                            })
                        st.success("All survey responses saved to database!")
                    except Exception as e:
                        st.error(f"Failed to save: {e}")
