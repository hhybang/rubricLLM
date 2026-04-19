"""Tab UI — extracted from git HEAD app.py (monolith)."""
from rubric_writer.ui._deps import *
from rubric_writer.widget_keys import project_scoped_key


def render_compare_rubrics_tab():
    st.subheader("🔍 Compare Rubrics")
    st.markdown("Select two different rubric versions to see how they would affect the same writing task.")

    # Get rubric history
    _, _, rubric_history = get_active_rubric()

    if len(rubric_history) < 2:
        st.warning("You need at least 2 rubrics to compare. Create more rubric versions first.")
    else:
        _valid_idx = list(range(len(rubric_history)))
        _k_a = project_scoped_key("tab2_rubric_a_select")
        _k_b = project_scoped_key("tab2_rubric_b_select")
        for _rk in (_k_a, _k_b):
            if _rk in st.session_state:
                _rv = st.session_state[_rk]
                if _rv not in _valid_idx:
                    st.session_state.pop(_rk, None)

        rubric_options = [f"v{r.get('version', 1)}" for r in rubric_history]
        col1, col2 = st.columns(2)

        with col1:
            rubric_a_idx = st.selectbox("Rubric A:", options=list(range(len(rubric_history))),
                                        format_func=lambda x: rubric_options[x], key=_k_a)
        with col2:
            rubric_b_idx = st.selectbox("Rubric B:", options=list(range(len(rubric_history))),
                                        format_func=lambda x: rubric_options[x], key=_k_b)

        # Display selected rubrics with mapping summary
        st.markdown("---")

        # Build criteria lists and matching info
        rubric_a_version = rubric_history[rubric_a_idx].get('version', 1)
        rubric_b_version = rubric_history[rubric_b_idx].get('version', 1)
        _src_a = rubric_history[rubric_a_idx].get("source", "")
        _src_b = rubric_history[rubric_b_idx].get("source", "")
        _crit_a = rubric_history[rubric_a_idx].get("rubric", [])
        _crit_b = rubric_history[rubric_b_idx].get("rubric", [])

        # Build name-to-index maps for matching
        _names_a = [(i, c.get("name", "").lower().strip(), c.get("name", "")) for i, c in enumerate(_crit_a)]
        _names_b = {c.get("name", "").lower().strip(): (i, c.get("name", "")) for i, c in enumerate(_crit_b)}
        _names_b_set = set(_names_b.keys())
        _names_a_set = set(n for _, n, _ in _names_a)

        # Build mapping rows for the summary table
        _mapping_rows = []
        _n_unchanged = 0
        _n_modified = 0
        _n_removed = 0
        _n_new = 0
        for _a_i, a_key, a_name in _names_a:
            if a_key in _names_b:
                _b_i, b_name = _names_b[a_key]
                _a_crit = _crit_a[_a_i]
                _b_crit = _crit_b[_b_i]
                _changes = []
                if _a_crit.get("description", "") != _b_crit.get("description", ""):
                    _changes.append("description")
                if _a_crit.get("priority", 0) != _b_crit.get("priority", 0):
                    _changes.append("priority")
                if [d.get("label", "") for d in _a_crit.get("dimensions", [])] != [d.get("label", "") for d in _b_crit.get("dimensions", [])]:
                    _changes.append("dimensions")
                if _changes:
                    _n_modified += 1
                    _mapping_rows.append({
                        f"v{rubric_a_version} Criterion": a_name,
                        "Status": "Modified",
                        f"v{rubric_b_version} Criterion": b_name,
                        "Changes": ", ".join(_changes)
                    })
                else:
                    _n_unchanged += 1
                    _mapping_rows.append({
                        f"v{rubric_a_version} Criterion": a_name,
                        "Status": "Unchanged",
                        f"v{rubric_b_version} Criterion": b_name,
                        "Changes": ""
                    })
            else:
                _n_removed += 1
                _mapping_rows.append({
                    f"v{rubric_a_version} Criterion": a_name,
                    "Status": "Removed",
                    f"v{rubric_b_version} Criterion": "---",
                    "Changes": ""
                })
        for b_key in _names_b:
            if b_key not in _names_a_set:
                _, b_name = _names_b[b_key]
                _n_new += 1
                _mapping_rows.append({
                    f"v{rubric_a_version} Criterion": "---",
                    "Status": "New",
                    f"v{rubric_b_version} Criterion": b_name,
                    "Changes": ""
                })

        # Show summary stats
        _stats_parts = []
        if _n_unchanged:
            _stats_parts.append(f"**{_n_unchanged}** unchanged")
        if _n_modified:
            _stats_parts.append(f"**{_n_modified}** modified")
        if _n_removed:
            _stats_parts.append(f"**{_n_removed}** removed")
        if _n_new:
            _stats_parts.append(f"**{_n_new}** new")
        st.markdown(f"**Criteria Mapping:** {' | '.join(_stats_parts)}")

        # Show mapping table
        if _mapping_rows:
            import pandas as pd
            _map_df = pd.DataFrame(_mapping_rows)

            def _color_status(val):
                colors = {
                    "Unchanged": "background-color: #E8F5E9; color: #2E7D32",
                    "Modified": "background-color: #FFF3E0; color: #E65100",
                    "Removed": "background-color: #FFEBEE; color: #C62828",
                    "New": "background-color: #E3F2FD; color: #1565C0",
                }
                return colors.get(val, "")

            try:
                _styled_df = _map_df.style.map(_color_status, subset=["Status"]).set_properties(**{"text-align": "left"})
            except AttributeError:
                _styled_df = _map_df.style.applymap(_color_status, subset=["Status"]).set_properties(**{"text-align": "left"})
            st.dataframe(_styled_df, width="stretch", hide_index=True)

        # Render 2-column rubric display
        col_rubric_a, col_rubric_b = st.columns(2)

        with col_rubric_a:
            st.markdown(f"### Rubric v{rubric_a_version}")
            display_rubric_criteria(rubric_history[rubric_a_idx], st, comparison_rubric_data=rubric_history[rubric_b_idx])

        with col_rubric_b:
            st.markdown(f"### Rubric v{rubric_b_version}")
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

                    # Save comparison results to database
                    _cmp_project_id = st.session_state.get("current_project_id")
                    if _cmp_project_id:
                        save_project_data(supabase, _cmp_project_id, "rubric_comparison", {
                            "timestamp": datetime.now().isoformat(),
                            "writing_task": compare_input.strip(),
                            "rubric_a": {
                                "version": rubric_a_version,
                                "source": _src_a,
                                "criteria_count": len(_crit_a),
                                "criteria_names": [c.get("name", "") for c in _crit_a],
                            },
                            "rubric_b": {
                                "version": rubric_b_version,
                                "source": _src_b,
                                "criteria_count": len(_crit_b),
                                "criteria_names": [c.get("name", "") for c in _crit_b],
                            },
                            "criteria_mapping": {
                                "unchanged": _n_unchanged,
                                "modified": _n_modified,
                                "removed": _n_removed,
                                "new": _n_new,
                                "details": _mapping_rows,
                            },
                            "comparison_output": {
                                "base_draft": result.get("base_txt", ""),
                                "rubric_a_revision": result.get("a_txt", ""),
                                "rubric_b_revision": result.get("b_txt", ""),
                                "key_differences": result.get("key_diffs", ""),
                                "summary": result.get("summary", ""),
                            },
                        })
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
