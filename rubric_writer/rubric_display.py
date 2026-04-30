"""Rubric criteria display, comparison UI, assessments."""
from rubric_writer.imports import *
from rubric_writer.config import MODEL_PRIMARY, client
from rubric_writer.api_client import _api_call_with_retry
from rubric_writer.diff_html import _word_level_diff
from rubric_writer.rubric_edit_log import classify_rubric_edits
from rubric_writer.draft_rubric_llm import regenerate_draft_from_rubric_changes, get_last_draft_from_messages
from rubric_writer.persistence import (
    _auto_save_conversation,
    get_active_rubric,
    load_rubric_history,
    save_rubric_history,
    invalidate_rubric_cache,
    next_version_number,
)
from rubric_writer.widget_keys import project_scoped_key

def display_rubric_criteria(rubric_data, container, comparison_rubric_data=None):
    """
    Display rubric criteria in a user-friendly format with headings, descriptions,
    priority icons, and expandable evidence sections. Criteria are grouped by category.

    If comparison_rubric_data is provided, highlights criteria that differ with ‼️ emoji
    and bolds the specific elements that are different (name, description, priority, dimensions).
    """
    if not rubric_data or 'rubric' not in rubric_data:
        container.warning("No rubric data available")
        return

    # Deep-copy so this display function never mutates the caller's rubric.
    # Previously we attached a `_diff` key with set() values directly to each
    # criterion dict in rubric_data, which leaked into st.session_state and
    # blew up json.dumps the next time the rubric got saved.
    import copy as _copy
    rubric_list = _copy.deepcopy(rubric_data.get('rubric', []))

    if not rubric_list:
        container.info("No criteria defined")
        return

    # Build comparison map for efficient lookup
    comparison_map = {}
    if comparison_rubric_data and 'rubric' in comparison_rubric_data:
        for c in comparison_rubric_data['rubric']:
            name = c.get('name', '').lower().strip()
            comparison_map[name] = c

    # Helper function to get dimension labels as a set for comparison
    def get_dimension_labels(dims):
        return set(d.get('label', '').strip() for d in dims if d.get('label', '').strip())

    # Group criteria by category
    from collections import defaultdict
    categories = defaultdict(list)
    for criterion in rubric_list:
        category = criterion.get('category', 'Uncategorized')

        # Track specific differences for comparison
        criterion['_diff'] = {
            'is_new_criterion': False,
            'name_changed': False,
            'description_changed': False,
            'priority_changed': False,
            'dimensions_changed': False,
            'added_dimensions': set(),
            'removed_dimensions': set()
        }

        if comparison_rubric_data:
            criterion_name = criterion.get('name', '').lower().strip()

            if criterion_name not in comparison_map:
                # Completely new criterion
                criterion['_diff']['is_new_criterion'] = True
            else:
                old_criterion = comparison_map[criterion_name]

                # Check name (case-sensitive comparison for display purposes)
                if criterion.get('name', '') != old_criterion.get('name', ''):
                    criterion['_diff']['name_changed'] = True

                # Check description
                if criterion.get('description', '') != old_criterion.get('description', ''):
                    criterion['_diff']['description_changed'] = True

                # Check priority
                old_priority = old_criterion.get('priority', old_criterion.get('weight', 0))
                new_priority = criterion.get('priority', criterion.get('weight', 0))
                if old_priority != new_priority:
                    criterion['_diff']['priority_changed'] = True

                # Check dimensions
                old_dims = get_dimension_labels(old_criterion.get('dimensions', []))
                new_dims = get_dimension_labels(criterion.get('dimensions', []))
                if old_dims != new_dims:
                    criterion['_diff']['dimensions_changed'] = True
                    criterion['_diff']['added_dimensions'] = new_dims - old_dims
                    criterion['_diff']['removed_dimensions'] = old_dims - new_dims

        categories[category].append(criterion)

    # Display each category group
    for category_name, criteria in categories.items():
        # Start the category box with HTML
        container.markdown(f"""
            <div style="border: 1px solid #2196F3; border-radius: 6px; padding: 8px; margin-bottom: 16px; background-color: rgba(33, 150, 243, 0.1);">
                <div style="font-size: 18px; font-weight: 600; margin-bottom: 8px; color: #2196F3; text-transform: capitalize; text-align: center;">{category_name}</div>
        """, unsafe_allow_html=True)

        # Display criteria within this category
        for criterion in criteria:
            diff = criterion.get('_diff', {})

            # Check if any difference exists
            has_any_diff = (
                diff.get('is_new_criterion') or
                diff.get('name_changed') or
                diff.get('description_changed') or
                diff.get('priority_changed') or
                diff.get('dimensions_changed')
            )

            # Build criterion label
            criterion_name = criterion.get('name', 'Unnamed')
            if has_any_diff:
                criterion_label = f"‼️  {criterion_name}"
            else:
                criterion_label = criterion_name

            with container.expander(criterion_label, expanded=False):
                # Description - bold if changed
                description = criterion.get('description', 'No description provided')
                if diff.get('description_changed'):
                    st.markdown(f"**{description}**")
                else:
                    st.markdown(description)

                # Priority display - bold if changed
                priority = criterion.get('priority', criterion.get('weight', 0))
                if priority > 0:
                    if diff.get('priority_changed'):
                        st.markdown(f"***Priority: #{priority}***")
                    else:
                        st.markdown(f"*Priority: #{priority}*")
                    st.markdown("")

                # Dimensions section (expandable)
                dimensions = criterion.get('dimensions', [])
                added_dims = diff.get('added_dimensions', set())

                if dimensions:
                    dims_label = f"📊 Dimensions ({len(dimensions)})"
                    if diff.get('dimensions_changed'):
                        dims_label = f"**📊 Dimensions ({len(dimensions)})**"

                    with st.expander(dims_label, expanded=False):
                        for dim in dimensions:
                            dim_label = dim.get('label', 'Unnamed dimension')
                            # Bold dimensions that are new/added
                            if dim_label.strip() in added_dims:
                                st.markdown(f"• **{dim_label}**")
                            else:
                                st.markdown(f"• {dim_label}")
                else:
                    st.caption("No dimensions defined")

        # Close the category box
        container.markdown("</div>", unsafe_allow_html=True)
def display_rubric_comparison(current_rubric: list, updated_rubric: list, apply_context: dict = None, criterion_reasons: dict = None):
    """
    Display a comparison of current and updated rubric using collapsible sections.
    Each criterion is shown as an expander with status badge visible when collapsed.
    Word-level diffing highlights specific changes.
    criterion_reasons: optional dict mapping criterion name -> reason string for why the change is suggested.
    If apply_context is provided, adds an "Apply all suggestions" button.
    apply_context = {"safe_msg_id", "message", "message_id"}.
    """
    # Build a map of current criteria by name for comparison
    current_map = {c.get('name', '').lower().strip(): c for c in current_rubric}
    updated_map = {c.get('name', '').lower().strip(): c for c in updated_rubric}
    safe_msg_id = (apply_context or {}).get("safe_msg_id", "")
    message = (apply_context or {}).get("message")
    message_id = (apply_context or {}).get("message_id", "")
    # Build case-insensitive reasons lookup
    _reasons_map = {}
    if criterion_reasons:
        for _rk, _rv in criterion_reasons.items():
            _reasons_map[_rk.lower().strip()] = _rv

    def _find_reason(crit_name_key):
        """Look up reason by exact match, then fuzzy substring match."""
        if crit_name_key in _reasons_map:
            return _reasons_map[crit_name_key]
        # Fuzzy: check if any reason key is contained in the criterion name or vice versa
        for _rk, _rv in _reasons_map.items():
            if _rk in crit_name_key or crit_name_key in _rk:
                return _rv
        return ""


    # CSS for highlighting
    st.markdown("""
    <style>
    .diff-field {
        margin: 8px 0;
        padding: 10px;
        background: #f8f9fa;
        border-radius: 6px;
        border-left: 3px solid #e0e0e0;
    }
    .diff-field-changed {
        border-left: 3px solid #ff9800;
        background: #fff8e1;
    }
    .diff-field-label {
        font-weight: 600;
        font-size: 12px;
        color: #555;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .diff-field-content {
        font-size: 14px;
        color: #333;
        line-height: 1.6;
    }
    .text-removed {
        text-decoration: line-through;
        color: #c62828;
        background-color: rgba(198, 40, 40, 0.1);
        padding: 1px 4px;
        border-radius: 3px;
    }
    .text-added {
        color: #2e7d32;
        background-color: rgba(46, 125, 50, 0.15);
        padding: 1px 4px;
        border-radius: 3px;
    }
    .weight-change {
        font-size: 13px;
        margin-bottom: 10px;
        padding: 6px 10px;
        background: #fff3e0;
        border-radius: 4px;
        display: inline-block;
    }
    .no-change-badge {
        font-size: 12px;
        color: #9e9e9e;
        font-style: italic;
    }
    .status-badge {
        font-size: 11px;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 500;
        margin-left: 8px;
    }
    .status-modified {
        background: #fff3e0;
        color: #e65100;
    }
    .status-new {
        background: #e8f5e9;
        color: #2e7d32;
    }
    .status-removed {
        background: #ffebee;
        color: #c62828;
    }
    .status-unchanged {
        background: #f5f5f5;
        color: #757575;
    }
    .dimension-block {
        margin: 6px 0;
        padding: 8px 12px;
        background: #f5f5f5;
        border-radius: 4px;
        border-left: 3px solid #e0e0e0;
        font-size: 14px;
    }
    .dimensions-container {
        margin-top: 6px;
    }
    .dimensions-section {
        margin-top: 10px;
    }
    .dimensions-section .dimensions-label {
        font-weight: 600;
        font-size: 12px;
        color: #555;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: block;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("#### 📋 Rubric Changes")

    # Show criteria in the updated rubric (maintains order), then removed ones
    for criterion in updated_rubric:
        name = criterion.get('name', 'Unnamed')
        name_key = name.lower().strip()
        priority = criterion.get('priority', criterion.get('weight', 0))

        if name_key not in current_map:
            # NEW criterion
            status_html = '<span class="status-badge status-new">NEW</span>'
            _crit_reason = _find_reason(name_key)
            expander_label = f"➕ {name} (Priority #{priority})"

            with st.expander(expander_label, expanded=False):
                st.markdown(status_html, unsafe_allow_html=True)
                if _crit_reason:
                    st.info(f"💡 {_crit_reason}")
                st.markdown(f"**Priority:** #{priority}")
                desc = criterion.get('description', '') or 'Not specified'
                st.markdown(f"""
                <div class="diff-field">
                    <div class="diff-field-label">Description</div>
                    <div class="diff-field-content"><span class="text-added">{html_module.escape(desc)}</span></div>
                </div>
                """, unsafe_allow_html=True)
                dims = criterion.get('dimensions', [])
                dim_labels = [d.get('label', '') for d in dims if d.get('label')]
                st.markdown('<div class="dimensions-section"><span class="dimensions-label">Dimensions</span><div class="dimensions-container">', unsafe_allow_html=True)
                if dim_labels:
                    for lbl in dim_labels:
                        st.markdown(f'<div class="dimension-block"><span class="text-added">{html_module.escape(lbl)}</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="dimension-block"><span class="text-added">None</span></div>', unsafe_allow_html=True)
                st.markdown('</div></div>', unsafe_allow_html=True)

        elif _criterion_changed(current_map[name_key], criterion):
            # MODIFIED criterion
            old = current_map[name_key]
            old_priority = old.get('priority', old.get('weight', 0))

            status_html = '<span class="status-badge status-modified">MODIFIED</span>'
            _crit_reason = _find_reason(name_key)
            expander_label = f"🔄 {name} (Priority #{priority})"

            with st.expander(expander_label, expanded=False):
                st.markdown(status_html, unsafe_allow_html=True)
                if _crit_reason:
                    st.info(f"💡 {_crit_reason}")

                # Show priority change if applicable
                if old_priority != priority:
                    st.markdown(f"""
                    <div class="priority-change">
                        <strong>Priority:</strong> <span class="text-removed">#{old_priority}</span> → <span class="text-added">#{priority}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"**Priority:** #{priority}")

                # Description
                old_desc = old.get('description', '') or ''
                new_desc = criterion.get('description', '') or ''
                if old_desc != new_desc:
                    diff_html = _word_level_diff(old_desc, new_desc)
                    field_class = "diff-field diff-field-changed"
                else:
                    diff_html = new_desc if new_desc else '<span class="no-change-badge">Not specified</span>'
                    field_class = "diff-field"
                st.markdown(f"""
                <div class="{field_class}">
                    <div class="diff-field-label">Description</div>
                    <div class="diff-field-content">{diff_html}</div>
                </div>
                """, unsafe_allow_html=True)

                # Dimensions (one block per dimension)
                old_dims = [d.get('label', '') for d in old.get('dimensions', [])]
                new_dims = [d.get('label', '') for d in criterion.get('dimensions', [])]
                dims_changed = old_dims != new_dims
                st.markdown('<div class="dimensions-section"><span class="dimensions-label">Dimensions</span><div class="dimensions-container">', unsafe_allow_html=True)
                if new_dims:
                    for lbl in new_dims:
                        if dims_changed and lbl not in old_dims:
                            st.markdown(f'<div class="dimension-block"><span class="text-added">{html_module.escape(lbl)}</span></div>', unsafe_allow_html=True)
                        elif dims_changed and lbl in old_dims:
                            st.markdown(f'<div class="dimension-block">{html_module.escape(lbl)}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="dimension-block">{html_module.escape(lbl)}</div>', unsafe_allow_html=True)
                    if dims_changed:
                        for lbl in old_dims:
                            if lbl not in new_dims:
                                st.markdown(f'<div class="dimension-block"><span class="text-removed">{html_module.escape(lbl)}</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="dimension-block"><span class="no-change-badge">None</span></div>' if not dims_changed else '<div class="dimension-block"><span class="text-removed">None</span></div>', unsafe_allow_html=True)
                st.markdown('</div></div>', unsafe_allow_html=True)

        else:
            # UNCHANGED criterion
            status_html = '<span class="status-badge status-unchanged">UNCHANGED</span>'
            _crit_reason = _find_reason(name_key)
            expander_label = f"⚪ {name} (Priority #{priority})"

            with st.expander(expander_label, expanded=False):
                st.markdown(status_html, unsafe_allow_html=True)
                if _crit_reason:
                    st.success(f"✓ {_crit_reason}")
                st.markdown(f"**Priority:** #{priority}")
                desc = criterion.get('description', '') or '<span class="no-change-badge">Not specified</span>'
                st.markdown(f"""
                <div class="diff-field">
                    <div class="diff-field-label">Description</div>
                    <div class="diff-field-content">{html_module.escape(desc) if isinstance(desc, str) and not desc.startswith('<') else desc}</div>
                </div>
                """, unsafe_allow_html=True)
                dims = criterion.get('dimensions', [])
                dim_labels = [d.get('label', '') for d in dims if d.get('label')]
                st.markdown('<div class="dimensions-section"><span class="dimensions-label">Dimensions</span><div class="dimensions-container">', unsafe_allow_html=True)
                if dim_labels:
                    for lbl in dim_labels:
                        st.markdown(f'<div class="dimension-block">{html_module.escape(lbl)}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="dimension-block"><span class="no-change-badge">None</span></div>', unsafe_allow_html=True)
                st.markdown('</div></div>', unsafe_allow_html=True)

    # Show removed criteria
    for criterion in current_rubric:
        name = criterion.get('name', 'Unnamed')
        name_key = name.lower().strip()

        if name_key not in updated_map:
            priority = criterion.get('priority', criterion.get('weight', 0))
            status_html = '<span class="status-badge status-removed">REMOVED</span>'
            _crit_reason = _find_reason(name_key)
            expander_label = f"❌ {name} (Priority #{priority}) - REMOVED"

            with st.expander(expander_label, expanded=False):
                st.markdown(status_html, unsafe_allow_html=True)
                if _crit_reason:
                    st.info(f"💡 {_crit_reason}")
                st.markdown(f"**Priority:** ~~#{priority}~~")
                desc = criterion.get('description', '') or 'Not specified'
                st.markdown(f"""
                <div class="diff-field">
                    <div class="diff-field-label">Description</div>
                    <div class="diff-field-content"><span class="text-removed">{html_module.escape(desc)}</span></div>
                </div>
                """, unsafe_allow_html=True)
                dims = criterion.get('dimensions', [])
                dim_labels = [d.get('label', '') for d in dims if d.get('label')]
                st.markdown('<div class="dimensions-section"><span class="dimensions-label">Dimensions</span><div class="dimensions-container">', unsafe_allow_html=True)
                if dim_labels:
                    for lbl in dim_labels:
                        st.markdown(f'<div class="dimension-block"><span class="text-removed">{html_module.escape(lbl)}</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="dimension-block"><span class="text-removed">None</span></div>', unsafe_allow_html=True)
                st.markdown('</div></div>', unsafe_allow_html=True)

    # Apply all suggestions (when in apply_context)
    if apply_context and message is not None and updated_rubric:
        # Check if there are actual differences before showing Apply All
        _any_change = False
        _cur_map_check = {c.get('name', '').lower().strip(): c for c in current_rubric}
        _upd_map_check = {c.get('name', '').lower().strip(): c for c in updated_rubric}
        for _uk in _upd_map_check:
            if _uk not in _cur_map_check:
                _any_change = True
                break
            if _criterion_changed(_cur_map_check[_uk], _upd_map_check[_uk]):
                _any_change = True
                break
        if not _any_change:
            for _ck in _cur_map_check:
                if _ck not in _upd_map_check:
                    _any_change = True
                    break
        st.markdown("---")
        _acol, _ = st.columns([0.3, 0.7])
        with _acol:
            if st.button("✅ Apply all", key=f"apply_all_{safe_msg_id}", type="primary", width="stretch", disabled=not _any_change):
                # Merge updated_rubric into the full current rubric (not replace)
                # This handles cases where updated_rubric is a subset (e.g., single criterion from probe)
                full_criteria = list(st.session_state.editing_criteria or [])
                _updated_map = {c.get('name', '').lower().strip(): c for c in updated_rubric}
                _current_names = {c.get('name', '').lower().strip() for c in current_rubric}
                # Update existing criteria that match
                for i, c in enumerate(full_criteria):
                    c_key = (c.get('name') or '').lower().strip()
                    if c_key in _updated_map:
                        full_criteria[i] = copy.deepcopy(_updated_map[c_key])
                # Add new criteria (in updated but not in current)
                for c in updated_rubric:
                    c_key = c.get('name', '').lower().strip()
                    if c_key not in _current_names:
                        full_criteria.append(copy.deepcopy(c))
                # Remove criteria (in current but not in updated — only if current_rubric was the full set)
                if len(current_rubric) == len(full_criteria):
                    _removed_names = _current_names - set(_updated_map.keys())
                    full_criteria = [c for c in full_criteria if (c.get('name') or '').lower().strip() not in _removed_names]
                new_criteria = copy.deepcopy(full_criteria)
                hist = load_rubric_history()
                new_version = next_version_number()
                # Snapshot the prev version so we can compute a diff for the
                # rubric_edit_event log below.
                _prev_rubric_criteria = (hist[-1].get("rubric", []) if hist else [])
                _prev_version = (hist[-1].get("version") if hist else None)
                hist.append({"version": new_version, "rubric": copy.deepcopy(new_criteria), "source": "edit_feedback", "conversation_id": st.session_state.get("selected_conversation")})
                db_version = save_rubric_history(hist)
                # Use DB-assigned version (may differ from local next_version_number)
                if db_version is not None:
                    new_version = db_version

                # P0.4: rubric_edit_event with trigger="manual_config_edit"
                # so post-hoc we can attribute every version bump to the
                # affordance that caused it. edit_summary captures the diff
                # between consecutive versions.
                try:
                    _sb = st.session_state.get("supabase")
                    _pid = st.session_state.get("current_project_id")
                    if _sb and _pid:
                        from auth_supabase import save_project_data as _save_pd
                        _prev_names = {(c.get("name") or "").strip(): c for c in _prev_rubric_criteria}
                        _new_names = {(c.get("name") or "").strip(): c for c in new_criteria}
                        _added = [{"criterion": n} for n in (_new_names.keys() - _prev_names.keys())]
                        _removed = [{"criterion": n} for n in (_prev_names.keys() - _new_names.keys())]
                        _modified = []
                        for n in (_prev_names.keys() & _new_names.keys()):
                            pv = _prev_names[n]
                            nv = _new_names[n]
                            if (pv.get("description") != nv.get("description")
                                    or (pv.get("dimensions") or []) != (nv.get("dimensions") or [])):
                                _modified.append({"criterion": n})
                        from datetime import datetime as _dt
                        _save_pd(_sb, _pid, "rubric_edit_event", {
                            "timestamp": _dt.now().isoformat(),
                            "conversation_id": st.session_state.get("selected_conversation"),
                            "from_version": _prev_version,
                            "to_version": new_version,
                            "trigger": "manual_config_edit",
                            "edit_summary": {
                                "added_dims": _added,
                                "removed_dims": _removed,
                                "modified_dims": _modified,
                            },
                        })
                except Exception:
                    pass
                st.session_state.rubric = new_criteria
                st.session_state.editing_criteria = new_criteria
                st.session_state.editing_criteria_ui_version = st.session_state.get("editing_criteria_ui_version", 0) + 1
                # Force the sidebar version selector to update to the new version
                st.session_state[project_scoped_key("rubric_version_selector")] = f"v{new_version}"
                if message and "rubric_suggestion" in message:
                    message["rubric_suggestion"]["applied"] = True
                    message["rubric_suggestion"]["applied_version"] = new_version
                    message["rubric_version"] = new_version
                if message and "probe_result" in message:
                    message["probe_result"]["applied"] = True
                    message["probe_result"]["applied_version"] = new_version
                    _mark_probe_rubric_updated(message["probe_result"].get("criterion_name", ""))
                # If this is a conversation-start alignment check, inject the suggested draft
                _ac_draft_injected = False
                if message and message.get("_ac_pending_draft"):
                    _ac_sd = message.get("_ac_suggested_draft", "")
                    if _ac_sd:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Here is your starting draft, generated using the **updated rubric** (with the improvements you just applied):\n\n<draft>\n{_ac_sd}\n</draft>",
                            "message_id": f"ac_draft_{int(time.time() * 1000000)}",
                        })
                        _ac_draft_injected = True
                    message["_ac_pending_draft"] = False
                # Save rubric application event to database
                _apply_pid = st.session_state.get("current_project_id")
                if _apply_pid:
                    _apply_source = "rubric_suggestion" if (message and "rubric_suggestion" in message) else "probe_result" if (message and "probe_result" in message) else "unknown"
                    save_project_data(st.session_state.get("supabase"), _apply_pid, "rubric_edit_applied", {
                        "timestamp": datetime.now().isoformat(),
                        "source": _apply_source,
                        "new_rubric_version": new_version,
                        "conversation_id": st.session_state.get("selected_conversation", ""),
                        "message_id": message_id,
                        "previous_rubric": _rubric_list_for_json(current_rubric),
                        "applied_rubric": _rubric_list_for_json(new_criteria),
                        "criteria_changes": {
                            "added": [c.get("name", "") for c in updated_rubric if c.get("name", "").lower().strip() not in {x.get("name", "").lower().strip() for x in current_rubric}],
                            "removed": [c.get("name", "") for c in current_rubric if c.get("name", "").lower().strip() not in {x.get("name", "").lower().strip() for x in updated_rubric}],
                            "modified": [c.get("name", "") for c in updated_rubric if c.get("name", "").lower().strip() in current_map and _criterion_changed(current_map[c.get("name", "").lower().strip()], c)],
                        },
                    })
                    # Also persist as log_changes for analysis
                    _apply_edit_class = classify_rubric_edits(current_rubric, updated_rubric)
                    _apply_old_ver = hist[-2].get("version", 0) if len(hist) >= 2 else 0
                    save_project_data(st.session_state.get("supabase"), _apply_pid, "log_changes", {
                        "timestamp": datetime.now().isoformat(),
                        "source": f"apply_{_apply_source}",
                        "edit_classification": _apply_edit_class,
                        "rubric_version": new_version,
                        "old_version": _apply_old_ver,
                        "draft_regenerated": False,
                    })
                # Use the preview draft from suggestion if available, otherwise regenerate
                # Skip if we already injected a draft from the alignment check
                _apply_last_draft, _ = get_last_draft_from_messages() if not _ac_draft_injected else (None, None)
                _apply_preview = message.get("rubric_suggestion", {}).get("preview_draft") if message else None
                if _apply_preview and _apply_preview.get("revised_draft"):
                    # Use the already-generated preview draft
                    _apply_new_draft = _apply_preview["revised_draft"]
                    _apply_draft_msg = {
                        "role": "assistant",
                        "content": f"Here's an updated draft based on the rubric changes you just applied (v{new_version}):\n\n<draft>{_apply_new_draft}</draft>",
                        "display_content": f"Here's an updated draft based on the rubric changes you just applied (v{new_version}):\n\n<draft>{_apply_new_draft}</draft>",
                        "rubric_version": new_version,
                        "is_system_generated": True,
                        "is_post_apply_draft": True,
                        "post_apply_data": {
                            "previous_draft": _apply_last_draft or "",
                            "new_draft": _apply_new_draft,
                            "rubric_version": new_version,
                        },
                        "message_id": f"post_apply_{int(time.time() * 1000000)}",
                    }
                    st.session_state.messages.append(_apply_draft_msg)
                elif _apply_last_draft and current_rubric:
                    # Fallback: regenerate draft (for non-suggestion Apply All, e.g. probe results)
                    _conv_parts = []
                    for _cm in st.session_state.messages:
                        _cm_role = _cm.get("role", "")
                        _cm_content = _cm.get("display_content") or _cm.get("content", "")
                        if _cm_role in ("user", "assistant") and _cm_content:
                            _conv_parts.append(f"[{_cm_role.upper()}]: {_cm_content[:2000]}")
                    _conv_history = "\n\n".join(_conv_parts[-20:]) if _conv_parts else None

                    _apply_suggestion_text = None
                    if message:
                        _rs = message.get("rubric_suggestion", {})
                        if _rs:
                            _apply_suggestion_text = _rs.get("suggestion_text", "")
                        _dd = message.get("diagnostic_data", {})
                        if not _apply_suggestion_text and _dd:
                            _apply_suggestion_text = _dd.get("suggestion_text", "")

                    _apply_edit_fb = None
                    if message:
                        _rr = message.get("rubric_revision", {})
                        _ann = _rr.get("annotated_changes", []) if _rr else []
                        _fb_parts = []
                        for _fb_ac in _ann:
                            _fb_text = _fb_ac.get("user_feedback", "")
                            if _fb_text and _fb_text.strip():
                                _fb_parts.append(f"- Edit: \"{_fb_ac.get('original_text', '')}\" → \"{_fb_ac.get('new_text', '')}\"\n  User feedback: {_fb_text}")
                        if _fb_parts:
                            _apply_edit_fb = "\n".join(_fb_parts)

                    _apply_regen = regenerate_draft_from_rubric_changes(
                        current_rubric, new_criteria, _apply_last_draft,
                        conversation_history=_conv_history,
                        rubric_suggestion_text=_apply_suggestion_text,
                        user_edit_feedback=_apply_edit_fb,
                    )
                    if _apply_regen and _apply_regen.get("revised_draft") and not _apply_regen.get("error"):
                        _apply_new_draft = _apply_regen["revised_draft"]
                        _apply_draft_msg = {
                            "role": "assistant",
                            "content": f"Here's an updated draft based on the rubric changes you just applied (v{new_version}):\n\n<draft>{_apply_new_draft}</draft>",
                            "display_content": f"Here's an updated draft based on the rubric changes you just applied (v{new_version}):\n\n<draft>{_apply_new_draft}</draft>",
                            "rubric_version": new_version,
                            "is_system_generated": True,
                            "is_post_apply_draft": True,
                            "post_apply_data": {
                                "previous_draft": _apply_last_draft,
                                "new_draft": _apply_new_draft,
                                "rubric_version": new_version,
                            },
                            "message_id": f"post_apply_{int(time.time() * 1000000)}",
                        }
                        st.session_state.messages.append(_apply_draft_msg)
                st.toast(f"All suggestions applied and saved as v{new_version}.")
                _auto_save_conversation()
                st.rerun()
def _criterion_changed(old: dict, new: dict) -> bool:
    """Check if a criterion has changed between old and new versions."""
    if old.get('description') != new.get('description'):
        return True
    if old.get('priority') != new.get('priority'):
        return True
    if old.get('category') != new.get('category'):
        return True
    old_dims = [d.get('label', '') for d in old.get('dimensions', [])]
    new_dims = [d.get('label', '') for d in new.get('dimensions', [])]
    if old_dims != new_dims:
        return True
    return False


def _build_rubric_version_changelog(old_rubric_list, new_rubric_list, old_version, new_version):
    """Build a concise text summary of changes between two rubric versions.

    Returns a string suitable for injecting into api_messages as context.
    """
    old_by_name = {c.get('name', '').strip().lower(): c for c in (old_rubric_list or [])}
    new_by_name = {c.get('name', '').strip().lower(): c for c in (new_rubric_list or [])}

    added = [new_by_name[n] for n in new_by_name if n not in old_by_name]
    removed = [old_by_name[n] for n in old_by_name if n not in new_by_name]
    modified = []
    for name_key in old_by_name:
        if name_key in new_by_name and _criterion_changed(old_by_name[name_key], new_by_name[name_key]):
            modified.append((old_by_name[name_key], new_by_name[name_key]))

    if not added and not removed and not modified:
        return ""

    parts = [f"[Rubric updated from v{old_version} to v{new_version}]", "", "Key changes:"]

    if added:
        names = ", ".join(c.get('name', '?') for c in added)
        parts.append(f"- Added criteria: {names}")

    if removed:
        names = ", ".join(c.get('name', '?') for c in removed)
        parts.append(f"- Removed criteria: {names}")

    for old_c, new_c in modified:
        name = new_c.get('name', '?')
        changes = []
        if old_c.get('description') != new_c.get('description'):
            changes.append("description updated")
        old_dims = set(d.get('label', '') for d in old_c.get('dimensions', []))
        new_dims = set(d.get('label', '') for d in new_c.get('dimensions', []))
        added_dims = new_dims - old_dims
        removed_dims = old_dims - new_dims
        if added_dims:
            changes.append(f"+dims: {', '.join(added_dims)}")
        if removed_dims:
            changes.append(f"-dims: {', '.join(removed_dims)}")
        if old_c.get('priority') != new_c.get('priority'):
            changes.append(f"priority {old_c.get('priority')} → {new_c.get('priority')}")
        if changes:
            parts.append(f"- Modified \"{name}\": {'; '.join(changes)}")

    return "\n".join(parts)
def display_rubric_assessment(assessment_data, message_id=None, draft_text=None):
    """Display rubric assessment in a card-based layout with dimension checklists and evidence highlights"""
    if not assessment_data:
        return

    st.markdown("---")

    # Initialize feedback storage in session state if needed
    if 'assessment_feedback' not in st.session_state:
        st.session_state.assessment_feedback = {}

    # Use message_id for unique key, fallback to id(assessment_data)
    assessment_key = f"assessment_{message_id}" if message_id else f"assessment_{id(assessment_data)}"
    if assessment_key not in st.session_state.assessment_feedback:
        st.session_state.assessment_feedback[assessment_key] = {}

    # Get assessment summary from JSON
    overall_assessment = None
    criteria_scores = []
    evidence_highlights = []

    if assessment_data.get('json_summary'):
        json_data = assessment_data['json_summary']
        overall_assessment = json_data.get('overall_assessment')
        criteria_scores = json_data.get('criteria_scores', [])
        evidence_highlights = json_data.get('evidence_highlights', [])

    # Helper function to get level info (star-based system)
    def get_level_info(achievement_level):
        if not achievement_level:
            return "#999", "❓"
        level_lower = achievement_level.lower()
        if 'excellent' in level_lower:
            return "#4CAF50", "⭐⭐⭐"
        elif 'good' in level_lower:
            return "#2196F3", "⭐⭐"
        elif 'fair' in level_lower:
            return "#FF9800", "⭐"
        elif 'needs work' in level_lower or 'needs_work' in level_lower:
            return "#E65100", "◇"
        else:  # weak
            return "#F44336", "☆"

    # Display assessment header
    with st.expander("📊 Rubric Assessment", expanded=False):
        # Show thinking if available
        if assessment_data.get('thinking'):
            with st.expander("🧠 Thinking", expanded=False):
                st.markdown(assessment_data['thinking'])

        # Sort criteria by priority
        sorted_criteria = sorted(criteria_scores, key=lambda x: x.get('priority', 99))

        # --- Summary dashboard ---
        if sorted_criteria:
            # Count levels
            level_counts = {"excellent": 0, "good": 0, "fair": 0, "needs work": 0, "weak": 0}
            total_dims_met = 0
            total_dims_all = 0
            for crit in sorted_criteria:
                lvl = crit.get('achievement_level', '').lower()
                for k in level_counts:
                    if k in lvl:
                        level_counts[k] += 1
                        break
                total_dims_met += crit.get('dimensions_met', 0)
                total_dims_all += crit.get('dimensions_total', 0)
            overall_pct = round(total_dims_met / total_dims_all * 100) if total_dims_all > 0 else 0

            # Overall score bar — compute 1-5 rating and label from percentage
            if overall_pct >= 90:
                overall_rating, overall_label, bar_color = 5, "Excellent", "#4CAF50"
            elif overall_pct >= 75:
                overall_rating, overall_label, bar_color = 4, "Good", "#2196F3"
            elif overall_pct >= 50:
                overall_rating, overall_label, bar_color = 3, "Fair", "#FF9800"
            elif overall_pct >= 25:
                overall_rating, overall_label, bar_color = 2, "Needs Work", "#E65100"
            else:
                overall_rating, overall_label, bar_color = 1, "Weak", "#F44336"
            summary_html = f'''<div style="background:linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);border:1px solid #e0e0e0;border-radius:12px;padding:20px;margin-bottom:16px;">
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
  <div>
    <span style="font-size:1.3em;font-weight:700;">Overall Score</span>
    <span style="display:inline-block;margin-left:10px;padding:3px 12px;background:{bar_color}20;color:{bar_color};border-radius:14px;font-size:0.95em;font-weight:700;">{overall_label} ({overall_rating})</span>
  </div>
  <span style="font-size:2em;font-weight:800;color:{bar_color};">{overall_pct}%</span>
</div>
<div style="background:#e9ecef;border-radius:8px;height:12px;overflow:hidden;margin-bottom:14px;">
  <div style="background:{bar_color};height:100%;width:{overall_pct}%;border-radius:8px;transition:width 0.5s;"></div>
</div>
<div style="font-size:0.85em;color:#666;margin-bottom:10px;">{total_dims_met} of {total_dims_all} dimensions met across {len(sorted_criteria)} criteria</div>
<div style="display:flex;gap:6px;flex-wrap:wrap;">'''
            level_configs = [
                ("excellent", "Excellent", "#E8F5E9", "#2E7D32"),
                ("good", "Good", "#E3F2FD", "#1565C0"),
                ("fair", "Fair", "#FFF3E0", "#E65100"),
                ("needs work", "Needs Work", "#FBE9E7", "#BF360C"),
                ("weak", "Weak", "#FFEBEE", "#C62828"),
            ]
            for key, label, bg, fg in level_configs:
                count = level_counts.get(key, 0)
                if count > 0:
                    summary_html += f'<span style="padding:4px 12px;background:{bg};color:{fg};border-radius:16px;font-size:0.85em;font-weight:600;">{count} {label}</span>'
            summary_html += '</div></div>'
            st.markdown(summary_html, unsafe_allow_html=True)

        # Overall assessment narrative
        if overall_assessment:
            st.markdown(f"*{overall_assessment}*")
            st.markdown("")

        # Build a lookup from (criterion, dimension_id) to evidence_highlights quotes
        highlight_lookup = {}
        for ev in evidence_highlights:
            key = (ev.get('criterion', ''), ev.get('dimension_id', ''))
            if key not in highlight_lookup:
                highlight_lookup[key] = []
            highlight_lookup[key].append({
                'quote': ev.get('quote', ''),
                'relevance': ev.get('relevance', '')
            })

        # --- Criteria cards ---
        for crit in sorted_criteria:
            crit_name = crit.get('name', 'Unknown')
            priority = crit.get('priority', 'N/A')
            achievement_level = crit.get('achievement_level', 'N/A')
            dimensions_detail = crit.get('dimensions_detail', [])
            dims_met = crit.get('dimensions_met', 0)
            dims_total = crit.get('dimensions_total', 0)
            improvement_explanation = crit.get('improvement_explanation', '')

            level_color, level_emoji = get_level_info(achievement_level)
            pct = round(dims_met / dims_total * 100) if dims_total > 0 else 0
            # Compute 1-5 rating for this criterion
            crit_pct_frac = dims_met / dims_total if dims_total > 0 else 0
            if crit_pct_frac >= 0.90: crit_rating = 5
            elif crit_pct_frac >= 0.75: crit_rating = 4
            elif crit_pct_frac >= 0.50: crit_rating = 3
            elif crit_pct_frac >= 0.25: crit_rating = 2
            else: crit_rating = 1

            # Criterion card header with progress bar
            card_header = f'''<div style="background:white;border:1px solid #e0e0e0;border-left:4px solid {level_color};border-radius:8px;padding:14px 16px;margin-bottom:4px;">
<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
  <div>
    <span style="font-size:0.75em;color:#888;text-transform:uppercase;letter-spacing:0.5px;">Priority #{priority}</span>
    <div style="font-size:1.05em;font-weight:600;margin-top:2px;">{level_emoji} {crit_name}</div>
  </div>
  <div style="text-align:right;">
    <span style="background:{level_color}20;color:{level_color};padding:3px 10px;border-radius:12px;font-size:0.8em;font-weight:600;">{achievement_level} ({crit_rating})</span>
    <div style="font-size:0.8em;color:#888;margin-top:4px;">{dims_met}/{dims_total} dimensions · {pct}%</div>
  </div>
</div>
<div style="background:#e9ecef;border-radius:6px;height:6px;overflow:hidden;margin-top:10px;">
  <div style="background:{level_color};height:100%;width:{pct}%;border-radius:6px;"></div>
</div>
</div>'''
            st.markdown(card_header, unsafe_allow_html=True)

            with st.expander(f"View details — {crit_name}", expanded=False):
                # Dimension checklist as styled items
                for dim in dimensions_detail:
                    dim_id = dim.get('id', '')
                    dim_label = dim.get('label', 'Unknown dimension')
                    dim_met = dim.get('met', False)
                    dim_evidence = dim.get('evidence', '')
                    linked_highlights = highlight_lookup.get((crit_name, dim_id), [])

                    if dim_met:
                        st.markdown(f'''<div style="display:flex;align-items:flex-start;gap:8px;padding:8px 12px;background:#f0faf0;border-radius:6px;margin-bottom:4px;border:1px solid #c8e6c9;">
<span style="color:#2E7D32;font-size:1.1em;flex-shrink:0;">✅</span>
<div><span style="font-weight:500;">{dim_label}</span></div>
</div>''', unsafe_allow_html=True)
                        evidence_text = ""
                        if linked_highlights:
                            for hl in linked_highlights:
                                quote = hl.get('quote', '')
                                if quote:
                                    evidence_text = quote
                                    break
                        if not evidence_text and dim_evidence:
                            evidence_text = dim_evidence
                        if evidence_text:
                            st.markdown(f'''<div style="margin-left:32px;padding:6px 12px;border-left:3px solid #a5d6a7;color:#555;font-size:0.88em;font-style:italic;margin-bottom:6px;">"{evidence_text}"</div>''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''<div style="display:flex;align-items:flex-start;gap:8px;padding:8px 12px;background:#fef0ef;border-radius:6px;margin-bottom:4px;border:1px solid #ffcdd2;">
<span style="color:#C62828;font-size:1.1em;flex-shrink:0;">❌</span>
<div><span style="font-weight:500;">{dim_label}</span></div>
</div>''', unsafe_allow_html=True)
                        evidence_text = ""
                        if linked_highlights:
                            for hl in linked_highlights:
                                quote = hl.get('quote', '')
                                if quote:
                                    evidence_text = quote
                                    break
                        if not evidence_text and dim_evidence:
                            evidence_text = dim_evidence
                        if evidence_text:
                            st.markdown(f'''<div style="margin-left:32px;padding:6px 12px;border-left:3px solid #ef9a9a;color:#555;font-size:0.88em;font-style:italic;margin-bottom:6px;">"{evidence_text}"</div>''', unsafe_allow_html=True)

                # Improvement explanation
                if improvement_explanation and 'excellent' not in achievement_level.lower():
                    st.markdown(f'''<div style="background:#FFF8E1;border:1px solid #FFE082;border-radius:8px;padding:12px 14px;margin-top:8px;">
<div style="font-weight:600;font-size:0.9em;color:#F57F17;margin-bottom:4px;">💡 To improve</div>
<div style="font-size:0.9em;color:#555;">{improvement_explanation}</div>
</div>''', unsafe_allow_html=True)

        # Evidence Highlights Section
        if evidence_highlights and draft_text:
            st.markdown("---")
            st.markdown("### 📄 Draft with Evidence Highlights")
            st.markdown("*Hover over highlighted text to see the criterion and relevance explanation.*")

            # Build evidence map with actual positions
            all_evidence = []
            for ev in evidence_highlights:
                quote = ev.get("quote", "").strip()
                if quote:
                    # Find actual position by searching for the quote
                    start_idx = draft_text.find(quote)
                    if start_idx == -1:
                        # Try finding first 30 chars as a fallback
                        if len(quote) > 30:
                            start_idx = draft_text.find(quote[:30])

                    if start_idx != -1:
                        all_evidence.append({
                            "criterion": ev.get("criterion", ""),
                            "quote": quote,
                            "start_index": start_idx,
                            "end_index": start_idx + len(quote),
                            "relevance": ev.get("relevance", ""),
                            "dimension_id": ev.get("dimension_id", ""),
                            "dimension_met": ev.get("dimension_met", True)
                        })

            if all_evidence:
                # Sort by start_index
                sorted_evidence = sorted(all_evidence, key=lambda x: x.get("start_index", 0))

                # Remove overlapping highlights
                non_overlapping = []
                last_end = 0
                for ev in sorted_evidence:
                    if ev["start_index"] >= last_end:
                        non_overlapping.append(ev)
                        last_end = ev["end_index"]
                sorted_evidence = non_overlapping

                # Define colors for different criteria
                criterion_colors = {}
                colors = ["#ffeb3b", "#81d4fa", "#a5d6a7", "#ffcc80", "#ce93d8", "#ef9a9a", "#80cbc4"]
                for i, crit in enumerate(sorted_criteria):
                    criterion_colors[crit.get("name", f"Criterion {i+1}")] = colors[i % len(colors)]

                # Build dimension label lookup: (criterion_name, dimension_id) -> dimension_label
                dimension_labels = {}
                for crit in sorted_criteria:
                    crit_name = crit.get("name", "")
                    for dim in crit.get("dimensions_detail", []):
                        dim_id = dim.get("id", "")
                        dim_label = dim.get("label", "")
                        if dim_id:
                            dimension_labels[(crit_name, dim_id)] = dim_label

                # Legend showing criterion colors with colored boxes
                st.markdown("**Legend:**")
                legend_html = '<div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 15px;">'
                for crit in sorted_criteria:
                    crit_name = crit.get("name", "")
                    color = criterion_colors.get(crit_name, "#ffeb3b")
                    legend_html += f'<span style="background-color: {color}; padding: 2px 8px; border-radius: 3px; font-size: 0.9em;">{crit_name}</span>'
                legend_html += '</div>'
                st.markdown(legend_html, unsafe_allow_html=True)

                # Build inline highlighted draft with tooltips (same as Evaluate: Alignment tab)
                tooltip_css = """<style>
.evidence-highlight { position: relative; cursor: help; padding: 1px 2px; border-radius: 2px; }
.evidence-highlight .tooltip-text { visibility: hidden; background-color: #333; color: #fff; text-align: left; padding: 8px 12px; border-radius: 6px; position: absolute; z-index: 1000; bottom: 125%; left: 50%; transform: translateX(-50%); width: 280px; font-size: 0.85em; line-height: 1.4; box-shadow: 0 2px 8px rgba(0,0,0,0.3); }
.evidence-highlight .tooltip-text::after { content: ""; position: absolute; top: 100%; left: 50%; margin-left: -5px; border-width: 5px; border-style: solid; border-color: #333 transparent transparent transparent; }
.evidence-highlight:hover .tooltip-text { visibility: visible; }
.tooltip-criterion { font-weight: bold; color: #ffc107; margin-bottom: 5px; }
</style>"""

                highlighted_html = tooltip_css
                highlighted_html += '<div style="line-height: 1.8; padding: 10px; background-color: #f9f9f9; border-radius: 8px; border: 1px solid #ddd;">'

                last_end = 0
                for ev in sorted_evidence:
                    start = ev.get("start_index", 0)
                    end = ev.get("end_index", 0)
                    criterion = ev.get("criterion", "")
                    dimension_id = ev.get("dimension_id", "")
                    dimension_met = ev.get("dimension_met", True)
                    relevance = ev.get("relevance", "")
                    color = criterion_colors.get(criterion, "#ffeb3b")

                    # Look up dimension label
                    dimension_label = dimension_labels.get((criterion, dimension_id), dimension_id)

                    # Build status indicator
                    status_icon = "✅" if dimension_met else "❌"
                    status_color = "#4CAF50" if dimension_met else "#F44336"

                    if start >= last_end and end > start and start < len(draft_text):
                        # Add unhighlighted text before this highlight
                        highlighted_html += draft_text[last_end:start].replace("\n", "<br>")

                        # Add highlighted text with tooltip showing criterion + dimension + status
                        quote_text = draft_text[start:end].replace("\n", "<br>")
                        tooltip_content = f'<div class="tooltip-criterion">{criterion}</div>'
                        if dimension_label:
                            tooltip_content += f'<div style="color: {status_color}; margin-bottom: 5px;">{status_icon} {dimension_label}</div>'
                        if relevance:
                            tooltip_content += f'<div style="font-size: 0.9em; opacity: 0.9;">{relevance}</div>'

                        highlighted_html += f'<span class="evidence-highlight" style="background-color: {color};">{quote_text}<span class="tooltip-text">{tooltip_content}</span></span>'

                        last_end = end

                # Add remaining text
                if last_end < len(draft_text):
                    highlighted_html += draft_text[last_end:].replace("\n", "<br>")

                highlighted_html += '</div>'

                st.markdown(highlighted_html, unsafe_allow_html=True)
            else:
                # No evidence found in draft text - show without highlights
                st.info("No evidence quotes could be matched to the draft text.")
                st.markdown(draft_text)
        elif draft_text and not evidence_highlights:
            # Show draft without highlights if no evidence_highlights provided
            st.markdown("---")
            st.markdown("### 📄 Draft")
            st.markdown(draft_text)
def simple_markdown_to_html(text):
    """Convert simple markdown formatting to HTML"""
    import re
    # Bold: **text** or __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)
    # Italic: *text* or _text_
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    text = re.sub(r'_(.+?)_', r'<em>\1</em>', text)
    # Line breaks
    text = text.replace('\n', '<br>')
    return text
